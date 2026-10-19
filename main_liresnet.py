import torch
import argparse
import sys
import torch.nn as nn
import time

# Import your custom modules
from supplementary_tools import load_liresnet, liresnet_vanilla_export, compute_lipschitz_liresnet_vra_and_time, compute_alphacrown_vra_and_time, compute_sdp_crown_vra
from project_utils import load_dataset_benchmark_auto, add_result_and_sort, compute_hybrid_vra, get_model_splits
from supplementary_tools import ConvIdentityWrapper

# Assuming compute_hybrid_vra is imported from your tools
# from your_hybrid_module import compute_hybrid_vra

def compute_truncated_network_lipschitz(model, drop_n=1):
    """Computes the mathematically exact Prefix Lipschitz constant."""
    model.eval()
    with torch.no_grad():
        mlp_depth = model.linear.depth if hasattr(model, 'linear') else 0
        has_head = hasattr(model, 'head')
        total_linear = mlp_depth + (1 if has_head else 0)

        if drop_n < 1 or drop_n > total_linear:
            raise ValueError(f"drop_n must be between 1 and {total_linear}")

        K_current = model.sub_lipschitz().item()

        if drop_n == 1:
            return K_current
            
        else:
            mlp_drop_count = drop_n - 1 
            mlp_weights = model.linear.eval_weight.detach().to(torch.float64)
            start_drop_idx = mlp_depth - mlp_drop_count
            
            for i in range(start_drop_idx, mlp_depth):
                W_layer = mlp_weights[i]
                K_layer = torch.linalg.matrix_norm(W_layer, ord=2).item()
                K_current /= K_layer 
                
            return K_current

class Flatten(nn.Module):
    """Explicit flatten module so auto_LiRPA can see the dimensionality change."""
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class PrefixWrapper(nn.Module):
    """Wraps the prefix and restores the crucial x.sub(0.5) centering."""
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x.sub(0.5))

def custom_hybrid_verification(images, targets, classes, original_model, vanilla_model, epsilon, clean_indices, drop_n, args, device):
    """
    A custom hybrid pipeline that perfectly unpacks LiResNet, slices it securely, 
    and verifies the suffix using CROWN.
    """
    # 1. Compute Exact Lipschitz Prefix Bound
    L_prefix = compute_truncated_network_lipschitz(original_model, drop_n)
    
    # 2. FLATTEN THE NETWORK COMPLETELY
    # This prevents the VanillaMap2Vec hidden reshape from crashing CROWN
    flat_layers = []
    
    # Stem
    if isinstance(vanilla_model.stem, nn.Sequential): flat_layers.extend(list(vanilla_model.stem.children()))
    else: flat_layers.append(vanilla_model.stem)
    
    # Conv (LiResConv)
    if isinstance(vanilla_model.conv, nn.Sequential): flat_layers.extend(list(vanilla_model.conv.children()))
    else: flat_layers.append(vanilla_model.conv)
    
    # Neck (Map2Vec - Unpacked manually to expose the Flatten)
    flat_layers.append(vanilla_model.neck.conv)
    flat_layers.append(Flatten())
    flat_layers.append(vanilla_model.neck.linear)
    flat_layers.append(vanilla_model.neck.activation)
    
    # Linear (LiResMLP)
    if isinstance(vanilla_model.linear, nn.Sequential): flat_layers.extend(list(vanilla_model.linear.children()))
    else: flat_layers.append(vanilla_model.linear)
    
    # Head
    flat_layers.append(vanilla_model.head)

    # 3. LOCATE SPLIT INDEX
    linear_indices = [i for i, layer in enumerate(flat_layers) if isinstance(layer, nn.Linear)]
    if drop_n < 1 or drop_n > len(linear_indices):
        raise ValueError(f"Cannot drop {drop_n} layers. Model only has {len(linear_indices)} linear layers.")
        
    split_idx = linear_indices[-drop_n]

    # 4. CREATE PREFIX AND SUFFIX
    f1_prefix = PrefixWrapper(flat_layers[:split_idx]).eval().to(device)
    f2_suffix = nn.Sequential(*flat_layers[split_idx:]).eval().to(device)

    print(f" -> Splitting at layer {split_idx}. Suffix contains {len(flat_layers[split_idx:])} modules.")

    # 5. EXECUTE HYBRID PIPELINE
    start_time = time.time()
    
    with torch.no_grad():
        z_k = f1_prefix(images)
        
    intermediate_epsilon = float(epsilon * L_prefix)
    print(f" -> Intermediate Epsilon for Suffix: {intermediate_epsilon:.6f}")
    
    print(f2_suffix)
    # Verify suffix using auto_LiRPA backend (Alpha-CROWN or IBP-CROWN)
    vra, t_v, idx_robust = compute_sdp_crown_vra(
        z_k, targets, f2_suffix, intermediate_epsilon, 
        clean_indices, torch.device("cuda" if torch.cuda.is_available() else "cpu"), classes, args, batch_size=args.batch_size, return_robust_points=True, x_U=None, x_L=None
    )
    
    total_time = time.time() - start_time
    return vra, total_time, L_prefix


def main():
    parser = argparse.ArgumentParser(description='Custom Hybrid Robustness Evaluation for LiResNet')
    parser.add_argument('--dataset', type=str, default='cifar10_liresnet')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_csv', type=str, default='results/hybrid_study.csv')
    parser.add_argument('--start', default=0, type=int, help='Start index for the dataset')
    parser.add_argument('--end', default=200, type=int, help='End index for the dataset')
    
    # Controls how many linear layers from the end belong to the suffix (e.g., 2)
    parser.add_argument('--drop_n', default=2, type=int, help='Number of linear layers in the suffix.')
    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')
    parser.add_argument('--high_tau', default=False, type=bool, help='Training temperature high/low')
    
    parser.add_argument('--epsilon', type=float, default=0.14118, help='Epsilon radius for verification.')
    parser.add_argument('--norm', type=int, default=2, choices=[2, 'inf'], help='Lp norm for perturbation')
    args = parser.parse_args()

    args.model = "liresnet_cifar10"

    CONFIG_FILE = "configs/cifar10.yaml"
    CHECKPOINT_FILE = "cifar10-12x512_799.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.dataset} dataset...")
    images, targets, classes = load_dataset_benchmark_auto(args)
    images, targets = images.to(DEVICE), targets.to(DEVICE)

    print("Loading PyTorch model...")
    original_model = load_liresnet(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    original_model.eval()
    
    # ==========================================
    # PASTE THE DIAGNOSTIC SNIPPET RIGHT HERE
    # ==========================================
    print("\n" + "="*40)
    print(" ENVIRONMENT & DATA DIAGNOSTICS")
    print("="*40)
    
    # 1. Check Data Preprocessing
    print(f"[Data] Images Shape: {images.shape}")
    print(f"[Data] Images Min:  {images.min().item():.5f}")
    print(f"[Data] Images Max:  {images.max().item():.5f}")
    print(f"[Data] Images Mean: {images.mean().item():.5f}")
    
    # 2. Check Model Weights
    head_weight_sum = original_model.head.weight.sum().item()
    print(f"[Model] Head Weight Sum: {head_weight_sum:.5f}")
    
    # 3. Check Deterministic Forward Pass
    with torch.no_grad():
        dummy_input = torch.ones(1, 3, 32, 32).to(DEVICE)
        dummy_logits = original_model(dummy_input)
        print(f"[Model] Dummy Logits (First 5): {dummy_logits[0, :5].cpu().numpy()}")
        
    print("="*40 + "\n")
    # ==========================================
    
    print(original_model)
    
    model_for_verification = original_model.to(DEVICE)
    model_for_verification = liresnet_vanilla_export(model_for_verification).to(DEVICE)
    model_for_verification.eval()

    # Compute Clean Accuracy
    with torch.no_grad():
        predictions = original_model(images).argmax(dim=1)
        clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]
        clean_acc = (len(clean_indices) / len(targets)) * 100.0
        print(f"\nClean Accuracy: {clean_acc:.2f}% ({len(clean_indices)}/{len(targets)} samples correct)")

    # 1. Closed-Form Lipschitz Verification (Baseline check)
    print(f"\n--- RUNNING LIPSCHITZ CERTIFICATION (EPSILON: {args.epsilon:.5f}) ---")
    cra_val, t_cra = compute_lipschitz_liresnet_vra_and_time(
            images=images, targets=targets, model=original_model, 
            epsilon=args.epsilon, clean_indices=clean_indices, 
            args=args, batch_size=args.batch_size, return_robust_points=False
        )
    print(f"Certified Robust Accuracy (CRA): {cra_val:.2f}%")

    # 2. Custom Hybrid Verification
    print(f"\n--- RUNNING HYBRID VERIFICATION (EPSILON: {args.epsilon:.5f} | CUT: {args.drop_n} linear layers) ---")
    vra, total_time, L_prefix = custom_hybrid_verification(
            images=images, 
            targets=targets,
            classes=classes, 
            original_model=original_model, 
            vanilla_model=model_for_verification, 
            epsilon=args.epsilon, 
            clean_indices=clean_indices, 
            drop_n=args.drop_n, 
            args=args, 
            device=DEVICE
        )
        
    print(f"\nVerification Complete!")
    print(f"Hybrid Verified Robust Accuracy (VRA): {vra:.2f}%")
    print(f"Total Verification Time: {total_time:.2f} seconds")

    result_dict = {
        'drop_n': args.drop_n,        
        'epsilon': args.epsilon, 
        'clean_acc': clean_acc,
        'lipschitz_cra': cra_val, 
        'hybrid_vra': vra, 
        'L_prefix': L_prefix
    }
    add_result_and_sort(result_dict, args.output_csv, norm=args.norm)

if __name__ == '__main__':
    main()