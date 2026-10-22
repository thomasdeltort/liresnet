import os
from contextlib import redirect_stdout
import argparse
import time
import json
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- External Libraries ---
import schedulefree

# --- Import your custom modules ---
from supplementary_tools import liresnet_vanilla_export, load_liresnet
from project_utils import load_dataset_benchmark_auto, compute_sdp_crown_vra

# v2.RandAugment(num_ops=2, magnitude=9),
# v2.RandomHorizontalFlip(), 
# v2.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')

def get_cifar10_loaders(batch_size=128):
    """Provides standard CIFAR-10 training data for head fine-tuning."""
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([transforms.RandAugment()]),
    #     transforms.ToTensor(),
    # ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
    ])
    train_dataset = datasets.CIFAR10(root="/lustre/fswork/projects/rech/syo/utf64nw/robust-benchmark/train_models/data/", train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader

def compute_truncated_network_lipschitz(model, drop_n=1):
    """Computes the mathematically exact Prefix Lipschitz constant directly from LiResNet."""
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
    """Explicit flatten module to safely expose shape changes to auto_LiRPA."""
    def forward(self, x): return x.reshape(x.shape[0], -1)

class PrefixWrapper(nn.Module):
    """Wraps the prefix and restores LiResNet's crucial x.sub(0.5) centering."""
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x.sub(0.5))

class RecombinedModel(nn.Module):
    """Combines Prefix and Suffix for holistic inference and saving."""
    def __init__(self, p, s):
        super().__init__()
        self.prefix = p
        self.suffix = s
    def forward(self, x):
        return self.suffix(self.prefix(x))

def get_cifar10_test_loader(batch_size=128):
    """Provides standard CIFAR-10 test data (10,000 images)."""
    transform_test = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root="/lustre/fswork/projects/rech/syo/utf64nw/robust-benchmark/train_models/data", train=False, download=True, transform=transform_test)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def evaluate_full_test_set(prefix, suffix, L_prefix, args, device, run_dir, classes=10):
    print("\n" + "="*50)
    print("  FINAL EVALUATION ON FULL 10,000 TEST SET")
    if args.eval_full_test: print("  (Including SDP-CROWN VRA - This may take a while!)")
    print("="*50)

    test_loader = get_cifar10_test_loader(batch_size=args.batch_size)
    eval_model = RecombinedModel(prefix, suffix).eval()
    
    # Calculate Final Global Lipschitz Constant
    suffix_linears = [m for m in suffix.modules() if isinstance(m, nn.Linear)]
    L_head_final = 1.0
    for l in suffix_linears:
        L_head_final *= torch.linalg.matrix_norm(l.weight, ord=2).item()
    
    L_global_final = L_prefix * L_head_final
    cert_threshold = (2 ** 0.5) * args.epsilon * L_global_final
    intermediate_epsilon = float(args.epsilon * L_prefix)
    
    total_samples, correct_clean, correct_cra, correct_vra = 0, 0, 0, 0

    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        bs = targets.size(0)
        total_samples += bs
        
        with torch.no_grad():
            # 1. Forward Pass & Clean Accuracy
            logits = eval_model(data)
            preds = logits.argmax(dim=1)
            is_correct = (preds == targets)
            correct_clean += is_correct.sum().item()
            
            # 2. Closed-Form CRA
            logits_true = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            logits_clone = logits.clone().scatter_(1, targets.unsqueeze(1), -float('inf'))
            margins = logits_true - logits_clone.max(dim=1)[0]
            
            is_robust_cra = is_correct & (margins > cert_threshold)
            correct_cra += is_robust_cra.sum().item()
            
            # 3. Hybrid VRA (SDP-CROWN) - Corrected Batch Pre-Slicing
            if args.eval_full_test:
                clean_indices_local = is_correct.nonzero(as_tuple=True)[0]
                
                if len(clean_indices_local) > 0:
                    # Pre-slice latent features and targets so the solver only processes clean points
                    z_k_clean = prefix(data)[clean_indices_local]
                    targets_clean = targets[clean_indices_local]
                    
                    # Local indices for the pre-sliced batch are simply 0 to len(clean_indices_local) - 1
                    solver_indices = torch.arange(len(clean_indices_local), device=device)
                    
                    with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
                        batch_vra_percent, _, _ = compute_sdp_crown_vra(
                            z_k_clean, targets_clean, suffix, intermediate_epsilon, 
                            solver_indices, device, classes, args, 
                            batch_size=len(clean_indices_local), return_robust_points=True, 
                            x_U=None, x_L=None
                        )
                        
                    # Convert returned percentage of CLEAN points back to absolute counts
                    n_robust_points = int(round((batch_vra_percent / 100.0) * len(clean_indices_local)))
                    correct_vra += n_robust_points
                    
        print(f"Processed {total_samples}/10000 images...", end='\r')

    print(f"\n\n[*] Final Global Lipschitz Constant: {L_global_final:.4f}")
    print(f"[*] Full Test Set Clean Accuracy : {(correct_clean / total_samples) * 100.0:.2f}% ({correct_clean}/{total_samples})")
    print(f"[*] Full Test Set Analytical CRA : {(correct_cra / total_samples) * 100.0:.2f}% ({correct_cra}/{total_samples})")
    
    if args.eval_full_test:
        print(f"[*] Full Test Set Hybrid VRA     : {(correct_vra / total_samples) * 100.0:.2f}% ({correct_vra}/{total_samples})")
    print("="*50 + "\n")

    # --- Save Results to CSV ---
    final_clean = (correct_clean / total_samples) * 100.0
    final_cra = (correct_cra / total_samples) * 100.0
    final_vra = (correct_vra / total_samples) * 100.0 if args.eval_full_test else "N/A"
    
    results_file = os.path.join(run_dir, 'final_test_metrics.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Clean_Accuracy', 'Analytical_CRA', 'Hybrid_VRA', 'Global_Lipschitz'])
        writer.writerow([
            f"{final_clean:.2f}", 
            f"{final_cra:.2f}", 
            f"{final_vra:.2f}" if isinstance(final_vra, float) else final_vra, 
            f"{L_global_final:.4f}"
        ])
    print(f"[*] Final full-test results saved to: {results_file}\n")

def main():
    parser = argparse.ArgumentParser(description="LiResNet Head Fine-Tuning & SDP Verification")
    
    # Model & Config
    parser.add_argument('--config_path', type=str, required=True, help="Path to LiResNet YAML config")
    parser.add_argument('--model_path', type=str, required=True, help="Path to initial .pth checkpoint")
    parser.add_argument('--dataset', type=str, default='cifar10_liresnet')
    parser.add_argument('--save_dir', type=str, default='./saved_models/liresnet_head')
    
    # Training Arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4, help="Weight decay for the head")
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Architecture Splitting
    parser.add_argument('--drop_n', default=2, type=int, help='Number of linear layers in the trainable Suffix.')
    parser.add_argument('--replace_relu', action='store_true', help='Replace non-linear activations in the suffix with ReLU.')
    
    # Verification Arguments
    parser.add_argument('--val_freq', type=int, default=20, help="Run formal verification every X epochs")
    parser.add_argument('--epsilon', type=float, default=0.14118)
    parser.add_argument('--norm', type=int, default=2)
    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')
    parser.add_argument('--high_tau', default=False, type=bool, help='Training temperature high/low')
    parser.add_argument('--eval_full_test', action='store_true', help='Run full 10k test evaluation (including SDP) at the end of training.')
    
    args = parser.parse_args()
    # Mocking arg for auto_LiRPA utils compatibility
    args.model = "liresnet_cifar10" 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. Setup & Directory Initialization
    # ==========================================
    run_name = f"LiRes_Head_Drop{args.drop_n}_WD{args.wd}_Eps{args.epsilon}_{int(time.time())}"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    metrics_file = os.path.join(run_dir, 'metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Bench_Clean', 'Bench_CRA', 'Bench_Hybrid_VRA'])

    print(f"[*] Starting Run: {run_dir}")

    # ==========================================
    # 2. Load Models & Calculate Lipschitz
    # ==========================================
    print(f"[*] Loading LiResNet Architecture...")
    original_model = load_liresnet(args.config_path, args.model_path, device=device)
    
    # Compute mathematically exact prefix bound before vanillizing
    L_prefix = compute_truncated_network_lipschitz(original_model, drop_n=args.drop_n)
    print(f"[*] Computed Prefix LC (frozen backbone): {L_prefix:.6f}")

    print("[*] Exporting to Vanilla PyTorch for Verification & Splitting...")
    vanilla_model = original_model.to(device)
    vanilla_model = liresnet_vanilla_export(vanilla_model).to(device)
    
#    # ==========================================
#    # 3. Physically Split the Network
#    # ==========================================
#    all_blocks = [
#        vanilla_model.stem,
#        vanilla_model.conv,
#        vanilla_model.neck.conv,
#        Flatten(),
#        vanilla_model.neck.linear,
#        vanilla_model.neck.activation
#    ]
#    for layer in vanilla_model.linear: 
#        all_blocks.append(layer)
#    all_blocks.append(vanilla_model.head)
#
#    # Locate cut point
#    linear_indices = [i for i, m in enumerate(all_blocks) if isinstance(m, nn.Linear)]
#    split_idx = linear_indices[-args.drop_n]
#
#    prefix = PrefixWrapper(all_blocks[:split_idx]).to(device)
#    suffix = nn.Sequential(*all_blocks[split_idx:]).to(device)
#
#    # Freeze Backbone, Train Suffix
#    for param in prefix.parameters(): param.requires_grad = False
#    for param in suffix.parameters(): param.requires_grad = True
#
#    print(f"[*] Split complete. Trainable Suffix contains {len(all_blocks[split_idx:])} modules.")

    # ==========================================
    # 3. Physically Split the Network & Replace GS2 with ReLU
    # ==========================================
    all_blocks = [
        vanilla_model.stem,
        vanilla_model.conv,
        vanilla_model.neck.conv,
        Flatten(),
        vanilla_model.neck.linear,
        vanilla_model.neck.activation
    ]
    for layer in vanilla_model.linear: 
        all_blocks.append(layer)
    all_blocks.append(vanilla_model.head)

    # Locate cut point
    linear_indices = [i for i, m in enumerate(all_blocks) if isinstance(m, nn.Linear)]
    split_idx = linear_indices[-args.drop_n]

    # Build Prefix (Frozen Backbone)
    prefix_modules = all_blocks[:split_idx]
    prefix = PrefixWrapper(prefix_modules).to(device)

    # Build Suffix (Trainable Head) & (Optionally) Replace GS2 with ReLU
    suffix_modules = []
    for m in all_blocks[split_idx:]:
        # If the flag is ON and the module is a non-linear activation, replace with ReLU
        if args.replace_relu and not isinstance(m, (nn.Linear, Flatten)):
            suffix_modules.append(nn.ReLU())
        else:
            suffix_modules.append(m)

    suffix = nn.Sequential(*suffix_modules).to(device)

    # Freeze Backbone, Train Suffix
    for param in prefix.parameters(): param.requires_grad = False
    for param in suffix.parameters(): param.requires_grad = True

    replacement_status = "with ReLU activations" if args.replace_relu else "with original activations"
    print(f"[*] Split complete. Trainable Suffix contains {len(suffix_modules)} modules {replacement_status}.")
    print(f"[*] Suffix Architecture:\n{suffix}")

    # ==========================================
    # 4. Training Setup (Using ScheduleFree)
    # ==========================================
    # Use AdamWScheduleFree strictly on the suffix
    optimizer = schedulefree.AdamWScheduleFree(suffix.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    train_loader = get_cifar10_loaders(args.batch_size)

    # Prepare Benchmark Dataset
    args.start = 0
    args.end = 200
    bench_images, bench_targets, classes = load_dataset_benchmark_auto(args)
    bench_images, bench_targets = bench_images.to(device), bench_targets.to(device)

    # ==========================================
    # 5. Main Training Loop
    # ==========================================
    for epoch in range(1, args.epochs + 1):
        prefix.eval() # Always frozen
        suffix.train()
        optimizer.train() # CRITICAL for ScheduleFree
        
        total_loss, correct, total = 0, 0, 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                features = prefix(data)
                
            outputs = suffix(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == target).sum().item()
            total += target.size(0)
            
        train_acc = 100. * correct / total
        print(f"Epoch [{epoch}/{args.epochs}] | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

        # --- VALIDATION PHASE ---
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            suffix.eval()
            optimizer.eval() # CRITICAL for ScheduleFree inference
            print(f"\n--- Running Verification Benchmark (Epoch {epoch}) ---")
            
            eval_model = RecombinedModel(prefix, suffix).eval()
            
            with torch.no_grad():
                preds = eval_model(bench_images).argmax(dim=1)
                clean_indices = (preds == bench_targets).nonzero(as_tuple=True)[0]
                bench_clean = (len(clean_indices) / len(bench_targets)) * 100.0
            print(f"  > Clean Accuracy: {bench_clean:.2f}%")

            # 1. Closed-Form CRA 
            # We recalculate the global LC dynamically using the newly trained suffix weights
            suffix_linears = [m for m in suffix.modules() if isinstance(m, nn.Linear)]
            L_head_new = 1.0
            for l in suffix_linears:
                L_head_new *= torch.linalg.matrix_norm(l.weight, ord=2).item()
            
            L_global_new = L_prefix * L_head_new
            cert_threshold = (2 ** 0.5) * args.epsilon * L_global_new
            
            with torch.no_grad():
                logits = eval_model(bench_images)
                logits_true = logits.gather(1, bench_targets.unsqueeze(1)).squeeze(1)
                logits_clone = logits.clone().scatter_(1, bench_targets.unsqueeze(1), -float('inf'))
                margins = logits_true - logits_clone.max(dim=1)[0]
                
                is_robust = (preds == bench_targets) & (margins > cert_threshold)
                bench_cra = (is_robust.sum().item() / len(bench_targets)) * 100.0
            print(f"  > Analytical CRA: {bench_cra:.2f}% (Global LC: {L_global_new:.4f})")

            # 2. Hybrid VRA (SDP-CROWN)
            intermediate_epsilon = float(args.epsilon * L_prefix)
            
            with torch.no_grad():
                z_k = prefix(bench_images)

            # Mute all prints from SDP-CROWN
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):    
                bench_vra, _, _ = compute_sdp_crown_vra(
                    z_k, bench_targets, suffix, intermediate_epsilon, 
                    clean_indices, device, classes, args, 
                    batch_size=args.batch_size, return_robust_points=True, 
                    x_U=None, x_L=None
                )
            print(f"  > Hybrid VRA    : {bench_vra:.2f}%\n")

            # Log to CSV
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{total_loss/len(train_loader):.4f}", f"{train_acc:.2f}", f"{bench_clean:.2f}", f"{bench_cra:.2f}", f"{bench_vra:.2f}"])

            # Save Checkpoint
            ckpt_path = os.path.join(run_dir, f"model_ep{epoch:03d}.pth")
            torch.save({'state_dict': eval_model.state_dict(), 'config': vars(args)}, ckpt_path)
            
    print(f"[*] Training Complete. Logs and models saved to {run_dir}")
    
    # ==========================================
    # Final Evaluation on Full Test Set
    # ==========================================
    prefix.eval()
    suffix.eval()
    evaluate_full_test_set(prefix, suffix, L_prefix, args, device, run_dir, classes=10)

if __name__ == '__main__':
    main()