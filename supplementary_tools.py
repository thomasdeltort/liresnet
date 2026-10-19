import torch
<<<<<<< HEAD
=======
import yaml
import sys
sys.path.insert(0,"/lustre/fswork/projects/rech/syo/utf64nw/robust-benchmark/liresnet/")
import models  # This imports the models folder from the LiResNet repo

import torch
import torch.nn as nn
import copy

import numpy as np
import torchattacks
import matplotlib.pyplot as plt
from torch.nn.utils.parametrize import is_parametrized

import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader

from models import GloroNet

sys.path.insert(0,"/lustre/fswork/projects/rech/syo/utf64nw/robust-benchmark/SDP-CROWN/")
import auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

sys.path.insert(0,"./..")
from project_utils import *

def compare_logits(model1, model2, inputs, atol=1e-5):
    """
    Compares the logits of two models given the same input batch
    to verify they are mathematically equivalent.
    """
    model1.eval()
    model2.eval()

    with torch.no_grad():
        logits1 = model1(inputs)
        logits2 = model2(inputs)

    # Calculate the maximum absolute difference between corresponding logits
    max_diff = torch.max(torch.abs(logits1 - logits2)).item()
    
    # Check if they match within the specified tolerance
    is_close = torch.allclose(logits1, logits2, atol=atol)

    print("\n--- Logit Comparison ---")
    print(f"Max absolute difference: {max_diff:.8e}")
    print(f"Do the models output the exact same logits? {is_close}")
    
    # Print a small sample to see them side-by-side
    print("\nModel 1 Predictions (first 3):", logits1[:3].argmax(dim=1).tolist())
    print("Model 2 Predictions (first 3):", logits2[:3].argmax(dim=1).tolist())
    
    return is_close, max_diff
>>>>>>> 3150bb3 (adding hybrid verification framework)

def evaluate_clean_accuracy(model, testloader, model_name="Model", device="cuda"):
    """
    Evaluates standard top-1 clean accuracy.
    This identical function works for both the dynamic LiResNet 
    and the static Vanilla PyTorch export.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Standard forward pass (works for both models)
            outputs = model(inputs)
            
            # Get the index of the max logit
            preds = outputs.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    accuracy = 100.0 * correct / total
    print(f"{model_name} Clean Accuracy: {accuracy:.2f}%")
<<<<<<< HEAD
    return accuracy
=======
    return accuracy

def export_model_to_onnx(model, output_path="model.onnx", input_shape=(1, 3, 32, 32), device="cuda"):
    """
    Exports a PyTorch model to ONNX format.
    
    Args:
        model: The loaded PyTorch model.
        output_path: The filename for the exported ONNX model.
        input_shape: The shape of the dummy input (Batch, Channels, Height, Width).
                     Default is (1, 3, 32, 32) for CIFAR-10.
        device: The device the model is currently on ("cuda" or "cpu").
    """
    model.eval() # Ensure model is in inference mode
    
    # Create a dummy input tensor matching the expected input shape
    dummy_input = torch.randn(*input_shape, device=device)
    
    print(f"Exporting model to {output_path}...")
    
    torch.onnx.export(
        model,                         # Model being run
        dummy_input,                   # Model input (or a tuple for multiple inputs)
        output_path,                   # Where to save the model
        export_params=True,            # Store the trained parameter weights inside the model file
        opset_version=14,              # ONNX version to export the model to
        do_constant_folding=True,      # Optimize the ONNX graph
        input_names=['input'],         # Define the input node's name
        output_names=['output'],       # Define the output node's name
        dynamic_axes={                 # Allow dynamic batch sizes during inference
            'input': {0: 'batch_size'},    
            'output': {0: 'batch_size'}
        }
    )
    
    print("ONNX export complete!")

class VanillaMap2Vec(nn.Module):
    """A standard PyTorch equivalent for LiResNet's Map2Vec"""
    def __init__(self, conv, linear, activation):
        super().__init__()
        self.conv = conv
        self.linear = linear
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return self.activation(x)

class VanillaGloroNet(nn.Module):
    """A standard PyTorch equivalent for GloroNet"""
    def __init__(self, stem, conv, neck, linear, head):
        super().__init__()
        self.stem = stem
        self.conv = conv
        self.neck = neck
        self.linear = linear
        self.head = head

    def forward(self, x, return_feat=False):
        x = x.sub(0.5) # LiResNet's built-in input centering
        x = self.stem(x)
        x = self.conv(x)
        x = self.neck(x)
        x = self.linear(x)
        if return_feat:
            return x
        return self.head(x)

class GroupSort_General(nn.Module):
    """
    Applies GroupSort specifically on the channel dimension.
    
    It permutes the input from (N, C, ...) to (N, ..., C), applies the 
    sort logic so that pairs (c_2k, c_2k+1) are sorted, and then restores
    the original layout.
    """
    def __init__(self, axis=1):
        super(GroupSort_General, self).__init__()
        self.axis = axis
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Permute to Channel Last
        # We assume the channel is at self.axis (usually 1).
        # We move that axis to the very end (-1).
        dims = list(range(x.dim()))
        # Remove the channel axis from its current spot and append to end
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        
        # permute returns a view, but we usually need contiguous memory for reshaping
        x_permuted = x.permute(dims).contiguous()
        
        # Capture the shape after permutation: (N, D1, D2, ..., C)
        permuted_shape = x_permuted.shape
        batch_size = permuted_shape[0]
        num_channels = permuted_shape[-1]
        
        # if num_channels % 2 != 0:
        #      raise ValueError(
        #         f"The number of channels must be even, but got {num_channels} "
        #         f"for input shape {x.shape}."
        #     )

        # 2. Flatten for the sorting logic
        # We flatten everything except batch. Since Channel is now last,
        # adjacent elements in this flattened view correspond to adjacent channels.
        x_flat = x_permuted.reshape(batch_size, -1)

        # --- Sort Logic (Verifiable / Auto_Lirpa compatible) ---
        
        # Group into pairs. 
        # Because we are Channel Last, the last dim is C.
        # This reshaping groups (c0, c1), (c2, c3), etc.
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        
        # Calculate diff and apply ReLU to determine min/max without conditional branching
        # min(a,b) = x2 - ReLU(x2 - x1)
        # max(a,b) = x1 + ReLU(x2 - x1)
        diff = x2s + (-1*x1s)
        relu_diff = self.relu(diff)
        
        y1 = x2s + (-1*relu_diff) # The smaller value
        y2 = x1s + relu_diff # The larger value
        
        sorted_pairs = torch.stack((y1, y2), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        
        # --- End Logic ---

        # 3. Restore Shape
        
        # First reshape back to the permuted shape (N, ..., C)
        output_permuted = sorted_flat.reshape(permuted_shape)
        
        # Finally, permute back to Channel First (N, C, ...)
        # We need to calculate the inverse permutation indices
        inv_dims = list(range(x.dim()))
        # Move the last dim (which is now channels) back to self.axis
        last_dim = inv_dims.pop(-1)
        inv_dims.insert(self.axis, last_dim)
        
        output = output_permuted.permute(inv_dims)
        
        return output
        

class CircularConv2d(nn.Module):
    def __init__(self, original_conv):
        super().__init__()
        self.weight = nn.Parameter(original_conv.weight.data.clone())
        if getattr(original_conv, 'bias', None) is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.bias = None
            
        self.stride = original_conv.stride
        self.padding = original_conv.padding[0] if isinstance(original_conv.padding, tuple) else original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        
        self.static_H = None
        self.static_W = None

    def forward(self, x):
        p = self.padding
        if p > 0:
            if self.static_H is None:
                self.static_H = int(x.shape[2])
                self.static_W = int(x.shape[3])
            
            H, W = self.static_H, self.static_W
            x = torch.cat([x[:, :, H-p:H], x, x[:, :, :p]], dim=2)
            x = torch.cat([x[:, :, :, W-p:W], x, x[:, :, :, :p]], dim=3)
            
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

def liresnet_vanilla_export(model):
    """
    Converts a dynamic LiResNet (GloroNet) into a static, ONNX-friendly model.
    """
    model.eval() 
    
    # 1. BAKE THE STEM
    orig_conv = model.stem[0]
    v_stem_conv = nn.Conv2d(
        orig_conv.in_channels, orig_conv.out_channels, 
        orig_conv.kernel_size, stride=orig_conv.stride, 
        padding=orig_conv.padding, bias=(orig_conv.bias is not None)
    )
    v_stem_conv.weight.data = orig_conv.weight.detach().clone()
    if orig_conv.bias is not None:
        v_stem_conv.bias.data = orig_conv.bias.detach().clone()
    v_stem = nn.Sequential(v_stem_conv, copy.deepcopy(model.stem[1]))

    # 2. BAKE THE LIRESCONV (Unpack the depth loop into a Sequential)
    orig_liresconv = model.conv
    weights = orig_liresconv.get_weight().detach()
    biases = (orig_liresconv.bias - orig_liresconv.running_mean).detach()
    
    conv_layers = []
    for i in range(orig_liresconv.depth):
        c = nn.Conv2d(orig_liresconv.width, orig_liresconv.width, kernel_size=3, padding=1)
        c.weight.data = weights[i].clone()
        c.bias.data = biases[i].clone()
        conv_layers.append(c)
        conv_layers.append(copy.deepcopy(orig_liresconv.act))
    v_conv = nn.Sequential(*conv_layers)

    # 3. BAKE MAP2VEC
    orig_neck = model.neck
    kernel, weight = orig_neck.get_weight()
    kernel = kernel.detach()
    weight = weight.detach()
    bias = orig_neck.bias.detach()

    v_neck_conv = nn.Conv2d(orig_neck.feat_dim, orig_neck.feat_dim, kernel_size=4, stride=4, bias=False)
    v_neck_conv.weight.data = kernel.clone()

    v_neck_linear = nn.Linear(weight.shape[1], weight.shape[0])
    v_neck_linear.weight.data = weight.clone()
    v_neck_linear.bias.data = bias.clone()
    
    v_neck = VanillaMap2Vec(v_neck_conv, v_neck_linear, copy.deepcopy(orig_neck.activation))

    # 4. BAKE LIRESMLP (Unpack the depth loop into a Sequential)
    orig_mlp = model.linear
    weights = orig_mlp.eval_weight.detach() # Uses eval_weight as per LiResMLP logic
    biases = orig_mlp.bias.detach()
    
    mlp_layers = []
    for i in range(orig_mlp.depth):
        l = nn.Linear(orig_mlp.width, orig_mlp.width)
        l.weight.data = weights[i].clone()
        l.bias.data = biases[i].clone()
        mlp_layers.append(l)
        mlp_layers.append(copy.deepcopy(orig_mlp.act))
    v_linear = nn.Sequential(*mlp_layers)

    # 5. BAKE THE HEAD
    orig_head = model.head
    v_head = nn.Linear(orig_head.in_features, orig_head.out_features, bias=(orig_head.bias is not None))
    v_head.weight.data = orig_head.get_weight().detach().clone()
    if orig_head.bias is not None:
        v_head.bias.data = orig_head.bias.detach().clone()

    # 6. ASSEMBLE
    vanilla_model = VanillaGloroNet(v_stem, v_conv, v_neck, v_linear, v_head)
    vanilla_model.eval()
    return vanilla_model

# sys.path.insert(0, "./..")


def load_liresnet(config_path, checkpoint_path, device="cuda"):
    # 1. Load the architecture settings from the YAML file
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
        
    model_cfg = cfg['model']
    dataset_cfg = cfg['dataset']
    
    # 2. Instantiate the blank model
    # GloroNet is the main network wrapper used in this repository
    model = models.GloroNet(**model_cfg, **dataset_cfg)
    
    # 3. Load the weights from the checkpoint
    # Map to CPU first to avoid GPU RAM spikes
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # The train.py script saves the weights under the 'backbone' key
    model.load_state_dict(ckpt['backbone'])
    
    # 4. Finalize
    model = model.to(device)
    model.eval()
    
    return model

def vanilla_export(model1):
    model1.eval()
    model2 = copy.deepcopy(model1)
    model2.eval()
    dict_modified_layers = {}
    for (n1,p1), (n2,p2) in zip(model1.named_modules(), model2.named_modules()):
        #print(n1,type(p1), type(p2))
        assert n1 == n2
        if isinstance(p1, torch.nn.Conv2d) and is_parametrized(p1):
            new_conv = torch.nn.Conv2d(p1.in_channels, p1.out_channels, kernel_size=p1.kernel_size, stride=p1.stride, padding=p1.padding, padding_mode=p1.padding_mode,bias=(p1.bias is not None))
            new_conv.weight.data = p1.weight.data.clone()
            if p1.bias is not None:
              new_conv.bias.data = p1.bias.data.clone()
            dict_modified_layers[n2] = new_conv
            #print("modified",n2,type(p1), type(new_conv),p1.in_channels, p1.out_channels, p1.kernel_size[0], p1.stride[0], p1.padding[0], p1.padding_mode,(p1.bias is not None))
            #setattr(model2, n2, new_conv)
            #print(n1,type(p1), type(getattr(model2, n2)))
        if isinstance(p1, torch.nn.Linear) and is_parametrized(p1):
            new_lin = torch.nn.Linear(p1.in_features, p1.out_features, bias=(p1.bias is not None))
            new_lin.weight.data = p1.weight.data.clone()
            new_lin.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            #setattr(model2, n2, new_lin)
            dict_modified_layers[n2] = new_lin
            #print("modified",n2,type(p1), type(new_lin))
            #print(n1,type(p1), type(getattr(model2, n2)))
    for n2, new_layer in dict_modified_layers.items():
        split_hierarchy = n2.split('.')
        lay = model2
        for h in split_hierarchy[:-1]:
            lay = getattr(lay, h)
        # print("modified",n2, type(getattr(lay, split_hierarchy[-1])),type(new_layer))
        setattr(lay, split_hierarchy[-1], new_layer)
    return model2

import time
import torch

def compute_lipschitz_liresnet_vra_and_time(images, targets, model, epsilon, clean_indices, args=None, batch_size=1, return_robust_points=False, **kwargs):
    """
    Computes Certified Robust Accuracy (CRA) utilizing the closed-form Lipschitz 
    margin guarantees of a GloRo-style network.
    
    Note: kwargs absorbs auto_LiRPA specific args like x_U, x_L, and norm 
    so you can hot-swap this function into your existing evaluation loop.
    """
    # Safer device detection
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = images.device
        
    total_num_images = images.shape[0]
    model.eval()
    
    if not isinstance(clean_indices, torch.Tensor):
        clean_indices = torch.tensor(clean_indices)

    # --- Step 1: Filter for correctly classified samples ---
    correct_images = images[clean_indices]
    correct_targets = targets[clean_indices]

    if len(correct_images) == 0:
        if return_robust_points:
            return 0.0, 0.0, torch.tensor([])
        return 0.0, 0.0

    num_robust_points = 0
    total_time = 0.0
    # Lipschitz networks are fast; you can often use a larger batch size here
    batch_size = getattr(args, 'batch_size', batch_size) 
    num_batches = (len(correct_images) + batch_size - 1) // batch_size
    robust_indices_list = []

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time_setup = time.time()

    # --- Step 2: Precompute Lipschitz Constants (Done ONCE) ---
    with torch.no_grad():
        # Get feature extractor Lipschitz constant
        K_sub = model.sub_lipschitz().item() 
        # K_sub = 1
        
        # Get linear head weights
        W = model.head.get_weight()  # Shape: (num_classes, out_dim)
        
        # Vectorized computation of the distance between ALL class weights
        # ||W_c - W_i||_2 -> Output shape: (num_classes, num_classes)
        W_expanded1 = W.unsqueeze(1) # (num_classes, 1, out_dim)
        W_expanded2 = W.unsqueeze(0) # (1, num_classes, out_dim)
        W_dists = torch.linalg.norm(W_expanded1 - W_expanded2, ord=2, dim=2)
        
        # Multiply by K_sub to get the specific Lipschitz constant for every margin
        margin_lipschitz_matrix = K_sub * W_dists
        
        # Fill diagonal with infinity so we ignore the distance from a class to itself
        margin_lipschitz_matrix.fill_diagonal_(float('inf')) # if we do'nt do that, we divide by 0

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time += (time.time() - start_time_setup)

    print(f"Verifying {len(correct_images)} samples using GloRo Lipschitz certificates...")

    # --- Step 3: Batch Loop ---
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(correct_images))
        
        batch_images = correct_images[start_idx:end_idx].to(device)
        batch_targets = correct_targets[start_idx:end_idx].to(device)
        current_bs = batch_images.shape[0]

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_batch = time.time()
        
        with torch.no_grad():
            # 1. Get the logits for the batch
            logits = model(batch_images) # (batch_size, num_classes)
            
            # 2. Extract the logit of the true class for each image
            logits_c = logits.gather(1, batch_targets.unsqueeze(1))
            
            # 3. Calculate margins: y_c - y_i 
            margins = logits_c - logits # for the true class, we get 0
            
            # 4. Fetch the precomputed margin Lipschitz constants
            batch_margin_lipschitz = margin_lipschitz_matrix[batch_targets]
            
            # 5. Divide to get radii (true class will currently be 0.0 or NaN)
            radii_matrix = margins / batch_margin_lipschitz
            
            # 🚨 THE FIX: Overwrite the true class radius with infinity! 🚨
            # This ensures .min() will only look at the OTHER classes.
            radii_matrix.scatter_(1, batch_targets.unsqueeze(1), float('inf'))
            
            # 6. Calculate Certified Radii!
            radii = radii_matrix.min(dim=1).values
            
            # 7. A point is robust if its guaranteed safe radius is >= epsilon
            is_robust = radii >= epsilon
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += (time.time() - start_time_batch)

        num_robust_points += torch.sum(is_robust).item()
        
        if return_robust_points:
            batch_global_indices = clean_indices[start_idx:end_idx]
            robust_indices_list.append(batch_global_indices[is_robust.cpu()])
            
        print(f"  Batch {i+1}/{num_batches}: {torch.sum(is_robust).item()}/{current_bs} robust.", end='\r')

    print("\nBatch verification finished.") 
    
    cra = (num_robust_points / total_num_images) * 100.0
    mean_time_per_image = total_time / len(correct_images) if len(correct_images) > 0 else 0.0

    if return_robust_points:
        all_robust_indices = torch.cat(robust_indices_list) if robust_indices_list else torch.tensor([])
        return cra, mean_time_per_image, all_robust_indices

    return cra, mean_time_per_image

def compute_ibpcrown_vra_and_time(images, targets, model, epsilon, clean_indices, args, batch_size=1, norm=2, return_robust_points=False, x_U=None, x_L=None):
    """
    Computes Certified Robust Accuracy (CRA) using IBP+CROWN.
    
    CRITICAL NOTE: 
    x_L and x_U here should represent the GLOBAL valid data range (e.g. 0 and 1), 
    NOT the local epsilon bounds. The function handles the epsilon intersection internally.
    """
    batch_size = args.batch_size
    # Safer device detection
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # If no parameters, use the device of the input tensor
        device = images.device
    total_num_images = images.shape[0]
    model.eval()
    
    if not isinstance(clean_indices, torch.Tensor):
        clean_indices = torch.tensor(clean_indices)

    # --- Step 1: Filter for correctly classified samples ---
    correct_images = images[clean_indices]
    correct_targets = targets[clean_indices]

    if len(correct_images) == 0:
        if return_robust_points:
            return 0.0, 0.0, torch.tensor([])
        return 0.0, 0.0

    # --- Step 2: Initialize variables ---
    num_robust_points = 0
    total_time = 0.0
    num_batches = (len(correct_images) + batch_size - 1) // batch_size
    robust_indices_list = []

    # --- Step 3: Setup BoundedModule ---
    # Check if the model contains any residual blocks
    has_residuals = any(type(m).__name__ in ['BasicBlockLipschitz', 'BottleneckBlockLipschitz']
                        for m in model.modules())
    
    # Determine the safest and most efficient conv_mode
    selected_conv_mode = "matrix" if has_residuals else "patches"
    
    print(f"Structure check: {'Residuals detected' if has_residuals else 'Sequential architecture'}.")
    print(f"Using auto_LiRPA conv_mode: {selected_conv_mode}")
    
    dummy_input = correct_images[0:1].to(device)
    bounded_model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": selected_conv_mode}, verbose=False)
    bounded_model.eval()

    print(f"Verifying {len(correct_images)} samples in {num_batches} batches using IBP+CROWN...")

    # --- Step 4: Batch Loop ---
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(correct_images))
        
        # Clone to avoid modifying original dataset
        batch_images = correct_images[start_idx:end_idx].clone().to(device)
        batch_targets = correct_targets[start_idx:end_idx]
        current_bs = batch_images.shape[0]

        # --- A. Prepare Global Domain Bounds (0 to 1) ---
        if x_L is not None:
            batch_global_L = x_L.expand(current_bs, *x_L.shape[1:]).contiguous()
        else:
            batch_global_L = None
            
        if x_U is not None:
            batch_global_U = x_U.expand(current_bs, *x_U.shape[1:]).contiguous()
        else:
            batch_global_U = None

        # --- B. CLAMP IMAGES (Crucial for Stability) ---
        if batch_global_L is not None and batch_global_U is not None:
            batch_images = torch.max(torch.min(batch_images, batch_global_U), batch_global_L)

        # --- C. Define Perturbation Constraints ---
        if norm == 'inf' or norm == float('inf'):
            # STRATEGY: TIGHT BOX INTERSECTION
            if batch_global_L is not None and batch_global_U is not None:
                ptb_L = torch.max(batch_global_L, batch_images - epsilon)
                ptb_U = torch.min(batch_global_U, batch_images + epsilon)
                
                ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon, x_L=ptb_L, x_U=ptb_U)
            else:
                ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
                
        else:
            # STRATEGY: GLOBAL BOUNDS + EPSILON SPHERE (Best for L2)
            ptb = PerturbationLpNorm(norm=norm, eps=epsilon, x_L=batch_global_L, x_U=batch_global_U)

        bounded_input = BoundedTensor(batch_images, ptb)
        
        num_classes = 10 
        c = build_C(batch_targets.to("cpu"), num_classes).to(device)

        # --- Time the verification ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_batch = time.time()
        
        lb_diff = bounded_model.compute_bounds(x=(bounded_input,), C=c, method='IBP+CROWN')[0]
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_batch = time.time()
        total_time += (end_time_batch - start_time_batch)

        # --- Check Robustness ---
        is_robust = (lb_diff.view(current_bs, num_classes - 1) > 0).all(dim=1)
        num_robust_points += torch.sum(is_robust).item()
        
        if return_robust_points:
            batch_global_indices = clean_indices[start_idx:end_idx]
            robust_indices_list.append(batch_global_indices[is_robust.cpu()])

        print(f"  Batch {i+1}/{num_batches}: {torch.sum(is_robust).item()}/{current_bs} robust.", end='\r')

    print("\nBatch verification finished.") 
    
    cra = (num_robust_points / total_num_images) * 100.0
    mean_time_per_image = total_time / len(correct_images) if len(correct_images) > 0 else 0.0

    if return_robust_points:
        all_robust_indices = torch.cat(robust_indices_list) if robust_indices_list else torch.tensor([])
        return cra, mean_time_per_image, all_robust_indices

    return cra, mean_time_per_image

import torch.nn as nn

def recursive_unroll(module):
    """
    Recursively unrolls a PyTorch module into a flat list of atomic layers.
    """
    # 1. Base Case: If the module has a 'lipschitz' method, we MUST treat it as 
    # an atomic block. If we break open a residual block (x + f(x)), we destroy 
    # the skip connection and the mathematical Lipschitz bound.
    if hasattr(module, 'lipschitz') and callable(module.lipschitz):
        return [module]
        
    # 2. Base Case: It's a standard layer with no children (Conv2d, Linear, ReLU, etc.)
    children = list(module.children())
    if not children:
        return [module]
        
    # 3. Recursive Step: It's a container (Sequential, ModuleList, or custom container)
    blocks = []
    for child in children:
        blocks.extend(recursive_unroll(child))
        
    return blocks

def unroll_gloronet_deep(model):
    blocks = []
    
    if hasattr(model, 'stem'):
        blocks.extend(recursive_unroll(model.stem))
    if hasattr(model, 'conv'):
        blocks.extend(recursive_unroll(model.conv))
    if hasattr(model, 'neck'):
        blocks.extend(recursive_unroll(model.neck))
    if hasattr(model, 'linear'):
        blocks.extend(recursive_unroll(model.linear))
        
    # NEW: Add the head so it gets included in the slice and LC calculations
    if hasattr(model, 'head'):
        blocks.extend(recursive_unroll(model.head))
        
    return blocks

def unroll_gloronet(model):
    """
    Unrolls the GloroNet into a flat, sequential list of atomic blocks.
    Residual connections inside the blocks are preserved.
    """
    blocks = []
    
    # 1. Stem (Treated as atomic since it has its own .lipschitz())
    blocks.append(model.stem)
    
    # 2. Convolutional Blocks (Unpack LiResConv)
    for child in model.conv.children():
        if isinstance(child, nn.Sequential):
            blocks.extend(list(child.children()))
        else:
            blocks.append(child)
            
    # 3. Neck (Atomic)
    blocks.append(model.neck)
    
    # 4. Linear Blocks (Unpack LiResMLP)
    for child in model.linear.children():
        if isinstance(child, nn.Sequential):
            blocks.extend(list(child.children()))
        else:
            blocks.append(child)
            
    # Note: We omit model.head because sub_lipschitz() typically 
    # evaluates the feature extractor, but you can append it if needed.
    
    return blocks

class UnrolledGloroNetWrapper(nn.Module):
    """
    Wraps an unrolled list of GloroNet blocks and the classification head 
    back into a callable PyTorch Module for accuracy evaluation.
    """
    def __init__(self, blocks, head):
        super().__init__()
        # nn.Sequential magically turns a list of layers into a callable block
        self.features = nn.Sequential(*blocks) 
        self.head = head

    def forward(self, x):
        x = x.sub(0.5)  # Re-apply the crucial LiResNet input centering
        x = self.features(x)
        x = self.head(x)
        return x

def compute_slice_output_and_lc(submodel_blocks, x, is_start_of_network=False):
    """
    Computes the forward pass and exact Lipschitz constant of a network slice.
    
    Args:
        submodel_blocks (list): A list/slice of unrolled nn.Module blocks.
        x (torch.Tensor): The input tensor to this slice.
        is_start_of_network (bool): Set to True if this slice starts from the very 
                                    first layer, so it applies the required x - 0.5.
                                    
    Returns:
        out (torch.Tensor): The output features of the slice.
        combined_lc (float): The exact Lipschitz constant of the slice.
    """
    # GloroNet specifically expects the raw image to be shifted by 0.5 
    if is_start_of_network:
        x = x.sub(0.5)
        
    combined_lc = 1.0
    
    for block in submodel_blocks:
        # --- 1. Forward Pass ---
        # The block executes its own forward(), safely computing any residuals
        x = block(x)
        
        # --- 2. Extract Lipschitz Constant ---
        if hasattr(block, 'lipschitz') and callable(block.lipschitz):
            # The block knows the mathematical bound of its own residual connection
            lc = block.lipschitz()
            combined_lc *= (lc.item() if isinstance(lc, torch.Tensor) else float(lc))
            
        elif hasattr(block, 'weight') and block.weight is not None:
            # Fallback for standard parameterized layers (Conv/Linear)
            weights = block.weight.view(block.weight.shape[0], -1).to(torch.float64)
            lc = torch.linalg.matrix_norm(weights, ord=2)
            combined_lc *= float(lc.item())
            
        else:
            # Non-parameterized layer (ReLU, Pooling, Flatten) -> LC is 1.0
            combined_lc *= 1.0
            
    return x, combined_lc

    import torch
import torch.nn as nn

class ConvIdentityWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # 1x1 Conv that acts as an identity function for 3-channel images
        self.identity = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
        
        # Initialize weights to an identity matrix equivalent
        self.identity.weight.data = torch.zeros(3, 3, 1, 1)
        for i in range(3):
            self.identity.weight.data[i, i, 0, 0] = 1.0
            
        self.identity.weight.requires_grad = False
        self.model = model
        
    def forward(self, x):
        return self.model(self.identity(x))
>>>>>>> 3150bb3 (adding hybrid verification framework)
