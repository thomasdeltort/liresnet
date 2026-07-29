import torch
import yaml
import models  # This imports the models folder from the LiResNet repo

import torch
import torch.nn as nn
import copy

from supplementary_tools import *

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

# --- Example Usage ---
if __name__ == "__main__":
    CONFIG_FILE = "configs/cifar10.yaml"              # Update with your config path
    CHECKPOINT_FILE = "cifar10-12x512_799.pth" # Update with your .pth path
    ONNX_OUTPUT = "liresnet_cifar10.onnx"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the PyTorch model
    print("Loading PyTorch model...")
    model = load_liresnet(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    print("Model loaded successfully!")
    model2 = liresnet_vanilla_export(model)
    # 2. Export to ONNX
    export_model_to_onnx(model2, output_path=ONNX_OUTPUT, input_shape=(1, 3, 32, 32), device=DEVICE)

    print(model, model2)

    print("\n--- Evaluating Clean Accuracy ---")
    
    # Evaluate Original LiResNet
    acc1 = evaluate_clean_accuracy(model, testloader, model_name="Model 1 (Original)", device=DEVICE)
    
    # Evaluate Vanilla Export
    acc2 = evaluate_clean_accuracy(model2, testloader, model_name="Model 2 (Vanilla)", device=DEVICE)
    
    # Verify they match perfectly
    assert abs(acc1 - acc2) < 1e-4, "Accuracy mismatch! The export altered the model's predictions."
    print("Success: Both models yield the exact same clean accuracy.")