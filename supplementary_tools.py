import torch

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
    return accuracy