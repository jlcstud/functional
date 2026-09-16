"""
Analyze the number and types of layers in different architectures.
Helps understand baseline dropout and layer composition.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from collections import defaultdict


def analyze_architecture(arch_name: str, model: nn.Module):
    """Analyze and print layer counts for a model."""
    
    layer_counts = defaultdict(int)
    total_params = 0
    
    for module in model.modules():
        module_type = type(module).__name__
        layer_counts[module_type] += 1
        
        # Count parameters
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            total_params += sum(p.numel() for p in module.parameters())
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Architecture: {arch_name}")
    print(f"{'='*60}")
    
    # Group layers by category
    categories = {
        'Convolution': ['Conv1d', 'Conv2d', 'Conv3d'],
        'Linear': ['Linear'],
        'Dropout': ['Dropout', 'Dropout2d', 'Dropout3d'],
        'Normalization': ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d'],
        'Activation': ['ReLU', 'GELU', 'Sigmoid', 'Tanh', 'Softmax'],
        'Pooling': ['MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d'],
        'Other': []
    }
    
    categorized = {cat: 0 for cat in categories}
    uncategorized = defaultdict(int)
    
    for module_type, count in layer_counts.items():
        found = False
        for category, types in categories.items():
            if module_type in types:
                categorized[category] += count
                found = True
                break
        if not found and module_type not in ['Sequential', 'ModuleList', 'ModuleDict']:
            uncategorized[module_type] += count
    
    # Print categorized counts
    print(f"\nLayer Counts by Category:")
    print(f"  Convolution layers:      {categorized['Convolution']:>4d}")
    print(f"  Linear layers:           {categorized['Linear']:>4d}")
    print(f"  Dropout layers:          {categorized['Dropout']:>4d}")
    print(f"  Normalization layers:    {categorized['Normalization']:>4d}")
    print(f"  Activation layers:       {categorized['Activation']:>4d}")
    print(f"  Pooling layers:          {categorized['Pooling']:>4d}")
    
    if uncategorized:
        print(f"\nOther Layers:")
        for module_type, count in sorted(uncategorized.items()):
            print(f"  {module_type:>25s}: {count:>4d}")
    
    print(f"\nTotal modules: {len(layer_counts)}")
    print(f"Total parameters: {total_params:,}")
    
    # Print all layer types if verbose needed
    print(f"\nDetailed Module Count:")
    for module_type in sorted(layer_counts.keys()):
        print(f"  {module_type:>25s}: {layer_counts[module_type]:>4d}")


def main():
    architectures = [
        ("resnet18", lambda: models.resnet18(weights=None)),
        ("resnet50", lambda: models.resnet50(weights=None)),
        ("efficientnet_b0", lambda: models.efficientnet_b0(weights=None)),
        ("vit_b16", lambda: models.vision_transformer.vit_b_16(weights=None)),
    ]
    
    print("\n" + "="*60)
    print("ARCHITECTURE LAYER ANALYSIS")
    print("="*60)
    
    for arch_name, model_fn in architectures:
        try:
            model = model_fn()
            analyze_architecture(arch_name, model)
        except Exception as e:
            print(f"\nError loading {arch_name}: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey Findings:")
    print("- Dropout count shows how much stochasticity is in baseline models")
    print("- Higher dropout → better uncertainty quantification with MC dropout")
    print("- Lower dropout → less computational overhead but less diversity in predictions")
    print("- Most CNN models have MINIMAL dropout (often 0)")
    print("- ViT models typically have more regularization built-in")


if __name__ == "__main__":
    main()
