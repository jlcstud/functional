"""
Quick script to count parameters in available models.
Usage: python count_params.py
"""

import torch
from models import build_model


def count_parameters(model):
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def format_params(num_params):
    """Format parameter count in human-readable form."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def main():
    """Initialize all models and print parameter counts."""
    models_to_test = ["resnet18", "resnet50", "efficientnet_b0", "vit_b16"]
    
    print("Model Parameter Counts")
    print("=" * 70)
    print(f"{'Model':<25} {'Pretrained':<15} {'Parameters':<15}")
    print("=" * 70)
    
    for arch in models_to_test:
        for pretrained in [False, True]:
            try:
                model = build_model(
                    arch=arch,
                    num_classes=10,
                    pretrained=pretrained,
                    dropout_type="none"
                )
                num_params = count_parameters(model)
                pretrain_str = "Yes" if pretrained else "No"
                print(f"{arch:<25} {pretrain_str:<15} {format_params(num_params):<15}")
            except Exception as e:
                print(f"{arch:<25} {str(pretrained):<15} Error: {str(e)}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
