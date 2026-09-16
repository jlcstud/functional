"""
Model definitions with MC Dropout support and pretrained ImageNet weights.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class MCDropout(nn.Module):
    """MC Dropout wrapper - dropout is enabled at test time too."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        # Always keep this dropout in training mode so it applies stochasticity at inference
        self.dropout.train()
    
    def forward(self, x):
        # Ensure dropout is in training mode (stays enabled at inference)
        self.dropout.train()
        return self.dropout(x)


class ActivationWithDropout(nn.Module):
    """Combines an activation layer with MC Dropout."""
    
    def __init__(self, activation: nn.Module, dropout_p: float = 0.5):
        super().__init__()
        self.activation = activation
        self.dropout = MCDropout(p=dropout_p)
    
    def forward(self, x):
        x = self.activation(x)
        # Ensure dropout stays in training mode for stochasticity at inference
        self.dropout.dropout.train()
        x = self.dropout(x)
        return x


def add_mc_dropout_to_model(model: nn.Module, dropout_p: float = 0.5):
    """Replace all Dropout layers with MCDropout."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], MCDropout(p=module.p))


def add_activation_dropout_to_model(model: nn.Module, dropout_p: float = 0.5):
    """Replace ReLU and SiLU activations with ActivationWithDropout."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.SiLU, nn.GELU, nn.Hardswish)):
            # Get parent module
            parent = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Create new activation with same configuration
            if isinstance(module, nn.ReLU):
                activation_copy = nn.ReLU(inplace=module.inplace)
            elif isinstance(module, nn.SiLU):
                activation_copy = nn.SiLU(inplace=module.inplace)
            elif isinstance(module, nn.GELU):
                activation_copy = nn.GELU(approximate=module.approximate if hasattr(module, 'approximate') else 'none')
            elif isinstance(module, nn.Hardswish):
                activation_copy = nn.Hardswish(inplace=module.inplace)
            else:
                continue
            
            wrapped = ActivationWithDropout(activation_copy, dropout_p=dropout_p)
            setattr(parent, parts[-1], wrapped)


def build_model(
    arch: str,
    num_classes: int = 10,
    pretrained: bool = False,
    dropout_type: str = "none",
    dropout_p: float = 0.5
) -> nn.Module:
    """
    Build a model for CIFAR-10.
    
    Args:
        arch: Model architecture (resnet18, resnet50, efficientnet_b0-b7, vit_b16)
        num_classes: Number of output classes (10 for CIFAR-10)
        pretrained: Use ImageNet pretrained weights
        dropout_type: "none", "mc_dropout", or "standard"
        dropout_p: Dropout probability
    
    Returns:
        Model with replaced classification head if pretrained
    """
    
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif arch == "efficientnet_b0":  # ~5.3M params
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif arch == "efficientnet_b1":  # ~7.8M params
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif arch == "efficientnet_b2":  # ~9.2M params
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif arch == "efficientnet_b3":  # ~12.2M params
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif arch == "efficientnet_b4":  # ~19.3M params
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif arch == "efficientnet_b5":  # ~30.4M params
        weights = models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b5(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif arch == "efficientnet_b6":  # ~43.2M params
        weights = models.EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b6(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif arch == "efficientnet_b7":  # ~66.3M params
        weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b7(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif arch == "vit_b16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vision_transformer.vit_b_16(weights=weights)
        in_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Apply MC Dropout if requested
    if dropout_type == "mc_dropout":
        add_mc_dropout_to_model(model, dropout_p=dropout_p)
    elif dropout_type == "standard":
        # Standard dropout - would need to be added as part of model definition
        pass
    
    return model
