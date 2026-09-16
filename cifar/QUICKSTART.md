# CIFAR-10 & MNIST Training Pipeline - Quick Start Guide

## Installation & Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (one-time)

# For CIFAR-10 (default)
python download_cifar10.py

# OR for MNIST
python download_mnist.py
```

## Dataset Options

### Train on CIFAR-10 (default, RGB 32x32)
```bash
python train.py
```

### Train on MNIST (grayscale 28x28 → converted to RGB 32x32)
```bash
# Download MNIST first (one-time)
python download_mnist.py

# Then train on MNIST
python train.py dataset=mnist
```

Note: MNIST images are automatically:
- Padded from 28x28 to 32x32 (2 pixels black padding on all edges)
- Converted from 1 channel to 3 channels (single channel repeated)
- This makes them compatible with all architectures designed for RGB 32x32 images

## Basic Usage

### Train a single model
```bash
python train.py
```
This trains on CIFAR-10 with default settings (seed=42). Output files:
- `logs/<timestamp>/val_logits.csv` - Validation set logits (10 classes + label)
- `logs/<timestamp>/optimal_temperature.txt` - Temperature for calibration

### Quick experiment with different architecture
```bash
# EfficientNet-B0
python train.py model.arch=efficientnet_b0

# Vision Transformer
python train.py model.arch=vit_b16

# ResNet-50 with pretrained ImageNet weights
python train.py model.arch=resnet50 model.pretrained=true
```

### Enable MC Dropout for uncertainty
```bash
python train.py model.dropout_type=mc_dropout

# With custom dropout probability
python train.py model.dropout_type=mc_dropout model.mc_dropout_p=0.3
```

### Train with different random seed
```bash
python train.py seed=123
```

## Ensemble Training

### Train 5 ensemble members (ResNet-18)
```bash
python train.py num_ensemble=5
```
This trains 5 models sequentially with seeds: 42, 43, 44, 45, 46 (base seed + 0,1,2,3,4)

### Train 3 models with MC Dropout
```bash
python train.py num_ensemble=3 model.dropout_type=mc_dropout
```

### Train 5 EfficientNet models with ImageNet pretraining
```bash
python train.py num_ensemble=5 model.arch=efficientnet_b0 model.pretrained=true
```

### Custom base seed for ensemble
```bash
python train.py num_ensemble=3 seed=100
```
This trains with seeds: 100, 101, 102

## Common Hyperparameter Configurations

### Longer training (300 epochs)
```bash
python train.py training.epochs=300
```

### Higher learning rate
```bash
python train.py training.initial_lr=0.2
```

### Smaller batch size (for limited GPU memory)
```bash
python train.py data.batch_size=64
```

### Validate 10 times per epoch instead of 5
```bash
python train.py training.val_per_epoch=10
```

### Different weight decay
```bash
python train.py training.weight_decay=5e-4
```

## Understanding the Output

### val_logits.csv
```
logit_0,logit_1,...,logit_9,label
-2.341,1.234,...,0.524,3
0.521,-1.231,...,2.341,7
...
```
- Each row is a validation sample
- `logit_i` is the model's output for class i
- `label` is the ground truth class
- Use for ensemble methods, calibration analysis, or uncertainty estimation

### optimal_temperature.txt
```
optimal_temperature: 1.2543
```
To use temperature scaling:
```python
scaled_logits = logits / 1.2543
scaled_probs = softmax(scaled_logits)
```

## Creating an Ensemble

See `ensemble_inference.py` for a complete example:

```python
from pathlib import Path
from ensemble_inference import example_ensemble

model_dirs = [
    Path("logs/2024-01-15/10-30-45"),  # Model 1
    Path("logs/2024-01-15/10-35-22"),  # Model 2
    Path("logs/2024-01-15/10-40-10"),  # Model 3
]

example_ensemble(model_dirs)
```

## MC Dropout Usage

MC Dropout allows dropout to remain active at test time for uncertainty estimation:

```python
import torch
from models import build_model

# Train with MC Dropout enabled
model = build_model(..., dropout_type="mc_dropout")

# At inference time
model.eval()  # Still use eval mode
with torch.no_grad():
    # Make multiple stochastic predictions
    predictions = []
    for _ in range(10):  # 10 MC samples
        logits = model(x)
        predictions.append(logits)
    
    # Combine for uncertainty
    logits_mean = torch.stack(predictions).mean(0)
    logits_std = torch.stack(predictions).std(0)
```

## GPU Memory Troubleshooting

If you run out of GPU memory:

```bash
# Reduce batch size
python train.py data.batch_size=32

# Use CPU (slow!)
python train.py device=cpu

# Reduce num_workers
python train.py data.num_workers=0
```

## All Available Architectures

- `resnet18` - ResNet-18 (11.7M params)
- `resnet50` - ResNet-50 (25.6M params)
- `efficientnet_b0` - EfficientNet-B0 (5.3M params)
- `vit_b16` - Vision Transformer-B16 (86.6M params)

All support `model.pretrained=true` to load ImageNet weights (with replaced classification head).

## Tips for Best Results

1. **Ensemble for better performance**: Use `num_ensemble=3` or higher to train multiple models
2. **Use temperature scaling**: Improves calibration on new data
3. **Adjust validation frequency**: Increase `val_per_epoch` for more frequent monitoring
4. **Pretrained models**: Use `model.pretrained=true` for faster convergence
5. **MC Dropout**: Better uncertainty estimates than standard ensemble methods
6. **Save everything**: All runs automatically save to timestamped directories

## Reproducibility

To reproduce exact results:
1. Use same `seed` value
2. Use same hyperparameter configuration
3. Ensure same CIFAR-10 data version
4. Use deterministic GPU operations (set by default)

## Need Help?

- Check `configs/config.yaml` for all available parameters
- Run `python train.py --help` for Hydra CLI options
- Review README.md for detailed documentation
