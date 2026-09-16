# CIFAR-10 Training - Command Reference

## Setup (One Time)
```bash
pip install -r requirements.txt
python download_cifar10.py
```

## Basic Training Commands

### Default (ResNet-18)
```bash
python train.py
```

### Different Architectures
```bash
python train.py model.arch=resnet50
python train.py model.arch=efficientnet_b0
python train.py model.arch=vit_b16
```

### With Pretrained Weights
```bash
python train.py model.pretrained=true
python train.py model.arch=efficientnet_b0 model.pretrained=true
```

### MC Dropout for Uncertainty
```bash
python train.py model.dropout_type=mc_dropout
python train.py model.dropout_type=mc_dropout model.mc_dropout_p=0.3
```

### Different Random Seeds
```bash
python train.py seed=123
python train.py seed=456
```

## Ensemble Training

### Train N ensemble members
```bash
python train.py num_ensemble=5                                          # 5 members, ResNet-18
python train.py num_ensemble=3 model.arch=efficientnet_b0 model.pretrained=true  # 3 members, EfficientNet pretrained
python train.py num_ensemble=5 model.dropout_type=mc_dropout            # 5 members, MC Dropout
```

## Hyperparameter Overrides

### Training Configuration
```bash
python train.py training.epochs=300              # More epochs
python train.py training.initial_lr=0.2          # Higher learning rate
python train.py training.weight_decay=5e-4       # L2 regularization
python train.py training.momentum=0.95           # SGD momentum
```

### Data Configuration
```bash
python train.py data.batch_size=64               # Smaller batch
python train.py data.batch_size=256              # Larger batch
python train.py data.num_workers=8               # More workers
python train.py data.num_workers=0               # No workers
```

### Validation Schedule
```bash
python train.py training.val_per_epoch=1         # Validate only at end
python train.py training.val_per_epoch=10        # 10 checkpoints per epoch
```

## Complex Examples

### Full setup: EfficientNet with pretraining, ensemble
```bash
python train.py num_ensemble=5 model.arch=efficientnet_b0 model.pretrained=true

# Then load results from logs/ and create ensemble
python ensemble_inference.py
```

### High-variance baseline: ResNet-50, long training, high dropout
```bash
python train.py \
    num_ensemble=3 \
    model.arch=resnet50 \
    model.dropout_type=mc_dropout \
    model.mc_dropout_p=0.4 \
    training.epochs=300
```

### Memory-constrained training
```bash
python train.py \
    model.arch=efficientnet_b0 \
    data.batch_size=32 \
    data.num_workers=2
```

## Output Interpretation

### Training Progress
```
Epoch 5/200 | Loss: 1.2345 | Train Acc: 0.8234 | Val Acc: 0.8156
^           ^              ^                    ^
Epoch progress, Train loss, Train accuracy, Validation accuracy (updated 5x per epoch)
```

### Files Generated
```
logs/2024-01-15/10-30-45/
├── val_logits.csv              # 10,000 samples × 10 logits + 1 label
├── optimal_temperature.txt      # T value for calibration
└── .hydra/
    └── config.yaml              # Exact configuration used
```

## Using Outputs

### Load logits and temperature
```python
import pandas as pd

# Load logits
df = pd.read_csv("logs/2024-01-15/10-30-45/val_logits.csv")
logits = df.iloc[:, :-1].values  # All but last column
labels = df.iloc[:, -1].values   # Last column

# Load temperature
with open("logs/2024-01-15/10-30-45/optimal_temperature.txt") as f:
    temp = float(f.read().split(": ")[1])

# Apply scaling
calibrated_logits = logits / temp
```

### Create ensemble
```python
from ensemble_inference import ensemble_predictions, evaluate_ensemble
import numpy as np

# Load multiple models
logits_list = [load_logits_from_dir(d) for d in model_dirs]
temps_list = [load_temperature_from_dir(d) for d in model_dirs]

# Create ensemble
ensemble_logits = ensemble_predictions(logits_list, temps_list, "mean")

# Evaluate
results = evaluate_ensemble(ensemble_logits, labels)
print(f"Accuracy: {results['accuracy']:.4f}")
```

## Monitoring Training

### Real-time training
```bash
# Terminal 1: Run training
python train.py model.arch=resnet50

# Terminal 2: Monitor logs
watch -n 5 'ls -lh logs/*/*/val_logits.csv'  # Check for output files
```

### Track multiple runs
```bash
# Create ensemble with tracking
for seed in 1 2 3 4 5; do
    echo "Training seed=$seed..."
    python train.py seed=$seed model.arch=resnet50
    echo "Seed=$seed complete"
done
```

## Troubleshooting

### GPU out of memory
```bash
python train.py data.batch_size=32
python train.py model.arch=efficientnet_b0  # Lighter model
```

### Training is slow
```bash
# Check GPU usage: nvidia-smi
# Use fewer workers
python train.py data.num_workers=2
# Or use CPU for debugging only
python train.py device=cpu
```

### Loss is NaN
```bash
# Reduce learning rate
python train.py training.initial_lr=0.01
# Check data isn't corrupted: python download_cifar10.py again
```

### Want to resume old run
```bash
# Configuration is saved, just re-run with same seed/params
python train.py seed=42 model.arch=resnet18
```

## Performance Benchmarks (A100 GPU)

| Model | Batch 128 | Epochs | Time/Epoch | Final Acc |
|-------|-----------|--------|-----------|-----------|
| ResNet-18 | 128 | 200 | ~18min | ~95% |
| ResNet-50 | 128 | 200 | ~28min | ~96% |
| EfficientNet-B0 | 128 | 200 | ~12min | ~96% |
| ViT-B16 | 128 | 200 | ~35min | ~97% |

## Environment Variables

```bash
# To use specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py

# Multiple GPUs (if supported)
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

## Quick Start Checklist

- [ ] `pip install -r requirements.txt`
- [ ] `python download_cifar10.py`
- [ ] `python train.py` (test single model)
- [ ] Check `logs/` for output files
- [ ] `python train.py num_ensemble=3` (train ensemble)
- [ ] Review outputs in `logs/*/`
- [ ] Use `ensemble_inference.py` for combining predictions

## More Information

- Full docs: `README.md`
- Quick start: `QUICKSTART.md`
- Technical details: `TECHNICAL.md`
- Example code: `ensemble_inference.py`
