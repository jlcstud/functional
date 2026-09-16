# Technical Documentation

## Architecture

### Project Structure

```
cifar/
├── configs/
│   └── config.yaml           # Hydra configuration with all hyperparameters
├── models/
│   └── __init__.py           # Model building functions with MC Dropout support
├── data/
│   └── __init__.py           # Data loading module (currently empty)
├── download_cifar10.py       # One-time CIFAR-10 download script
├── train.py                  # Main training script (Hydra-based, supports ensemble)
├── utils.py                  # Temperature scaling and logits saving utilities
├── ensemble_inference.py     # Multi-model ensemble inference example
├── requirements.txt          # Python dependencies
├── README.md                 # Full documentation
├── QUICKSTART.md             # Quick start guide
└── .gitignore               # Git ignore patterns
```

## Key Components

### 1. Model Building (`models/__init__.py`)

**Supported Architectures:**
- ResNet-18/50: Standard CNNs, ImageNet pretrained available
- EfficientNet-B0: Lightweight, efficient CNN
- Vision Transformer-B16: Transformer-based vision model

**MC Dropout Implementation:**
- `MCDropout`: Custom dropout that applies dropout at both train and test time
- `add_mc_dropout_to_model()`: Replaces all standard Dropout layers with MCDropout
- Enables uncertainty estimation through multiple stochastic forward passes

**Head Replacement:**
- When `pretrained=true`, loads ImageNet weights
- Replaces final classification layer to output 10 classes (CIFAR-10)
- Allows fine-tuning with pretrained features

### 2. Training Loop (`train.py`)

**Learning Rate Schedule:**
```
lr(t) = lr_0 * 0.5 * (1 + cos(π * t / T))
```
- Reaches 0 at the end of training (T = total_steps)
- No early stopping - trains for full number of epochs
- Smooth decay suitable for SGD with momentum

**Validation Schedule:**
- `val_per_epoch` parameter controls validation frequency
- Default `val_per_epoch=5` means:
  - 5 validation checkpoints during each epoch
  - At 20%, 40%, 60%, 80%, 100% of epoch progress
  - Each checkpoint evaluates the corresponding fraction of validation data
  - Example: at 20% epoch progress, validate on 20% of validation set
  - By epoch end, all validation data has been validated

**Progress Bar:**
```
Epoch 5/200 | Loss: 1.2345 | Train Acc: 0.8234 | Val Acc: 0.8156
```
- Updates every batch
- Total iterations shown as cumulative across all epochs
- Displays real-time metrics during training

**Data Augmentation:**
- Training: Random crop (32px with 4px padding), random horizontal flip
- Validation: No augmentation
- Normalization: CIFAR-10 standard (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

### 3. Temperature Scaling (`utils.py`)

**Purpose:** Calibrate model confidence after training

**Method:**
- Finds temperature T that maximizes likelihood on validation set
- Uses L-BFGS optimizer for fast convergence
- Applied as: `scaled_logits = original_logits / T`
- T > 1: Model is overconfident
- T < 1: Model is underconfident

**Optimal Temperature Computation:**
```python
scaler = TemperatureScaler(device)
optimal_temp = scaler.find_optimal_temperature(val_logits, val_labels)
```

### 4. Ensemble Inference (`ensemble_inference.py`)

**Aggregation Methods:**

1. **Mean Logits** (default)
   - Average logits: `E_logits = mean(scaled_logits_1, scaled_logits_2, ...)`
   - Simple, fast, often effective

2. **Product of Probabilities**
   - Convert to probabilities: `p_i = softmax(logits_i)`
   - Multiply: `P_ensemble = ∏ p_i / Z` (normalized)
   - Geometric mean in probability space
   - Better uncertainty calibration

3. **Max Logits**
   - Takes maximum logit across models
   - Less commonly used

## Hydra Configuration System

**Benefits:**
- Override any parameter from CLI: `python train.py param=value`
- Automatic output directory creation with timestamp
- Config composition and groups (extendable)
- Reproducibility: config saved in output directory

**Default Config Structure:**
```yaml
seed: 42                              # Random seed
device: cuda                          # Device

model:
  arch: resnet18                      # Architecture choice
  pretrained: false                   # ImageNet pretraining
  dropout_type: none                  # MC dropout switch
  mc_dropout_p: 0.5                   # Dropout rate

data:
  batch_size: 128
  num_workers: 4
  data_dir: ./data

training:
  epochs: 200
  initial_lr: 0.1
  weight_decay: 1e-4
  momentum: 0.9
  val_per_epoch: 5                    # Validation frequency
```

## Output Files

### val_logits.csv
CSV with one row per validation sample:
- `logit_0` to `logit_9`: Model output for each class
- `label`: Ground truth class label

Useful for:
- Post-hoc calibration
- Ensemble methods
- Uncertainty analysis
- ROC curves and other metrics

### optimal_temperature.txt
Single line format:
```
optimal_temperature: 1.2543
```

Usage:
```python
temperature = 1.2543
calibrated_probs = softmax(logits / temperature)
```

### Hydra Outputs
- `.hydra/config.yaml`: Exact configuration used
- `.hydra/hydra.yaml`: Execution metadata
- Timestamp-based directory structure in `logs/`

## Performance Considerations

### GPU Memory
- ResNet-18: ~4GB at batch_size=128
- EfficientNet-B0: ~3GB at batch_size=128
- ViT-B16: ~6GB at batch_size=128

### Training Time
- ResNet-18: ~15-20 minutes per epoch on A100
- ResNet-50: ~25-30 minutes per epoch on A100
- EfficientNet-B0: ~10-15 minutes per epoch on A100
- ViT-B16: ~30-40 minutes per epoch on A100

### Validation Overhead
- `val_per_epoch=5` adds ~5-10% overhead
- Each validation fraction takes proportional time

## Reproducibility

All components support reproducibility:

1. **Random Seeds:**
   - `np.random.seed(seed)`
   - `torch.manual_seed(seed)`
   - `torch.cuda.manual_seed(seed)`

2. **Deterministic Operations:**
   - DataLoader uses same seed for shuffling
   - CUDA determinism enabled by default

3. **Config Saving:**
   - Complete config saved in output directory
   - Exact hyperparameters available for reference

## Extensions

### Adding New Architectures
Edit `models/__init__.py`:
```python
elif arch == "my_new_model":
    model = build_my_model(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
```

### Custom Augmentation
Edit `get_data_loaders()` in `train.py`:
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # Add custom augmentations here
    transforms.ToTensor(),
    ...
])
```

### Advanced Ensemble Methods
Extend `ensemble_inference.py` with custom aggregation:
```python
def custom_ensemble(logits_list, weights=None):
    # Custom weighted ensemble
    if weights is None:
        weights = [1.0] * len(logits_list)
    
    weighted_sum = sum(w * l for w, l in zip(weights, logits_list))
    return weighted_sum / sum(weights)
```

## Debugging

### Check data loading
```python
from train import get_data_loaders
train_loader, val_loader = get_data_loaders(cfg)
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)
```

### Verify model
```python
from models import build_model
model = build_model("resnet18", pretrained=False)
print(model)
```

### Test temperature scaling
```python
from utils import TemperatureScaler
scaler = TemperatureScaler("cpu")
temp = scaler.find_optimal_temperature(test_logits, test_labels)
print(f"Optimal temperature: {temp}")
```

## Common Issues & Solutions

**Issue:** CUDA out of memory
- Solution: Reduce `batch_size` or use smaller model (`efficientnet_b0`)

**Issue:** Slow training with CPU
- Solution: Ensure `device: cuda` in config, check GPU is available

**Issue:** Unstable training (loss NaN)
- Solution: Reduce learning rate (`training.initial_lr=0.01`)

**Issue:** Validation doesn't improve
- Solution: Increase training time (`training.epochs=300`), check data quality

**Issue:** Weights not loading with pretrained=true
- Solution: Ensure torchvision version matches requirements.txt

## References

- PyTorch: https://pytorch.org/
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- Hydra: https://hydra.cc/
- Temperature Scaling: https://arxiv.org/abs/1706.04599
- MC Dropout: https://arxiv.org/abs/1506.02142
