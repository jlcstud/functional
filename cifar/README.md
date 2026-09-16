# CIFAR-10 Model Training Pipeline

A comprehensive PyTorch training pipeline for CIFAR-10 with support for multiple architectures, MC Dropout, ImageNet pretraining, and advanced training features.

## Features

- **Multiple Architectures**: ResNet-18/50, EfficientNet-B0, Vision Transformer-B16
- **Pretrained Weights**: Load ImageNet pretrained models with replaced classification head
- **MC Dropout**: Enable dropout at test time for uncertainty estimation
- **Cosine Annealing**: Learning rate decay schedule that reaches 0
- **Temperature Scaling**: Automatic optimal temperature scaling computation
- **Validation Schedule**: Configurable validation frequency within each epoch via `val_per_epoch`
- **Logits Export**: Save validation set logits to CSV for ensemble methods
- **Hydra Configuration**: CLI-based hyperparameter management

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CIFAR-10

```bash
python download_cifar10.py
```

This creates a `./data` directory with the CIFAR-10 dataset.

## Usage

### Basic Training

```bash
python train.py
```

### Custom Configuration via CLI

```bash
# Train ResNet-50 with MC Dropout
python train.py model.arch=resnet50 model.dropout_type=mc_dropout

# Use pretrained ImageNet weights
python train.py model.pretrained=true

# Change seed for ensemble member
python train.py seed=123

# Adjust batch size and validation frequency
python train.py data.batch_size=256 training.val_per_epoch=10

# Custom learning rate
python train.py training.initial_lr=0.05
```

### Configuration Override Examples

```bash
# EfficientNet with pretrained weights
python train.py model.arch=efficientnet_b0 model.pretrained=true

# Vision Transformer
python train.py model.arch=vit_b16

# MC Dropout with custom dropout probability
python train.py model.dropout_type=mc_dropout model.mc_dropout_p=0.3

# Different validation schedule (validate 10 times per epoch)
python train.py training.val_per_epoch=10
```

## Configuration

Edit `configs/config.yaml` to customize defaults:

```yaml
seed: 42                    # Base random seed for reproducibility
num_ensemble: 1             # Number of ensemble members (1=single model, >1=trains with seeds: seed, seed+1, ...)
device: cuda                # cuda or cpu

model:
  arch: resnet18            # Model architecture
  pretrained: false         # ImageNet pretraining
  dropout_type: none        # none, mc_dropout
  mc_dropout_p: 0.5         # MC Dropout probability

data:
  batch_size: 128           # Batch size
  num_workers: 4            # DataLoader workers
  data_dir: ./data          # CIFAR-10 data directory

training:
  epochs: 200               # Number of epochs
  initial_lr: 0.1           # Initial learning rate
  weight_decay: 1e-4        # L2 regularization
  momentum: 0.9             # SGD momentum
  val_per_epoch: 5          # Validation checkpoints per epoch (default: 5 = 20% of data each)
```

### Validation Schedule Explanation

The `val_per_epoch` parameter controls validation frequency:
- `val_per_epoch=5` (default): Validate at 20%, 40%, 60%, 80%, 100% of epoch iterations
- `val_per_epoch=10`: Validate at 10%, 20%, ..., 100%
- Each validation checkpoint evaluates the corresponding fraction of the validation set
- A full validation pass (all validation data) occurs after each epoch

## Output

After training, the following files are saved in the output directory:

- `val_logits.csv`: Validation set logits for each class + ground truth labels
- `optimal_temperature.txt`: Optimal NLL temperature scaling value
- `.hydra/config.yaml`: Final configuration used for training
- `.hydra/hydra.yaml`: Hydra execution metadata

### Using Outputs

**Validation Logits** (`val_logits.csv`):
- Use for ensemble methods or additional downstream analysis
- Columns: `logit_0`, `logit_1`, ..., `logit_9`, `label`

**Temperature Scaling** (`optimal_temperature.txt`):
- Scale model logits by dividing: `scaled_logits = logits / temperature`
- Improves calibration on new data

## Training Details

### Learning Rate Schedule

Cosine annealing with warm restart:
```
lr(t) = lr_0 * 0.5 * (1 + cos(π * t / T))
```
where `t` is the current step and `T` is total training steps.

### Progress Bar

Shows:
- Current iteration / total iterations
- Current epoch / total epochs
- Training loss
- Training accuracy
- Validation accuracy

### MC Dropout

When enabled, dropout layers remain active during inference, allowing multiple stochastic forward passes for uncertainty estimation:

```python
model.eval()  # Standard evaluation mode
with torch.no_grad():
    predictions = [model(x) for _ in range(num_samples)]  # Multiple passes
```

## Ensemble Training

To create an ensemble, simply use the `num_ensemble` parameter:

```bash
# Train 5 ensemble members (seeds: 42, 43, 44, 45, 46)
python train.py num_ensemble=5

# Train 3 ensemble members with different base seed
python train.py num_ensemble=3 seed=100  # seeds: 100, 101, 102

# Train 5 ensemble members with MC Dropout
python train.py num_ensemble=5 model.dropout_type=mc_dropout

# Train ensemble with different architecture
python train.py num_ensemble=5 model.arch=efficientnet_b0 model.pretrained=true
```

Each model trains sequentially and creates its own timestamped output directory with `val_logits.csv` and `optimal_temperature.txt` files.

Then combine predictions using `ensemble_inference.py`:

## Device Support

The pipeline automatically detects CUDA availability. To force CPU:

```bash
python train.py device=cpu
```

## Logging

Training progress and results are logged to the console and saved in the output directory via Hydra's configuration system.
