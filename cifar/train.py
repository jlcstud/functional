"""
Main training script for CIFAR-10 and MNIST with Hydra configuration.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
import logging

from models import build_model, add_activation_dropout_to_model
from utils import (
    TemperatureScaler,
    save_validation_logits,
    save_mc_dropout_logits,
    save_temperature_scaling,
    compute_nll,
    compute_accuracy
)

log = logging.getLogger(__name__)


class CIFAR10NoValidation(CIFAR10):
    """CIFAR10 dataset that skips checksum validation for existing data."""
    
    def _check_integrity(self):
        """Override to skip checksum validation."""
        # Just check if the base folder exists
        return True


class MNISTNoValidation(MNIST):
    """MNIST dataset that skips checksum validation for existing data."""
    
    def _check_integrity(self):
        """Override to skip checksum validation."""
        # Just check if the base folder exists
        return True


class RepeatChannelTo3:
    """Convert single channel to 3 channels by repeating."""
    
    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


def get_data_loaders(cfg: DictConfig):
    """Load data (CIFAR-10 or MNIST) with train/val split, filtering to in-distribution classes."""
    
    # Convert data_dir to absolute path (Hydra changes cwd, so relative paths break)
    data_dir = cfg.data.data_dir
    if not os.path.isabs(data_dir):
        # If relative, resolve from the original working directory
        # Hydra stores the original cwd in HydraConfig
        from hydra.core.hydra_config import HydraConfig
        try:
            hydra_cfg = HydraConfig.get()
            original_cwd = hydra_cfg.runtime.cwd
            data_dir = os.path.join(original_cwd, data_dir)
        except:
            # Fallback: assume data is relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, data_dir)
    
    # Parse id_classes (string of digit indices)
    id_classes_str = cfg.data.get("id_classes", "0123456789")
    id_classes = [int(c) for c in id_classes_str]
    id_classes_set = set(id_classes)
    
    # Get dataset name from config
    dataset_name = cfg.get("dataset", "cifar10").lower()
    
    # Check if using ViT (needs 224x224 input)
    use_resize = cfg.model.arch == "vit_b16"
    
    if dataset_name == "cifar10":
        # CIFAR-10: RGB, 32x32
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if use_resize:
            train_transform_list.append(transforms.Resize(224))
        train_transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])
        train_transform = transforms.Compose(train_transform_list)
        
        test_transform_list = []
        if use_resize:
            test_transform_list.append(transforms.Resize(224))
        test_transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])
        test_transform = transforms.Compose(test_transform_list)
        
        # Load train and test sets using custom class that skips checksum validation
        train_dataset = CIFAR10NoValidation(
            root=data_dir,
            train=True,
            download=False,
            transform=train_transform
        )
        
        test_dataset = CIFAR10NoValidation(
            root=data_dir,
            train=False,
            download=False,
            transform=test_transform
        )
        
        log.info(f"Loaded CIFAR-10 dataset")
        
    elif dataset_name == "mnist":
        # MNIST: Grayscale 28x28 -> convert to RGB, pad to 32x32
        train_transform_list = [
            transforms.Pad(2, fill=0),  # Pad from 28x28 to 32x32 (2 pixels on each edge)
            transforms.RandomHorizontalFlip(),
        ]
        if use_resize:
            train_transform_list.append(transforms.Resize(224))
        train_transform_list.extend([
            transforms.ToTensor(),
            RepeatChannelTo3(),  # Convert 1 channel to 3 channels by repeating
            transforms.Normalize(
                mean=[0.1307, 0.1307, 0.1307],  # MNIST mean, repeated for 3 channels
                std=[0.3081, 0.3081, 0.3081]   # MNIST std, repeated for 3 channels
            ),
        ])
        train_transform = transforms.Compose(train_transform_list)
        
        test_transform_list = []
        if use_resize:
            test_transform_list.append(transforms.Resize(224))
        test_transform_list.extend([
            transforms.ToTensor(),
            RepeatChannelTo3(),  # Convert 1 channel to 3 channels by repeating
            transforms.Normalize(
                mean=[0.1307, 0.1307, 0.1307],  # MNIST mean, repeated for 3 channels
                std=[0.3081, 0.3081, 0.3081]   # MNIST std, repeated for 3 channels
            ),
        ])
        test_transform = transforms.Compose(test_transform_list)
        
        # Load train and test sets using custom class - allow download on first run
        train_dataset = MNIST(
            root=data_dir,
            train=True,
            download=True,  # Allow download to process data if needed
            transform=train_transform
        )
        
        test_dataset = MNIST(
            root=data_dir,
            train=False,
            download=True,  # Allow download to process data if needed
            transform=test_transform
        )
        
        log.info(f"Loaded MNIST dataset")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'cifar10' or 'mnist'")
    
    # Filter training data to only include in-distribution classes
    # Handle both CIFAR10 (list targets) and MNIST (tensor targets)
    targets_data = train_dataset.targets
    if hasattr(targets_data, 'tolist'):
        # Convert tensor to list if needed (MNIST)
        targets_list = targets_data.tolist() if hasattr(targets_data, 'tolist') else list(targets_data)
    else:
        targets_list = list(targets_data)
    
    train_indices = [i for i in range(len(targets_list)) 
                    if targets_list[i] in id_classes_set]
    train_dataset = Subset(train_dataset, train_indices)
    log.info(f"Filtered training set to {len(train_indices)} samples (in-distribution classes: {id_classes_str})")
    
    # Apply data ratio if specified
    data_ratio = cfg.data.get("data_ratio", 1.0)
    if data_ratio < 1.0:
        if not (0 < data_ratio <= 1.0):
            raise ValueError(f"data_ratio must be in (0, 1], got {data_ratio}")
        
        # Use fixed seed to create reproducible subset across models
        rng = np.random.RandomState(0)  # Fixed seed for consistency across runs
        n_train = len(train_dataset)
        n_keep = int(np.ceil(data_ratio * n_train))
        
        # Create a random permutation and take first n_keep indices
        perm = rng.permutation(n_train)
        subset_indices = perm[:n_keep]
        subset_indices = np.sort(subset_indices)  # Sort for consistency
        
        train_dataset = Subset(train_dataset, subset_indices)
        log.info(f"Applied data_ratio={data_ratio}: reduced training set to {len(subset_indices)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_cosine_schedule(optimizer, total_epochs: int, num_batches_per_epoch: int, initial_lr: float):
    """
    Cosine annealing learning rate schedule.
    """
    total_steps = total_epochs * num_batches_per_epoch
    
    def lr_schedule(step):
        return initial_lr * 0.5 * (1 + np.cos(np.pi * step / total_steps))
    
    return lr_schedule


def validate(
    model: nn.Module,
    val_dataset,
    device: str,
    fraction: float = 1.0,
    batch_size: int = 512,
    num_workers: int = 4,
    id_classes: str = "0123456789",
    id_only_head: bool = False,
    data_ratio: float = 1.0
):
    """
    Validate on full or fraction of validation set.
    
    Args:
        fraction: Fraction of validation data to use (0.0 to 1.0)
        id_classes: String of in-distribution class indices
        id_only_head: If True, model has len(id_classes) neurons (no padding). If False, pad logits to 10.
        data_ratio: Additional ratio to apply to validation data (for tensorboard logging). Only applied when fraction < 1.0
    """
    model.eval()
    
    # Parse id_classes
    id_classes_list = [int(c) for c in id_classes]
    id_classes_set = set(id_classes_list)
    class_to_model_idx = {cls: idx for idx, cls in enumerate(id_classes_list)}
    model_idx_to_class = {idx: cls for cls, idx in class_to_model_idx.items()}
    
    # Get subset of data based on fraction and data_ratio
    total_samples = len(val_dataset)
    # Apply data_ratio only when doing partial validation (fraction < 1.0) for tensorboard logging
    effective_fraction = fraction * data_ratio if fraction < 1.0 else fraction
    num_samples = int(total_samples * effective_fraction)
    indices = np.arange(num_samples)
    
    subset = Subset(val_dataset, indices)
    val_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_id_samples = 0
    criterion = nn.CrossEntropyLoss(reduction='none')  # Use reduction='none' to compute per-sample loss
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            
            # For loss computation, handle based on model type
            if id_only_head:
                # Model has reduced outputs
                logits_for_loss = logits
            else:
                # Model has full 10 outputs
                logits_for_loss = logits
            
            # Always expand to full 10-class space for output storage
            batch_size_actual = logits.shape[0]
            if id_only_head:
                # Map reduced logits to full 10-class space
                full_logits = torch.zeros((batch_size_actual, 10), device=device)
                for model_idx, class_idx in model_idx_to_class.items():
                    full_logits[:, class_idx] = logits[:, model_idx]
                logits_for_storage = full_logits
            else:
                logits_for_storage = logits
            
            # Compute loss only for in-distribution samples
            is_id = torch.tensor([l.item() in id_classes_set for l in labels], device=device)
            if is_id.any():
                if id_only_head:
                    # Model has reduced outputs, map labels to model indices
                    id_logits = logits_for_loss[is_id]
                    id_labels_raw = labels[is_id]
                    id_labels = torch.tensor([class_to_model_idx[l.item()] for l in id_labels_raw], device=device)
                else:
                    # Model outputs full 10 classes, use logits and labels directly
                    id_logits = logits_for_loss[is_id]
                    id_labels = labels[is_id]
                loss_per_sample = criterion(id_logits, id_labels)
                total_loss += loss_per_sample.sum().item()
                num_id_samples += loss_per_sample.shape[0]
            
            all_logits.append(logits_for_storage.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Compute validation loss (only for ID samples)
    val_loss = total_loss / num_id_samples if num_id_samples > 0 else 0.0
    
    # Compute validation accuracy (only for ID samples)
    id_mask = np.array([l in id_classes_set for l in labels])
    if id_mask.any():
        if id_only_head:
            # Model has reduced outputs - map GT labels to model indices for accuracy
            id_logits = torch.from_numpy(logits[id_mask]).float()
            id_labels_raw = labels[id_mask]
            id_labels = torch.tensor([class_to_model_idx[int(l)] for l in id_labels_raw]).long()
        else:
            # Model has full 10 outputs
            id_logits = torch.from_numpy(logits[id_mask]).float()
            id_labels = torch.from_numpy(labels[id_mask]).long()
        val_acc = compute_accuracy(id_logits, id_labels)
    else:
        val_acc = 0.0
    
    return val_loss, val_acc, logits, labels


def validate_mc_dropout(
    model: nn.Module,
    val_dataset,
    device: str,
    num_mc_passes: int = 10,
    fraction: float = 1.0,
    batch_size: int = 512,
    num_workers: int = 4,
    id_classes: str = "0123456789",
    id_only_head: bool = False,
    data_ratio: float = 1.0
):
    """
    Validate with MC Dropout - run inference multiple times with different dropout masks.
    
    Args:
        num_mc_passes: Number of stochastic inference passes
        
    Returns:
        List of (logits, labels) tuples, one for each MC pass
    """
    results = []
    
    for pass_idx in range(num_mc_passes):
        val_loss, val_acc, logits, labels = validate(
            model, val_dataset, device, fraction, batch_size, num_workers,
            id_classes, id_only_head, data_ratio
        )
        results.append((logits, labels))
    
    return results


def train_single(cfg: DictConfig, seed: int, member_idx: int, total_members: int):
    """Train a single model (or ensemble member).
    
    Args:
        cfg: Hydra configuration
        seed: Random seed for this training run
        member_idx: Index of this ensemble member (1-indexed)
        total_members: Total number of ensemble members
    """
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = cfg.device if torch.cuda.is_available() else "cpu"
    
    # Log ensemble info if training multiple
    if total_members > 1:
        log.info(f"Training ensemble member {member_idx}/{total_members} (seed={seed})")
    else:
        log.info(f"Using device: {device}")
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directory and TensorBoard writer
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize TensorBoard writer
    tb_dir = output_dir / "runs" / f"member_{member_idx}" if total_members > 1 else output_dir / "runs"
    writer = SummaryWriter(str(tb_dir))
    
    # Load data
    train_loader, val_loader = get_data_loaders(cfg)
    val_dataset = val_loader.dataset  # Get the raw dataset for subset sampling
    
    ood_classes = cfg.data.get("ood_classes", 0)
    id_only_head = cfg.data.get("id_only_head", False)
    
    # Parse id_classes
    id_classes_str = cfg.data.get("id_classes", "0123456789")
    id_classes_list = [int(c) for c in id_classes_str]
    num_id_classes = len(id_classes_list)
    
    # Determine number of output classes
    if id_only_head:
        # Use head with only in-distribution classes
        num_classes = num_id_classes
    else:
        # Default: use full 10 classes (OOD classes are dummy)
        num_classes = 10
    
    # Build model
    model = build_model(
        arch=cfg.model.arch,
        num_classes=num_classes,
        pretrained=cfg.model.pretrained,
        dropout_type=cfg.model.dropout_type,
        dropout_p=cfg.model.mc_dropout_p
    )
    model = model.to(device)
    
    # Add dropout to activations if using MC dropout (except for ViT which has 37 dropout layers already)
    if cfg.model.dropout_type == "mc_dropout" and cfg.model.arch != "vit_b16":
        add_activation_dropout_to_model(model, dropout_p=cfg.model.mc_dropout_p)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    if total_members == 1:
        log.info(f"Built model: {cfg.model.arch} (pretrained={cfg.model.pretrained})")
        log.info(f"Model parameters: {num_params:,}")
        if num_id_classes < 10:
            if id_only_head:
                log.info(f"Training on {num_classes} classes (ID-only head), ID classes: {id_classes_str}")
            else:
                log.info(f"Training on 10 classes (with {10 - num_id_classes} OOD classes as dummy), ID classes: {id_classes_str}")
    
    # Optimizer
    if cfg.training.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
    elif cfg.training.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")
    
    criterion = nn.CrossEntropyLoss()
    lr_schedule_fn = get_cosine_schedule(
        optimizer,
        cfg.training.epochs,
        len(train_loader),
        cfg.training.learning_rate
    )
    
    # Validation schedule: validate at fractions 1/n, 2/n, ..., n/n of epoch
    num_val_checks = cfg.training.val_per_epoch
    steps_per_val_check = len(train_loader) / num_val_checks
    val_fractions = [(i + 1) / num_val_checks for i in range(num_val_checks)]
    
    # Create class index mapping for training
    class_to_model_idx = {cls: idx for idx, cls in enumerate(id_classes_list)}
    
    # Training loop
    total_steps = cfg.training.epochs * len(train_loader)
    step = 0
    save_logits_every_n_epochs = cfg.training.get("save_logits_every_n_epochs", 0)
    
    # Create logits folder for checkpoints if needed
    logits_dir = output_dir / "logits"
    if save_logits_every_n_epochs > 0:
        logits_dir.mkdir(exist_ok=True)
    
    desc = f"Member {member_idx}/{total_members}" if total_members > 1 else "Training"
    with tqdm(total=total_steps, desc=desc, unit="step") as pbar:
        for epoch in range(cfg.training.epochs):
            model.train()
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            next_val_check = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Update learning rate
                current_lr = lr_schedule_fn(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits = model(images)
                
                # If using id_only_head, map labels to model indices
                if id_only_head:
                    model_labels = torch.tensor([class_to_model_idx[l.item()] for l in labels], device=device)
                    loss = criterion(logits, model_labels)
                else:
                    loss = criterion(logits, labels)
                loss.backward()
                
                # Gradient clipping
                if cfg.training.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                
                optimizer.step()
                
                # Track metrics
                if id_only_head:
                    acc = compute_accuracy(logits, model_labels)
                else:
                    acc = compute_accuracy(logits, labels)
                epoch_train_loss += loss.item()
                epoch_train_acc += acc
                
                step += 1
                
                # Perform validation at scheduled fractions
                current_progress = (batch_idx + 1) / len(train_loader)
                if next_val_check < num_val_checks:
                    if current_progress >= val_fractions[next_val_check]:
                        val_fraction = val_fractions[next_val_check]
                        val_loss, val_acc, _, _ = validate(
                            model, val_dataset, device, val_fraction, id_classes=id_classes_str, id_only_head=id_only_head,
                            data_ratio=cfg.data.get("data_ratio", 1.0)
                        )
                        avg_train_loss = epoch_train_loss / (batch_idx + 1)
                        avg_train_acc = epoch_train_acc / (batch_idx + 1)
                        
                        # Log to TensorBoard
                        writer.add_scalar('train/loss', avg_train_loss, step)
                        writer.add_scalar('train/accuracy', avg_train_acc, step)
                        writer.add_scalar('val/loss', val_loss, step)
                        writer.add_scalar('val/accuracy', val_acc, step)
                        writer.add_scalar('learning_rate', current_lr, step)
                        
                        pbar.set_description(
                            f"Epoch {epoch+1}/{cfg.training.epochs} "
                            f"| Loss: {avg_train_loss:.4f} "
                            f"| Train Acc: {avg_train_acc:.4f} "
                            f"| Val Acc: {val_acc:.4f}"
                        )
                        next_val_check += 1
                
                pbar.update(1)
            
            # Save logits at specified intervals
            if save_logits_every_n_epochs > 0 and (epoch + 1) % save_logits_every_n_epochs == 0:
                log.info(f"Saving logits at epoch {epoch + 1}...")
                model.eval()
                
                if cfg.model.dropout_type == "mc_dropout":
                    # MC dropout: save multiple passes
                    num_mc_passes = cfg.model.get("num_mc_passes", 10)
                    mc_results = validate_mc_dropout(
                        model, val_dataset, device, num_mc_passes=num_mc_passes,
                        fraction=1.0, id_classes=id_classes_str, id_only_head=id_only_head
                    )
                    logits_list = [logits for logits, _ in mc_results]
                    checkpoint_labels = mc_results[0][1]  # Labels are the same for all passes
                    
                    # Save to epoch-specific subfolder
                    epoch_folder = logits_dir / f"epoch_{epoch + 1:03d}"
                    epoch_folder.mkdir(exist_ok=True)
                    for pass_idx, logits in enumerate(logits_list):
                        checkpoint_logits_path = epoch_folder / f"logits_{pass_idx}.csv"
                        save_validation_logits(logits, checkpoint_labels, str(checkpoint_logits_path))
                else:
                    # Standard: single pass
                    _, _, checkpoint_logits, checkpoint_labels = validate(
                        model, val_dataset, device, fraction=1.0, id_classes=id_classes_str, id_only_head=id_only_head
                    )
                    checkpoint_logits_path = logits_dir / f"val_logits_epoch_{epoch + 1:03d}.csv"
                    save_validation_logits(checkpoint_logits, checkpoint_labels, str(checkpoint_logits_path))
                
                model.train()  # Return to training mode
    
    # Final evaluation on full validation set
    log.info("Computing final logits on full validation set...")
    model.eval()
    
    if cfg.model.dropout_type == "mc_dropout":
        # MC dropout: save multiple passes
        num_mc_passes = cfg.model.get("num_mc_passes", 10)
        mc_results = validate_mc_dropout(
            model, val_dataset, device, num_mc_passes=num_mc_passes,
            fraction=1.0, id_classes=id_classes_str, id_only_head=id_only_head
        )
        val_logits_list = [logits for logits, _ in mc_results]
        val_labels = mc_results[0][1]  # Labels are the same for all passes
        val_loss, val_acc, _, _ = validate(
            model, val_dataset, device, fraction=1.0, id_classes=id_classes_str, id_only_head=id_only_head
        )  # Compute loss/acc from single pass
    else:
        # Standard: single pass
        val_loss, val_acc, val_logits, val_labels = validate(
            model, val_dataset, device, fraction=1.0, id_classes=id_classes_str, id_only_head=id_only_head
        )
        val_logits_list = [val_logits]
    
    # Log final metrics
    writer.add_scalar('final/val_loss', val_loss, cfg.training.epochs)
    writer.add_scalar('final/val_accuracy', val_acc, cfg.training.epochs)
    writer.flush()
    
    # Save validation logits
    if cfg.model.dropout_type == "mc_dropout":
        # Save MC dropout passes to subfolder structure
        save_mc_dropout_logits(val_logits_list, val_labels, str(output_dir))
    else:
        # Standard: single file
        logits_path = output_dir / "val_logits.csv"
        save_validation_logits(val_logits_list[0], val_labels, str(logits_path))
    
    # Compute and save optimal temperature scaling
    # If using MC dropout, average the logits across passes
    if cfg.model.dropout_type == "mc_dropout":
        logits_for_temp_orig = np.mean(val_logits_list, axis=0)
    else:
        logits_for_temp_orig = val_logits_list[0]
    
    # If using id_only_head, expand logits to full 10-class space for temperature scaling
    if id_only_head:
        # Create mapping from model output indices to original class indices
        class_to_model_idx = {cls: idx for idx, cls in enumerate(id_classes_list)}
        model_idx_to_class = {idx: cls for cls, idx in class_to_model_idx.items()}
        
        # Expand reduced logits to 10-class space using class mapping
        val_logits_expanded = np.zeros((logits_for_temp_orig.shape[0], 10))
        for model_idx, class_idx in model_idx_to_class.items():
            val_logits_expanded[:, class_idx] = logits_for_temp_orig[:, model_idx]
        logits_for_temp = val_logits_expanded
    else:
        logits_for_temp = logits_for_temp_orig
    
    # Only compute temperature on in-distribution samples
    id_classes_set = set(id_classes_list)
    id_mask = np.array([l in id_classes_set for l in val_labels])
    
    scaler = TemperatureScaler(device=device)
    optimal_temp = scaler.find_optimal_temperature(
        torch.from_numpy(logits_for_temp[id_mask]).float().to(device),
        torch.from_numpy(val_labels[id_mask]).long().to(device)
    )
    temp_path = output_dir / "optimal_temperature.txt"
    save_temperature_scaling(optimal_temp, str(temp_path))
    
    # Save model checkpoint if enabled
    if cfg.training.save_model:
        model_path = output_dir / "model.pt"
        torch.save(model.state_dict(), str(model_path))
        log.info(f"Saved model checkpoint to {model_path}")
    
    # Log optimal temperature and close writer
    writer.add_scalar('final/temperature', optimal_temp, 0)
    writer.close()
    
    if total_members == 1:
        log.info(f"Training complete!")
        log.info(f"Final validation accuracy: {val_acc:.4f}")
        log.info(f"Optimal temperature: {optimal_temp:.4f}")
        log.info(f"Output directory: {output_dir.absolute()}")
        log.info(f"TensorBoard logs: tensorboard --logdir {output_dir / 'runs'}")


@hydra_main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Train single model or ensemble of models."""
    
    num_ensemble = cfg.get("num_ensemble", 1)
    base_seed = cfg.seed
    
    if num_ensemble == 1:
        # Single model training
        train_single(cfg, base_seed, 1, 1)
    else:
        # Ensemble training with different seeds
        log.info(f"Training ensemble of {num_ensemble} models")
        log.info(f"Base seed: {base_seed}")
        log.info(f"Seeds: {[base_seed + i for i in range(num_ensemble)]}")
        
        for member_idx in range(num_ensemble):
            member_seed = base_seed + member_idx
            train_single(cfg, member_seed, member_idx + 1, num_ensemble)


if __name__ == "__main__":
    train()
