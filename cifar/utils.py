"""
Utility functions for temperature scaling and logits saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


class TemperatureScaler(nn.Module):
    """Temperature scaling for calibration."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, device=device))
        self.device = device
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
    
    def find_optimal_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4
    ) -> float:
        """
        Find optimal temperature via maximum likelihood on validation set.
        
        Args:
            logits: (N, C) tensor of logits
            labels: (N,) tensor of labels
            lr: Learning rate
            max_iter: Maximum iterations
            tol: Convergence tolerance
        
        Returns:
            Optimal temperature value
        """
        self.temperature.data.fill_(1.0)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        return self.temperature.item()


def save_validation_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    output_path: str
) -> None:
    """
    Save validation logits to CSV with full precision.
    
    Args:
        logits: (N, C) array of logits
        labels: (N,) array of labels
        output_path: Path to save CSV file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with logits for each class
    data = {f"logit_{i}": logits[:, i] for i in range(logits.shape[1])}
    data["label"] = labels
    
    df = pd.DataFrame(data)
    # Save with full float64 precision (17 significant digits)
    df.to_csv(output_path, index=False, float_format='%.17g')
    print(f"Saved logits to {output_path}")


def save_mc_dropout_logits(
    logits_list: list,
    labels: np.ndarray,
    output_dir: str
) -> None:
    """
    Save multiple MC dropout logits to separate files in a subfolder structure.
    
    Args:
        logits_list: List of (N, C) logit arrays, one for each MC pass
        labels: (N,) array of labels (same for all passes)
        output_dir: Parent directory for logits folder (e.g., "outputs")
                    Files will be saved as output_dir/val_logits/logits_0.csv, logits_1.csv, ...
    """
    logits_folder = Path(output_dir) / "val_logits"
    logits_folder.mkdir(parents=True, exist_ok=True)
    
    for pass_idx, logits in enumerate(logits_list):
        output_path = logits_folder / f"logits_{pass_idx}.csv"
        
        # Create DataFrame with logits for each class
        data = {f"logit_{i}": logits[:, i] for i in range(logits.shape[1])}
        data["label"] = labels
        
        df = pd.DataFrame(data)
        # Save with full float64 precision (17 significant digits)
        df.to_csv(str(output_path), index=False, float_format='%.17g')
        print(f"Saved MC dropout pass {pass_idx} to {output_path}")



def save_temperature_scaling(
    temperature: float,
    output_path: str
) -> None:
    """
    Save optimal temperature scaling value.
    
    Args:
        temperature: Temperature value
        output_path: Path to save temperature file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"optimal_temperature: {temperature}\n")
    
    print(f"Saved temperature scaling to {output_path} (T={temperature:.4f})")


def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute negative log-likelihood."""
    criterion = nn.CrossEntropyLoss()
    return criterion(logits, labels).item()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
