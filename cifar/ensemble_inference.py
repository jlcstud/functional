"""
Example: Ensemble inference using saved logits and temperature scaling.

Shows how to:
1. Load logits from multiple trained models
2. Apply temperature scaling
3. Combine predictions for ensemble inference
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F


def load_logits_and_temperature(model_dir: Path):
    """Load logits and optimal temperature from a trained model."""
    logits_df = pd.read_csv(model_dir / "val_logits.csv")
    
    # Extract logits and labels
    logit_cols = [col for col in logits_df.columns if col.startswith('logit_')]
    logits = logits_df[logit_cols].values.astype(np.float32)
    labels = logits_df['label'].values.astype(np.int64)
    
    # Load temperature
    with open(model_dir / "optimal_temperature.txt") as f:
        temp_line = f.read().strip()
        temperature = float(temp_line.split(': ')[1])
    
    return logits, labels, temperature


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    return logits / temperature


def ensemble_predictions(
    model_logits_list: List[np.ndarray],
    model_temperatures_list: List[float],
    aggregate_method: str = "mean"
) -> np.ndarray:
    """
    Combine predictions from multiple models.
    
    Args:
        model_logits_list: List of logits arrays from different models
        model_temperatures_list: List of temperature values
        aggregate_method: "mean", "max", or "product" (in probability space)
    
    Returns:
        Ensemble logits or probabilities
    """
    scaled_logits = []
    
    for logits, temp in zip(model_logits_list, model_temperatures_list):
        scaled = apply_temperature_scaling(logits, temp)
        scaled_logits.append(scaled)
    
    scaled_logits = np.array(scaled_logits)  # (num_models, num_samples, num_classes)
    
    if aggregate_method == "mean":
        # Average logits (simple ensemble)
        ensemble_logits = scaled_logits.mean(axis=0)
        return ensemble_logits
    
    elif aggregate_method == "max":
        # Max logit
        ensemble_logits = scaled_logits.max(axis=0)
        return ensemble_logits
    
    elif aggregate_method == "product":
        # Product of probabilities
        probs = torch.softmax(torch.from_numpy(scaled_logits).float(), dim=-1)
        prod_probs = probs.prod(dim=0)
        prod_probs = prod_probs / prod_probs.sum(dim=-1, keepdim=True)  # Normalize
        
        # Convert back to log-space for consistency
        ensemble_logits = torch.log(prod_probs + 1e-10).numpy()
        return ensemble_logits
    
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate_method}")


def evaluate_ensemble(
    ensemble_logits: np.ndarray,
    labels: np.ndarray
) -> dict:
    """Evaluate ensemble performance."""
    predictions = ensemble_logits.argmax(axis=1)
    accuracy = (predictions == labels).mean()
    
    # Calibration: NLL and ECE
    probs = torch.softmax(torch.from_numpy(ensemble_logits).float(), dim=-1)
    nll = F.cross_entropy(torch.from_numpy(ensemble_logits).float(), 
                          torch.from_numpy(labels).long()).item()
    
    return {
        "accuracy": accuracy,
        "nll": nll,
    }


def example_ensemble(model_dirs: List[Path]):
    """
    Example: Create and evaluate an ensemble from multiple trained models.
    
    Args:
        model_dirs: List of directories containing trained model outputs
    """
    print(f"Loading {len(model_dirs)} models...")
    
    all_logits = []
    all_temperatures = []
    labels = None
    
    for model_dir in model_dirs:
        logits, model_labels, temp = load_logits_and_temperature(model_dir)
        all_logits.append(logits)
        all_temperatures.append(temp)
        
        if labels is None:
            labels = model_labels
        else:
            assert np.array_equal(labels, model_labels), "Labels mismatch!"
        
        print(f"  Model: {model_dir}")
        print(f"    Temperature: {temp:.4f}")
        print(f"    Accuracy: {(logits.argmax(1) == labels).mean():.4f}")
    
    print(f"\nCreating ensemble from {len(all_logits)} models...")
    
    # Create ensemble with different aggregation methods
    for method in ["mean", "product"]:
        ensemble_logits = ensemble_predictions(
            all_logits,
            all_temperatures,
            aggregate_method=method
        )
        
        results = evaluate_ensemble(ensemble_logits, labels)
        print(f"\nEnsemble ({method}):")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  NLL: {results['nll']:.4f}")


# Example usage:
# model_dirs = [
#     Path("logs/2024-01-15/10-30-45"),  # Model 1 (seed=42)
#     Path("logs/2024-01-15/10-35-22"),  # Model 2 (seed=123)
#     Path("logs/2024-01-15/10-40-10"),  # Model 3 (seed=456)
# ]
# example_ensemble(model_dirs)
