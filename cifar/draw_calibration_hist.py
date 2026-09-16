"""
Utility to draw calibration histograms before and after temperature scaling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import softmax
import argparse
import torch
import torch.nn as nn
from scipy.optimize import minimize


class TemperatureScaler(nn.Module):
    """Temperature scaling using L-BFGS (from utils.py)."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature
    
    def compute_temperature(self, logits, labels):
        """Compute optimal temperature via L-BFGS on validation set."""
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval_fn(temp):
            self.temperature.data = torch.tensor([temp[0]], dtype=torch.float32)
            logits_scaled = self.forward(logits)
            loss = nll_criterion(logits_scaled, labels)
            return loss.item()
        
        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        result = minimize(
            lambda t: eval_fn(t),
            [1.0],
            method='Nelder-Mead',
            options={'maxiter': 500}
        )
        
        self.temperature.data = torch.tensor([result.x[0]], dtype=torch.float32)
        return float(result.x[0])


def compute_temperature_torch(logits_np, labels_np):
    """Compute optimal temperature from numpy arrays."""
    logits = torch.from_numpy(logits_np).float()
    labels = torch.from_numpy(labels_np).long()
    
    scaler = TemperatureScaler()
    temp = scaler.compute_temperature(logits, labels)
    return temp


def draw_calibration_hist(logits_csv_path="./logs/2026-08-27/15-42-42-C7-Enet0/val_logits.csv", 
                         num_bins=10, 
                         recompute=False, 
                         id_classes="0123456789"):
    """
    Plot calibration histograms before and after temperature scaling.
    
    Args:
        logits_csv_path: Path to the validation logits CSV file
        num_bins: Number of bins for calibration histogram (default: 10)
        recompute: If True, compute temperature from scratch on in-distribution samples
        id_classes: String of in-distribution class indices (auto-detected from config if available)
    """
    logits_csv_path = Path(logits_csv_path)
    
    # Auto-detect id_classes from training config if not explicitly set
    if id_classes == "0123456789":
        config_file = logits_csv_path.parent / ".hydra" / "config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                if 'data' in config and 'id_classes' in config['data']:
                    id_classes = config['data']['id_classes']
                    print(f"Auto-detected id_classes={id_classes} from training config")
    
    # Parse id_classes
    id_classes_list = [int(c) for c in id_classes]
    id_classes_set = set(id_classes_list)
    
    # Load logits and labels from CSV
    df = pd.read_csv(logits_csv_path)
    
    # Extract logits (all columns except label)
    logit_cols = [col for col in df.columns if col.startswith('logit_')]
    logits = df[logit_cols].values
    labels = df['label'].values
    
    # Determine in-distribution samples
    id_mask = np.array([label in id_classes_set for label in labels])
    
    # Prepare data
    if recompute:
        print(f"Recomputing temperature using {id_mask.sum()} in-distribution samples (id_classes={id_classes})")
        logits_for_temp = logits[id_mask]
        labels_for_temp = labels[id_mask]
        temperature = compute_temperature_torch(logits_for_temp, labels_for_temp)
        
        # Save the new temperature
        temp_file_path = logits_csv_path.parent / "optimal_temperature.txt"
        with open(temp_file_path, 'w') as f:
            f.write(f"optimal_temperature: {temperature}")
        print(f"Saved new temperature {temperature:.6f} to {temp_file_path}")
    else:
        # Infer temperature file path
        temp_file_path = logits_csv_path.parent / "optimal_temperature.txt"
        
        if not temp_file_path.exists():
            raise FileNotFoundError(f"Temperature file not found: {temp_file_path}")
        
        # Load temperature value (last line of the file)
        with open(temp_file_path, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1].strip()
            # Parse format: "optimal_temperature: 3.926959276199341"
            if ':' in last_line:
                temperature = float(last_line.split(':')[1].strip())
            else:
                temperature = float(last_line)
    
    print(f"Loaded logits from: {logits_csv_path}")
    print(f"Loaded temperature: {temperature:.4f} from {temp_file_path}")
    print(f"Using {id_mask.sum()} in-distribution samples (id_classes={id_classes})")
    
    # Filter to in-distribution samples only
    logits = logits[id_mask]
    labels = labels[id_mask]
    
    # Compute probabilities before and after temperature scaling
    logits_before = logits
    logits_after = logits / temperature
    
    probs_before = softmax(logits_before, axis=1)
    probs_after = softmax(logits_after, axis=1)
    
    # Get predicted class and max probability
    preds = np.argmax(logits_before, axis=1)
    max_probs_before = probs_before[np.arange(len(preds)), preds]
    max_probs_after = probs_after[np.arange(len(preds)), preds]
    
    # Compute correctness
    is_correct = (preds == labels).astype(int)
    
    # Compute calibration for each bin
    def compute_calibration_curve(confidences, correctness, num_bins=10):
        """Compute calibration curve (expected accuracy vs confidence)."""
        bins = np.linspace(0, 1, num_bins + 1)
        bin_sums = np.zeros(num_bins)
        bin_true = np.zeros(num_bins)
        bin_total = np.zeros(num_bins)
        
        for i in range(num_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if i == num_bins - 1:
                mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
            
            bin_total[i] = mask.sum()
            if bin_total[i] > 0:
                bin_true[i] = correctness[mask].sum()
                bin_sums[i] = bin_true[i] / bin_total[i]
        
        # Bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return bin_centers, bin_sums, bin_total
    
    # Compute calibration curves
    bin_centers_before, accs_before, counts_before = compute_calibration_curve(
        max_probs_before, is_correct, num_bins
    )
    bin_centers_after, accs_after, counts_after = compute_calibration_curve(
        max_probs_after, is_correct, num_bins
    )
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot before temperature scaling
    ax1.bar(bin_centers_before, accs_before, width=0.08, alpha=0.7, edgecolor='black', label='Accuracy')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    # Add sample counts on top of bars
    for i, (center, count) in enumerate(zip(bin_centers_before, counts_before)):
        if count > 0:
            ax1.text(center, accs_before[i] + 0.02, f'n={int(count)}', ha='center', va='bottom', fontsize=9)
    ax1.set_xlabel('Confidence (max probability)')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Before Temperature Scaling')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot after temperature scaling
    ax2.bar(bin_centers_after, accs_after, width=0.08, alpha=0.7, edgecolor='black', label='Accuracy')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    # Add sample counts on top of bars
    for i, (center, count) in enumerate(zip(bin_centers_after, counts_after)):
        if count > 0:
            ax2.text(center, accs_after[i] + 0.02, f'n={int(count)}', ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Confidence (max probability)')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(f'After Temperature Scaling (T={temperature:.4f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = logits_csv_path.parent / "calibration_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nCalibration histogram saved to: {output_path}")
    
    # Compute and print ECE (Expected Calibration Error)
    def compute_ece(confidences, correctness, num_bins=10):
        bin_centers, accs, counts = compute_calibration_curve(confidences, correctness, num_bins)
        ece = np.sum(np.abs(bin_centers - accs) * counts) / np.sum(counts)
        return ece
    
    ece_before = compute_ece(max_probs_before, is_correct, num_bins)
    ece_after = compute_ece(max_probs_after, is_correct, num_bins)
    
    print(f"\nExpected Calibration Error (ECE):")
    print(f"  Before: {ece_before:.4f}")
    print(f"  After:  {ece_after:.4f}")
    print(f"  Improvement: {(ece_before - ece_after):.4f}")
    
    plt.show()
    
    return fig, (ece_before, ece_after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw calibration histograms before/after temperature scaling')
    parser.add_argument('csv_path', nargs='?', default="./logs/2026-08-27/15-42-42-C7-Enet0/val_logits.csv",
                        help='Path to validation logits CSV file')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute temperature from scratch on in-distribution samples')
    parser.add_argument('--id_classes', type=str, default="0123456789",
                        help='String of in-distribution class indices (default: "0123456789", auto-detect from config if available)')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of bins for calibration histogram (default: 10)')
    
    args = parser.parse_args()
    
    draw_calibration_hist(
        logits_csv_path=args.csv_path,
        num_bins=args.num_bins,
        recompute=args.recompute,
        id_classes=args.id_classes
    )
