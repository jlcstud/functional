"""
One-time script to download and prepare CIFAR-10 dataset.
Run this once before training: python download_cifar10.py
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def download_cifar10(data_dir="./data"):
    """Download CIFAR-10 dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading CIFAR-10 to {data_dir}...")
    
    # Download training set
    CIFAR10(root=data_dir, train=True, download=True)
    
    # Download test set
    CIFAR10(root=data_dir, train=False, download=True)
    
    print("CIFAR-10 download complete!")


if __name__ == "__main__":
    download_cifar10()
