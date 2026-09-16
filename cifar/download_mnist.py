"""
One-time script to download and prepare MNIST dataset.
Run this once before training: python download_mnist.py
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def download_mnist(data_dir="./data"):
    """Download and process MNIST dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading and processing MNIST to {data_dir}...")
    
    # Download training set (download=True forces processing)
    train_data = MNIST(root=data_dir, train=True, download=True)
    print(f"Downloaded training set: {len(train_data)} samples")
    
    # Download test set
    test_data = MNIST(root=data_dir, train=False, download=True)
    print(f"Downloaded test set: {len(test_data)} samples")
    
    print("MNIST download and processing complete!")


if __name__ == "__main__":
    download_mnist()
