import os
import pickle
from torchvision.datasets import CIFAR10

# Check what files are present
data_dir = "./data"
print("Contents of data folder:")
for item in os.listdir(data_dir):
    print(f"  {item}")

# Check inside the extracted folder
cifar_path = os.path.join(data_dir, "cifar-10-batches-py")
print(f"\nContents of {cifar_path}:")
for item in os.listdir(cifar_path):
    print(f"  {item}")

# Verify data files by trying to load them
print("\n--- Verifying data files ---")
required_files = ["batches.meta", "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
for fname in required_files:
    fpath = os.path.join(cifar_path, fname)
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    
    # Try to load and verify
    if exists and fname != "batches.meta":
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            print(f"{fname}: ✓ ({size / 1e6:.1f}MB, {len(data[b'data'])} samples)")
        except Exception as e:
            print(f"{fname}: ✗ Error - {e}")
    else:
        print(f"{fname}: {'✓' if exists else '✗'} ({size / 1e6:.1f}MB)")

# Try loading with CIFAR10
print("\n--- Testing CIFAR10 loader ---")
try:
    dataset = CIFAR10(root=data_dir, train=True, download=False)
    print(f"✓ Successfully loaded CIFAR10 train set: {len(dataset)} samples")
except Exception as e:
    print(f"✗ Error loading train set: {e}")

try:
    dataset = CIFAR10(root=data_dir, train=False, download=False)
    print(f"✓ Successfully loaded CIFAR10 test set: {len(dataset)} samples")
except Exception as e:
    print(f"✗ Error loading test set: {e}")
