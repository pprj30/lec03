# Reuse train_data, valid_data and test_data from loaddataset.py
import torch
from torch.utils.data import DataLoader
from loaddataset import train_data, valid_data, test_data

# Set the random seed right before creating the loaders
torch.manual_seed(42)

# Create the DataLoaders, all with batch size 32
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print(f"train_loader batches: {len(train_loader)}")
    print(f"valid_loader batches: {len(valid_loader)}")
    print(f"test_loader batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"first batch image shape: {images.shape}")
    print(f"first batch label shape: {labels.shape}")
