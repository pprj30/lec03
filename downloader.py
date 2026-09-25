# Import torch for manual_seed and random_split
import torch
from torch.utils.data import random_split

# Import torchvision datasets module to load FashionMNIST and ToTensor to convert images to tensors
from torchvision import datasets
from torchvision.transforms import ToTensor

# Report status: beginning download of the FashionMNIST training/validation dataset
print("Downloading FashionMNIST training data (train=True)...")

# Download FashionMNIST training split (60000 samples) into train_and_valid_data, converting images to tensors
train_and_valid_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Report status: training/validation dataset downloaded, with total sample count
print(f"train_and_valid_data ready. Total samples: {len(train_and_valid_data)}")

# Report status: beginning download of the FashionMNIST test dataset
print("Downloading FashionMNIST test data (train=False)...")

# Download FashionMNIST test split (10000 samples) into test_data, converting images to tensors
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Report status: test dataset downloaded, with total sample count
print(f"test_data ready. Total samples: {len(test_data)}")

# Manually set the random seed to 42 so the train/validation split is reproducible
torch.manual_seed(42)

# Report status: seed has been set
print("Random seed manually set to 42.")

# Split train_and_valid_data into 55000 training samples and 5000 validation samples
train_data, valid_data = random_split(train_and_valid_data, [55000, 5000])

# Report status: split complete, with resulting sample counts for each subset
print(f"Split complete. train_data: {len(train_data)} samples, valid_data: {len(valid_data)} samples")
