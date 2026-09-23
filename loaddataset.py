# Import the needed libraries
import torch
from torchvision import datasets
from torchvision.transforms import v2

# Define toTensor: convert images to a tensor, then to float32 scaled to [0, 1]
toTensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Load FashionMNIST training+validation set and test set, downloading into ./datasets if needed
train_and_valid_data = datasets.FashionMNIST(
    root="datasets",
    train=True,
    download=True,
    transform=toTensor,
)

test_data = datasets.FashionMNIST(
    root="datasets",
    train=False,
    download=True,
    transform=toTensor,
)

# Set the random seed right before splitting, then split into train (55000) and valid (5000)
torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(train_and_valid_data, [55000, 5000])

# Take the first training sample
X_sample, y_sample = train_data[0]

if __name__ == "__main__":
    print(f"train_data size: {len(train_data)}")
    print(f"valid_data size: {len(valid_data)}")
    print(f"test_data size: {len(test_data)}")
    print(f"X_sample shape: {X_sample.shape}, dtype: {X_sample.dtype}")
