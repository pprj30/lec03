# Import torch and DataLoader to batch and iterate over the datasets
import torch
from torch.utils.data import DataLoader

# Import train_data, valid_data, test_data, and train_and_valid_data (for its class labels) from downloader.py
from downloader import train_data, valid_data, test_data, train_and_valid_data

# Manually set the random seed to 42
torch.manual_seed(42)

# Load the training set with batch size 32, shuffling enabled
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

# Load the validation set with batch size 32, no shuffling
valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False)

# Load the testing set with batch size 32, no shuffling
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# Each sample in train_data is a (image, target) tuple; assign its first element's parts to X_sample and y_sample
X_sample, y_sample = train_data[0]

# Print X_sample's shape and data type
print(f"X_sample shape: {X_sample.shape}, dtype: {X_sample.dtype}")

# Look up the class name for y_sample using the class labels from train_and_valid_data
print(f"y_sample: {y_sample}, class: {train_and_valid_data.classes[y_sample]}")
