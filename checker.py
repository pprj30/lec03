# Reuse train_data and train_and_valid_data from loaddataset.py
from loaddataset import train_data, train_and_valid_data

# Take the first element of train_data
X_sample, y_sample = train_data[0]

# Each image has shape [channels, rows, columns]; FashionMNIST images are
# grayscale (one channel) and 28x28 pixels, so the shape is [1, 28, 28]

# Look up the class name for y_sample in the classes list
class_name = train_and_valid_data.classes[y_sample]

if __name__ == "__main__":
    print(f"X_sample shape: {X_sample.shape}")
    print(f"X_sample dtype: {X_sample.dtype}")
    print(f"y_sample class name: {class_name}")
