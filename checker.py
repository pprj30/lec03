# Import train_data and train_and_valid_data (for its class labels) from downloader.py
from downloader import train_data, train_and_valid_data

# Unpack the first sample of train_data into X_sample (image) and y_sample (target)
X_sample, y_sample = train_data[0]

# Look up the class name for y_sample from the classes list
class_name = train_and_valid_data.classes[y_sample]

# Print the sample's data type, shape, and class_name
print(f"dtype: {X_sample.dtype}, shape: {X_sample.shape}, class_name: {class_name}")
