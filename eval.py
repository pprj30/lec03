# Reuse the trained model and device from training.py, and the valid_loader
# and train_and_valid_data from loader.py and loaddataset.py
import torch

from training import model, device
from loaddataset import train_and_valid_data
from loader import valid_loader

model.eval()

# Get one batch and keep only the first 3 samples
X_batch, y_batch = next(iter(valid_loader))
X_new, y_new = X_batch[:3], y_batch[:3]
X_new = X_new.to(device)

# Predict without tracking gradients
with torch.no_grad():
    y_pred_logits = model(X_new)

# Predicted class is the index of the largest logit
y_pred = y_pred_logits.argmax(dim=1)
print(y_pred)

# Look up the class name for each predicted index
class_names = [train_and_valid_data.classes[idx] for idx in y_pred]
print(class_names)

# True labels for the same 3 samples
print(y_new)
