# Import torch for the no_grad inference context
import torch

# Import the trained model (this runs training.py's full training loop as a side effect)
from training import model

# Import the validation DataLoader from loader.py
from loader import valid_dataloader

# Import the selected device
from imageclassifier import device

# Set the model to evaluation mode (disables dropout/batchnorm training behavior if present)
model.eval()

# Pull one batch of (images, targets) from the validation DataLoader
X_batch, y_batch = next(iter(valid_dataloader))

# Take a small batch of 3 samples from that batch
X_small, y_small = X_batch[:3], y_batch[:3]

# Move the small batch to the selected device
X_small, y_small = X_small.to(device), y_small.to(device)

# Report status: evaluating the small batch, on which device
print(f"Evaluating a small batch of {X_small.size(0)} samples on device: {device}")

# Disable gradient tracking since this is inference, not training
with torch.no_grad():
    # Forward pass: compute predicted logits for the small batch
    logits = model(X_small)

# Report status: evaluation forward pass complete, with the resulting logits shape
print(f"Evaluation complete. logits shape: {logits.shape}")
