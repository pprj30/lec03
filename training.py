# Import torch for the optimizer and torchmetrics for the accuracy metric
import torch
from torch import nn
import torchmetrics

# Import the training DataLoader built in loader.py
from loader import train_dataloader

# Import the model class and the selected device from imageclassifier.py
from imageclassifier import ImageClassifier, device

# Instantiate the model and move it to the selected device
model = ImageClassifier().to(device)

# Define the loss function: cross-entropy loss for multi-class classification on raw logits
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer: Stochastic Gradient Descent with learning rate 0.001, no momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Define the accuracy metric (10 FashionMNIST classes) and move it to the selected device
accuracy_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)

# Set the number of training epochs
epochs = 20

# Track training accuracy across epochs for later plotting
accuracy_history = []

# Report status: training is starting, on which device
print(f"Starting training on device: {device}")

# Loop over the dataset multiple times (one full pass per epoch)
for epoch in range(epochs):
    # Set the model to training mode (enables dropout/batchnorm training behavior if present)
    model.train()

    # Track the running loss total for this epoch
    running_loss = 0.0

    # Iterate over batches of (image, target) pairs from the training DataLoader
    for X, y in train_dataloader:
        # Move the batch to the selected device
        X, y = X.to(device), y.to(device)

        # Reset gradients from the previous step
        optimizer.zero_grad()

        # Forward pass: compute predicted logits
        pred = model(X)

        # Compute the loss between predictions and targets
        loss = loss_fn(pred, y)

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters using the computed gradients
        optimizer.step()

        # Accumulate this batch's loss, weighted by batch size
        running_loss += loss.item() * X.size(0)

        # Update the running accuracy metric with this batch's predictions and targets
        accuracy_metric.update(pred, y)

    # Compute the average loss over all training samples this epoch
    epoch_loss = running_loss / len(train_dataloader.dataset)

    # Compute the accuracy accumulated over the whole epoch
    epoch_accuracy = accuracy_metric.compute()

    # Record this epoch's accuracy for later plotting
    accuracy_history.append(epoch_accuracy.item())

    # Report status: epoch number, average loss, and accuracy
    print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}")

    # Reset the accuracy metric so it doesn't accumulate across epochs
    accuracy_metric.reset()

# Report status: training loop complete
print("Training complete.")
