# Reuse the model, loss function and device from imageclassifier.py,
# and the DataLoaders from loader.py
import torch
from torchmetrics import Accuracy

from imageclassifier import model, xentropy, device
from loader import train_loader, valid_loader

n_epochs = 20

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Metric
accuracy = Accuracy(task="multiclass", num_classes=10).to(device)


def train2(model, optimizer, criterion, metric, train_loader, valid_loader, n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    for epoch in range(n_epochs):
        model.train()
        metric.reset()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            metric.update(y_pred, y_batch)

        train_loss /= len(train_loader.dataset)
        train_acc = metric.compute().item()

        model.eval()
        metric.reset()
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                metric.update(y_pred, y_batch)
        valid_acc = metric.compute().item()

        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_acc)
        history["valid_metrics"].append(valid_acc)

        print(
            f"epoch {epoch + 1}/{n_epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- train_acc: {train_acc:.4f} "
            f"- valid_acc: {valid_acc:.4f}"
        )

    return history


if __name__ == "__main__":
    history = train2(model, optimizer, xentropy, accuracy, train_loader, valid_loader, n_epochs)
