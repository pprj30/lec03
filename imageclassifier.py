# Import the needed libraries
import torch
from torch import nn


class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),
        )

    def forward(self, X):
        return self.mlp(X)


# Select device: "cuda" if available, else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Set the random seed right before creating the model
torch.manual_seed(42)

model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100, n_classes=10)
model = model.to(device)

# Loss function
xentropy = nn.CrossEntropyLoss()

if __name__ == "__main__":
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total model parameters: {n_params}")

    X_random = torch.rand(32, 1, 28, 28, device=device)
    output = model(X_random)
    print(f"output shape: {output.shape}")
