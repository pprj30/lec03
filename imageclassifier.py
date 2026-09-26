# Import torch's neural network module to build the model
from torch import nn


# Define the image classifier as a subclass of nn.Module
class ImageClassifier(nn.Module):
    # Define the layers used in the forward pass
    def __init__(self):
        # Initialize the parent nn.Module class
        super().__init__()

        # Flatten each (1, 28, 28) image tensor into a 784-length vector
        self.flatten = nn.Flatten()

        # Define the hidden layer stack: 784 -> 300 -> 100 -> 10, with ReLU between hidden layers
        self.linear_relu_stack = nn.Sequential(
            # First hidden layer: 784 input features to 300 neurons
            nn.Linear(784, 300),
            # ReLU activation after the first hidden layer
            nn.ReLU(),
            # Second hidden layer: 300 neurons to 100 neurons
            nn.Linear(300, 100),
            # ReLU activation after the second hidden layer
            nn.ReLU(),
            # Output layer: 100 neurons to 10 class logits
            nn.Linear(100, 10),
        )

    # Define how input data flows through the layers
    def forward(self, x):
        # Flatten the input image tensor
        x = self.flatten(x)

        # Pass the flattened tensor through the hidden layer stack to get logits
        logits = self.linear_relu_stack(x)

        # Return the raw logits (no activation applied)
        return logits
