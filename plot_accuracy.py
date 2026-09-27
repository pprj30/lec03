# Import matplotlib's plotting interface
import matplotlib.pyplot as plt

# Import the per-epoch training accuracy history and epoch count (this runs training.py's
# full training loop as a side effect)
from training import accuracy_history, epochs

# Build the list of epoch numbers (1-indexed) to use as the x-axis
epoch_numbers = list(range(1, epochs + 1))

# Plot training accuracy against epoch number
plt.plot(epoch_numbers, accuracy_history, marker="o")

# Label the x-axis
plt.xlabel("Epoch")

# Label the y-axis
plt.ylabel("Training Accuracy")

# Give the plot a title
plt.title("Training Accuracy per Epoch")

# Display the plot
plt.show()
