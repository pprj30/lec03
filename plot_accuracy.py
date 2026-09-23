# Reuse history from training.py
import matplotlib.pyplot as plt

from training import history

epochs = range(1, len(history["train_metrics"]) + 1)

plt.plot(epochs, history["train_metrics"])
plt.xlabel("Epoch")
plt.ylabel("Training accuracy")
plt.title("Training accuracy over epochs")
plt.show()
