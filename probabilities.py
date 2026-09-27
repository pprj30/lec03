# Import torch for softmax/topk operations
import torch

# Import the evaluation results: this runs evaluation.py (which itself trains the model
# and runs the small-batch forward pass) as a side effect
from evaluation import logits, X_small, y_small

# Import the class labels for displaying human-readable predictions
from downloader import train_and_valid_data

# Convert logits to probabilities using softmax across the class dimension
probabilities = torch.softmax(logits, dim=1)

# Pick out the top 4 probabilities (and their class indices) for each sample
top4_probs, top4_indices = torch.topk(probabilities, k=4, dim=1)

# Print the true label and the top-4 predicted classes with probabilities, for each sample
for i in range(X_small.size(0)):
    # Look up the true class name for this sample
    true_class = train_and_valid_data.classes[y_small[i]]

    # Build a list of (class_name, probability) pairs for this sample's top-4 predictions
    top4 = [
        (train_and_valid_data.classes[idx], prob.item())
        for prob, idx in zip(top4_probs[i], top4_indices[i])
    ]

    # Print the true class alongside the top-4 predictions
    print(f"Sample {i}: true class = {true_class}, top-4 predictions = {top4}")
