# Reuse y_pred_logits and device from eval.py
import torch

from eval import y_pred_logits, device

# Softmax across all 10 classes
y_proba = torch.softmax(y_pred_logits, dim=1)
print(torch.round(y_proba, decimals=3))

# Top 4 logits and their class indices per sample
y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)

# Softmax renormalized over just the top 4 logits
y_top4_probas = torch.softmax(y_top4_values, dim=1)
print(torch.round(y_top4_probas, decimals=3))

print(y_top4_indices)
