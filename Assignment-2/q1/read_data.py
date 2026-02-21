import torch

data = torch.load("Q1/train_labels.pt")
labels = torch.unique(data)
print(labels.shape)