import torch
from torchvision import models
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18().to(device)
# You must provide the input size (Channels, H, W)
summary(model, (3, 36, 36))