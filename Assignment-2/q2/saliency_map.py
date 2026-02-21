import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Setup and Model Loading
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
CLASS_NAMES = ['arctic fox', 'corgi', 'electric ray', 'goldfish', 'hammerhead shark', 
               'horse', 'hummingbird', 'indigo finch', 'puma', 'red panda']

def get_model(path='Q2/network_visualization.pth'):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    # Load your custom pretrained weights
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# 2. Preprocessing Function
def preprocess(img_path):
    """Resizes, normalizes, and returns a tensor with grad enabled."""
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    tensor = transform(img).unsqueeze(0)
    tensor.requires_grad_() # CRITICAL for saliency
    return tensor

def deprocess(tensor):
    """Converts a normalized tensor back to a displayable image."""
    img = tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)

# 3. Saliency Map Computation
def compute_saliency(model, tensor, target_class):
    model.zero_grad()
    output = model(tensor)
    
    # Get the logit for the correct class
    score = output[0, target_class]
    score.backward()
    
    # Saliency is the max absolute gradient across color channels
    saliency, _ = torch.max(torch.abs(tensor.grad[0]), dim=0)
    return saliency.cpu().numpy()

# 4. Masking and Classification Test
def apply_mask(tensor, saliency, threshold=0.7, mode='constant'):
    # Normalize saliency to [0, 1]
    sal_min, sal_max = saliency.min(), saliency.max()
    norm_sal = (saliency - sal_min) / (sal_max - sal_min + 1e-8)
    
    mask = norm_sal < threshold
    masked_tensor = tensor.clone().detach()
    
    if mode == 'constant':
        masked_tensor[0, :, mask] = 0
    elif mode == 'noise':
        noise = torch.randn_like(masked_tensor)
        masked_tensor[0, :, mask] = noise[0, :, mask]
        
    return masked_tensor

# --- Execution Example ---
model = get_model()
# Let's assume you have a list of paths for 10 images (one per class)
# image_paths = ["Q2/network visualization/arctic fox/arctic fox_2.JPEG"]
image_paths = ["Q2/network visualization/hummingbird/hummingbird_1.JPEG"]

for i, path in enumerate(image_paths):
    img_tensor = preprocess(path)
    saliency = compute_saliency(model, img_tensor, i)
    
    # Visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(deprocess(img_tensor))
    plt.title(f"GT: {CLASS_NAMES[i]}")
    
    plt.subplot(1, 3, 2)
    plt.imshow(saliency, cmap='hot')
    plt.title("Saliency Map")
    
    # Masking test
    masked = apply_mask(img_tensor, saliency, threshold=0.1, mode='noise')
    plt.subplot(1, 3, 3)
    plt.imshow(deprocess(masked))
    plt.title("Masked Image")
    plt.show()

    # Predict on masked
    with torch.no_grad():
        output = model(masked)
        pred = torch.argmax(output).item()
        print(f"Original: {CLASS_NAMES[i]} | Masked Prediction: {CLASS_NAMES[pred]}")