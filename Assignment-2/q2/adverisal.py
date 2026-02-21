import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Constants and Configuration ---
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
CLASS_NAMES = [
    'arctic fox', 'corgi', 'electric ray', 'goldfish', 'hammerhead shark',
    'horse', 'hummingbird', 'indigo finch', 'puma', 'red panda'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Model Setup ---
def get_model(weights_path='Q2/network_visualization.pth'):
    # Load ResNet18 and modify the last layer (fc) from 1000 to 10 classes
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    
    # Load the specific weight file provided
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- 3. Preprocessing and Deprocessing ---
def preprocess(img_path_or_pil):
    """Resizes to 224x224 and normalizes."""
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert('RGB')
    else:
        img = img_path_or_pil
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor

def deprocess(tensor):
    """Converts a normalized tensor back to a displayable image."""
    img = tensor.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


# --- 5. Adversarial Attack Logic (Optimization-based) ---
def adversarial_attack(model, input_tensor, ground_truth_idx, target_type="next_highest"):
    # Create a clone of the image to optimize (adversarial example)
    adv_image = input_tensor.clone().detach().to(device)
    adv_image.requires_grad = True
    
    # Determine the target class
    with torch.no_grad():
        logits = model(input_tensor)
        if target_type == "next_highest":
            # Class with the second highest logit
            target_class = torch.argsort(logits, descending=True)[0, 1].item()
        else: # target_type == "lowest"
            target_class = torch.argmin(logits).item()

    # Optimization: Use Adam to update image pixels to maximize target logit
    optimizer = optim.Adam([adv_image], lr=0.01)
    
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(adv_image)
        # Minimize negative logit (equivalent to maximizing logit)
        loss = -output[0, target_class]
        loss.backward()
        optimizer.step()
        
        # Stop early if the model is successfully fooled
        if torch.argmax(model(adv_image)).item() == target_class:
            break
            
    return adv_image.detach(), target_class

# --- 6. Main Execution Script ---
def run_all_tasks(image_list):
    """
    image_list: list of 10 image paths (one per class)
    """
    model = get_model()
    
    for i, path in enumerate(image_list):
        print(f"\nProcessing Class: {CLASS_NAMES[i]}")
        
        img_tensor = preprocess(path)
        
        # --- Task 3: Adversarial Attacks ---
        # (a) Target Next-Highest Class
        adv_next, target_next = adversarial_attack(model, img_tensor, i, "next_highest")
        # (b) Target Lowest Probability Class
        adv_low, target_low = adversarial_attack(model, img_tensor, i, "lowest")
        
        # --- Visualization ---
        plt.figure(figsize=(15, 8))
        
        # Original Image
        plt.subplot(2, 3, 1)
        plt.imshow(deprocess(img_tensor))
        plt.title(f"Original: {CLASS_NAMES[i]}")

        
        # Difference/Noise for Next-Highest
        plt.subplot(2, 3, 3)
        noise = (adv_next - img_tensor).abs().squeeze().cpu().numpy().transpose(1, 2, 0)
        plt.imshow(noise / (noise.max() + 1e-8)) # Normalized for visibility
        plt.title("Adversarial Noise (Target Next)")

        # Adversarial Example (Next-Highest)
        plt.subplot(2, 3, 4)
        plt.imshow(deprocess(adv_next))
        plt.title(f"Pred: {CLASS_NAMES[target_next]}")

        # Adversarial Example (Lowest)
        plt.subplot(2, 3, 5)
        plt.imshow(deprocess(adv_low))
        plt.title(f"Pred: {CLASS_NAMES[target_low]}")

        plt.tight_layout()
        plt.show()

# To run:
my_images = ["Q2/network visualization/arctic fox/arctic fox_5.JPEG"] # Add your paths here
run_all_tasks(my_images)