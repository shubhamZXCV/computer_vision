import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split , Dataset
from torchvision import models , transforms
import wandb

def get_data_loaders():
    # Load .pt files
    full_train_x = torch.load('Q1/train_data.pt')
    full_train_y = torch.load('Q1/train_labels.pt')
    test_x = torch.load('Q1/test_data.pt')
    test_y = torch.load('Q1/test_labels.pt')

    # Create full training dataset
    full_dataset = TensorDataset(full_train_x.float(), full_train_y.long())
    test_dataset = TensorDataset(test_x.float(), test_y.long())

    # Split training into Train and Validation
    train_size = int((1 - CONFIG["val_split"]) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_modified_resnet(mod_type, num_classes=10, pretrained=False):
    # Load model
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Change the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if mod_type == "A":
        # Modification A: 3x3 Conv, Stride 1, No MaxPool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity() 
        
    elif mod_type == "B":
        # Modification B: 3x3 Conv, Stride 2, No MaxPool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        model.maxpool = nn.Identity()

    elif mod_type == "C":
        # Modification C: Two 3x3 Convs (simulating a 5x5 receptive field)
        model.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        model.maxpool = nn.Identity()

    return model

def validate(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return val_loss / len(loader), 100. * correct / total


def train_and_log(mod_type, is_pretrained):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_modified_resnet(mod_type, pretrained=is_pretrained).to(device)
    
    # WandB Setup
    run_type = "pretrained" if is_pretrained else "scratch"
    wandb.init(project="resnet-mods", name=f"mod_{mod_type}_{run_type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Assume train_loader and test_loader are defined as in previous steps
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            
            wandb.log({"train_loss": loss.item()})
            
        # Evaluation step here...
        # wandb.log({"val_acc": val_acc})
    
    wandb.finish()

# --- Main Execution ---
train_loader, val_loader, test_loader = get_data_loaders()

# Execute for Part 1 (Scratch)
for m in ["A", "B", "C"]:
    train_and_log(m, is_pretrained=False)

# Execute for Part 2 (Pretrained)
for m in ["A", "B", "C"]:
    train_and_log(m, is_pretrained=True)