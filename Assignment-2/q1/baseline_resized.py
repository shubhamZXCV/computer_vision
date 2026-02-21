import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split , Dataset
from torchvision import models , transforms
import wandb

# --- 1. Configuration & Hyperparameters ---
CONFIG = {
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
    "num_classes": 10,  # UPDATE this to your actual class count
    "val_split": 0.2,    # 20% of training data goes to validation
    "input_size": 36
}

# --- 1. Custom Dataset Wrapper for Resizing ---
class ResizedTensorDataset(Dataset):
    def __init__(self, data_path, label_path, target_size=(224, 224)):
        self.data = torch.load(data_path).float()
        self.labels = torch.load(label_path).long()
        # ResNet expects 3 channels. If data is (N, 1, 36, 36), we repeat it.
        if self.data.shape[1] == 1:
            self.data = self.data.repeat(1, 3, 1, 1)
            
        self.transform = transforms.Resize(target_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return self.transform(img), label

def get_data_loaders():
    # Load .pt files
    # full_train_x = torch.load('Q1/train_data.pt')
    # full_train_y = torch.load('Q1/train_labels.pt')
    # test_x = torch.load('Q1/test_data.pt')
    # test_y = torch.load('Q1/test_labels.pt')

    # Create full training dataset
    # full_dataset = TensorDataset(full_train_x.float(), full_train_y.long())
    # test_dataset = TensorDataset(test_x.float(), test_y.long())
    full_dataset = ResizedTensorDataset(data_path="Q1/train_data.pt",label_path="Q1/train_labels.pt")
    test_dataset = ResizedTensorDataset(data_path="Q1/test_data.pt",label_path="Q1/test_labels.pt")

    # Split training into Train and Validation
    train_size = int((1 - CONFIG["val_split"]) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_and_evaluate(model, train_loader, val_loader, test_loader, run_name):
    # Initialize WandB
    wandb.init(project="resnet-36x36-comparison", name=run_name, config=CONFIG)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    for epoch in range(CONFIG["epochs"]):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # --- Validation Phase ---
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # --- Final Test Evaluation ---
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nFINAL TEST RESULTS for {run_name}: Accuracy: {test_acc:.2f}%\n")
    wandb.log({"test_accuracy": test_acc})
    wandb.finish()

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

# --- Main Execution ---

train_loader, val_loader, test_loader = get_data_loaders()

# 1. Standard ResNet18 (Random Weights)
print("Starting Standard ResNet18 Training...")
model_std = models.resnet18(weights=None)
# Modify FC layer for custom classes
model_std.fc = nn.Linear(model_std.fc.in_features, CONFIG["num_classes"])
train_and_evaluate(model_std, train_loader, val_loader, test_loader, "standard_resnet18_resized")

# 2. Pretrained ResNet18 (ImageNet Weights)
print("Starting Pretrained ResNet18 Training...")
model_pre = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Modify FC layer for custom classes
model_pre.fc = nn.Linear(model_pre.fc.in_features, CONFIG["num_classes"])
train_and_evaluate(model_pre, train_loader, val_loader, test_loader, "pretrained_resnet18_resized")