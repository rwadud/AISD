"""
Lecture 6: PyTorch Example
Reconstructed from the code the lecturer walked through in class.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image


# ============================================================
# 1. Autograd Demo
# ============================================================
# f(x) = 3 * (x + 2)^2
# Computational graph:
#   X ──► [+2] ──► A ──┐
#                       ├──► [A*B] ──► C ──► [*3] ──► f
#   X ──► [+2] ──► B ──┘
#
# Chain rule:
#   df/dX = df/dC * dC/dA * dA/dX  +  df/dC * dC/dB * dB/dX
#         =   3   *   B   *   1    +    3    *   A   *   1
#         =   3*(x+2) + 3*(x+2) = 6*(x+2)
#   At x=1: 6*(1+2) = 18

def autograd_demo():
    # Without requires_grad, backward() will fail
    x = torch.tensor(1.0, requires_grad=True)

    a = x + 2
    b = x + 2
    c = a * b           # (x+2)^2
    f = 3 * c           # 3(x+2)^2

    f.backward()

    print(f"f(1) = {f.item()}")        # 3 * 9 = 27
    print(f"x.grad = {x.grad.item()}")  # 18


# ============================================================
# 2. Custom Dataset
# ============================================================
# Needs 3 methods: __init__, __len__, __getitem__
#
# Folder structure expected:
#   root/
#     class_a/
#       img1.jpg
#       img2.jpg
#     class_b/
#       img1.jpg
#       img2.jpg

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Read folder structure and assign labels
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 3. Custom Linear Layer
# ============================================================
# Implements f(x) = Wx + b
# Needs 2 methods: __init__, forward
# Backward is handled automatically by autograd

class CustomLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        # Two learnable parameters initialized randomly
        self.weight = nn.Parameter(torch.randn(input_features, output_features))
        self.bias = nn.Parameter(torch.randn(output_features))

    def forward(self, x):
        return x @ self.weight + self.bias


# ============================================================
# 4. Custom MSE Loss
# ============================================================
# mean( (prediction - target)^2 )
# Needs 2 methods: __init__, forward

class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        return torch.mean((prediction - target) ** 2)


# ============================================================
# 5. Data Preparation
# ============================================================
# Uses ImageFolder (shortcut that reads folder structure as labels)
# Transforms: resize (so all images are the same size for batching),
#             crop, to tensor, normalize with ImageNet mean/std

def load_data(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ImageFolder reads folder names as class labels
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split into train / validation / test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoader is a generator: call next() to get the next batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# 6. Model Definition
# ============================================================
# Like Lego: each layer (conv, relu, pool, linear) is a predefined block.
# __init__ defines which blocks you need (and their sizes).
# forward assembles them into your custom architecture.
#
# Architecture:
#   conv1 -> relu -> maxpool -> conv2 -> relu -> maxpool -> flatten -> fc1 -> fc2

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After conv1 + pool on 64x64 input: 32 x 32 x 32
        # After conv2 + pool:                64 x 16 x 16
        # Flattened: 64 * 16 * 16 = 16384
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool(torch.relu(self.conv1(x)))
        # Conv block 2
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten: -1 tells PyTorch to infer the batch size
        x = x.view(-1, 64 * 16 * 16)
        # Fully connected
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ============================================================
# 7. Training
# ============================================================
# Two most important things: loss function and optimizer.
# For each batch:
#   1. zero_grad   (flush gradient buffers, otherwise they accumulate)
#   2. forward     (model(inputs) is same as model.forward(inputs))
#   3. loss        (compare output to labels)
#   4. backward    (compute gradients via autograd)
#   5. step        (update weights)

def train(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()          # 1. zero gradients
            outputs = model(inputs)        # 2. forward pass
            loss = criterion(outputs, labels)  # 3. compute loss
            loss.backward()                # 4. backward pass (compute gradients)
            optimizer.step()               # 5. update weights

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}")

        # Validate every 5 epochs to check for overfitting
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            print(f"  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")


# ============================================================
# 8. Evaluation
# ============================================================
# torch.no_grad() sets requires_grad=False for everything inside,
# so gradients are not tracked. Best practice for eval/inference.

def evaluate(model, data_loader, criterion=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    avg_loss = running_loss / len(data_loader) if criterion else 0
    return avg_loss, accuracy


# ============================================================
# 9. Main
# ============================================================

def main():
    print("=== Autograd Demo ===")
    autograd_demo()
    print()

    data_dir = "path/to/your/data"  # e.g. data/ with cats/ and dogs/ subfolders
    train_loader, val_loader, test_loader = load_data(data_dir)

    model = CNN(num_classes=2)

    print("=== Training ===")
    train(model, train_loader, val_loader, num_epochs=10)

    print("\n=== Test Results ===")
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
    print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
