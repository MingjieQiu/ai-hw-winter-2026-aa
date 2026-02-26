import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader = DataLoader(
    datasets.MNIST("data", train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

test_loader = DataLoader(
    datasets.MNIST("data", train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=64)

# Model
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), "cnn.pth")

# Clean evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print("Clean Test Accuracy:", accuracy)