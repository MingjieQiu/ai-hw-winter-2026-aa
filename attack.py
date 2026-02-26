import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("cnn.pth", map_location=device))
model.eval()

# Data
test_loader = DataLoader(
    datasets.MNIST("data", train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=64)

criterion = nn.CrossEntropyLoss()
epsilon = 0.3


# ---------------- FGSM ----------------
def fgsm(model, x, y, epsilon):
    x.requires_grad = True
    output = model(x)
    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    grad = x.grad.data
    x_adv = x + epsilon * grad.sign()
    return torch.clamp(x_adv, 0, 1)


# ---------------- PGD ----------------
def pgd(model, x, y, epsilon=0.3, alpha=0.01, iters=40):
    x_adv = x.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()

        grad = x_adv.grad
        x_adv = x_adv + alpha * grad.sign()
        eta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + eta, 0, 1).detach()

    return x_adv


# ---------------- Momentum I-FGSM ----------------
def mifgsm(model, x, y, epsilon=0.3, alpha=0.01, iters=40, mu=0.9):
    x_adv = x.clone().detach()
    g = 0

    for _ in range(iters):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()

        grad = x_adv.grad
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        g = mu * g + grad
        x_adv = x_adv + alpha * g.sign()

        eta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + eta, 0, 1).detach()

    return x_adv


# ---------------- Evaluation ----------------
def evaluate(attack_fn, name):
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(model, x, y, epsilon)
        pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / total
    asr = 1 - acc

    print(f"\n{name}")
    print("Accuracy under attack:", acc)
    print("Attack Success Rate (ASR):", asr)


evaluate(fgsm, "FGSM")
evaluate(pgd, "PGD")
evaluate(mifgsm, "Momentum I-FGSM")