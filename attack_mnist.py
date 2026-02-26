import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Target Model
class SimpleCNN(nn.Module):
    """Basic Neural Network."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# Training
def train(model, loader, device, epochs=5):
    """
    Standard training: updates weights to minimize loss on clean data
    Args:
        model: PyTorch model to train
        loader: Training data loader
        epochs: Number of training epochs
        device: Device to train on
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")


# Accuracy Evaluation
def test_accuracy(model, loader, device):
    """
    Measures performance on original, non-attacked data.
    Args:
        model: Trained model
        loader: Data loader to evaluate
        device: Device to evaluate on
    Returns:
        float: Accuracy in [0,1]
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\nClean Accuracy: {acc * 100:.2f}%")
    return acc


# Attacks
def fgsm(model, images, labels, epsilon):
    """
    Fast Gradient Sign Method attack.
    Args:
        model: Model to attack
        images: Original images, shape [B,1,28,28]
        labels: True labels, shape [B]
        epsilon: Attack strength
    Returns:
        torch.Tensor: Adversarial images
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()


def pgd(model, images, labels, epsilon=0.3, alpha=0.01, iters=40):
    """
    Iterative PGD / I-FGSM attack.
    Args:
        model: Model to attack
        images: Original images
        labels: True labels
        epsilon: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
    Returns:
        torch.Tensor: Adversarial images
    """
    original = images.clone().detach()
    images = original.clone()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv = images + alpha * images.grad.sign()
        eta = torch.clamp(adv - original, min=-epsilon, max=epsilon)
        images = torch.clamp(original + eta, 0, 1).detach()

    return images



def mifgsm(model, images, labels, epsilon=0.3, alpha=0.01, iters=40, mu=1.0):
    """
    Momentum I-FGSM attack.
    Args:
        model Model to attack
        images: Original images
        labels: True labels
        epsilon: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
        mu Momentum factor
    Returns:
        torch.Tensor: Adversarial images
    """
    original = images.clone().detach()
    images = original.clone()
    momentum = torch.zeros_like(images)

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = images.grad
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

        momentum = mu * momentum + grad
        adv = images + alpha * momentum.sign()

        eta = torch.clamp(adv - original, min=-epsilon, max=epsilon)
        images = torch.clamp(original + eta, 0, 1).detach()

    return images


# Evaluate Attack
def evaluate_attack(model, loader, device, attack_fn, **kwargs):
    """
    Measures Post-Attack Accuracy and Attack Success Rate (ASR)
    Args:
        model: Trained model
        loader: Test data loader
        attack_fn: Attack function, e.g. fgsm, pgd, mifgsm
        kwargs: Arguments for attack function (epsilon, alpha, iters, mu)
    Returns:
        tuple: (Post-attack accuracy, ASR)
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        adv_images = attack_fn(model, images, labels, **kwargs)
        outputs = model(adv_images)
        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    asr = 1 - acc

    print(f"Post-Attack Accuracy: {acc * 100:.2f}%")
    print(f"Attack Success Rate (ASR): {asr * 100:.2f}%")


if __name__ == "__main__":
    # Prepare MNIST Data
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = SimpleCNN().to(device)

    # Train
    print("Starting Training...")
    train(model, train_loader, device, epochs=5)

    # Clean accuracy
    test_accuracy(model, test_loader, device)

    # Attacks
    print("\n--- Running FGSM Attack ---")
    evaluate_attack(model, test_loader, device, fgsm, epsilon=0.2)

    print("\n--- Running PGD Attack ---")
    evaluate_attack(model, test_loader, device, pgd, epsilon=0.3, alpha=0.01, iters=40)

    print("\n--- Running Momentum I-FGSM Attack ---")
    evaluate_attack(model, test_loader, device, mifgsm, epsilon=0.3, alpha=0.01, iters=40)
