import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 1. Setup device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Load MNIST dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# 3. Define MLP model

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

model = MLP().to(device)

# 4. Loss + optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5. Training loop

epochs = 5

train_losses = []
accuracies = []

for epoch in range(epochs):
    running_loss = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate accuracy after each epoch
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    train_losses.append(running_loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch+1}, Loss: {running_loss:.2f}, Accuracy: {accuracy:.2f}%")


# 6. Save model

torch.save(model.state_dict(), "models/mlp_model.pth")

# 7. Plot 

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.legend()
plt.title("Training Loss")
plt.savefig("results/loss.png")

plt.figure()
plt.plot(accuracies, label="Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig("results/accuracy.png")

print("Training complete. Graphs saved.")


# 8. Test on handwritten digits

import os
import numpy as np
from PIL import Image

custom_path = "Handwritten-Digits-Spring-2026/digits"

images = []
labels = []

for file in os.listdir(custom_path):
    if file.endswith(".png"):
        label = int(file.split("-")[0])
        img_path = os.path.join(custom_path, file)

        img = Image.open(img_path).convert("L")
        img = img.resize((28, 28)) 
        img = np.array(img)
        img = np.array(img)

        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Apply same transform as MNIST
images = images / 255.0
images = (images - 0.5) / 0.5  # normalize to [-1,1]
images = images.reshape(-1, 1, 28, 28)

images_tensor = torch.tensor(images, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(labels).to(device)

# Run model
model.eval()
with torch.no_grad():
    outputs = model(images_tensor)
    _, predicted = torch.max(outputs, 1)

# Accuracy
correct = (predicted == labels_tensor).sum().item()
total = labels_tensor.size(0)
accuracy = 100 * correct / total

print(f"\nHandwritten Digit Accuracy: {accuracy:.2f}%")


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(labels_tensor.cpu(), predicted.cpu())

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Handwritten Digits)")
plt.savefig("results/confusion_matrix.png")