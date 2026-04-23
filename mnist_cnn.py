import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load real data: handwritten digits 0-9 
transform = transforms.ToTensor()

train_data = datasets.MNIST(root="data", train=True,
                             download=True, transform=transform)
test_data  = datasets.MNIST(root="data", train=False,
                             download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"Training images: {len(train_data)}")
print(f"Test images:     {len(test_data)}")

# Build the CNN
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), # filter
            nn.ReLU(),
            nn.MaxPool2d(2),                            # shrink
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)                             # shrink again
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),           # 2D image → 1D vector
            nn.Linear(32*7*7, 128), # fully connected
            nn.ReLU(),
            nn.Linear(128, 10)      # 10 outputs: digits 0-9
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model     = DigitCNN()
loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for 3 epochs 
for epoch in range(3):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        pred = model(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Test accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred    = model(images)
            correct += (pred.argmax(1) == labels).sum().item()
    accuracy = correct / len(test_data)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.3f} | "
          f"Accuracy: {accuracy:.1%}")
