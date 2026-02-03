"""
CNN Image Classification
========================

This example demonstrates Convolutional Neural Networks using Nalyst.nn.

Topics covered:
- Building CNN architectures
- Convolutional and pooling layers
- Image classification training
- Using pre-built models (VGG, ResNet)
"""

import numpy as np
from nalyst import nn

# Example 1: Simple CNN Architecture

# print a header so each cnn example is easy to spot
print("=" * 60)
print("Example 1: Building a Simple CNN")
print("=" * 60)


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # After 3 poolings: 28->14->7->3
        self.fc2 = nn.Linear(256, num_classes)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# spin up the simple cnn and inspect its size
model = SimpleCNN(num_classes=10)
print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# send a tiny random batch through to verify shapes
x = nn.Tensor(np.random.randn(4, 1, 28, 28))  # Batch of 4, 1 channel, 28x28
output = model(x)
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")

# Example 2: VGG-style Network

# Example 2: VGG-style Network

# call out the vgg style architecture
print("\n" + "=" * 60)
print("Example 2: VGG-style Network")
print("=" * 60)


def make_vgg_block(in_channels, out_channels, num_convs):
    """Create a VGG-style convolutional block."""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels if i == 0 else out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGGLike(nn.Module):
    """VGG-inspired network."""

    def __init__(self, num_classes=10):
        super().__init__()

        # VGG blocks
        self.block1 = make_vgg_block(3, 64, 2)   # 64 channels
        self.block2 = make_vgg_block(64, 128, 2) # 128 channels
        self.block3 = make_vgg_block(128, 256, 3) # 256 channels

        # Global average pooling instead of flatten
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# instantiate the vgg inspired model for reference
model = VGGLike(num_classes=10)
print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# Example 3: ResNet-style Network with Skip Connections

# Example 3: ResNet-style Network with Skip Connections

# add a banner for the residual network portion
print("\n" + "=" * 60)
print("Example 3: ResNet-style Network")
print("=" * 60)


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        out = self.relu(out)

        return out


class MiniResNet(nn.Module):
    """Simplified ResNet for demonstration."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


# build the mini resnet to demonstrate skip connections
model = MiniResNet(num_classes=10)
print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# double check tensor shapes using imagenet sized inputs
x = nn.Tensor(np.random.randn(2, 3, 224, 224))
output = model(x)
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")

# Example 4: Training a CNN

# Example 4: Training a CNN

# clearly mark the training demo
print("\n" + "=" * 60)
print("Example 4: Training a CNN")
print("=" * 60)

# generate small synthetic images so training finishes quickly
np.random.seed(42)
n_samples = 500
X = np.random.randn(n_samples, 1, 16, 16).astype(np.float32)
# Simple classification: images with positive mean = class 1
y = (X.mean(axis=(1, 2, 3)) > 0).astype(np.int64)

# reserve a validation split to measure accuracy
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# define a compact cnn suited for 16x16 inputs
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 4 * 4, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

optimizer = nn.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# run a short minibatch training loop
print("\nTraining CNN...")
model.train()

epochs = 10
batch_size = 32

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = nn.Tensor(X_train[batch_idx])
        y_batch = nn.Tensor(y_train[batch_idx])

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# evaluate on the held out set and compute accuracy
model.eval()
outputs = model(nn.Tensor(X_test))
predictions = np.argmax(outputs.data, axis=1)
accuracy = (predictions == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.2%}")

# Example 5: Using Pre-built Models

# Example 5: Using Pre-built Models

# wrap up by pointing to bundled model zoo entries
print("\n" + "=" * 60)
print("Example 5: Using Pre-built Models")
print("=" * 60)

# Import pre-built models (if available)
try:
    from nalyst.nn.models import VGG, ResNet, SimpleCNN as PrebuiltCNN

    # VGG-16
    vgg16 = VGG(config='vgg16', num_classes=1000)
    print(f"VGG-16 parameters: {nn.count_parameters(vgg16):,}")

    # ResNet-18
    resnet18 = ResNet(config='resnet18', num_classes=1000)
    print(f"ResNet-18 parameters: {nn.count_parameters(resnet18):,}")

except ImportError:
    print("Pre-built models available in nn.models module")
    print("- VGG (11, 13, 16, 19)")
    print("- ResNet (18, 34, 50, 101, 152)")
    print("- SimpleCNN")
    print("- MLP")

print("\n CNN examples completed!")
