"""
Neural Network Basics
=====================

This example demonstrates the fundamentals of deep learning with Nalyst.nn.

Topics covered:
- Creating tensors with autograd
- Building neural network modules
- Training loops
- Model saving and loading
"""

import numpy as np
from nalyst import nn

# Example 1: Tensor Basics and Autograd

# print a banner so each concept is easy to spot
print("=" * 60)
print("Example 1: Tensor Basics and Autograd")
print("=" * 60)

# create a tensor that tracks gradients for autograd demos
x = nn.Tensor([1, 2, 3, 4], requires_grad=True)
print(f"Tensor x: {x}")
print(f"Shape: {x.shape}")
print(f"Requires grad: {x.requires_grad}")

# run a simple computation to set up gradients
y = x * 2
z = y.sum()
print(f"\nz = sum(x * 2) = {z.data}")

# backpropagate to see gradients on x
z.backward()
print(f"Gradient of x: {x.grad}")  # Should be [2, 2, 2, 2]

# redo with a slightly different operation to reinforce intuition
print("\n--- Gradient of x ---")
x = nn.Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(f"x = {x.data}")
print(f"y = sum(x) = {y.data}")
print(f"dy/dx = 2x = {x.grad}")  # Should be [2, 4, 6, 8]

# Example 2: Building a Simple Network

# clearly separate the first model example
print("\n" + "=" * 60)
print("Example 2: Building a Simple Network")
print("=" * 60)


class SimpleNet(nn.Module):
    """A simple 2-layer neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# create and print the toy model to understand its shape
model = SimpleNet(input_size=10, hidden_size=32, output_size=2)
print(model)

# show how many trainable parameters the model holds
num_params = nn.count_parameters(model)
print(f"\nTotal parameters: {num_params:,}")

# check that a forward pass works on random data
x = nn.Tensor(np.random.randn(5, 10))  # Batch of 5
output = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Example 3: Training Loop

# Example 3: Training Loop

# label the end to end training example
print("\n" + "=" * 60)
print("Example 3: Complete Training Loop")
print("=" * 60)

# synthesize an easy binary classification dataset
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

# keep a holdout chunk for evaluation
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Define model
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# instantiate the classifier and supporting training objects
model = Classifier()
optimizer = nn.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# basic minibatch training routine
print("\nTraining...")
model.train()

epochs = 20
batch_size = 32

for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        # Get batch
        batch_idx = indices[i:i+batch_size]
        X_batch = nn.Tensor(X_train[batch_idx])
        y_batch = nn.Tensor(y_train[batch_idx])

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# run evaluation and compute accuracy
model.eval()
X_test_tensor = nn.Tensor(X_test)
outputs = model(X_test_tensor)
predictions = np.argmax(outputs.data, axis=1)
accuracy = (predictions == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.2%}")

# Example 4: Using Sequential API

# Example 4: Using Sequential API

# demonstrate the sequential helper for quick prototypes
print("\n" + "=" * 60)
print("Example 4: Sequential API")
print("=" * 60)

# Quick model definition with Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)

print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# Example 5: Model Save and Load

# Example 5: Model Save and Load

# mark the persistence example clearly
print("\n" + "=" * 60)
print("Example 5: Model Save and Load")
print("=" * 60)

# create a small model whose weights we can inspect
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# inspect the state dict to show stored tensors
state_dict = model.state_dict()
print("State dict keys:")
for key in state_dict.keys():
    print(f"  {key}: shape {state_dict[key].shape}")

# save the checkpoint with a tiny bit of metadata
nn.save_model(model, '/tmp/model_checkpoint.pkl', extra={'epoch': 10})
print("\n Model saved to /tmp/model_checkpoint.pkl")

# reload into a fresh model instance
new_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)
checkpoint = nn.load_model(new_model, '/tmp/model_checkpoint.pkl')
print(f" Model loaded from epoch {checkpoint.get('epoch')}")

# Example 6: Different Layers

# Example 6: Different Layers

# finish with a quick tour of common layer types
print("\n" + "=" * 60)
print("Example 6: Layer Types")
print("=" * 60)

# run a few activation functions on the same input for comparison
activations = [
    ("ReLU", nn.ReLU()),
    ("LeakyReLU", nn.LeakyReLU(0.1)),
    ("GELU", nn.GELU()),
    ("Sigmoid", nn.Sigmoid()),
    ("Tanh", nn.Tanh()),
    ("Swish", nn.Swish()),
]

x = nn.Tensor(np.linspace(-2, 2, 5))
print(f"\nInput: {x.data}")
print(f"\n{'Activation':<15} {'Output'}")
print("-" * 50)
for name, layer in activations:
    out = layer(x)
    print(f"{name:<15} {out.data}")

# show how normalization layers reshape distributions
print("\n--- Normalization Layers ---")
batch = nn.Tensor(np.random.randn(4, 8))  # (batch, features)
bn = nn.BatchNorm1d(8)
ln = nn.LayerNorm(8)

print(f"Input mean: {batch.data.mean():.4f}, std: {batch.data.std():.4f}")
print(f"BatchNorm output mean: {bn(batch).data.mean():.4f}, std: {bn(batch).data.std():.4f}")
print(f"LayerNorm output mean: {ln(batch).data.mean():.4f}, std: {ln(batch).data.std():.4f}")

print("\n Neural network basics completed!")
