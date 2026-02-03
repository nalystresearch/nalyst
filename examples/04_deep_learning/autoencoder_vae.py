"""
Autoencoder and VAE
===================

This example demonstrates Autoencoders and Variational Autoencoders using Nalyst.nn.

Topics covered:
- Basic Autoencoder
- Convolutional Autoencoder
- Variational Autoencoder (VAE)
- Reconstruction and generation
"""

import numpy as np
from nalyst import nn

# Example 1: Basic Autoencoder

# print a banner so console output stays organized
print("=" * 60)
print("Example 1: Basic Autoencoder")
print("=" * 60)


class Autoencoder(nn.Module):
    """Simple autoencoder for dimensionality reduction."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # For normalized inputs
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z


# instantiate the autoencoder with a compact latent bottleneck
model = Autoencoder(input_dim=784, latent_dim=32)
print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# run a tiny random batch through the network to inspect shapes
x = nn.Tensor(np.random.rand(16, 784).astype(np.float32))
reconstructed, latent = model(x)

print(f"\nInput shape:        {x.shape}")
print(f"Latent shape:       {latent.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")

# Example 2: Training Autoencoder

# Example 2: Training Autoencoder

# call out the training phase
print("\n" + "=" * 60)
print("Example 2: Training Autoencoder")
print("=" * 60)

# generate synthetic data with mild structure so the autoencoder has signal
np.random.seed(42)
n_samples = 1000
input_dim = 50

# Create data with some structure
X = np.random.randn(n_samples, 10) @ np.random.randn(10, input_dim)
X = (X - X.min()) / (X.max() - X.min())  # Normalize to [0, 1]
X = X.astype(np.float32)

# mimic a simple train/test split
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]

# set up the autoencoder, optimizer, and loss
model = Autoencoder(input_dim=input_dim, latent_dim=10)
optimizer = nn.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# lightweight training loop with manual batching
print("\nTraining Autoencoder...")
model.train()

epochs = 30
batch_size = 32

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = nn.Tensor(X_train[batch_idx])

        optimizer.zero_grad()
        reconstructed, _ = model(X_batch)
        loss = criterion(reconstructed, X_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Reconstruction Loss: {total_loss:.6f}")

# evaluate on the held out data to see reconstruction quality
model.eval()
X_test_tensor = nn.Tensor(X_test)
reconstructed, latent = model(X_test_tensor)
test_loss = criterion(reconstructed, X_test_tensor)
print(f"\nTest Reconstruction Loss: {test_loss.data:.6f}")

# Example 3: Convolutional Autoencoder

# Example 3: Convolutional Autoencoder

# flag the convolutional variant for image data
print("\n" + "=" * 60)
print("Example 3: Convolutional Autoencoder")
print("=" * 60)


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for images."""

    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_fc(z)
        x = nn.Tensor(x.data.reshape(-1, 64, 7, 7))
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# create the convolutional model with a modest latent size
model = ConvAutoencoder(latent_dim=32)
print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# push a fake image batch through to validate tensor dimensions
x = nn.Tensor(np.random.rand(4, 1, 28, 28).astype(np.float32))
reconstructed, latent = model(x)
print(f"\nInput shape:        {x.shape}")
print(f"Latent shape:       {latent.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")

# Example 4: Variational Autoencoder (VAE)

# Example 4: Variational Autoencoder (VAE)

# move into probabilistic modeling with a vae banner
print("\n" + "=" * 60)
print("Example 4: Variational Autoencoder (VAE)")
print("=" * 60)


class VAE(nn.Module):
    """Variational Autoencoder for generative modeling."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation."""
        std = nn.Tensor(np.exp(0.5 * logvar.data))
        eps = nn.Tensor(np.random.randn(*mu.shape))
        return mu + std * eps

    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z


def vae_loss(reconstructed, original, mu, logvar):
    """VAE loss = Reconstruction loss + KL divergence."""
    # Reconstruction loss (binary cross entropy for normalized inputs)
    recon_loss = nn.F.binary_cross_entropy(reconstructed, original, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * np.sum(1 + logvar.data - mu.data ** 2 - np.exp(logvar.data))

    return recon_loss.data + kl_loss


# initialize the dense vae for mnist sized vectors
model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)
print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# verify all tensors line up by passing random noise
x = nn.Tensor(np.random.rand(16, 784).astype(np.float32))
reconstructed, mu, logvar, z = model(x)

print(f"\nInput shape:        {x.shape}")
print(f"Mu shape:           {mu.shape}")
print(f"LogVar shape:       {logvar.shape}")
print(f"Latent z shape:     {z.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")

# Example 5: Training VAE

# Example 5: Training VAE

# clearly label the vae training loop
print("\n" + "=" * 60)
print("Example 5: Training VAE")
print("=" * 60)

# craft synthetic latent structure so the vae has meaningful patterns
np.random.seed(42)
n_samples = 1000
input_dim = 100

# Data with latent structure
true_latent = np.random.randn(n_samples, 10)
X = np.random.rand(10, input_dim)  # Mixing matrix
X = 1 / (1 + np.exp(-true_latent @ X))  # Sigmoid activation
X = X.astype(np.float32)

# perform a simple split for evaluation
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]

# prepare the vae plus optimizer and reconstruction loss helper
model = VAE(input_dim=input_dim, hidden_dim=128, latent_dim=10)
optimizer = nn.optim.Adam(model.parameters(), lr=0.001)
recon_criterion = nn.BCELoss(reduction='sum')

# run a basic minibatch training routine
print("\nTraining VAE...")
model.train()

epochs = 30
batch_size = 32

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = nn.Tensor(X_train[batch_idx])

        optimizer.zero_grad()
        reconstructed, mu, logvar, _ = model(X_batch)

        # VAE loss
        loss = vae_loss(reconstructed, X_batch, mu, logvar)

        # Manual gradient computation for VAE
        # (In practice, use autograd for full backprop)
        recon_loss = recon_criterion(reconstructed, X_batch)
        recon_loss.backward()

        optimizer.step()

        total_loss += loss

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.2f}")

# sample from the latent prior to show generation capability
print("\n--- Generating New Samples ---")
model.eval()

# Sample from prior
z_samples = nn.Tensor(np.random.randn(10, 10))
generated = model.decode(z_samples)
print(f"Generated samples shape: {generated.shape}")
print(f"Generated values range: [{generated.data.min():.3f}, {generated.data.max():.3f}]")

# Example 6: Convolutional VAE

# Example 6: Convolutional VAE

# finish with a convolutional vae suited for images
print("\n" + "=" * 60)
print("Example 6: Convolutional VAE")
print("=" * 60)


class ConvVAE(nn.Module):
    """Convolutional VAE for image generation."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = nn.Tensor(np.exp(0.5 * logvar.data))
        eps = nn.Tensor(np.random.randn(*mu.shape))
        return mu + std * eps

    def decode(self, z):
        h = self.fc_decode(z)
        h = nn.Tensor(h.data.reshape(-1, 64, 7, 7))
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = ConvVAE(latent_dim=16)
print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# Test generation
z_samples = nn.Tensor(np.random.randn(8, 16))
generated = model.decode(z_samples)
print(f"\nGenerated images shape: {generated.shape}")

# Example 7: Using Pre-built VAE

print("\n" + "=" * 60)
print("Example 7: Using Pre-built VAE Model")
print("=" * 60)

try:
    from nalyst.nn.models import VAE as PrebuiltVAE

    vae = PrebuiltVAE(input_dim=784, hidden_dims=[512, 256], latent_dim=32)
    print(f"Pre-built VAE parameters: {nn.count_parameters(vae):,}")

except ImportError:
    print("Pre-built VAE available in nn.models:")
    print("  from nalyst.nn.models import VAE")
    print("  vae = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=32)")

print("\n Autoencoder and VAE examples completed!")
