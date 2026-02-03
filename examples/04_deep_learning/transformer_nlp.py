"""
Transformer and NLP
===================

This example demonstrates Transformer architectures using Nalyst.nn.

Topics covered:
- Multi-Head Attention
- Transformer Encoder/Decoder
- Positional Encoding
- Text Classification with Transformers
"""

import numpy as np
from nalyst import nn

# Example 1: Multi-Head Attention

# print a banner so each transformer topic stands out
print("=" * 60)
print("Example 1: Multi-Head Attention")
print("=" * 60)

# create a multi head attention block to inspect shapes
attention = nn.MultiHeadAttention(
    embed_dim=64,
    num_heads=8,
    dropout=0.1
)

# inputs follow (batch, seq_len, embed_dim)
query = nn.Tensor(np.random.randn(4, 10, 64))
key = nn.Tensor(np.random.randn(4, 10, 64))
value = nn.Tensor(np.random.randn(4, 10, 64))

# Self-attention (Q=K=V)
output, attn_weights = attention(query, key, value)

print(f"Query shape:   {query.shape}")
print(f"Key shape:     {key.shape}")
print(f"Value shape:   {value.shape}")
print(f"Output shape:  {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")

# Example 2: Transformer Encoder Layer

# highlight a single encoder layer example
print("\n" + "=" * 60)
print("Example 2: Transformer Encoder Layer")
print("=" * 60)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=64,
    nhead=8,
    dim_feedforward=256,
    dropout=0.1
)

# Input sequence
x = nn.Tensor(np.random.randn(4, 20, 64))  # (batch, seq_len, d_model)
output = encoder_layer(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")

# Example 3: Full Transformer Encoder

# call out the stacked encoder variant
print("\n" + "=" * 60)
print("Example 3: Stacked Transformer Encoder")
print("=" * 60)

# Create encoder layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=8,
    dim_feedforward=512,
    dropout=0.1
)

# Stack multiple layers
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Test
x = nn.Tensor(np.random.randn(4, 50, 128))
output = encoder(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Encoder has {6} layers")

# Example 4: Positional Encoding

# banner for positional encoding demos
print("\n" + "=" * 60)
print("Example 4: Positional Encoding")
print("=" * 60)

# Sinusoidal positional encoding
pos_encoder = nn.PositionalEncoding(d_model=64, max_len=100, dropout=0.1)

# Input embeddings
x = nn.Tensor(np.random.randn(4, 30, 64))
output = pos_encoder(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Max sequence length: 100")

# Learned positional encoding
learned_pos = nn.LearnedPositionalEncoding(d_model=64, max_len=100)
output_learned = learned_pos(x)
print(f"\nLearned positional encoding output: {output_learned.shape}")

# Example 5: Text Classification with Transformer

# clearly mark the transformer classifier example
print("\n" + "=" * 60)
print("Example 5: Transformer for Text Classification")
print("=" * 60)


class TransformerClassifier(nn.Module):
    """Transformer-based text classifier."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, max_len=512):
        super().__init__()

        # embed incoming token ids
        self.embedding = nn.Embedding(vocab_size, d_model)

        # add positional information to the tokens
        self.pos_encoder = nn.PositionalEncoding(d_model, max_len, dropout=0.1)

        # stack transformer encoder layers for sequence understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # simple mlp head for classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len) - token indices

        # embed tokens
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # add positions for order awareness
        x = self.pos_encoder(x)

        # run through the encoder stack
        x = self.transformer(x)

        # use the first token as a classification summary
        cls_output = nn.Tensor(x.data[:, 0, :])  # (batch, d_model)

        # classify the sentence
        logits = self.classifier(cls_output)

        return logits


model = TransformerClassifier(
    vocab_size=10000,
    d_model=256,
    nhead=8,
    num_layers=4,
    num_classes=5,
    max_len=256
)

print(model)
print(f"\nTotal parameters: {nn.count_parameters(model):,}")

# Test with token indices
tokens = nn.Tensor(np.random.randint(0, 10000, (4, 50)))  # (batch, seq_len)
output = model(tokens)
print(f"\nInput shape (tokens): {tokens.shape}")
print(f"Output shape:         {output.shape}")

# Example 6: Full Transformer (Encoder-Decoder)

# banner for the complete seq2seq transformer
print("\n" + "=" * 60)
print("Example 6: Full Transformer (Seq2Seq)")
print("=" * 60)

transformer = nn.Transformer(
    d_model=128,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=512,
    dropout=0.1
)

print(transformer)

# create fake source and target embeddings to test shapes
src = nn.Tensor(np.random.randn(4, 30, 128))  # (batch, src_len, d_model)
tgt = nn.Tensor(np.random.randn(4, 20, 128))  # (batch, tgt_len, d_model)

# Create causal mask for decoder
from nalyst.nn.layers.attention import generate_square_subsequent_mask
tgt_mask = generate_square_subsequent_mask(20)

output = transformer(src, tgt, tgt_mask=tgt_mask)

print(f"\nSource shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Output shape: {output.shape}")

# Example 7: Training a Simple Transformer

# clearly mark the training walkthrough
print("\n" + "=" * 60)
print("Example 7: Training Transformer Classifier")
print("=" * 60)

# generate tiny synthetic token sequences so training is fast
np.random.seed(42)
n_samples = 500
seq_len = 20
vocab_size = 100
n_classes = 3

# create random token ids and simple labels
X = np.random.randint(0, vocab_size, (n_samples, seq_len)).astype(np.int64)
# Class based on average token value
y = np.digitize(X.mean(axis=1), bins=[33, 66]).astype(np.int64)

# hold out part of the data for evaluation
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.pos_enc = nn.PositionalEncoding(32, max_len=seq_len, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(32, nhead=4, dim_feedforward=64)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        # mean pool across time to get a sentence vector
        x = nn.Tensor(x.data.mean(axis=1))
        return self.fc(x)


model = SimpleTransformer()
optimizer = nn.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# train with a lightweight minibatch loop
print("\nTraining Transformer...")
model.train()

epochs = 15
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

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# evaluate on the held out set and compute accuracy
model.eval()
outputs = model(nn.Tensor(X_test))
predictions = np.argmax(outputs.data, axis=1)
accuracy = (predictions == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.2%}")

print("\n Transformer examples completed!")
