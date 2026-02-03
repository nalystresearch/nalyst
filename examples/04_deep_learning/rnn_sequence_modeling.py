"""
RNN and Sequence Modeling
=========================

This example demonstrates Recurrent Neural Networks using Nalyst.nn.

Topics covered:
- RNN, LSTM, GRU layers
- Sequence classification
- Sequence-to-sequence models
- Text generation basics
"""

import numpy as np
from nalyst import nn

# Example 1: Basic RNN

# print a banner so sequence demos stay organized
print("=" * 60)
print("Example 1: Basic RNN")
print("=" * 60)

# build a compact stacked rnn to highlight tensor shapes
rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    bidirectional=False
)

# inputs follow the (batch, seq_len, features) convention
x = nn.Tensor(np.random.randn(4, 15, 10))  # 4 sequences, 15 steps, 10 features
output, h_n = rnn(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")  # (batch, seq_len, hidden_size)
print(f"Hidden state: {h_n.shape}")     # (num_layers, batch, hidden_size)

# Example 2: LSTM with Bidirectional

# flag the bidirectional lstm example clearly
print("\n" + "=" * 60)
print("Example 2: Bidirectional LSTM")
print("=" * 60)

lstm = nn.LSTM(
    input_size=10,
    hidden_size=32,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

x = nn.Tensor(np.random.randn(4, 20, 10))
output, (h_n, c_n) = lstm(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")  # (batch, seq_len, 2*hidden_size)
print(f"Hidden state: {h_n.shape}")     # (2*num_layers, batch, hidden_size)
print(f"Cell state:   {c_n.shape}")

# Example 3: GRU for Sequence Classification

# banner for the gru classifier portion
print("\n" + "=" * 60)
print("Example 3: GRU for Sequence Classification")
print("=" * 60)


class SequenceClassifier(nn.Module):
    """Classify sequences using GRU."""

    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # run the gru and collect the stacked hidden states
        output, h_n = self.gru(x)

        # use the last hidden state from both directions
        # h_n shape: (num_layers * 2, batch, hidden)
        # concatenate last layer's forward and backward hidden states
        hidden = nn.Tensor(np.concatenate([
            h_n.data[-2, :, :],  # Last forward
            h_n.data[-1, :, :]   # Last backward
        ], axis=-1))

        hidden = self.dropout(hidden)
        output = self.fc(hidden)

        return output


model = SequenceClassifier(input_size=10, hidden_size=64, num_classes=5)
print(model)

# sanity check with random sequences
x = nn.Tensor(np.random.randn(8, 30, 10))  # 8 sequences, 30 steps
output = model(x)
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")

# Example 4: Training LSTM for Sequence Classification

# clearly mark the training walkthrough
print("\n" + "=" * 60)
print("Example 4: Training LSTM Classifier")
print("=" * 60)

# generate synthetic sequences so training stays lightweight
np.random.seed(42)
n_samples = 500
seq_len = 20
n_features = 5
n_classes = 3

# define labels based on a simple statistic for clarity
X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
y = np.digitize(X.mean(axis=(1, 2)), bins=[-0.3, 0.3]) - 1
y = y.clip(0, n_classes - 1).astype(np.int64)

# keep a validation chunk for evaluation
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        # Use last time step
        last_hidden = nn.Tensor(h_n.data[-1])
        return self.fc(last_hidden)


# instantiate the classifier and optimizer pieces
model = LSTMClassifier()
optimizer = nn.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# run a minimal minibatch training loop
print("\nTraining LSTM...")
model.train()

epochs = 20
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

# evaluate on held out sequences and report accuracy
model.eval()
outputs = model(nn.Tensor(X_test))
predictions = np.argmax(outputs.data, axis=1)
accuracy = (predictions == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.2%}")

# Example 5: Sequence-to-Sequence (Encoder-Decoder)

# banner for the seq2seq encoder decoder pair
print("\n" + "=" * 60)
print("Example 5: Sequence-to-Sequence Model")
print("=" * 60)


class Seq2SeqEncoder(nn.Module):
    """Encoder for Seq2Seq model."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs, (h_n, c_n)


class Seq2SeqDecoder(nn.Module):
    """Decoder for Seq2Seq model."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        outputs, (h_n, c_n) = self.lstm(x, hidden)
        predictions = self.fc(outputs)
        return predictions, (h_n, c_n)


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.encoder = Seq2SeqEncoder(input_size, hidden_size, num_layers)
        self.decoder = Seq2SeqDecoder(output_size, hidden_size, output_size, num_layers)
        self.hidden_size = hidden_size

    def forward(self, src, tgt):
        # encode the source sequence into hidden states
        _, encoder_hidden = self.encoder(src)

        # decode the target sequence with teacher forcing input
        decoder_output, _ = self.decoder(tgt, encoder_hidden)

        return decoder_output


# instantiate the seq2seq stack and inspect it
model = Seq2Seq(input_size=10, hidden_size=64, output_size=8)
print(model)

# feed random tensors through to confirm output shapes
src = nn.Tensor(np.random.randn(4, 15, 10))  # Source sequence
tgt = nn.Tensor(np.random.randn(4, 10, 8))   # Target sequence (teacher forcing)
output = model(src, tgt)
print(f"\nSource shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Output shape: {output.shape}")

# Example 6: Using RNN Cells for Custom Logic

# finish with a low level cell example for manual control
print("\n" + "=" * 60)
print("Example 6: Using RNN Cells")
print("=" * 60)


class CustomRNN(nn.Module):
    """RNN with custom cell-level control."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # initialize hidden and cell states explicitly
        h = nn.Tensor(np.zeros((batch_size, self.hidden_size)))
        c = nn.Tensor(np.zeros((batch_size, self.hidden_size)))

        outputs = []

        # walk through the sequence one timestep at a time
        for t in range(seq_len):
            x_t = nn.Tensor(x.data[:, t, :])
            h, c = self.cell(x_t, (h, c))
            out = self.fc(h)
            outputs.append(out.data)

        # Stack outputs
        return nn.Tensor(np.stack(outputs, axis=1))


# verify the custom cell returns the expected shapes
model = CustomRNN(input_size=5, hidden_size=16)
x = nn.Tensor(np.random.randn(4, 10, 5))
output = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")

print("\n RNN examples completed!")
