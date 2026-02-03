Deep Learning (nn)
==================

Nalyst.nn provides a PyTorch-inspired deep learning framework built from scratch using NumPy.

Module Overview
---------------

- ``nn.Tensor`` - Tensor with automatic differentiation
- ``nn.Module`` - Base class for neural network layers
- ``nn.Sequential`` - Container for sequential layers
- ``nn.optim`` - Optimizers (SGD, Adam, etc.)
- ``nn.functional`` - Functional operations
- ``nn.layers`` - Pre-built layers

Quick Start
-----------

.. code-block:: python

    from nalyst import nn
    import numpy as np
    
    # Define a simple neural network
    class SimpleNet(nn.Module):
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
    
    # Create model
    model = SimpleNet(10, 32, 2)
    
    # Training
    optimizer = nn.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(nn.Tensor(X_train))
        loss = criterion(output, nn.Tensor(y_train))
        loss.backward()
        optimizer.step()

Tensor and Autograd
-------------------

Basic Tensor Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nalyst.nn import Tensor
    
    # Create tensors
    x = Tensor([1, 2, 3, 4], requires_grad=True)
    y = Tensor([5, 6, 7, 8])
    
    # Operations
    z = x + y
    z = x * y
    z = x @ y.reshape(4, 1)  # Matrix multiplication
    z = x.sum()
    z = x.mean()
    
    # Autograd
    z = (x ** 2).sum()
    z.backward()
    print(x.grad)  # [2, 4, 6, 8]

Layers
------

Linear Layers
~~~~~~~~~~~~~

.. code-block:: python

    from nalyst import nn
    
    # Fully connected layer
    linear = nn.Linear(in_features=64, out_features=32)
    
    # With bias disabled
    linear = nn.Linear(64, 32, bias=False)

Convolutional Layers
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # 2D Convolution
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1
    )
    
    # 1D Convolution
    conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
    
    # Transposed Convolution (deconvolution)
    convt = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

Recurrent Layers
~~~~~~~~~~~~~~~~

.. code-block:: python

    # LSTM
    lstm = nn.LSTM(
        input_size=10,
        hidden_size=32,
        num_layers=2,
        batch_first=True,
        bidirectional=True
    )
    output, (h_n, c_n) = lstm(x)
    
    # GRU
    gru = nn.GRU(input_size=10, hidden_size=32, num_layers=2)
    
    # Basic RNN
    rnn = nn.RNN(input_size=10, hidden_size=32)
    
    # Cell versions for manual control
    lstm_cell = nn.LSTMCell(10, 32)
    gru_cell = nn.GRUCell(10, 32)

Attention Layers
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Multi-Head Attention
    attention = nn.MultiHeadAttention(
        embed_dim=64,
        num_heads=8,
        dropout=0.1
    )
    output, attn_weights = attention(query, key, value)
    
    # Transformer Encoder Layer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=64,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1
    )
    
    # Full Transformer
    transformer = nn.Transformer(
        d_model=128,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )

Normalization Layers
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Batch Normalization
    bn1d = nn.BatchNorm1d(num_features=64)
    bn2d = nn.BatchNorm2d(num_features=64)
    
    # Layer Normalization
    ln = nn.LayerNorm(normalized_shape=64)
    
    # Instance Normalization
    inst = nn.InstanceNorm2d(num_features=64)
    
    # Group Normalization
    gn = nn.GroupNorm(num_groups=8, num_channels=64)

Activation Functions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # As layers
    relu = nn.ReLU()
    leaky_relu = nn.LeakyReLU(negative_slope=0.1)
    gelu = nn.GELU()
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    softmax = nn.Softmax(dim=-1)
    swish = nn.Swish()
    
    # Functional API
    from nalyst.nn import functional as F
    x = F.relu(x)
    x = F.gelu(x)
    x = F.softmax(x, dim=-1)

Pooling Layers
~~~~~~~~~~~~~~

.. code-block:: python

    # Max Pooling
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Average Pooling
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    # Adaptive Pooling (output size specified)
    adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    # Global Average Pooling
    gap = nn.GlobalAvgPool2d()

Dropout
~~~~~~~

.. code-block:: python

    dropout = nn.Dropout(p=0.5)
    dropout2d = nn.Dropout2d(p=0.2)  # Spatial dropout

Embedding
~~~~~~~~~

.. code-block:: python

    # For token embeddings
    embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128)
    
    x = nn.Tensor([[1, 2, 3], [4, 5, 6]])  # Token indices
    embedded = embedding(x)  # Shape: (2, 3, 128)

Optimizers
----------

.. code-block:: python

    # Stochastic Gradient Descent
    optimizer = nn.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Adam
    optimizer = nn.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    # AdamW (with weight decay)
    optimizer = nn.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # RMSprop
    optimizer = nn.optim.RMSprop(model.parameters(), lr=0.001)
    
    # Training loop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

Learning Rate Schedulers
------------------------

.. code-block:: python

    from nalyst.nn.optim import schedulers
    
    # Step decay
    scheduler = schedulers.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Exponential decay
    scheduler = schedulers.ExponentialLR(optimizer, gamma=0.95)
    
    # Cosine annealing
    scheduler = schedulers.CosineAnnealingLR(optimizer, T_max=100)
    
    # In training loop
    for epoch in range(epochs):
        train(...)
        scheduler.step()

Loss Functions
--------------

.. code-block:: python

    # Classification
    criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()  # Binary cross entropy
    criterion = nn.BCEWithLogitsLoss()
    criterion = nn.NLLLoss()
    
    # Regression
    criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    criterion = nn.SmoothL1Loss()
    criterion = nn.HuberLoss()
    
    # Other
    criterion = nn.KLDivLoss()
    criterion = nn.TripletMarginLoss()
    criterion = nn.CosineEmbeddingLoss()

Model Building Patterns
-----------------------

Sequential API
~~~~~~~~~~~~~~

.. code-block:: python

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10)
    )

ModuleList
~~~~~~~~~~

.. code-block:: python

    class DeepNet(nn.Module):
        def __init__(self, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(64, 64) for _ in range(num_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))
            return x

ModuleDict
~~~~~~~~~~

.. code-block:: python

    class MultiHeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(64, 32)
            self.heads = nn.ModuleDict({
                'classification': nn.Linear(32, 10),
                'regression': nn.Linear(32, 1)
            })

Pre-built Models
----------------

.. code-block:: python

    from nalyst.nn.models import VGG, ResNet, Transformer, VAE
    
    # VGG
    vgg16 = VGG(config='vgg16', num_classes=1000)
    
    # ResNet
    resnet18 = ResNet(config='resnet18', num_classes=1000)
    
    # Transformer
    transformer = Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    
    # VAE
    vae = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=32)

Model Utilities
---------------

.. code-block:: python

    # Count parameters
    num_params = nn.count_parameters(model)
    
    # Save model
    nn.save_model(model, 'model.pkl')
    
    # Load model
    checkpoint = nn.load_model(model, 'model.pkl')
    
    # Train/eval mode
    model.train()
    model.eval()
    
    # Get state dict
    state_dict = model.state_dict()
    model.load_state_dict(state_dict)
