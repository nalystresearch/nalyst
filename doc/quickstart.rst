==========
Quickstart
==========

This guide will help you get started with Nalyst in just a few minutes.

Machine Learning Example
------------------------

.. code-block:: python

    from nalyst.learners.linear import LogisticLearner
    from nalyst.evaluation import split_data, accuracy_score
    from nalyst.datasets import load_iris

    # Load data
    X, y = load_iris(return_X_y=True)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Train model
    model = LogisticLearner()
    model.train(X_train, y_train)

    # Make predictions
    predictions = model.infer(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

Deep Learning Example
---------------------

.. code-block:: python

    from nalyst import nn
    import numpy as np

    # Define a neural network
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Create model
    model = SimpleNet(784, 128, 10)

    # Define optimizer and loss
    optimizer = nn.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(nn.Tensor(X_train))
        loss = criterion(outputs, nn.Tensor(y_train))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.data:.4f}")

Statistical Analysis Example
----------------------------

.. code-block:: python

    from nalyst.stats import ttest_ind, shapiro, pearsonr
    import numpy as np

    # Generate sample data
    group1 = np.random.normal(100, 15, 50)
    group2 = np.random.normal(110, 15, 50)

    # Check normality
    stat, p_value = shapiro(group1)
    print(f"Shapiro-Wilk test: p-value = {p_value:.4f}")

    # Compare groups
    t_stat, p_value = ttest_ind(group1, group2)
    print(f"T-test: t = {t_stat:.3f}, p = {p_value:.4f}")

    # Correlation
    r, p = pearsonr(group1[:30], group2[:30])
    print(f"Correlation: r = {r:.3f}, p = {p:.4f}")

AutoML Example
--------------

.. code-block:: python

    from nalyst.automl import AutoClassifier
    from nalyst.datasets import load_breast_cancer

    # Load data
    X, y = load_breast_cancer(return_X_y=True)

    # AutoML finds the best model
    auto = AutoClassifier(time_budget=60)  # 60 seconds
    auto.train(X, y)

    # Get results
    print(auto.leaderboard())
    print(f"Best model: {auto.best_model_}")
