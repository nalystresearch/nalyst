"""
Tree-Based Models
=================

This example demonstrates tree-based models in Nalyst.learners.

Topics covered:
- Decision Trees
- Random Forest
- Gradient Boosting
- Feature importance
- Hyperparameter tuning
"""

import numpy as np
from nalyst import learners

# Example 1: Decision Tree Classifier

# add a banner so console logs stay organized
print("=" * 60)
print("Example 1: Decision Tree Classifier")
print("=" * 60)

# seed the generator for fully reproducible samples
np.random.seed(42)

# craft a tiny logical and dataset to keep rules obvious
n = 300
X = np.random.randn(n, 5)
y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)  # Logical AND

X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]

# train a shallow tree to keep the rules interpretable
model = learners.DecisionTreeLearner(max_depth=3)
model.train(X_train, y_train)

print(f"Tree depth: {model.max_depth}")
y_pred = model.infer(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test accuracy: {accuracy:.4f}")

# Example 2: Controlling Tree Complexity

# Example 2: Controlling Tree Complexity

# show how depth changes bias and variance
print("\n" + "=" * 60)
print("Example 2: Controlling Tree Complexity")
print("=" * 60)

depths = [1, 3, 5, 10, None]
print(f"{'Max Depth':<12} {'Train Acc':>12} {'Test Acc':>12}")
print("-" * 38)

for depth in depths:
    model = learners.DecisionTreeLearner(max_depth=depth)
    model.train(X_train, y_train)

    train_acc = (model.infer(X_train) == y_train).mean()
    test_acc = (model.infer(X_test) == y_test).mean()

    depth_str = str(depth) if depth else "None"
    print(f"{depth_str:<12} {train_acc:>12.4f} {test_acc:>12.4f}")

print("\n Shallow trees may underfit, deep trees may overfit")

# Example 3: Random Forest Classifier

# Example 3: Random Forest Classifier

# switch to a tougher dataset and compare ensembles
print("\n" + "=" * 60)
print("Example 3: Random Forest Classifier")
print("=" * 60)

# generate a nonlinear boundary so averaging helps
X = np.random.randn(500, 10)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 - X[:, 2] > 0.5).astype(int)

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# compare a single tree against a forest on identical splits
tree = learners.DecisionTreeLearner(max_depth=5)
forest = learners.RandomForestLearner(n_estimators=100, max_depth=5)

tree.train(X_train, y_train)
forest.train(X_train, y_train)

print("Model Comparison:")
print(f"  Decision Tree accuracy: {(tree.infer(X_test) == y_test).mean():.4f}")
print(f"  Random Forest accuracy: {(forest.infer(X_test) == y_test).mean():.4f}")

print("\n Random Forest reduces variance through ensemble")

# Example 4: Random Forest Hyperparameters

# Example 4: Random Forest Hyperparameters

# illustrate how estimator count affects accuracy
print("\n" + "=" * 60)
print("Example 4: Random Forest Hyperparameters")
print("=" * 60)

# Vary number of trees
n_trees = [10, 50, 100, 200]
print(f"{'n_estimators':<15} {'Test Accuracy':>15}")
print("-" * 32)

for n in n_trees:
    model = learners.RandomForestLearner(n_estimators=n, max_depth=5)
    model.train(X_train, y_train)
    acc = (model.infer(X_test) == y_test).mean()
    print(f"{n:<15} {acc:>15.4f}")

print("\n--- Other important parameters ---")
print(" max_features: number of features per split (sqrt, log2, or int)")
print(" min_samples_split: minimum samples to split a node")
print(" min_samples_leaf: minimum samples in a leaf")
print(" bootstrap: whether to use bootstrap samples")

# Example 5: Gradient Boosting Classifier

# Example 5: Gradient Boosting Classifier

# spotlight gradient boosting as another ensemble type
print("\n" + "=" * 60)
print("Example 5: Gradient Boosting Classifier")
print("=" * 60)

# Gradient Boosting
gb = learners.GradientBoostingLearner(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
gb.train(X_train, y_train)

print(f"Gradient Boosting accuracy: {(gb.infer(X_test) == y_test).mean():.4f}")

# Compare different learning rates
learning_rates = [0.01, 0.1, 0.5, 1.0]
print(f"\n{'Learning Rate':<15} {'Test Accuracy':>15}")
print("-" * 32)

for lr in learning_rates:
    model = learners.GradientBoostingLearner(
        n_estimators=100, learning_rate=lr, max_depth=3
    )
    model.train(X_train, y_train)
    acc = (model.infer(X_test) == y_test).mean()
    print(f"{lr:<15} {acc:>15.4f}")

# Example 6: Feature Importance

# Example 6: Feature Importance

# reset data so we know exactly which features matter
print("\n" + "=" * 60)
print("Example 6: Feature Importance")
print("=" * 60)

# generate data with known important features
X = np.random.randn(500, 10)
# Only first 3 features matter
y = (2 * X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

feature_names = [f'Feature_{i}' for i in range(10)]

# fit a random forest so we can read its importances
rf = learners.RandomForestLearner(n_estimators=100)
rf.train(X, y)

print("Random Forest Feature Importance:")
print("-" * 35)
indices = np.argsort(rf.feature_importances_)[::-1]
for i in range(10):
    idx = indices[i]
    bar = "" * int(rf.feature_importances_[idx] * 30)
    print(f"  {feature_names[idx]:<12} {rf.feature_importances_[idx]:.4f} {bar}")

print("\n Features 0, 1, 2 should have highest importance")

# Example 7: Decision Tree for Regression

# Example 7: Decision Tree for Regression

# pivot to regression to show versatility
print("\n" + "=" * 60)
print("Example 7: Decision Tree for Regression")
print("=" * 60)

# synthesize a smooth target with noise
X = np.random.randn(300, 5)
y = 3 * X[:, 0] + np.sin(X[:, 1] * np.pi) + np.random.randn(300) * 0.5

X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]

# compare a few depth values to see fit quality
depths = [3, 5, 10]
print(f"{'Max Depth':<12} {'Train R':>12} {'Test R':>12}")
print("-" * 38)

for depth in depths:
    model = learners.DecisionTreeLearner(max_depth=depth, task='regression')
    model.train(X_train, y_train)

    train_r2 = 1 - np.sum((y_train - model.infer(X_train))**2) / np.sum((y_train - y_train.mean())**2)
    test_r2 = 1 - np.sum((y_test - model.infer(X_test))**2) / np.sum((y_test - y_test.mean())**2)

    print(f"{depth:<12} {train_r2:>12.4f} {test_r2:>12.4f}")

# Example 8: Random Forest Regressor

# evaluate the regression variant of random forest
print("\n" + "=" * 60)
print("Example 8: Random Forest Regressor")
print("=" * 60)

rf_reg = learners.RandomForestLearner(n_estimators=100, task='regression')
rf_reg.train(X_train, y_train)

y_pred = rf_reg.infer(X_test)
r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
mae = np.abs(y_test - y_pred).mean()

print(f"Random Forest Regression:")
print(f"  R score: {r2:.4f}")
print(f"  MAE: {mae:.4f}")

# Example 9: Out-of-Bag Score

# turn on oob scoring to get a free validation estimate
print("\n" + "=" * 60)
print("Example 9: Out-of-Bag Estimation")
print("=" * 60)

# Random Forest with OOB score
rf_oob = learners.RandomForestLearner(
    n_estimators=100,
    oob_score=True
)
rf_oob.train(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.4f}")
print(f"Test Score: {(rf_oob.infer(X_test) == y_test).mean():.4f}")
print("\n OOB score is a free cross-validation estimate")

# Example 10: Visualizing Decision Rules

# finish with a tiny tree so we can print the learned rules
print("\n" + "=" * 60)
print("Example 10: Decision Rules")
print("=" * 60)

# generate a minimal dataset to keep the rule list short
X = np.random.randn(200, 3)
y = ((X[:, 0] > 0) & (X[:, 1] < 0.5)).astype(int)
feature_names = ['age', 'income', 'score']

tree = learners.DecisionTreeLearner(max_depth=2)
tree.train(X, y)

print("Decision Tree Rules (max_depth=2):")
print("-" * 40)
rules = tree.get_rules(feature_names=feature_names)
for rule in rules:
    print(f"  {rule}")

print("\n Simple trees are highly interpretable")

print("\n Tree-based models examples completed!")
