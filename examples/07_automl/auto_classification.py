"""
AutoML Examples
===============

This example demonstrates automatic machine learning with Nalyst.automl.

Topics covered:
- AutoClassifier
- AutoRegressor
- Hyperparameter tuning
- Model comparison
- Feature importance
"""

import numpy as np
from nalyst import automl, learners, evaluation, transform

# Example 1: AutoClassifier Basic Usage

# print a banner so each automl capability is easy to spot
print("=" * 60)
print("Example 1: AutoClassifier Basic Usage")
print("=" * 60)

np.random.seed(42)

# generate a mildly non linear classification problem
n_samples = 500
n_features = 20

X = np.random.randn(n_samples, n_features)
# Non-linear decision boundary
y = ((X[:, 0] ** 2 + X[:, 1] ** 2 > 1.5) | (X[:, 2] > 0.5)).astype(int)

# create a simple train test split
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# configure the automl search constraints
auto_clf = automl.AutoClassifier(
    time_limit=60,  # seconds
    metric='accuracy',
    n_trials=10
)

# kick off the automated search
print("Training AutoClassifier...")
auto_clf.train(X_train, y_train)

print(f"\nBest model: {auto_clf.best_model_name_}")
print(f"Best parameters: {auto_clf.best_params_}")
print(f"Cross-validation score: {auto_clf.best_score_:.4f}")

# evaluate the best discovered model on held out data
y_pred = auto_clf.infer(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\nTest accuracy: {accuracy:.4f}")

# Example 2: AutoRegressor

# Example 2: AutoRegressor

# highlight the regression variant
print("\n" + "=" * 60)
print("Example 2: AutoRegressor")
print("=" * 60)

# craft a nonlinear regression signal
X = np.random.randn(500, 10)
y = 3 * X[:, 0] - 2 * X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 3] + np.random.randn(500) * 0.5

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# set up the autoregressor search configuration
auto_reg = automl.AutoRegressor(
    time_limit=60,
    metric='r2',
    n_trials=10
)

# search for the best regressor automatically
print("Training AutoRegressor...")
auto_reg.train(X_train, y_train)

print(f"\nBest model: {auto_reg.best_model_name_}")
print(f"Cross-validation R: {auto_reg.best_score_:.4f}")

# score the discovered regressor on the test portion
y_pred = auto_reg.infer(X_test)
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
mae = np.abs(y_test - y_pred).mean()
print(f"\nTest R: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

# Example 3: Model Comparison Report

# Example 3: Model Comparison Report

# run a manual leaderboard for reference
print("\n" + "=" * 60)
print("Example 3: Model Comparison")
print("=" * 60)

# Generate data
X = np.random.randn(300, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]

# train a few classic learners for side by side metrics
models = {
    'Logistic': learners.LogisticLearner(),
    'Random Forest': learners.RandomForestLearner(),
    'Gradient Boosting': learners.GradientBoostingLearner(),
    'SVM': learners.SVMLearner(kernel='rbf'),
    'KNN': learners.KNeighborsLearner(n_neighbors=5)
}

print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 60)

results = {}
for name, model in models.items():
    model.train(X_train, y_train)
    y_pred = model.infer(X_test)

    accuracy = (y_pred == y_test).mean()
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    print(f"{name:<20} {accuracy:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

best_model = max(results, key=lambda x: results[x]['f1'])
print(f"\n Best model by F1: {best_model}")

# Example 4: Hyperparameter Tuning

# banner for the grid search helper
print("\n" + "=" * 60)
print("Example 4: Hyperparameter Tuning")
print("=" * 60)

# define a small grid for random forest tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

tuner = automl.HyperparameterTuner(
    model_class=learners.RandomForestLearner,
    param_grid=param_grid,
    cv=5,
    metric='accuracy',
    search_type='grid'  # or 'random'
)

# execute the grid search and report best params
print("Running grid search...")
tuner.train(X_train, y_train)

print(f"\nBest parameters:")
for param, value in tuner.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest CV score: {tuner.best_score_:.4f}")

# reuse the tuned model on the test split
best_model = tuner.best_model_
y_pred = best_model.infer(X_test)
print(f"Test accuracy: {(y_pred == y_test).mean():.4f}")

# Example 5: Feature Importance Analysis

# Example 5: Feature Importance Analysis

# add a banner for the interpretability example
print("\n" + "=" * 60)
print("Example 5: Feature Importance Analysis")
print("=" * 60)

# generate data where only the first few features matter
np.random.seed(42)
X = np.random.randn(500, 10)
# Only first 3 features matter
y = (2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + np.random.randn(500) * 0.5 > 0).astype(int)

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# train a random forest so we can inspect feature importances
rf = learners.RandomForestLearner(n_estimators=100)
rf.train(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = [f'Feature_{i}' for i in range(10)]

# Sort by importance
indices = np.argsort(importances)[::-1]

print("Feature Importance Ranking:")
print("-" * 40)
for i in range(10):
    idx = indices[i]
    print(f"  {feature_names[idx]:<15} {importances[idx]:.4f}")

print("\n First 3 features should have highest importance")

# Example 6: Automated Feature Engineering

# Example 6: Automated Feature Engineering

# show how polynomial features can lift linear models
print("\n" + "=" * 60)
print("Example 6: Automated Feature Engineering")
print("=" * 60)

# expand features with pairwise interactions
X = np.random.randn(200, 3)
y = X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 0] * X[:, 1] + np.random.randn(200) * 0.1

# Polynomial feature generator
poly = transform.PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.train_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")

# compare the baseline linear fit against polynomial expansion
from nalyst.learners import LinearLearner

# Without polynomial features
model1 = LinearLearner()
model1.train(X[:160], y[:160])
pred1 = model1.infer(X[160:])
r2_1 = 1 - np.sum((y[160:] - pred1) ** 2) / np.sum((y[160:] - y[160:].mean()) ** 2)

# With polynomial features
model2 = LinearLearner()
model2.train(X_poly[:160], y[:160])
pred2 = model2.infer(X_poly[160:])
r2_2 = 1 - np.sum((y[160:] - pred2) ** 2) / np.sum((y[160:] - y[160:].mean()) ** 2)

print(f"\nLinear model R: {r2_1:.4f}")
print(f"Polynomial features R: {r2_2:.4f}")

# Example 7: Automated Pipeline

# Example 7: Automated Pipeline

# demonstrate a turnkey preprocessing plus modeling pipeline
print("\n" + "=" * 60)
print("Example 7: Automated ML Pipeline")
print("=" * 60)

# mix missing values and noise to motivate preprocessing
X = np.random.randn(300, 10)
# Add some missing values
mask = np.random.random(X.shape) < 0.1
X[mask] = np.nan
y = (X[:, 0] + X[:, 1] > 0).astype(float)
y[np.isnan(y)] = 0

# Split
X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]

# build an autopipeline that handles cleanup and modeling
from nalyst.workflow import Pipeline

pipeline = automl.AutoPipeline(
    preprocessing=['impute', 'scale'],
    model_type='classifier',
    time_limit=30
)

# launch the automated pipeline search
print("Training automated pipeline...")
pipeline.train(X_train, y_train)

print(f"\nSelected preprocessing:")
for step in pipeline.preprocessing_steps_:
    print(f"  - {step}")
print(f"Selected model: {pipeline.best_model_name_}")

# evaluate the fitted pipeline on clean labels
y_pred = pipeline.infer(X_test)
y_test_clean = y_test.copy()
accuracy = (y_pred == y_test_clean).mean()
print(f"\nTest accuracy: {accuracy:.4f}")

# Example 8: Cross-Validation Strategies

# Example 8: Cross-Validation Strategies

# recap multiple cross validation flavors
print("\n" + "=" * 60)
print("Example 8: Cross-Validation Strategies")
print("=" * 60)

X = np.random.randn(200, 5)
y = (X[:, 0] > 0).astype(int)

model = learners.LogisticLearner()

# plain kfold evaluation
kfold_scores = evaluation.cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"K-Fold (5):          {kfold_scores.mean():.4f}  {kfold_scores.std():.4f}")

# Stratified K-Fold (default for classification)
strat_scores = evaluation.cross_val_score(model, X, y, cv=5, scoring='accuracy',
                                          stratified=True)
print(f"Stratified K-Fold:   {strat_scores.mean():.4f}  {strat_scores.std():.4f}")

# Leave-One-Out (for small datasets)
# loo_scores = evaluation.cross_val_score(model, X[:50], y[:50], cv='loo')
# print(f"Leave-One-Out:       {loo_scores.mean():.4f}")

# Example 9: Early Stopping and Checkpoints

# finish with a gradient boosting run that leverages early stopping
print("\n" + "=" * 60)
print("Example 9: Training with Early Stopping")
print("=" * 60)

X = np.random.randn(1000, 20)
y = (X[:, :5].sum(axis=1) > 0).astype(int)

X_train, X_val, X_test = X[:600], X[600:800], X[800:]
y_train, y_val, y_test = y[:600], y[600:800], y[800:]

# Gradient boosting with early stopping
model = learners.GradientBoostingLearner(
    n_estimators=1000,
    learning_rate=0.1,
    early_stopping_rounds=10,
    validation_fraction=0.2
)

# train while monitoring validation loss to stop early
print("Training with early stopping...")
model.train(X_train, y_train)

print(f"\nBest iteration: {model.best_iteration_}")
print(f"Best validation score: {model.best_score_:.4f}")

y_pred = model.infer(X_test)
print(f"Test accuracy: {(y_pred == y_test).mean():.4f}")

print("\n AutoML examples completed!")
