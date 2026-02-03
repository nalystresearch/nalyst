"""
Basic Regression Example
========================

This example demonstrates basic regression using Nalyst.

Topics covered:
- Loading regression datasets
- Training regression models
- Evaluating with regression metrics
- Visualization
"""

import numpy as np
from nalyst.learners.linear import (
    LinearLearner,
    RidgeLearner,
    LassoLearner,
    ElasticNetLearner
)
from nalyst.learners.tree import TreeRegressor, ForestRegressor
from nalyst.evaluation import split_data, cross_validate
from nalyst.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)
from nalyst.datasets import load_boston, load_diabetes

# Example 1: Linear Regression on Boston Housing

# print a header so results are easy to follow
print("=" * 60)
print("Example 1: Boston Housing - Linear Regression")
print("=" * 60)

# load the classic boston housing data for a simple regression target
X, y = load_boston(return_X_y=True)
print(f"Dataset shape: {X.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# reserve a portion for honest evaluation
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# train the vanilla linear learner as a baseline
model = LinearLearner()
model.train(X_train, y_train)

# run predictions on the holdout slice
y_pred = model.infer(X_test)

# print a small set of friendly regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResults:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R:   {r2:.4f}")

# peek at the largest coefficients for interpretability
print(f"\nTop 5 most important features:")
coefficients = model.coefficients_
indices = np.argsort(np.abs(coefficients))[::-1][:5]
for i in indices:
    print(f"  Feature {i}: {coefficients[i]:.4f}")

# Example 2: Regularized Regression Comparison

# start a tiny regularization comparison to show tradeoffs
print("\n" + "=" * 60)
print("Example 2: Regularization Comparison")
print("=" * 60)

# define a few linear variants with different penalties
regressors = {
    "Linear (No regularization)": LinearLearner(),
    "Ridge (L2)": RidgeLearner(alpha=1.0),
    "Lasso (L1)": LassoLearner(alpha=0.1),
    "ElasticNet (L1+L2)": ElasticNetLearner(alpha=0.1, l1_ratio=0.5),
}

print(f"\n{'Model':<30} {'RMSE':<12} {'MAE':<12} {'R':<12}")
print("-" * 65)

for name, reg in regressors.items():
    reg.train(X_train, y_train)
    y_pred = reg.infer(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name:<30} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f}")

# Example 3: Tree-based Regressors

# switch to tree ensembles to show nonlinear behavior
print("\n" + "=" * 60)
print("Example 3: Tree-based Regressors")
print("=" * 60)

# train a compact decision tree for comparison
tree = TreeRegressor(max_depth=5)
tree.train(X_train, y_train)
y_pred_tree = tree.infer(X_test)

# follow up with a random forest for more stability
forest = ForestRegressor(n_estimators=100, max_depth=10)
forest.train(X_train, y_train)
y_pred_forest = forest.infer(X_test)

print(f"\n{'Model':<20} {'RMSE':<12} {'R':<12}")
print("-" * 45)
print(f"{'Decision Tree':<20} {root_mean_squared_error(y_test, y_pred_tree):<12.4f} {r2_score(y_test, y_pred_tree):<12.4f}")
print(f"{'Random Forest':<20} {root_mean_squared_error(y_test, y_pred_forest):<12.4f} {r2_score(y_test, y_pred_forest):<12.4f}")

# print the top feature importances to highlight drivers
print(f"\nRandom Forest Feature Importances:")
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1][:5]
for i, idx in enumerate(indices):
    print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")

# Example 4: Cross-Validation for Regression

# give an example of cross validation on regression metrics
print("\n" + "=" * 60)
print("Example 4: Cross-Validation")
print("=" * 60)

model = ForestRegressor(n_estimators=100)

# evaluate multiple metrics at once to show flexibility
cv_results = cross_validate(
    model, X, y, cv=5,
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
)

print(f"\n5-Fold Cross-Validation Results:")
print(f"  R:   {cv_results['test_r2'].mean():.4f} (+/- {cv_results['test_r2'].std() * 2:.4f})")
print(f"  MSE:  {-cv_results['test_neg_mean_squared_error'].mean():.4f}")
print(f"  MAE:  {-cv_results['test_neg_mean_absolute_error'].mean():.4f}")

# Example 5: Prediction Intervals

# close with a lightweight confidence interval demo
print("\n" + "=" * 60)
print("Example 5: Predictions with Confidence")
print("=" * 60)

# reuse the forest to approximate prediction intervals
model = ForestRegressor(n_estimators=100)
model.train(X_train, y_train)
y_pred = model.infer(X_test)

# estimate a residual standard deviation for intervals
residuals = y_test - y_pred
std_error = np.std(residuals)

print(f"\nPredictions for first 5 samples:")
print(f"{'Actual':<12} {'Predicted':<12} {'95% CI':<20}")
print("-" * 45)

for i in range(5):
    lower = y_pred[i] - 1.96 * std_error
    upper = y_pred[i] + 1.96 * std_error
    print(f"{y_test[i]:<12.2f} {y_pred[i]:<12.2f} [{lower:.2f}, {upper:.2f}]")

print("\n Regression examples completed!")
