"""
Linear Models
=============

This example demonstrates linear models in Nalyst.learners.

Topics covered:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Polynomial Regression
- Regularization effects
"""

import numpy as np
from nalyst import learners

# Example 1: Simple Linear Regression

# print a banner so readers can follow console output
print("=" * 60)
print("Example 1: Simple Linear Regression")
print("=" * 60)

# fix the seed so the sample run is deterministic
np.random.seed(42)

# generate a noisy linear trend to regress on
n = 100
X = np.random.randn(n, 1) * 5
y = 3 * X.squeeze() + 7 + np.random.randn(n) * 2

# train the basic linear learner to recover slope and intercept
model = learners.LinearLearner()
model.train(X, y)

print(f"True relationship: y = 3x + 7")
print(f"Learned: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")

# score the fit against the generated points
y_pred = model.infer(X)
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"R score: {r2:.4f}")

# Example 2: Multiple Linear Regression

print("\n" + "=" * 60)
print("Example 2: Multiple Linear Regression")
print("=" * 60)

# synthesize a few informative and noisy features
n = 200
X = np.random.randn(n, 5)
true_coefs = [3.0, -2.0, 1.5, 0.0, 0.0]  # Last two are noise
y = X @ true_coefs + 5 + np.random.randn(n) * 0.5

# keep a manual split to mimic train and test sets
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# train the learner on the training portion only
model = learners.LinearLearner()
model.train(X_train, y_train)

print("True coefficients:", true_coefs)
print(f"Learned coefficients: {model.coef_.round(4)}")
print(f"Intercept: {model.intercept_:.4f}")

# compute a couple of friendly error metrics
y_pred = model.infer(X_test)
mae = np.abs(y_test - y_pred).mean()
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nTest MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# Example 3: Ridge Regression (L2 Regularization)

# Example 3: Ridge Regression (L2 Regularization)

# call out the next section
print("\n" + "=" * 60)
print("Example 3: Ridge Regression (L2 Regularization)")
print("=" * 60)

# build a dataset with deliberate multicollinearity to stress ridge
X = np.random.randn(100, 10)
X[:, 1] = X[:, 0] + np.random.randn(100) * 0.1  # Collinear
y = X[:, 0] * 3 + X[:, 2] * 2 + np.random.randn(100)

# sweep a few alpha values to show shrinkage
alphas = [0, 0.1, 1.0, 10.0]
print(f"{'Alpha':<10} {'Coef Norm':>12} {'Coef[0]':>10} {'Coef[1]':>10}")
print("-" * 45)

for alpha in alphas:
    if alpha == 0:
        model = learners.LinearLearner()
    else:
        model = learners.RidgeLearner(alpha=alpha)
    model.train(X, y)

    coef_norm = np.sqrt(np.sum(model.coef_ ** 2))
    print(f"{alpha:<10} {coef_norm:>12.4f} {model.coef_[0]:>10.4f} {model.coef_[1]:>10.4f}")

print("\n Ridge shrinks coefficients, especially for collinear features")

# Example 4: Lasso Regression (L1 Regularization)

# Example 4: Lasso Regression (L1 Regularization)

# announce the lasso block for clarity
print("\n" + "=" * 60)
print("Example 4: Lasso Regression (L1 Regularization)")
print("=" * 60)

# craft sparse coefficients so lasso has something to zero out
n, p = 100, 20
X = np.random.randn(n, p)
true_coefs = np.zeros(p)
true_coefs[:3] = [5, -3, 2]  # Only first 3 matter
y = X @ true_coefs + np.random.randn(n) * 0.5

# test multiple alpha strengths to see sparsity shift
alphas = [0.01, 0.1, 0.5, 1.0]
print(f"{'Alpha':<10} {'Non-zero':>12} {'Coef[:3]'}")
print("-" * 50)

for alpha in alphas:
    model = learners.LassoLearner(alpha=alpha)
    model.train(X, y)

    nonzero = np.sum(np.abs(model.coef_) > 0.01)
    print(f"{alpha:<10} {nonzero:>12} {model.coef_[:3].round(3)}")

print(f"\nTrue coefs[:3]: {true_coefs[:3]}")
print(" Lasso produces sparse solutions (feature selection)")

# Example 5: Elastic Net (L1 + L2)

# Example 5: Elastic Net (Combined L1 + L2)

# add a quick comparison that mixes l1 and l2
print("\n" + "=" * 60)
print("Example 5: Elastic Net (Combined L1 + L2)")
print("=" * 60)

# elastic net balances sparsity with stability
model_lasso = learners.LassoLearner(alpha=0.1)
model_ridge = learners.RidgeLearner(alpha=0.1)
model_enet = learners.ElasticNetLearner(alpha=0.1, l1_ratio=0.5)

model_lasso.train(X, y)
model_ridge.train(X, y)
model_enet.train(X, y)

print(f"{'Model':<15} {'Non-zero':>12} {'Coef norm':>12}")
print("-" * 40)
print(f"{'Lasso':<15} {np.sum(np.abs(model_lasso.coef_) > 0.01):>12} "
      f"{np.sqrt(np.sum(model_lasso.coef_**2)):>12.4f}")
print(f"{'Ridge':<15} {np.sum(np.abs(model_ridge.coef_) > 0.01):>12} "
      f"{np.sqrt(np.sum(model_ridge.coef_**2)):>12.4f}")
print(f"{'ElasticNet':<15} {np.sum(np.abs(model_enet.coef_) > 0.01):>12} "
      f"{np.sqrt(np.sum(model_enet.coef_**2)):>12.4f}")

print("\n Elastic Net: sparsity of Lasso + stability of Ridge")

# Example 6: Polynomial Regression

# Example 6: Polynomial Regression

# clearly mark the polynomial section
print("\n" + "=" * 60)
print("Example 6: Polynomial Regression")
print("=" * 60)

from nalyst.transform import PolynomialFeatures

# simulate a cubic relationship with noise
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = 0.5 * X.squeeze() ** 3 - 2 * X.squeeze() ** 2 + X.squeeze() + np.random.randn(50) * 2

# test several polynomial degrees to illustrate overfitting
degrees = [1, 2, 3, 5]
print(f"{'Degree':<10} {'Train R':>12} {'Test R':>12}")
print("-" * 35)

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.train_transform(X)

    # use the first chunk for training to keep things simple
    model = learners.LinearLearner()
    model.train(X_poly[:40], y[:40])

    train_r2 = 1 - np.sum((y[:40] - model.infer(X_poly[:40])) ** 2) / np.sum((y[:40] - y[:40].mean()) ** 2)
    test_r2 = 1 - np.sum((y[40:] - model.infer(X_poly[40:])) ** 2) / np.sum((y[40:] - y[40:].mean()) ** 2)

    print(f"{degree:<10} {train_r2:>12.4f} {test_r2:>12.4f}")

print("\n Degree 3 matches true relationship (y = x - 2x + x)")

# Example 7: Regularization Path

# Example 7: Regularization Path

# begin the path demo for lasso shrinkage
print("\n" + "=" * 60)
print("Example 7: Regularization Path (Lasso)")
print("=" * 60)

# reuse a simple sparse setup for the path plot
X = np.random.randn(100, 10)
true_coefs = [3, 2, 1, 0.5, 0, 0, 0, 0, 0, 0]
y = X @ true_coefs + np.random.randn(100) * 0.5

# iterate over a log-spaced grid of alpha values
alphas = np.logspace(-3, 1, 20)
coef_path = []

for alpha in alphas:
    model = learners.LassoLearner(alpha=alpha)
    model.train(X, y)
    coef_path.append(model.coef_.copy())

coef_path = np.array(coef_path)

print("Lasso Regularization Path:")
print(f"{'log()':<10} {'Feat1':>8} {'Feat2':>8} {'Feat3':>8} {'Feat4':>8}")
print("-" * 45)
for i in [0, 5, 10, 15, 19]:
    print(f"{np.log10(alphas[i]):<10.2f} {coef_path[i, 0]:>8.3f} {coef_path[i, 1]:>8.3f} "
          f"{coef_path[i, 2]:>8.3f} {coef_path[i, 3]:>8.3f}")

print("\n As alpha increases, coefficients shrink to zero")

# Example 8: Cross-Validation for Regularization

# Example 8: Cross-Validation for Regularization

# flag the tuning section with a banner
print("\n" + "=" * 60)
print("Example 8: Cross-Validation for Alpha Selection")
print("=" * 60)

from nalyst.evaluation import cross_val_score

# try a handful of alpha values and keep the best score
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
print(f"{'Alpha':<10} {'Mean CV R':>15} {'Std':>10}")
print("-" * 40)

best_alpha = None
best_score = -np.inf

for alpha in alphas:
    model = learners.RidgeLearner(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    mean_score = scores.mean()

    print(f"{alpha:<10} {mean_score:>15.4f} {scores.std():>10.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha} (CV R = {best_score:.4f})")

# Example 9: Interpreting Linear Model

# Example 9: Interpreting Linear Model

# wrap up with a quick interpretability example
print("\n" + "=" * 60)
print("Example 9: Interpreting Linear Model Coefficients")
print("=" * 60)

# standardize so coefficient magnitudes become comparable
from nalyst.transform import StandardScaler

feature_names = ['age', 'income', 'education', 'experience', 'hours_week']
X = np.random.randn(200, 5)
X[:, 1] = X[:, 1] * 10000 + 50000  # income in different scale
y = 0.5 * X[:, 0] + 0.0001 * X[:, 1] + 0.8 * X[:, 2] - 0.3 * X[:, 3] + 0.1 * X[:, 4]

# train once on raw scales to show skewed weights
model_raw = learners.LinearLearner()
model_raw.train(X, y)

# then fit again on standardized inputs for a fair comparison
scaler = StandardScaler()
X_scaled = scaler.train_transform(X)
model_std = learners.LinearLearner()
model_std.train(X_scaled, y)

print("Coefficients comparison:")
print(f"{'Feature':<15} {'Raw Coef':>12} {'Std Coef':>12}")
print("-" * 40)
for i, name in enumerate(feature_names):
    print(f"{name:<15} {model_raw.coef_[i]:>12.6f} {model_std.coef_[i]:>12.4f}")

print("\n Standardized coefficients show relative importance")
print(" Education has the largest effect")

print("\n Linear models examples completed!")
