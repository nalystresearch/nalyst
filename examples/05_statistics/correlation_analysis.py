"""
Correlation Analysis
====================

This example demonstrates correlation analysis methods with Nalyst.stats.

Topics covered:
- Pearson correlation
- Spearman rank correlation
- Kendall tau
- Partial correlation
- Correlation matrices
"""

import numpy as np
from nalyst import stats

# Example 1: Pearson Correlation

# print a banner so each correlation flavor stands out
print("=" * 60)
print("Example 1: Pearson Correlation")
print("=" * 60)

np.random.seed(42)

# synthesize two correlated variables with noise
n = 100
x = np.random.randn(n)
y = 0.7 * x + np.random.randn(n) * 0.5  # r  0.7

result = stats.pearsonr(x, y)

print(f"Pearson r: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.6f}")
print(f"95% CI: [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")

# Interpretation
r = result['statistic']
if abs(r) < 0.3:
    strength = "weak"
elif abs(r) < 0.7:
    strength = "moderate"
else:
    strength = "strong"
direction = "positive" if r > 0 else "negative"
print(f"\nInterpretation: {strength} {direction} linear relationship")

# Example 2: Spearman Rank Correlation

# highlight the rank based approach
print("\n" + "=" * 60)
print("Example 2: Spearman Rank Correlation")
print("=" * 60)

# craft a monotonic but nonlinear mapping
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # y = x

result_pearson = stats.pearsonr(x, y)
result_spearman = stats.spearmanr(x, y)

print(f"Data: y = x")
print(f"\nPearson r:  {result_pearson['statistic']:.4f} (assumes linearity)")
print(f"Spearman : {result_spearman['statistic']:.4f} (monotonic relationship)")

print("\n Spearman captures perfect monotonic relationship despite non-linearity")

# Example 3: Kendall Tau

# banner for kendall tau which handles ties gracefully
print("\n" + "=" * 60)
print("Example 3: Kendall Tau Correlation")
print("=" * 60)

# Rankings with ties
x_ranks = np.array([1, 2, 3, 4, 5, 5, 7, 8, 9, 10])
y_ranks = np.array([1, 3, 2, 4, 6, 5, 7, 9, 8, 10])

result = stats.kendalltau(x_ranks, y_ranks)

print(f"Kendall : {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")

print("\nKendall tau is robust to ties and outliers")

# Example 4: Correlation Matrix

# move to a multivariate correlation matrix demo
print("\n" + "=" * 60)
print("Example 4: Correlation Matrix")
print("=" * 60)

# create four variables with known relationships
n = 100
var1 = np.random.randn(n)
var2 = 0.6 * var1 + np.random.randn(n) * 0.5
var3 = -0.4 * var1 + np.random.randn(n) * 0.7
var4 = np.random.randn(n)  # Independent

data = np.column_stack([var1, var2, var3, var4])
var_names = ['Var1', 'Var2', 'Var3', 'Var4']

# Compute correlation matrix
corr_matrix, pvalue_matrix = stats.corrmatrix(data)

print("Correlation Matrix:")
print("=" * 40)
header = "       " + "  ".join([f"{v:>7}" for v in var_names])
print(header)
for i, name in enumerate(var_names):
    row = f"{name:>6}"
    for j in range(len(var_names)):
        row += f"  {corr_matrix[i, j]:>7.3f}"
    print(row)

print("\nP-value Matrix:")
print("=" * 40)
print(header)
for i, name in enumerate(var_names):
    row = f"{name:>6}"
    for j in range(len(var_names)):
        row += f"  {pvalue_matrix[i, j]:>7.4f}"
    print(row)

# Example 5: Partial Correlation

# before/after controlling for a confounder
print("\n" + "=" * 60)
print("Example 5: Partial Correlation")
print("=" * 60)

# Confounded relationship
n = 200
z = np.random.randn(n)  # Confounder
x = 0.6 * z + np.random.randn(n) * 0.5
y = 0.7 * z + np.random.randn(n) * 0.5

# Simple correlation (spurious due to confounder)
simple_corr = stats.pearsonr(x, y)
print(f"Simple correlation (x, y): {simple_corr['statistic']:.4f}")

# Partial correlation controlling for z
partial_corr = stats.partial_corr(x, y, z)
print(f"Partial correlation (x, y | z): {partial_corr['statistic']:.4f}")
print(f"p-value: {partial_corr['pvalue']:.4f}")

print("\n After controlling for the confounder, the relationship weakens/disappears")

# Example 6: Point-Biserial Correlation

# mix a binary outcome with continuous scores
print("\n" + "=" * 60)
print("Example 6: Point-Biserial Correlation")
print("=" * 60)

# Continuous variable vs binary variable
scores = np.array([85, 90, 78, 92, 88, 76, 95, 82, 87, 91,
                   65, 70, 72, 68, 75, 62, 71, 69, 74, 67])
passed = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

result = stats.pointbiserialr(passed, scores)

print(f"Point-biserial r: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")
print(f"\nMean score (passed):  {scores[passed == 1].mean():.1f}")
print(f"Mean score (failed):  {scores[passed == 0].mean():.1f}")

# Example 7: Correlation with Missing Data

# show how pairwise deletion affects sample size
print("\n" + "=" * 60)
print("Example 7: Handling Missing Data")
print("=" * 60)

# Data with missing values
x = np.array([1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10])
y = np.array([2, 4, 6, np.nan, 10, 12, 14, 16, np.nan, 20])

# drop rows where either variable is missing
mask = ~(np.isnan(x) | np.isnan(y))
x_clean = x[mask]
y_clean = y[mask]

result = stats.pearsonr(x_clean, y_clean)
print(f"Original n: {len(x)}")
print(f"Valid pairs: {sum(mask)}")
print(f"Correlation: {result['statistic']:.4f}")

# Example 8: Testing Correlation Significance

# compare how sample size influences significance
print("\n" + "=" * 60)
print("Example 8: Testing Correlation Significance")
print("=" * 60)

# Small correlation with small sample
small_n = 10
x_small = np.random.randn(small_n)
y_small = 0.5 * x_small + np.random.randn(small_n)

# Large correlation with large sample
large_n = 500
x_large = np.random.randn(large_n)
y_large = 0.15 * x_large + np.random.randn(large_n)

result_small = stats.pearsonr(x_small, y_small)
result_large = stats.pearsonr(x_large, y_large)

print("Effect of sample size on significance:")
print(f"\nSmall sample (n={small_n}):")
print(f"  r = {result_small['statistic']:.3f}, p = {result_small['pvalue']:.4f}")
print(f"  {'Significant' if result_small['pvalue'] < 0.05 else 'Not significant'}")

print(f"\nLarge sample (n={large_n}):")
print(f"  r = {result_large['statistic']:.3f}, p = {result_large['pvalue']:.4f}")
print(f"  {'Significant' if result_large['pvalue'] < 0.05 else 'Not significant'}")

print("\n Large samples can detect small correlations as significant")
print("   Always consider effect size, not just p-value!")

# Example 9: Correlation vs Causation Demo

# wrap up with a tongue in cheek spurious correlation
print("\n" + "=" * 60)
print("Example 9: Spurious Correlation Example")
print("=" * 60)

# Generate spuriously correlated data
years = np.arange(2000, 2020)
ice_cream_sales = 100 + 5 * (years - 2000) + np.random.randn(20) * 10
shark_attacks = 50 + 2.5 * (years - 2000) + np.random.randn(20) * 5

result = stats.pearsonr(ice_cream_sales, shark_attacks)

print("Spurious correlation example:")
print(f"Ice cream sales vs Shark attacks: r = {result['statistic']:.3f}")
print(f"p-value: {result['pvalue']:.6f}")

print("\n Both are driven by a hidden confounder (summer temperature)")
print(" Correlation does NOT imply causation!")

print("\n Correlation analysis examples completed!")
