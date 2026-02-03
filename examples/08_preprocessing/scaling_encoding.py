"""
Preprocessing and Pipelines
===========================

This example demonstrates data preprocessing and pipeline workflows with Nalyst.

Topics covered:
- Scaling and normalization
- Encoding categorical variables
- Missing value imputation
- Feature selection
- Building pipelines
"""

import numpy as np
from nalyst import transform, workflow, learners

# Example 1: Standard Scaling

# add a banner so each preprocessing topic stands out
print("=" * 60)
print("Example 1: Standard Scaling")
print("=" * 60)

# fix randomness for reproducible stats
np.random.seed(42)

# create three features on wildly different scales
X = np.column_stack([
    np.random.randn(100) * 10 + 50,     # Feature 1: mean=50, std=10
    np.random.randn(100) * 1000 + 5000,  # Feature 2: mean=5000, std=1000
    np.random.randn(100) * 0.01          # Feature 3: mean=0, std=0.01
])

print("Original data statistics:")
print(f"  Feature 1: mean={X[:, 0].mean():.2f}, std={X[:, 0].std():.2f}")
print(f"  Feature 2: mean={X[:, 1].mean():.2f}, std={X[:, 1].std():.2f}")
print(f"  Feature 3: mean={X[:, 2].mean():.4f}, std={X[:, 2].std():.4f}")

# standardize to zero mean and unit variance
scaler = transform.StandardScaler()
X_scaled = scaler.train_transform(X)

print("\nAfter StandardScaler:")
print(f"  Feature 1: mean={X_scaled[:, 0].mean():.4f}, std={X_scaled[:, 0].std():.4f}")
print(f"  Feature 2: mean={X_scaled[:, 1].mean():.4f}, std={X_scaled[:, 1].std():.4f}")
print(f"  Feature 3: mean={X_scaled[:, 2].mean():.4f}, std={X_scaled[:, 2].std():.4f}")

# Example 2: MinMax Scaling

# demonstrate range scaling as another option
print("\n" + "=" * 60)
print("Example 2: MinMax Scaling")
print("=" * 60)

# scale to the default [0, 1] interval
minmax = transform.MinMaxScaler(feature_range=(0, 1))
X_minmax = minmax.train_transform(X)

print("After MinMaxScaler [0, 1]:")
print(f"  Feature 1: min={X_minmax[:, 0].min():.4f}, max={X_minmax[:, 0].max():.4f}")
print(f"  Feature 2: min={X_minmax[:, 1].min():.4f}, max={X_minmax[:, 1].max():.4f}")
print(f"  Feature 3: min={X_minmax[:, 2].min():.4f}, max={X_minmax[:, 2].max():.4f}")

# also show how to stretch to a custom range
minmax_custom = transform.MinMaxScaler(feature_range=(-1, 1))
X_custom = minmax_custom.train_transform(X)
print(f"\nMinMaxScaler [-1, 1]:")
print(f"  Range: [{X_custom.min():.4f}, {X_custom.max():.4f}]")

# Example 3: Robust Scaling

# highlight an option that ignores outliers
print("\n" + "=" * 60)
print("Example 3: Robust Scaling (outlier-resistant)")
print("=" * 60)

# inject a few extreme values to stress the scalers
X_outliers = np.random.randn(100, 3)
X_outliers[0, :] = [100, 200, -150]  # Add outliers

# compare the default scaler with the robust variant
std_scaler = transform.StandardScaler()
robust_scaler = transform.RobustScaler()

X_std = std_scaler.train_transform(X_outliers)
X_robust = robust_scaler.train_transform(X_outliers)

print("StandardScaler (affected by outliers):")
print(f"  Feature 1 range: [{X_std[:, 0].min():.2f}, {X_std[:, 0].max():.2f}]")

print("\nRobustScaler (resistant to outliers):")
print(f"  Feature 1 range: [{X_robust[:, 0].min():.2f}, {X_robust[:, 0].max():.2f}]")

# Example 4: One-Hot Encoding

# make the categorical encoding section obvious
print("\n" + "=" * 60)
print("Example 4: One-Hot Encoding")
print("=" * 60)

# list a tiny set of color labels for encoding
categories = np.array(['red', 'blue', 'green', 'blue', 'red', 'green', 'blue', 'red'])

encoder = transform.OneHotEncoder()
encoded = encoder.train_transform(categories.reshape(-1, 1))

print(f"Original categories: {categories}")
print(f"Encoded shape: {encoded.shape}")
print(f"Categories: {encoder.categories_}")
print(f"\nFirst 3 samples encoded:")
for i in range(3):
    print(f"  {categories[i]}: {encoded[i]}")

# Example 5: Label Encoding

# now show the simpler label encoder
print("\n" + "=" * 60)
print("Example 5: Label Encoding")
print("=" * 60)

labels = np.array(['low', 'medium', 'high', 'medium', 'low', 'high'])

encoder = transform.LabelEncoder()
encoded = encoder.train_transform(labels)

print(f"Original: {labels}")
print(f"Encoded: {encoded}")
print(f"Mapping: {encoder.classes_}")

# decode back to strings to prove the mapping is reversible
decoded = encoder.inverse_transform(encoded)
print(f"Decoded: {decoded}")

# Example 6: Missing Value Imputation

# clearly separate the imputation walkthrough
print("\n" + "=" * 60)
print("Example 6: Missing Value Imputation")
print("=" * 60)

# craft a small matrix with assorted NaNs
X = np.array([
    [1, 2, np.nan],
    [4, np.nan, 6],
    [7, 8, 9],
    [np.nan, 11, 12],
    [13, 14, 15]
])

print("Original data:")
print(X)

# show three different imputers back to back
mean_imputer = transform.SimpleImputer(strategy='mean')
X_mean = mean_imputer.train_transform(X)
print("\nAfter mean imputation:")
print(X_mean.round(2))

# Median imputation
median_imputer = transform.SimpleImputer(strategy='median')
X_median = median_imputer.train_transform(X)
print("\nAfter median imputation:")
print(X_median.round(2))

# KNN imputation
knn_imputer = transform.KNNImputer(n_neighbors=2)
X_knn = knn_imputer.train_transform(X)
print("\nAfter KNN imputation:")
print(X_knn.round(2))

# Example 7: Feature Selection

# explore quick feature filters before modeling
print("\n" + "=" * 60)
print("Example 7: Feature Selection")
print("=" * 60)

# simulate a blend of informative and noisy features
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 10)
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

# remove nearly constant features
var_selector = transform.VarianceThreshold(threshold=0.1)
X_var = var_selector.train_transform(X)
print(f"VarianceThreshold: {X.shape[1]}  {X_var.shape[1]} features")

# keep the top scoring features with selectkbest
kbest = transform.SelectKBest(k=5, score_func='f_regression')
X_kbest = kbest.train_transform(X, y)
print(f"SelectKBest (k=5): {X.shape[1]}  {X_kbest.shape[1]} features")
print(f"Selected features: {kbest.selected_features_}")
print(f"Feature scores: {kbest.scores_[:5].round(3)}")

# Example 8: Building Pipelines

# switch to a small end to end pipeline demo
print("\n" + "=" * 60)
print("Example 8: Building Pipelines")
print("=" * 60)

# generate noisy data with missing entries to motivate preprocessing
np.random.seed(42)
X = np.random.randn(200, 10)
# Add missing values
mask = np.random.random(X.shape) < 0.05
X[mask] = np.nan

y = (X[:, 0] + X[:, 1] > 0).astype(int)
y[np.isnan(y)] = 0  # Handle NaN in target

X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# wire the preprocessing steps together with a classifier
pipeline = workflow.Pipeline([
    ('imputer', transform.SimpleImputer(strategy='mean')),
    ('scaler', transform.StandardScaler()),
    ('classifier', learners.LogisticLearner())
])

print("Pipeline steps:")
for name, step in pipeline.steps:
    print(f"  {name}: {step.__class__.__name__}")

# fit the pipeline end to end then score it
pipeline.train(X_train, y_train)
y_pred = pipeline.infer(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\nPipeline accuracy: {accuracy:.4f}")

# Example 9: Column Transformer

# demonstrate per column preprocessing branches
print("\n" + "=" * 60)
print("Example 9: Column Transformer")
print("=" * 60)

# create side by side numeric and categorical arrays
numeric_data = np.random.randn(100, 3)
categorical_data = np.random.choice(['A', 'B', 'C'], size=(100, 2))

print(f"Numeric features shape: {numeric_data.shape}")
print(f"Categorical features shape: {categorical_data.shape}")

# build lightweight pipelines for each data type
numeric_transformer = workflow.Pipeline([
    ('scaler', transform.StandardScaler())
])

categorical_transformer = workflow.Pipeline([
    ('encoder', transform.OneHotEncoder())
])

# stitch the numeric and categorical branches together
col_transformer = transform.ColumnTransformer([
    ('num', numeric_transformer, [0, 1, 2]),
    ('cat', categorical_transformer, [3, 4])
])

# Combine data
X_combined = np.column_stack([numeric_data, categorical_data])
X_transformed = col_transformer.train_transform(X_combined)

print(f"\nOriginal shape: {X_combined.shape}")
print(f"Transformed shape: {X_transformed.shape}")

# Example 10: Full ML Pipeline

# close with a more realistic pipeline that mixes several steps
print("\n" + "=" * 60)
print("Example 10: Complete ML Pipeline")
print("=" * 60)

# generate a messy dataset with duplicates, zero variance, and missing values
np.random.seed(42)
n = 500
X = np.random.randn(n, 15)

# Add some problematic features
X[:, 10] = X[:, 0]  # Duplicate feature
X[:, 11] = 0  # Zero variance
mask = np.random.random((n, 15)) < 0.03
X[mask] = np.nan

y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)
y = np.where(np.isnan(y), 0, y)

# Split
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# assemble a comprehensive pipeline to clean and model the data
full_pipeline = workflow.Pipeline([
    ('imputer', transform.KNNImputer(n_neighbors=3)),
    ('scaler', transform.RobustScaler()),
    ('var_filter', transform.VarianceThreshold(threshold=0.01)),
    ('selector', transform.SelectKBest(k=8)),
    ('classifier', learners.GradientBoostingLearner(n_estimators=50))
])

print("Full pipeline:")
for name, _ in full_pipeline.steps:
    print(f"   {name}")

print("\nTraining pipeline...")
full_pipeline.train(X_train, y_train)

y_pred = full_pipeline.infer(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\nFinal accuracy: {accuracy:.4f}")

print("\n Preprocessing and pipeline examples completed!")
