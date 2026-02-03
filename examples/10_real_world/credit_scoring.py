"""
Real-World Example: Credit Scoring
==================================

This example demonstrates a complete ML workflow for credit scoring.

Topics covered:
- Data preprocessing
- Feature engineering
- Model selection
- Model evaluation
- Production considerations
"""

import numpy as np
from nalyst import learners, transform, workflow, evaluation, explainability, imbalance

# Generate Synthetic Credit Data

print("=" * 60)
print("Credit Scoring Example - Complete ML Pipeline")
print("=" * 60)

np.random.seed(42)
# keep synthetic portfolio stable each run

n_samples = 5000

# Generate features
data = {
    'age': np.random.normal(40, 12, n_samples).clip(18, 75),
    'income': np.random.lognormal(10.5, 0.6, n_samples),
    'employment_length': np.random.exponential(5, n_samples).clip(0, 30),
    'debt_to_income': np.random.beta(2, 5, n_samples),
    'credit_history_length': np.random.exponential(10, n_samples).clip(0, 40),
    'num_credit_lines': np.random.poisson(5, n_samples),
    'num_delinquencies': np.random.poisson(0.5, n_samples),
    'credit_utilization': np.random.beta(3, 7, n_samples),
    'loan_amount': np.random.lognormal(9, 0.8, n_samples),
    'purpose': np.random.choice(['home', 'car', 'education', 'other'], n_samples),
}

# assemble a quick mix of numeric credit features
# Create feature matrix
feature_names = list(data.keys())[:-1]  # Exclude 'purpose' for now
X_numeric = np.column_stack([data[f] for f in feature_names])

# Generate target (default: 0=no default, 1=default)
# Defaults are more likely with high debt, low income, more delinquencies
default_prob = 1 / (1 + np.exp(
    -(-2 +
      0.5 * data['debt_to_income'] * 10 -
      0.00001 * data['income'] +
      0.3 * data['num_delinquencies'] +
      0.5 * data['credit_utilization'] * 5 -
      0.05 * data['employment_length'] +
      np.random.randn(n_samples) * 0.5)
))
y = (np.random.random(n_samples) < default_prob).astype(int)

# target mimics real drivers like high debt and utilization
print(f"Total samples: {n_samples}")
print(f"Default rate: {y.mean():.2%}")
print(f"Features: {len(feature_names)}")

# Data Preprocessing

print("\n" + "=" * 60)
print("Step 1: Data Preprocessing")
print("=" * 60)

# Split data
split_idx = int(0.8 * n_samples)
X_train, X_test = X_numeric[:split_idx], X_numeric[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# simple chronological split keeps story readable
print(f"Training set: {len(y_train)} samples")
print(f"Test set: {len(y_test)} samples")

# Check for missing values (simulate some)
mask = np.random.random(X_train.shape) < 0.02
X_train_missing = X_train.copy()
X_train_missing[mask] = np.nan

# sprinkle some missing data to push the preprocessing stack
print(f"\nSimulated missing values: {mask.sum()}")

# Preprocessing pipeline
preprocessor = workflow.Pipeline([
    ('imputer', transform.KNNImputer(n_neighbors=5)),
    ('scaler', transform.RobustScaler()),  # Robust to outliers
])

# pipeline keeps later deployment consistent
X_train_processed = preprocessor.train_transform(X_train_missing)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing applied: KNN Imputation + Robust Scaling")

# Handle Class Imbalance

print("\n" + "=" * 60)
print("Step 2: Handle Class Imbalance")
print("=" * 60)

print(f"Original class distribution:")
print(f"  No Default: {np.sum(y_train == 0)} ({np.mean(y_train == 0):.1%})")
print(f"  Default:    {np.sum(y_train == 1)} ({np.mean(y_train == 1):.1%})")

# defaults are rare, so we rebalance before model search
# Apply SMOTE
smote = imbalance.SMOTE(sampling_strategy='auto')
X_train_balanced, y_train_balanced = smote.train_resample(X_train_processed, y_train)

print(f"\nAfter SMOTE:")
print(f"  No Default: {np.sum(y_train_balanced == 0)}")
print(f"  Default:    {np.sum(y_train_balanced == 1)}")

# Model Selection

print("\n" + "=" * 60)
print("Step 3: Model Selection via Cross-Validation")
print("=" * 60)

models = {
    'Logistic Regression': learners.LogisticLearner(),
    'Random Forest': learners.RandomForestLearner(n_estimators=100),
    'Gradient Boosting': learners.GradientBoostingLearner(n_estimators=100),
    'SVM': learners.SVMLearner(kernel='rbf', probability=True),
}

# compare a few classic scorers using roc auc
print(f"{'Model':<22} {'CV AUC':>10} {'CV Std':>10}")
print("-" * 45)

results = {}
for name, model in models.items():
    scores = evaluation.cross_val_score(
        model, X_train_balanced, y_train_balanced,
        cv=5, scoring='roc_auc'
    )
    results[name] = scores.mean()
    print(f"{name:<22} {scores.mean():>10.4f} {scores.std():>10.4f}")

best_model_name = max(results, key=results.get)
print(f"\n Best model: {best_model_name} (AUC = {results[best_model_name]:.4f})")

# Train Final Model

print("\n" + "=" * 60)
print("Step 4: Train Final Model")
print("=" * 60)

# Use Gradient Boosting with tuned hyperparameters
final_model = learners.GradientBoostingLearner(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=10
)
final_model.train(X_train_balanced, y_train_balanced)

# gradient boosting usually shines on tabular risk data
print("Final model trained: Gradient Boosting")
print(f"  n_estimators: 200")
print(f"  learning_rate: 0.1")
print(f"  max_depth: 4")

# Model Evaluation

print("\n" + "=" * 60)
print("Step 5: Model Evaluation on Test Set")
print("=" * 60)

# Predictions
y_pred = final_model.infer(X_test_processed)
y_proba = final_model.infer_proba(X_test_processed)[:, 1]

# evaluate on untouched test set to mimic production
# Metrics
accuracy = (y_pred == y_test).mean()
roc_auc = evaluation.roc_auc_score(y_test, y_proba)
pr_auc = evaluation.average_precision_score(y_test, y_proba)

print("Performance Metrics:")
print(f"  Accuracy:   {accuracy:.4f}")
print(f"  ROC-AUC:    {roc_auc:.4f}")
print(f"  PR-AUC:     {pr_auc:.4f}")

# Confusion matrix
tp = np.sum((y_pred == 1) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

# confusion table helps risk teams balance recalls vs rejections
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              No    Yes")
print(f"  Actual No   {tn:<5} {fp:<5}")
print(f"  Actual Yes  {fn:<5} {tp:<5}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nDefault Detection:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Model Interpretation

print("\n" + "=" * 60)
print("Step 6: Model Interpretation")
print("=" * 60)

# Feature importance
importances = final_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

# quick view of which borrower traits matter most
print("Feature Importance:")
print("-" * 40)
for i in range(len(feature_names)):
    idx = sorted_idx[i]
    bar = "" * int(importances[idx] * 40)
    print(f"  {feature_names[idx]:<25} {importances[idx]:.4f} {bar}")

# SHAP values for a sample
print("\n--- Sample Explanation ---")
sample_idx = np.where(y_test == 1)[0][0]  # First default case
sample = X_test_processed[sample_idx:sample_idx+1]

shap_explainer = explainability.SHAPExplainer(final_model, X_train_balanced)
shap_values = shap_explainer.explain(sample)

# shap narrates the top pushes for one applicant
print(f"Prediction: {'Default' if y_pred[sample_idx] == 1 else 'No Default'}")
print(f"Probability: {y_proba[sample_idx]:.4f}")
print("\nTop factors for this prediction:")
top_features = np.argsort(np.abs(shap_values[0]))[::-1][:5]
for idx in top_features:
    direction = "" if shap_values[0][idx] > 0 else ""
    print(f"  {feature_names[idx]:<25}: {direction} {shap_values[0][idx]:+.4f}")

# Decision Threshold Optimization

print("\n" + "=" * 60)
print("Step 7: Threshold Optimization")
print("=" * 60)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':<12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
print("-" * 50)

# scan a few cutoff points to tune policy
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)

    tp = np.sum((y_pred_thresh == 1) & (y_test == 1))
    fp = np.sum((y_pred_thresh == 1) & (y_test == 0))
    fn = np.sum((y_pred_thresh == 0) & (y_test == 1))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"{thresh:<12} {prec:>12.4f} {rec:>12.4f} {f1:>12.4f}")

print("\n Lower threshold: catch more defaults (higher recall)")
print(" Higher threshold: fewer false positives (higher precision)")

# Business Impact Analysis

print("\n" + "=" * 60)
print("Step 8: Business Impact Analysis")
print("=" * 60)

# Assume:
# - Cost of missed default (FN): $10,000 (loan loss)
# - Cost of false rejection (FP): $500 (lost revenue)
# - Benefit of correct rejection (TP): $10,000 (saved loss)

cost_fn = 10000
cost_fp = 500
benefit_tp = 10000

# rough economics to show stakeholders real money impact
# Without model (approve everyone)
defaults_no_model = np.sum(y_test == 1)
cost_no_model = defaults_no_model * cost_fn
print(f"Without model (approve all):")
print(f"  Total defaults: {defaults_no_model}")
print(f"  Total loss: ${cost_no_model:,}")

# With model
cost_with_model = (fn * cost_fn) + (fp * cost_fp)
savings = tp * benefit_tp

print(f"\nWith model (threshold=0.5):")
print(f"  Missed defaults (FN): {fn}")
print(f"  False rejections (FP): {fp}")
print(f"  Caught defaults (TP): {tp}")
print(f"  FN cost: ${fn * cost_fn:,}")
print(f"  FP cost: ${fp * cost_fp:,}")
print(f"  Savings from TP: ${savings:,}")
print(f"\n  Net benefit: ${savings - cost_with_model:,}")
print(f"  Improvement over no model: ${cost_no_model - cost_with_model:,}")

# Model Persistence

print("\n" + "=" * 60)
print("Step 9: Model Saving")
print("=" * 60)

# Save model
import pickle

model_path = '/tmp/credit_scoring_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': final_model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'threshold': 0.5
    }, f)

# stash everything needed to serve the scorer later
print(f"Model saved to: {model_path}")
print("Includes: model, preprocessor, feature names, threshold")

print("\n" + "=" * 60)
print(" Credit Scoring Pipeline Complete!")
print("=" * 60)
