"""
Advanced Examples - Imbalanced Data
====================================

This example demonstrates handling imbalanced datasets with Nalyst.imbalance.

Topics covered:
- SMOTE (Synthetic Minority Over-sampling)
- ADASYN
- Random oversampling and undersampling
- Combining sampling with models
- Evaluation for imbalanced data
"""

import numpy as np
from nalyst import imbalance, learners, evaluation

# Example 1: Understanding Imbalanced Data

print("=" * 60)
print("Example 1: Understanding Imbalanced Data")
print("=" * 60)

np.random.seed(42)
# keep randomness stable so balancing tricks are comparable

# Create imbalanced dataset (10% minority class)
n_majority = 900
n_minority = 100

# craft a simple binary dataset with a heavy class skew
X_majority = np.random.randn(n_majority, 10) - 0.5
X_minority = np.random.randn(n_minority, 10) + 0.5
X = np.vstack([X_majority, X_minority])
y = np.array([0] * n_majority + [1] * n_minority)

# Shuffle
indices = np.random.permutation(len(y))
X, y = X[indices], y[indices]

print(f"Total samples: {len(y)}")
print(f"Class 0 (majority): {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
print(f"Class 1 (minority): {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
print(f"Imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.1f}:1")

# Train without handling imbalance
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# baseline model shows how poor recall looks before fixes
model = learners.LogisticLearner()
model.train(X_train, y_train)
y_pred = model.infer(X_test)

print("\n--- Without imbalance handling ---")
print(f"Accuracy: {(y_pred == y_test).mean():.4f}")
print(f"Minority class recall: {np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1):.4f}")

# Example 2: SMOTE (Synthetic Minority Over-sampling)

print("\n" + "=" * 60)
print("Example 2: SMOTE")
print("=" * 60)

# Apply SMOTE
smote = imbalance.SMOTE(sampling_strategy='auto', k_neighbors=5)
X_resampled, y_resampled = smote.train_resample(X_train, y_train)

# smote synthesizes new minority points between neighbors
print("After SMOTE:")
print(f"  Original: {len(y_train)} samples")
print(f"  Resampled: {len(y_resampled)} samples")
print(f"  Class 0: {np.sum(y_resampled == 0)}")
print(f"  Class 1: {np.sum(y_resampled == 1)}")

# Train on resampled data
model = learners.LogisticLearner()
model.train(X_resampled, y_resampled)
y_pred = model.infer(X_test)

print("\n--- With SMOTE ---")
print(f"Accuracy: {(y_pred == y_test).mean():.4f}")
print(f"Minority class recall: {np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1):.4f}")

# Example 3: ADASYN (Adaptive Synthetic Sampling)

print("\n" + "=" * 60)
print("Example 3: ADASYN")
print("=" * 60)

# ADASYN focuses on difficult-to-learn samples
adasyn = imbalance.ADASYN(sampling_strategy='auto', n_neighbors=5)
X_adasyn, y_adasyn = adasyn.train_resample(X_train, y_train)

# adasyn leans into boundary areas where mistakes happen
print("After ADASYN:")
print(f"  Resampled: {len(y_adasyn)} samples")
print(f"  Class 0: {np.sum(y_adasyn == 0)}")
print(f"  Class 1: {np.sum(y_adasyn == 1)}")

model = learners.LogisticLearner()
model.train(X_adasyn, y_adasyn)
y_pred = model.infer(X_test)

print(f"\nMinority class recall: {np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1):.4f}")

# Example 4: Random Undersampling and Oversampling

print("\n" + "=" * 60)
print("Example 4: Random Sampling Methods")
print("=" * 60)

# Random oversampling
ros = imbalance.RandomOverSampler(sampling_strategy='auto')
X_ros, y_ros = ros.train_resample(X_train, y_train)
print("Random Oversampling:")
print(f"  Samples: {len(y_ros)}")

# Random undersampling
rus = imbalance.RandomUnderSampler(sampling_strategy='auto')
X_rus, y_rus = rus.train_resample(X_train, y_train)
print("\nRandom Undersampling:")
print(f"  Samples: {len(y_rus)}")

# Combine both (under-sample majority, over-sample minority)
print("\n--- Comparison ---")
methods = {
    'Original': (X_train, y_train),
    'Oversampling': (X_ros, y_ros),
    'Undersampling': (X_rus, y_rus),
    'SMOTE': (X_resampled, y_resampled)
}

for name, (X_tr, y_tr) in methods.items():
    model = learners.LogisticLearner()
    model.train(X_tr, y_tr)
    y_pred = model.infer(X_test)

    # reuse recall to see who protects minority cases better
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
    print(f"{name:<15}: Recall = {recall:.4f}")

# Example 5: Borderline SMOTE

print("\n" + "=" * 60)
print("Example 5: Borderline SMOTE")
print("=" * 60)

# Only oversamples borderline minority samples
border_smote = imbalance.BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)
X_border, y_border = border_smote.train_resample(X_train, y_train)

print("Borderline SMOTE:")
print(f"  Samples: {len(y_border)}")
print(f"  New minority samples: {np.sum(y_border == 1) - np.sum(y_train == 1)}")

# Example 6: SMOTE + Tomek Links (SMOTETomek)

print("\n" + "=" * 60)
print("Example 6: SMOTETomek (Cleaning Overlap)")
print("=" * 60)

# Combine SMOTE with Tomek links cleaning
smotetomek = imbalance.SMOTETomek(sampling_strategy='auto')
X_smotetomek, y_smotetomek = smotetomek.train_resample(X_train, y_train)

print("SMOTETomek:")
print(f"  Original: {len(y_train)}")
print(f"  After SMOTE: ~{len(y_train) + (np.sum(y_train == 0) - np.sum(y_train == 1))}")
print(f"  After Tomek cleaning: {len(y_smotetomek)}")

# cleaned data removes overlapping pairs that confuse the separator
# Example 7: Evaluation Metrics for Imbalanced Data

print("\n" + "=" * 60)
print("Example 7: Proper Evaluation Metrics")
print("=" * 60)

# Train model with SMOTE
model = learners.RandomForestLearner(n_estimators=50)
model.train(X_resampled, y_resampled)
y_pred = model.infer(X_test)
y_proba = model.infer_proba(X_test)[:, 1]

# go beyond accuracy because skewed data needs richer metrics
# Compute various metrics
metrics = evaluation.classification_report(y_test, y_pred)

print("Classification Report:")
print("-" * 50)
print(f"{'Metric':<20} {'Class 0':>12} {'Class 1':>12}")
print("-" * 50)
print(f"{'Precision':<20} {metrics['precision'][0]:>12.4f} {metrics['precision'][1]:>12.4f}")
print(f"{'Recall':<20} {metrics['recall'][0]:>12.4f} {metrics['recall'][1]:>12.4f}")
print(f"{'F1-Score':<20} {metrics['f1'][0]:>12.4f} {metrics['f1'][1]:>12.4f}")
print("-" * 50)
print(f"{'Accuracy':<20} {metrics['accuracy']:>12.4f}")
print(f"{'Macro F1':<20} {np.mean(metrics['f1']):>12.4f}")
print(f"{'Weighted F1':<20} {metrics['weighted_f1']:>12.4f}")

# ROC-AUC and PR-AUC
roc_auc = evaluation.roc_auc_score(y_test, y_proba)
pr_auc = evaluation.average_precision_score(y_test, y_proba)

print(f"\n{'ROC-AUC':<20} {roc_auc:>12.4f}")
print(f"{'PR-AUC (AP)':<20} {pr_auc:>12.4f}")
print("\n PR-AUC is more informative for imbalanced data")

# Example 8: Class Weights

print("\n" + "=" * 60)
print("Example 8: Using Class Weights")
print("=" * 60)

# Train without class weights
model_no_weights = learners.LogisticLearner()
model_no_weights.train(X_train, y_train)
y_pred_no_weights = model_no_weights.infer(X_test)

# Train with class weights
class_weights = {0: 1, 1: 9}  # Weight minority class more
model_weighted = learners.LogisticLearner(class_weight=class_weights)
model_weighted.train(X_train, y_train)
y_pred_weighted = model_weighted.infer(X_test)

print("Without class weights:")
recall_0 = np.sum((y_pred_no_weights == 1) & (y_test == 1)) / np.sum(y_test == 1)
print(f"  Minority recall: {recall_0:.4f}")

print("\nWith class weights (1:9):")
recall_1 = np.sum((y_pred_weighted == 1) & (y_test == 1)) / np.sum(y_test == 1)
print(f"  Minority recall: {recall_1:.4f}")

# Example 9: Comparison of All Methods

print("\n" + "=" * 60)
print("Example 9: Comprehensive Method Comparison")
print("=" * 60)

samplers = {
    'Original': None,
    'RandomOver': imbalance.RandomOverSampler(),
    'RandomUnder': imbalance.RandomUnderSampler(),
    'SMOTE': imbalance.SMOTE(),
    'ADASYN': imbalance.ADASYN(),
    'BorderlineSMOTE': imbalance.BorderlineSMOTE(),
    'SMOTETomek': imbalance.SMOTETomek()
}

print(f"{'Method':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 60)

for name, sampler in samplers.items():
    if sampler is None:
        X_tr, y_tr = X_train, y_train
    else:
        X_tr, y_tr = sampler.train_resample(X_train, y_train)

    model = learners.LogisticLearner()
    model.train(X_tr, y_tr)
    y_pred = model.infer(X_test)

    acc = (y_pred == y_test).mean()
    prec = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    rec = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"{name:<18} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

print("\n Imbalanced data examples completed!")
