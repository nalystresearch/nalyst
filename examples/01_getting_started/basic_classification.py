"""
Basic Classification Example
============================

This example demonstrates basic classification using Nalyst.

Topics covered:
- Loading datasets
- Training a classifier
- Making predictions
- Evaluating model performance
"""

import numpy as np
from nalyst.learners.linear import LogisticLearner
from nalyst.learners.tree import TreeLearner, ForestLearner
from nalyst.learners.neighbors import KNNLearner
from nalyst.evaluation import split_data, cross_validate
from nalyst.metrics import accuracy_score, classification_report, confusion_matrix
from nalyst.datasets import load_iris, load_breast_cancer

# Example 1: Iris Classification with Logistic Regression

# print headings so the console output stays easy to scan
print("=" * 60)
print("Example 1: Iris Classification")
print("=" * 60)

# load the classic iris measurements for a gentle starting point
X, y = load_iris(return_X_y=True)
print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

# carve out a holdout set to check generalization
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# train a logistic learner because it is fast and well behaved here
model = LogisticLearner(max_iter=1000)
model.train(X_train, y_train)

# run inference on the held out block
y_pred = model.infer(X_test)

# capture a few quick metrics for reference
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example 2: Breast Cancer with Multiple Classifiers

# label the second example clearly so users can follow along
print("\n" + "=" * 60)
print("Example 2: Breast Cancer - Model Comparison")
print("=" * 60)

# pull in a binary medical dataset to compare learners
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# assemble a few baseline classifiers for side by side scores
classifiers = {
    "Logistic Regression": LogisticLearner(),
    "Decision Tree": TreeLearner(max_depth=5),
    "Random Forest": ForestLearner(n_estimators=100),
    "KNN (k=5)": KNNLearner(n_neighbors=5),
}

print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 60)

for name, clf in classifiers.items():
    # train each learner on the exact same split
    clf.train(X_train, y_train)

    # infer on the shared test slice
    y_pred = clf.infer(X_test)

    # keep metrics minimal so the table stays readable
    from nalyst.metrics import precision_score, recall_score
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"{name:<25} {acc:<12.2%} {prec:<12.2%} {rec:<12.2%}")

# Example 3: Cross-Validation

# highlight the cross validation showcase
print("\n" + "=" * 60)
print("Example 3: Cross-Validation")
print("=" * 60)

model = ForestLearner(n_estimators=100)

# run k-fold scoring to show variance and mean accuracy
cv_results = cross_validate(model, X, y, cv=5, scoring='accuracy')

print(f"\nCross-validation scores: {cv_results['test_score']}")
print(f"Mean accuracy: {cv_results['test_score'].mean():.2%} (+/- {cv_results['test_score'].std() * 2:.2%})")

# Example 4: Probability Predictions

# wrap up by showing probability outputs for interpretability
print("\n" + "=" * 60)
print("Example 4: Probability Predictions")
print("=" * 60)

model = LogisticLearner()
model.train(X_train, y_train)

# request per class probabilities to inspect confidence
y_proba = model.infer_proba(X_test)

print(f"\nPrediction probabilities for first 5 samples:")
print(f"{'Sample':<10} {'Prob(0)':<12} {'Prob(1)':<12} {'Predicted':<12} {'Actual':<12}")
print("-" * 60)

y_pred = model.infer(X_test)
for i in range(5):
    print(f"{i:<10} {y_proba[i, 0]:<12.3f} {y_proba[i, 1]:<12.3f} {y_pred[i]:<12} {y_test[i]:<12}")

print("\n Classification examples completed!")
