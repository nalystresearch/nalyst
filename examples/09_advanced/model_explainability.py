"""
Model Explainability
====================

This example demonstrates model interpretability with Nalyst.explainability.

Topics covered:
- Feature importance
- SHAP values
- LIME explanations
- Partial Dependence Plots
- Global vs local explanations
"""

import numpy as np
from nalyst import explainability, learners

# Example 1: Feature Importance

print("=" * 60)
print("Example 1: Feature Importance")
print("=" * 60)

np.random.seed(42)
# fix randomness so importances line up with true signal

# Generate data with known important features
n_samples = 1000
feature_names = ['income', 'age', 'education', 'experience', 'location',
                 'random1', 'random2', 'random3', 'random4', 'random5']

X = np.random.randn(n_samples, 10)
# Target depends mainly on first 4 features
y = (2 * X[:, 0] + 1.5 * X[:, 1] + X[:, 2] - 0.5 * X[:, 3] +
     np.random.randn(n_samples) * 0.5 > 0).astype(int)

# first few columns really drive the label, perfect for demos
# Split
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Train Random Forest
model = learners.RandomForestLearner(n_estimators=100)
model.train(X_train, y_train)

# tree importances give a quick global story
# Get feature importance
importances = model.feature_importances_

print("Feature Importance (Random Forest):")
print("-" * 40)
sorted_idx = np.argsort(importances)[::-1]
for i in range(10):
    idx = sorted_idx[i]
    bar = "" * int(importances[idx] * 50)
    print(f"  {feature_names[idx]:<12} {importances[idx]:.4f} {bar}")

# Example 2: Permutation Importance

print("\n" + "=" * 60)
print("Example 2: Permutation Importance")
print("=" * 60)

# Permutation importance (model-agnostic)
perm_importance = explainability.permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    scoring='accuracy'
)

# permutation importance is model agnostic and double checks the story
print("Permutation Importance:")
print("-" * 40)
sorted_idx = np.argsort(perm_importance['importances_mean'])[::-1]
for i in range(10):
    idx = sorted_idx[i]
    mean = perm_importance['importances_mean'][idx]
    std = perm_importance['importances_std'][idx]
    print(f"  {feature_names[idx]:<12} {mean:.4f}  {std:.4f}")

# Example 3: SHAP Values

print("\n" + "=" * 60)
print("Example 3: SHAP (SHapley Additive exPlanations)")
print("=" * 60)

# Create SHAP explainer
explainer = explainability.SHAPExplainer(model, X_train)

# Get SHAP values for test set
shap_values = explainer.explain(X_test[:100])

# shap gives per-feature push and pull for each row
print("SHAP Feature Importance (mean |SHAP|):")
print("-" * 40)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]
for i in range(10):
    idx = sorted_idx[i]
    bar = "" * int(mean_abs_shap[idx] * 30)
    print(f"  {feature_names[idx]:<12} {mean_abs_shap[idx]:.4f} {bar}")

# Single prediction explanation
print("\n--- Single Prediction Explanation ---")
sample_idx = 0
sample = X_test[sample_idx:sample_idx+1]
sample_shap = shap_values[sample_idx]

# single row view shows which signals move probability up or down
print(f"Predicted class: {model.infer(sample)[0]}")
print(f"True class: {y_test[sample_idx]}")
print("\nTop contributing features:")
top_features = np.argsort(np.abs(sample_shap))[::-1][:5]
for idx in top_features:
    direction = "+" if sample_shap[idx] > 0 else ""
    print(f"  {feature_names[idx]:<12}: {direction}{sample_shap[idx]:.4f}")

# Example 4: LIME Explanations

print("\n" + "=" * 60)
print("Example 4: LIME (Local Interpretable Model-agnostic Explanations)")
print("=" * 60)

# Create LIME explainer
lime_explainer = explainability.LIMEExplainer(
    model, X_train,
    feature_names=feature_names,
    mode='classification'
)

# lime fits tiny local models to mimic the big one
# Explain single prediction
explanation = lime_explainer.explain_instance(X_test[0])

print("LIME Explanation for single prediction:")
print("-" * 40)
for feature, weight in explanation['features'][:5]:
    direction = "+" if weight > 0 else ""
    print(f"  {feature:<20}: {direction}{weight:.4f}")

print(f"\nLocal model R: {explanation['local_r2']:.4f}")

# Example 5: Partial Dependence Plots

print("\n" + "=" * 60)
print("Example 5: Partial Dependence")
print("=" * 60)

# Calculate partial dependence for important features
pdp = explainability.partial_dependence(
    model, X_train,
    features=[0, 1],  # income, age
    grid_resolution=20
)

# pdp curves show direction of effect averaged over others
print("Partial Dependence for 'income' feature:")
print(f"  Grid points: {len(pdp[0]['grid'])}")
print(f"  PD range: [{pdp[0]['pd'].min():.4f}, {pdp[0]['pd'].max():.4f}]")

print("\nPartial Dependence for 'age' feature:")
print(f"  Grid points: {len(pdp[1]['grid'])}")
print(f"  PD range: [{pdp[1]['pd'].min():.4f}, {pdp[1]['pd'].max():.4f}]")

# Example 6: ICE Plots (Individual Conditional Expectation)

print("\n" + "=" * 60)
print("Example 6: ICE (Individual Conditional Expectation)")
print("=" * 60)

# ICE shows individual sample trajectories
ice = explainability.individual_conditional_expectation(
    model, X_test[:50],
    feature=0,  # income
    grid_resolution=20
)

# ice keeps each customer's line separate to expose heterogeneity
print(f"ICE for 'income' feature:")
print(f"  Grid points: {len(ice['grid'])}")
print(f"  Number of samples: {ice['ice'].shape[0]}")
print(f"  ICE values shape: {ice['ice'].shape}")

# Example 7: Global Surrogate Model

print("\n" + "=" * 60)
print("Example 7: Global Surrogate Model")
print("=" * 60)

# Train a simple decision tree to approximate the complex model
from nalyst.learners import DecisionTreeLearner

# Get predictions from complex model
y_pred_train = model.infer(X_train)

# Train surrogate
surrogate = DecisionTreeLearner(max_depth=4)
surrogate.train(X_train, y_pred_train)

# surrogate tree mimics the complex model with readable rules
# Evaluate fidelity
y_pred_surrogate = surrogate.infer(X_test)
y_pred_original = model.infer(X_test)
fidelity = (y_pred_surrogate == y_pred_original).mean()

print(f"Surrogate model fidelity: {fidelity:.4f}")
print(f"Surrogate depth: {surrogate.max_depth}")

# Get rules from surrogate
print("\nSurrogate Decision Rules (interpretable approximation):")
rules = surrogate.get_rules()
for i, rule in enumerate(rules[:3]):
    print(f"  Rule {i+1}: {rule}")

# Example 8: Feature Interaction

print("\n" + "=" * 60)
print("Example 8: Feature Interactions")
print("=" * 60)

# SHAP interaction values
shap_interaction = explainer.shap_interaction_values(X_test[:50])

print("Top Feature Interactions:")
print("-" * 40)

# interaction cube lets us see which feature pairs team up
# Get mean absolute interaction effects
n_features = len(feature_names)
interaction_effects = np.zeros((n_features, n_features))
for i in range(n_features):
    for j in range(n_features):
        if i != j:
            interaction_effects[i, j] = np.abs(shap_interaction[:, i, j]).mean()

# Find top interactions
top_interactions = []
for i in range(n_features):
    for j in range(i+1, n_features):
        top_interactions.append((i, j, interaction_effects[i, j]))

top_interactions.sort(key=lambda x: x[2], reverse=True)

for i, j, effect in top_interactions[:5]:
    print(f"  {feature_names[i]}  {feature_names[j]}: {effect:.4f}")

# Example 9: Counterfactual Explanations

print("\n" + "=" * 60)
print("Example 9: Counterfactual Explanations")
print("=" * 60)

# Find counterfactual for a negative prediction
cf_explainer = explainability.CounterfactualExplainer(model, X_train)

# Find a sample predicted as 0
sample_idx = np.where(model.infer(X_test) == 0)[0][0]
sample = X_test[sample_idx]

# counterfactuals reveal the smallest tweak that flips the model
print(f"Original prediction: {model.infer(sample.reshape(1, -1))[0]} (Class 0)")

# Find minimal change to flip prediction
counterfactual = cf_explainer.explain(sample, target_class=1)

print("\nCounterfactual explanation (minimal changes to get Class 1):")
print("-" * 40)
for i, (feat_name, original, cf_val) in enumerate(zip(
    feature_names, sample, counterfactual['counterfactual']
)):
    change = cf_val - original
    if abs(change) > 0.01:
        direction = "+" if change > 0 else ""
        print(f"  {feat_name:<12}: {original:.3f}  {cf_val:.3f} ({direction}{change:.3f})")

print(f"\nNew prediction: {model.infer(counterfactual['counterfactual'].reshape(1, -1))[0]} (Class 1)")

print("\n Model explainability examples completed!")
