"""
Real-World Example: Customer Churn Prediction
==============================================

This example demonstrates a complete ML workflow for customer churn.

Topics covered:
- Customer data analysis
- Churn prediction modeling
- Feature engineering for customer data
- Retention strategies from predictions
"""

import numpy as np
from nalyst import learners, transform, evaluation, explainability

# Generate Synthetic Customer Data

print("=" * 60)
print("Customer Churn Prediction - Complete Example")
print("=" * 60)

np.random.seed(42)
# use deterministic data so churn rates stay consistent in docs

n_customers = 7000

# Generate customer features
data = {
    'tenure_months': np.random.exponential(24, n_customers).clip(1, 72),
    'monthly_charges': np.random.normal(70, 25, n_customers).clip(20, 150),
    'total_charges': None,  # Will compute
    'contract_type': np.random.choice([0, 1, 2], n_customers, p=[0.5, 0.3, 0.2]),  # 0=month, 1=year, 2=2-year
    'payment_method': np.random.choice([0, 1, 2, 3], n_customers),
    'num_support_tickets': np.random.poisson(2, n_customers),
    'num_products': np.random.poisson(3, n_customers).clip(1, 8),
    'has_partner': np.random.binomial(1, 0.5, n_customers),
    'has_dependents': np.random.binomial(1, 0.3, n_customers),
    'senior_citizen': np.random.binomial(1, 0.15, n_customers),
    'online_security': np.random.binomial(1, 0.4, n_customers),
    'tech_support': np.random.binomial(1, 0.35, n_customers),
    'paperless_billing': np.random.binomial(1, 0.6, n_customers),
    'avg_data_usage_gb': np.random.exponential(5, n_customers),
    'late_payments_count': np.random.poisson(0.5, n_customers),
}

data['total_charges'] = data['tenure_months'] * data['monthly_charges']

# monthly cost times tenure gives a cumulative spend signal
# Generate churn (higher for short tenure, month-to-month, high support tickets)
churn_prob = 1 / (1 + np.exp(
    -(1.5 -
      0.05 * data['tenure_months'] +
      0.5 * (data['contract_type'] == 0) -
      0.3 * (data['contract_type'] == 2) +
      0.2 * data['num_support_tickets'] -
      0.15 * data['num_products'] +
      0.01 * data['monthly_charges'] -
      0.2 * data['online_security'] -
      0.2 * data['tech_support'] +
      0.3 * data['late_payments_count'] +
      np.random.randn(n_customers) * 0.5)
))
y = (np.random.random(n_customers) < churn_prob).astype(int)

# Create feature matrix
feature_names = list(data.keys())
X = np.column_stack([data[f] for f in feature_names])

# keep a plain matrix for downstream models
print(f"Total customers: {n_customers}")
print(f"Churn rate: {y.mean():.2%}")
print(f"Features: {len(feature_names)}")

# Exploratory Data Analysis

print("\n" + "=" * 60)
print("Step 1: Exploratory Data Analysis")
print("=" * 60)

print("\nChurn by Contract Type:")
for ct, name in [(0, 'Month-to-month'), (1, 'One year'), (2, 'Two year')]:
    mask = data['contract_type'] == ct
    rate = y[mask].mean()
    print(f"  {name:<15}: {rate:.2%}")

# quick aggregations to mimic analyst notes
print("\nChurn by Tenure Group:")
tenure = data['tenure_months']
for low, high, name in [(0, 12, '0-12 months'), (12, 36, '12-36 months'), (36, 72, '36+ months')]:
    mask = (tenure >= low) & (tenure < high)
    rate = y[mask].mean()
    print(f"  {name:<15}: {rate:.2%}")

print("\nAverage values by churn status:")
print(f"  {'Metric':<25} {'Churned':>12} {'Retained':>12}")
print("  " + "-" * 50)
for feat in ['tenure_months', 'monthly_charges', 'num_support_tickets', 'num_products']:
    churned_mean = data[feat][y == 1].mean()
    retained_mean = data[feat][y == 0].mean()
    print(f"  {feat:<25} {churned_mean:>12.2f} {retained_mean:>12.2f}")

# Feature Engineering

print("\n" + "=" * 60)
print("Step 2: Feature Engineering")
print("=" * 60)

# Create new features
charges_per_product = data['monthly_charges'] / (data['num_products'] + 0.1)
tenure_squared = data['tenure_months'] ** 2 / 100
tickets_per_month = data['num_support_tickets'] / (data['tenure_months'] + 1)
is_month_to_month = (data['contract_type'] == 0).astype(int)
high_value = ((data['monthly_charges'] > 80) & (data['tenure_months'] > 24)).astype(int)

# engineered fields capture non linear churn behaviors
# Add to feature matrix
X_enhanced = np.column_stack([
    X,
    charges_per_product,
    tenure_squared,
    tickets_per_month,
    is_month_to_month,
    high_value
])

enhanced_feature_names = feature_names + [
    'charges_per_product', 'tenure_squared', 'tickets_per_month',
    'is_month_to_month', 'high_value'
]

print(f"Original features: {len(feature_names)}")
print(f"Enhanced features: {len(enhanced_feature_names)}")
print(f"\nNew features created:")
print("  - charges_per_product: value efficiency")
print("  - tenure_squared: non-linear tenure effect")
print("  - tickets_per_month: support intensity")
print("  - is_month_to_month: contract flag")
print("  - high_value: valuable customer indicator")

# Data Preparation

print("\n" + "=" * 60)
print("Step 3: Data Preparation")
print("=" * 60)

# Split data
split_train = int(0.7 * n_customers)
split_val = int(0.85 * n_customers)

# train/val/test split mirrors a typical churn project
X_train = X_enhanced[:split_train]
X_val = X_enhanced[split_train:split_val]
X_test = X_enhanced[split_val:]

y_train = y[:split_train]
y_val = y[split_train:split_val]
y_test = y[split_val:]

print(f"Training:   {len(y_train)} ({y_train.mean():.2%} churn)")
print(f"Validation: {len(y_val)} ({y_val.mean():.2%} churn)")
print(f"Test:       {len(y_test)} ({y_test.mean():.2%} churn)")

# Scale features
scaler = transform.StandardScaler()
X_train_scaled = scaler.train_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# scaling keeps tree and linear models happy together
# Model Training and Comparison

print("\n" + "=" * 60)
print("Step 4: Model Comparison")
print("=" * 60)

models = {
    'Logistic': learners.LogisticLearner(),
    'Random Forest': learners.RandomForestLearner(n_estimators=100),
    'Gradient Boosting': learners.GradientBoostingLearner(n_estimators=100),
}

# compare a few common churn models using validation auc
print(f"{'Model':<20} {'Val AUC':>12} {'Val Recall':>12}")
print("-" * 46)

best_auc = 0
best_model = None

for name, model in models.items():
    model.train(X_train_scaled, y_train)
    y_proba = model.infer_proba(X_val_scaled)[:, 1]
    y_pred = model.infer(X_val_scaled)

    auc = evaluation.roc_auc_score(y_val, y_proba)
    recall = np.sum((y_pred == 1) & (y_val == 1)) / np.sum(y_val == 1)

    print(f"{name:<20} {auc:>12.4f} {recall:>12.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

print(f"\n Best model: {best_model_name} (AUC = {best_auc:.4f})")

# Final Model Evaluation

print("\n" + "=" * 60)
print("Step 5: Final Model Evaluation")
print("=" * 60)

y_pred = best_model.infer(X_test_scaled)
y_proba = best_model.infer_proba(X_test_scaled)[:, 1]

# test set gives our unbiased read on churn power
# Metrics
print("Test Set Performance:")
print(f"  ROC-AUC:  {evaluation.roc_auc_score(y_test, y_proba):.4f}")
print(f"  PR-AUC:   {evaluation.average_precision_score(y_test, y_proba):.4f}")

# Classification report
tp = np.sum((y_pred == 1) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nChurn Prediction:")
print(f"  Precision: {precision:.4f} (of predicted churners, how many actually churn)")
print(f"  Recall:    {recall:.4f} (of actual churners, how many we catch)")

# Feature Importance Analysis

print("\n" + "=" * 60)
print("Step 6: Feature Importance")
print("=" * 60)

importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

# highlight the signals retention teams care about
print("Top 10 Most Important Features:")
print("-" * 50)
for i in range(10):
    idx = sorted_idx[i]
    bar = "" * int(importances[idx] * 40)
    print(f"  {enhanced_feature_names[idx]:<25} {importances[idx]:.4f} {bar}")

# Customer Segmentation by Risk

print("\n" + "=" * 60)
print("Step 7: Customer Risk Segmentation")
print("=" * 60)

risk_segments = {
    'Low Risk': (0, 0.2),
    'Medium Risk': (0.2, 0.5),
    'High Risk': (0.5, 0.8),
    'Very High Risk': (0.8, 1.0)
}

# convert probabilities into easy business tiers
# Use all data for demonstration
X_all_scaled = scaler.transform(X_enhanced)
all_probas = best_model.infer_proba(X_all_scaled)[:, 1]

print(f"{'Segment':<18} {'Customers':>12} {'Actual Churn':>15} {'Action'}")
print("-" * 70)

for segment, (low, high) in risk_segments.items():
    mask = (all_probas >= low) & (all_probas < high)
    n_customers_seg = mask.sum()
    actual_churn = y[mask].mean() if mask.sum() > 0 else 0

    if segment == 'Low Risk':
        action = "Monitor"
    elif segment == 'Medium Risk':
        action = "Engagement program"
    elif segment == 'High Risk':
        action = "Retention offer"
    else:
        action = "Personal outreach"

    print(f"{segment:<18} {n_customers_seg:>12,} {actual_churn:>14.2%} {action}")

# Individual Customer Analysis

print("\n" + "=" * 60)
print("Step 8: Individual Customer Explanations")
print("=" * 60)

# Find high-risk customers
high_risk_mask = all_probas > 0.7
high_risk_indices = np.where(high_risk_mask)[0][:3]

# inspect a few risky customers to craft retention stories
for idx in high_risk_indices:
    print(f"\n--- Customer {idx} ---")
    print(f"Churn Probability: {all_probas[idx]:.2%}")

    # Key characteristics
    customer = X_enhanced[idx]
    print(f"Tenure: {customer[0]:.0f} months")
    print(f"Monthly Charges: ${customer[1]:.2f}")
    print(f"Contract: {'Month-to-month' if customer[3] == 0 else 'Annual+'}")
    print(f"Support Tickets: {customer[5]:.0f}")
    print(f"Products: {customer[6]:.0f}")

    # Top risk factors (using simple comparison to low-risk avg)
    print("Risk Factors:")
    if customer[0] < 12:
        print("  - Short tenure (< 12 months)")
    if customer[3] == 0:
        print("  - Month-to-month contract")
    if customer[5] > 3:
        print("  - High support ticket volume")
    if customer[10] == 0:
        print("  - No online security")

# Retention ROI Analysis

print("\n" + "=" * 60)
print("Step 9: Retention Campaign ROI")
print("=" * 60)

# Assumptions
avg_customer_value = data['monthly_charges'].mean() * 12  # Annual value
retention_offer_cost = 50  # Cost of retention offer
offer_success_rate = 0.3  # 30% of targeted churners are retained

# simple roi math helps justify outreach programs
# Target high-risk customers
high_risk_customers = all_probas > 0.5
n_targeted = high_risk_customers.sum()
actual_churners = (high_risk_customers & (y == 1)).sum()

# Calculate ROI
saved_customers = int(actual_churners * offer_success_rate)
revenue_saved = saved_customers * avg_customer_value
campaign_cost = n_targeted * retention_offer_cost
roi = (revenue_saved - campaign_cost) / campaign_cost * 100 if campaign_cost > 0 else 0

print(f"Campaign Targeting (threshold=0.5):")
print(f"  Customers targeted: {n_targeted:,}")
print(f"  Would-be churners in group: {actual_churners:,}")
print(f"  Expected saves ({offer_success_rate:.0%} rate): {saved_customers}")
print(f"\nFinancial Impact:")
print(f"  Revenue saved: ${revenue_saved:,.0f}")
print(f"  Campaign cost: ${campaign_cost:,.0f}")
print(f"  Net benefit: ${revenue_saved - campaign_cost:,.0f}")
print(f"  ROI: {roi:.0f}%")

print("\n" + "=" * 60)
print(" Churn Prediction Pipeline Complete!")
print("=" * 60)
