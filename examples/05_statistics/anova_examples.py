"""
ANOVA Examples
==============

This example demonstrates Analysis of Variance (ANOVA) with Nalyst.stats.

Topics covered:
- One-way ANOVA
- Two-way ANOVA
- Repeated measures ANOVA
- Post-hoc tests
- Effect sizes
"""

import numpy as np
from nalyst import stats

# Example 1: One-Way ANOVA

# print a banner so each test section is easy to spot
print("=" * 60)
print("Example 1: One-Way ANOVA")
print("=" * 60)

np.random.seed(42)

# synthesize three treatment groups with slightly different means
control = np.random.normal(100, 15, 30)
treatment_a = np.random.normal(108, 15, 30)
treatment_b = np.random.normal(115, 15, 30)

result = stats.f_oneway(control, treatment_a, treatment_b)

print("Group Statistics:")
print(f"  Control:     mean={control.mean():.2f}, std={control.std():.2f}")
print(f"  Treatment A: mean={treatment_a.mean():.2f}, std={treatment_a.std():.2f}")
print(f"  Treatment B: mean={treatment_b.mean():.2f}, std={treatment_b.std():.2f}")
print(f"\nF-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.6f}")

if result['pvalue'] < 0.05:
    print("\n Significant difference between at least two groups")
else:
    print("\n No significant differences between groups")

# Example 2: Effect Size (Eta-squared)

# Example 2: Effect Size (Eta-squared)

# call out the effect size follow up
print("\n" + "=" * 60)
print("Example 2: Effect Size Calculation")
print("=" * 60)

# compute eta squared manually for transparency
all_data = np.concatenate([control, treatment_a, treatment_b])
grand_mean = all_data.mean()

# Between-group sum of squares
ss_between = (len(control) * (control.mean() - grand_mean) ** 2 +
              len(treatment_a) * (treatment_a.mean() - grand_mean) ** 2 +
              len(treatment_b) * (treatment_b.mean() - grand_mean) ** 2)

# Total sum of squares
ss_total = np.sum((all_data - grand_mean) ** 2)

eta_squared = ss_between / ss_total

print(f" (Eta-squared): {eta_squared:.4f}")

# Interpretation
if eta_squared < 0.01:
    effect = "negligible"
elif eta_squared < 0.06:
    effect = "small"
elif eta_squared < 0.14:
    effect = "medium"
else:
    effect = "large"

print(f"Effect size interpretation: {effect}")
print(f"Variance explained: {eta_squared * 100:.1f}%")

# Example 3: Post-Hoc Tests (Tukey HSD)

# add a banner for the follow up test
print("\n" + "=" * 60)
print("Example 3: Post-Hoc Comparisons (Tukey HSD)")
print("=" * 60)

# Prepare data for Tukey test
groups = ['Control'] * len(control) + ['Treatment_A'] * len(treatment_a) + ['Treatment_B'] * len(treatment_b)
values = np.concatenate([control, treatment_a, treatment_b])

result = stats.tukey_hsd(control, treatment_a, treatment_b,
                         groups=['Control', 'Treatment A', 'Treatment B'])

print("Tukey HSD Pairwise Comparisons:")
print("-" * 50)
for comparison in result['comparisons']:
    g1, g2 = comparison['groups']
    diff = comparison['mean_diff']
    pval = comparison['p_adj']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"{g1:>12} vs {g2:<12}: diff={diff:>7.2f}, p={pval:.4f} {sig}")

print("\n* p < 0.05, ** p < 0.01, *** p < 0.001")

# Example 4: Two-Way ANOVA

# switch to a factorial design example
print("\n" + "=" * 60)
print("Example 4: Two-Way ANOVA")
print("=" * 60)

# 2x2 factorial design: Drug (A, B) vs Dose (Low, High)
np.random.seed(123)
n_per_cell = 20

drug_a_low = np.random.normal(50, 10, n_per_cell)
drug_a_high = np.random.normal(60, 10, n_per_cell)
drug_b_low = np.random.normal(55, 10, n_per_cell)
drug_b_high = np.random.normal(75, 10, n_per_cell)

# Create factor arrays
values = np.concatenate([drug_a_low, drug_a_high, drug_b_low, drug_b_high])
drug = np.array(['A'] * 2 * n_per_cell + ['B'] * 2 * n_per_cell)
dose = np.array(['Low'] * n_per_cell + ['High'] * n_per_cell +
                ['Low'] * n_per_cell + ['High'] * n_per_cell)

result = stats.f_twoway(values, drug, dose)

print("Cell Means:")
print(f"  Drug A, Low:  {drug_a_low.mean():.2f}")
print(f"  Drug A, High: {drug_a_high.mean():.2f}")
print(f"  Drug B, Low:  {drug_b_low.mean():.2f}")
print(f"  Drug B, High: {drug_b_high.mean():.2f}")

print("\nANOVA Results:")
print("-" * 50)
print(f"{'Source':<20} {'F':>10} {'p-value':>12}")
print("-" * 50)
print(f"{'Drug':<20} {result['F_drug']:>10.4f} {result['p_drug']:>12.4f}")
print(f"{'Dose':<20} {result['F_dose']:>10.4f} {result['p_dose']:>12.4f}")
print(f"{'Drug  Dose':<20} {result['F_interaction']:>10.4f} {result['p_interaction']:>12.4f}")

# Example 5: Repeated Measures ANOVA

# mark the within-subjects scenario
print("\n" + "=" * 60)
print("Example 5: Repeated Measures ANOVA")
print("=" * 60)

# simulate the same subjects across three visits
n_subjects = 15
np.random.seed(456)

baseline = np.random.normal(100, 15, n_subjects)
week_4 = baseline + np.random.normal(10, 5, n_subjects)
week_8 = baseline + np.random.normal(18, 5, n_subjects)

result = stats.f_oneway_rm(baseline, week_4, week_8)

print("Within-Subject Means:")
print(f"  Baseline: {baseline.mean():.2f}")
print(f"  Week 4:   {week_4.mean():.2f}")
print(f"  Week 8:   {week_8.mean():.2f}")
print(f"\nF-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.6f}")

# Example 6: Welch's ANOVA (Unequal Variances)

# demonstrate a variance-robust alternative
print("\n" + "=" * 60)
print("Example 6: Welch's ANOVA (Unequal Variances)")
print("=" * 60)

# Groups with unequal variances
g1 = np.random.normal(50, 5, 30)    # Small variance
g2 = np.random.normal(55, 15, 25)   # Large variance
g3 = np.random.normal(52, 10, 35)   # Medium variance

print("Group Variances:")
print(f"  Group 1:  = {g1.var():.2f}")
print(f"  Group 2:  = {g2.var():.2f}")
print(f"  Group 3:  = {g3.var():.2f}")

# Standard ANOVA
anova_result = stats.f_oneway(g1, g2, g3)
print(f"\nStandard ANOVA:  F = {anova_result['statistic']:.4f}, p = {anova_result['pvalue']:.4f}")

# Welch's ANOVA
welch_result = stats.welch_anova(g1, g2, g3)
print(f"Welch's ANOVA:   F = {welch_result['statistic']:.4f}, p = {welch_result['pvalue']:.4f}")

# Example 7: ANOVA Table

# reuse earlier data to show a full summary table
print("\n" + "=" * 60)
print("Example 7: Full ANOVA Table")
print("=" * 60)

# Reuse data from Example 1
all_groups = [control, treatment_a, treatment_b]
group_labels = ['Control', 'Treatment A', 'Treatment B']

result = stats.anova_table(control, treatment_a, treatment_b)

print("ANOVA Summary Table")
print("=" * 70)
print(f"{'Source':<15} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p':>10}")
print("-" * 70)
print(f"{'Between':<15} {result['ss_between']:>12.2f} {result['df_between']:>6} "
      f"{result['ms_between']:>12.2f} {result['F']:>10.4f} {result['p']:>10.4f}")
print(f"{'Within':<15} {result['ss_within']:>12.2f} {result['df_within']:>6} "
      f"{result['ms_within']:>12.2f}")
print(f"{'Total':<15} {result['ss_total']:>12.2f} {result['df_total']:>6}")
print("=" * 70)

# Example 8: Games-Howell Post-Hoc (for unequal variances)

# banner for the unequal variance friendly test
print("\n" + "=" * 60)
print("Example 8: Games-Howell Post-Hoc Test")
print("=" * 60)

# Groups with unequal variances
g1 = np.random.normal(50, 5, 30)
g2 = np.random.normal(60, 15, 25)
g3 = np.random.normal(70, 10, 35)

result = stats.games_howell(g1, g2, g3,
                            groups=['Group 1', 'Group 2', 'Group 3'])

print("Games-Howell Pairwise Comparisons (for unequal variances):")
print("-" * 50)
for comparison in result['comparisons']:
    g1_name, g2_name = comparison['groups']
    diff = comparison['mean_diff']
    pval = comparison['p_adj']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"{g1_name:>10} vs {g2_name:<10}: diff={diff:>7.2f}, p={pval:.4f} {sig}")

# Example 9: Power Analysis for ANOVA

# finish with a power analysis banner
print("\n" + "=" * 60)
print("Example 9: Power Analysis for ANOVA")
print("=" * 60)

# calculate how many samples are needed for a target power
effect_size = 0.25  # Cohen's f (medium effect)
alpha = 0.05
power = 0.80
n_groups = 3

result = stats.power_anova(effect_size=effect_size,
                           n_groups=n_groups,
                           alpha=alpha,
                           power=power)

print(f"Target parameters:")
print(f"  Effect size (f): {effect_size}")
print(f"  : {alpha}")
print(f"  Power: {power}")
print(f"  Number of groups: {n_groups}")
print(f"\nRequired sample size per group: {result['n_per_group']}")
print(f"Total sample size: {result['n_total']}")

print("\n ANOVA examples completed!")
