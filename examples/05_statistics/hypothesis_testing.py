"""
Hypothesis Testing
==================

This example demonstrates statistical hypothesis testing with Nalyst.stats.

Topics covered:
- t-tests (one-sample, two-sample, paired)
- Chi-square tests
- Non-parametric tests
- Multiple testing correction
"""

import numpy as np
from nalyst import stats

# Example 1: One-Sample t-test

# print a banner so each test is clearly separated
print("=" * 60)
print("Example 1: One-Sample t-test")
print("=" * 60)

# test whether a sample mean departs from a target value
np.random.seed(42)
sample = np.random.normal(loc=102, scale=15, size=50)  # True mean is 102

# Test against hypothesized mean of 100
result = stats.ttest_1samp(sample, popmean=100)

print(f"Sample mean: {sample.mean():.2f}")
print(f"Hypothesized mean: 100")
print(f"\nt-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")
print(f"Confidence interval (95%): [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")

if result['pvalue'] < 0.05:
    print("\n Reject H0: Sample mean significantly differs from 100")
else:
    print("\n Fail to reject H0: No significant difference from 100")

# Example 2: Two-Sample t-test

# move to comparing two independent groups
print("\n" + "=" * 60)
print("Example 2: Two-Sample t-test (Independent)")
print("=" * 60)

# Compare two independent groups
group_a = np.random.normal(loc=75, scale=10, size=30)  # Control
group_b = np.random.normal(loc=80, scale=12, size=35)  # Treatment

result = stats.ttest_ind(group_a, group_b, equal_var=True)

print(f"Group A mean: {group_a.mean():.2f} (n={len(group_a)})")
print(f"Group B mean: {group_b.mean():.2f} (n={len(group_b)})")
print(f"Mean difference: {group_b.mean() - group_a.mean():.2f}")
print(f"\nt-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")

# also show the welch adjustment for unequal variances
result_welch = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"\nWelch's t-test p-value: {result_welch['pvalue']:.4f}")

# Example 3: Paired t-test

# highlight a before vs after matched scenario
print("\n" + "=" * 60)
print("Example 3: Paired t-test")
print("=" * 60)

# Before-after measurements
before = np.array([200, 180, 220, 190, 210, 205, 195, 215, 185, 225])
after = np.array([185, 165, 200, 175, 195, 190, 180, 200, 170, 210])

result = stats.ttest_rel(before, after)

print(f"Before mean: {before.mean():.2f}")
print(f"After mean:  {after.mean():.2f}")
print(f"Mean difference: {(before - after).mean():.2f}")
print(f"\nt-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")

print("\n Significant reduction after treatment" if result['pvalue'] < 0.05
      else "\n No significant change")

# Example 4: Chi-Square Test

# show a contingency table example
print("\n" + "=" * 60)
print("Example 4: Chi-Square Test of Independence")
print("=" * 60)

# Contingency table: Gender vs Product Preference
observed = np.array([
    [30, 20, 50],  # Male: Product A, B, C
    [25, 35, 40]   # Female: Product A, B, C
])

result = stats.chi2_contingency(observed)

print("Observed frequencies:")
print(observed)
print(f"\nChi-square statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")
print(f"Degrees of freedom: {result['dof']}")

print("\nExpected frequencies:")
print(result['expected'].round(1))

if result['pvalue'] < 0.05:
    print("\n Significant association between gender and product preference")
else:
    print("\n No significant association")

# Example 5: Chi-Square Goodness of Fit

# test whether a die is fair
print("\n" + "=" * 60)
print("Example 5: Chi-Square Goodness of Fit")
print("=" * 60)

# Test if die is fair
observed_rolls = np.array([18, 22, 16, 25, 19, 20])  # 120 total rolls
expected_probs = np.ones(6) / 6  # Fair die

result = stats.chisquare(observed_rolls, f_exp=expected_probs * 120)

print(f"Observed: {observed_rolls}")
print(f"Expected: {(expected_probs * 120).astype(int)}")
print(f"\nChi-square statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")

# Example 6: Non-Parametric Tests

# bundle a few non parametric alternatives
print("\n" + "=" * 60)
print("Example 6: Non-Parametric Tests")
print("=" * 60)

# Mann-Whitney U test (non-parametric alternative to t-test)
group1 = np.array([12, 15, 18, 22, 25, 28, 32])
group2 = np.array([8, 10, 14, 16, 20, 24, 26, 30])

result = stats.mannwhitneyu(group1, group2)
print("--- Mann-Whitney U Test ---")
print(f"Group 1 median: {np.median(group1):.1f}")
print(f"Group 2 median: {np.median(group2):.1f}")
print(f"U-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")

# Wilcoxon signed-rank test (non-parametric paired test)
before = np.array([60, 65, 70, 75, 80, 85, 90])
after = np.array([55, 62, 68, 72, 78, 82, 88])

result = stats.wilcoxon(before, after)
print("\n--- Wilcoxon Signed-Rank Test ---")
print(f"W-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")

# Kruskal-Wallis test (non-parametric ANOVA)
g1 = np.array([5, 6, 7, 8, 9])
g2 = np.array([10, 11, 12, 13, 14])
g3 = np.array([15, 16, 17, 18, 19])

result = stats.kruskal(g1, g2, g3)
print("\n--- Kruskal-Wallis Test ---")
print(f"H-statistic: {result['statistic']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")

# Example 7: Multiple Testing Correction

# demonstrate popular correction strategies
print("\n" + "=" * 60)
print("Example 7: Multiple Testing Correction")
print("=" * 60)

# Multiple p-values from independent tests
pvalues = np.array([0.001, 0.015, 0.032, 0.048, 0.062, 0.085, 0.12])

# Bonferroni correction
bonferroni = stats.multipletests(pvalues, method='bonferroni')
print("--- Bonferroni Correction ---")
print(f"Original p-values:  {pvalues}")
print(f"Corrected p-values: {bonferroni['pvalues_corrected'].round(4)}")
print(f"Reject null (=0.05): {bonferroni['reject']}")

# Benjamini-Hochberg (FDR)
bh = stats.multipletests(pvalues, method='fdr_bh')
print("\n--- Benjamini-Hochberg (FDR) ---")
print(f"Corrected p-values: {bh['pvalues_corrected'].round(4)}")
print(f"Reject null (=0.05): {bh['reject']}")

# Holm-Bonferroni
holm = stats.multipletests(pvalues, method='holm')
print("\n--- Holm-Bonferroni ---")
print(f"Corrected p-values: {holm['pvalues_corrected'].round(4)}")
print(f"Reject null (=0.05): {holm['reject']}")

# Example 8: Normality Tests

# compare normal and skewed samples with standard tests
print("\n" + "=" * 60)
print("Example 8: Testing for Normality")
print("=" * 60)

# Normal data
normal_data = np.random.normal(0, 1, 100)

# Non-normal data (exponential)
nonnormal_data = np.random.exponential(1, 100)

# Shapiro-Wilk test
print("--- Shapiro-Wilk Test ---")
sw_normal = stats.shapiro(normal_data)
sw_nonnormal = stats.shapiro(nonnormal_data)

print(f"Normal data:     W={sw_normal['statistic']:.4f}, p={sw_normal['pvalue']:.4f}")
print(f"Non-normal data: W={sw_nonnormal['statistic']:.4f}, p={sw_nonnormal['pvalue']:.4f}")

# D'Agostino-Pearson test
print("\n--- D'Agostino-Pearson Test ---")
dp_normal = stats.normaltest(normal_data)
dp_nonnormal = stats.normaltest(nonnormal_data)

print(f"Normal data:     K={dp_normal['statistic']:.4f}, p={dp_normal['pvalue']:.4f}")
print(f"Non-normal data: K={dp_nonnormal['statistic']:.4f}, p={dp_nonnormal['pvalue']:.4f}")

# Example 9: Homogeneity of Variance

# finish with classic variance equality tests
print("\n" + "=" * 60)
print("Example 9: Tests for Homogeneity of Variance")
print("=" * 60)

# Groups with similar variances
g1 = np.random.normal(50, 10, 30)
g2 = np.random.normal(55, 11, 30)
g3 = np.random.normal(60, 10, 30)

# Levene's test
levene_result = stats.levene(g1, g2, g3)
print("--- Levene's Test ---")
print(f"W-statistic: {levene_result['statistic']:.4f}")
print(f"p-value: {levene_result['pvalue']:.4f}")

# Bartlett's test
bartlett_result = stats.bartlett(g1, g2, g3)
print("\n--- Bartlett's Test ---")
print(f"Chi-square: {bartlett_result['statistic']:.4f}")
print(f"p-value: {bartlett_result['pvalue']:.4f}")

print("\n Hypothesis testing examples completed!")
