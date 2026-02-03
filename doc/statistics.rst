Statistical Analysis
====================

Nalyst provides comprehensive statistical analysis capabilities through the ``stats``, ``timeseries``, ``glm``, and ``diagnostics`` modules.

Hypothesis Testing
------------------

t-Tests
~~~~~~~

.. code-block:: python

    from nalyst import stats
    
    # One-sample t-test
    result = stats.ttest_1samp(sample, popmean=100)
    print(f"t-statistic: {result['statistic']}")
    print(f"p-value: {result['pvalue']}")
    
    # Two-sample t-test (independent)
    result = stats.ttest_ind(group_a, group_b, equal_var=True)
    
    # Welch's t-test (unequal variances)
    result = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    # Paired t-test
    result = stats.ttest_rel(before, after)

Chi-Square Tests
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Chi-square test of independence
    observed = np.array([[30, 20], [25, 35]])
    result = stats.chi2_contingency(observed)
    
    # Chi-square goodness of fit
    result = stats.chisquare(observed, f_exp=expected)

ANOVA
~~~~~

.. code-block:: python

    # One-way ANOVA
    result = stats.f_oneway(group1, group2, group3)
    
    # Two-way ANOVA
    result = stats.f_twoway(values, factor1, factor2)
    
    # Repeated measures ANOVA
    result = stats.f_oneway_rm(time1, time2, time3)
    
    # Welch's ANOVA (unequal variances)
    result = stats.welch_anova(group1, group2, group3)

Post-Hoc Tests
~~~~~~~~~~~~~~

.. code-block:: python

    # Tukey HSD
    result = stats.tukey_hsd(g1, g2, g3, groups=['A', 'B', 'C'])
    
    # Games-Howell (for unequal variances)
    result = stats.games_howell(g1, g2, g3, groups=['A', 'B', 'C'])

Non-Parametric Tests
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Mann-Whitney U test
    result = stats.mannwhitneyu(group1, group2)
    
    # Wilcoxon signed-rank test
    result = stats.wilcoxon(before, after)
    
    # Kruskal-Wallis test
    result = stats.kruskal(g1, g2, g3)
    
    # Friedman test
    result = stats.friedmanchisquare(t1, t2, t3)

Normality Tests
~~~~~~~~~~~~~~~

.. code-block:: python

    # Shapiro-Wilk test
    result = stats.shapiro(data)
    
    # D'Agostino-Pearson test
    result = stats.normaltest(data)
    
    # Kolmogorov-Smirnov test
    result = stats.kstest(data, 'norm')

Correlation Analysis
--------------------

.. code-block:: python

    # Pearson correlation
    result = stats.pearsonr(x, y)
    
    # Spearman rank correlation
    result = stats.spearmanr(x, y)
    
    # Kendall tau
    result = stats.kendalltau(x, y)
    
    # Partial correlation
    result = stats.partial_corr(x, y, z)
    
    # Correlation matrix
    corr_matrix, pvalue_matrix = stats.corrmatrix(data)
    
    # Point-biserial correlation
    result = stats.pointbiserialr(binary_var, continuous_var)

Multiple Testing Correction
---------------------------

.. code-block:: python

    pvalues = [0.001, 0.015, 0.032, 0.048]
    
    # Bonferroni
    result = stats.multipletests(pvalues, method='bonferroni')
    
    # Benjamini-Hochberg (FDR)
    result = stats.multipletests(pvalues, method='fdr_bh')
    
    # Holm-Bonferroni
    result = stats.multipletests(pvalues, method='holm')

Power Analysis
--------------

.. code-block:: python

    # Power for t-test
    result = stats.power_ttest(effect_size=0.5, n=None, alpha=0.05, power=0.8)
    print(f"Required n: {result['n']}")
    
    # Power for ANOVA
    result = stats.power_anova(effect_size=0.25, n_groups=3, alpha=0.05, power=0.8)

Time Series Analysis
--------------------

ARIMA Models
~~~~~~~~~~~~

.. code-block:: python

    from nalyst import timeseries
    
    # ARIMA
    model = timeseries.ARIMA(order=(1, 1, 1))
    model.train(y)
    forecast = model.infer(steps=10)
    
    # SARIMA (Seasonal)
    model = timeseries.SARIMA(
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12)
    )
    model.train(y)

Exponential Smoothing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Simple Exponential Smoothing
    model = timeseries.SimpleExpSmoothing(alpha=0.3)
    
    # Holt's Linear Trend
    model = timeseries.HoltLinear(alpha=0.3, beta=0.1)
    
    # Holt-Winters
    model = timeseries.HoltWinters(
        alpha=0.3, beta=0.1, gamma=0.2,
        seasonal_periods=12,
        seasonal='additive'
    )

Decomposition
~~~~~~~~~~~~~

.. code-block:: python

    result = timeseries.seasonal_decompose(y, period=12, model='additive')
    trend = result['trend']
    seasonal = result['seasonal']
    residual = result['residual']

Stationarity Tests
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Augmented Dickey-Fuller test
    result = timeseries.adf_test(y)
    
    # KPSS test
    result = timeseries.kpss_test(y)

ACF and PACF
~~~~~~~~~~~~

.. code-block:: python

    # Autocorrelation function
    acf_values, conf_int = timeseries.acf(y, nlags=20)
    
    # Partial autocorrelation function
    pacf_values, conf_int = timeseries.pacf(y, nlags=20)

VAR (Vector Autoregression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Multivariate time series
    model = timeseries.VAR(lags=2)
    model.train(multivariate_data)
    forecast = model.infer(steps=5)

Generalized Linear Models
-------------------------

.. code-block:: python

    from nalyst import glm
    
    # Logistic Regression (Binomial)
    model = glm.GLM(family='binomial', link='logit')
    model.train(X, y)
    
    # Poisson Regression
    model = glm.GLM(family='poisson', link='log')
    model.train(X, count_data)
    
    # Gamma Regression
    model = glm.GLM(family='gamma', link='inverse')
    model.train(X, positive_continuous)
    
    # Negative Binomial
    model = glm.GLM(family='negative_binomial', link='log')

Regression Diagnostics
----------------------

.. code-block:: python

    from nalyst import diagnostics
    
    # Residual analysis
    residuals = diagnostics.get_residuals(model, X, y)
    
    # Heteroscedasticity tests
    result = diagnostics.breusch_pagan_test(model, X, y)
    result = diagnostics.white_test(model, X, y)
    
    # Multicollinearity
    vif = diagnostics.variance_inflation_factor(X)
    
    # Influential points
    cooks_d = diagnostics.cooks_distance(model, X, y)
    leverage = diagnostics.leverage(X)
    
    # Autocorrelation
    result = diagnostics.durbin_watson(residuals)

Quantile Regression
-------------------

.. code-block:: python

    from nalyst import quantile
    
    model = quantile.QuantileRegressor(quantile=0.5)  # Median
    model.train(X, y)
    
    # Multiple quantiles
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        model = quantile.QuantileRegressor(quantile=q)
        model.train(X, y)
        predictions = model.infer(X_test)

Nonparametric Methods
---------------------

.. code-block:: python

    from nalyst import nonparametric
    
    # Kernel Density Estimation
    kde = nonparametric.KernelDensityEstimator(bandwidth=0.5)
    kde.train(data)
    density = kde.evaluate(grid_points)
    
    # LOWESS (locally weighted regression)
    model = nonparametric.LOWESS(frac=0.3)
    smoothed = model.train_transform(x, y)

GAM (Generalized Additive Models)
---------------------------------

.. code-block:: python

    from nalyst import gam
    
    model = gam.GAM(
        terms=[
            gam.s(0, n_splines=10),  # Smooth term for feature 0
            gam.s(1, n_splines=10),  # Smooth term for feature 1
            gam.f(2)                  # Linear term for feature 2
        ],
        family='gaussian'
    )
    model.train(X, y)

Survival Analysis
-----------------

.. code-block:: python

    from nalyst import survival
    
    # Kaplan-Meier estimator
    km = survival.KaplanMeierEstimator()
    km.train(durations, event_observed)
    survival_function = km.survival_function_
    
    # Cox Proportional Hazards
    cph = survival.CoxPHModel()
    cph.train(X, durations, event_observed)
    hazard_ratios = cph.hazard_ratios_
    
    # Log-rank test
    result = survival.logrank_test(durations1, durations2, events1, events2)
