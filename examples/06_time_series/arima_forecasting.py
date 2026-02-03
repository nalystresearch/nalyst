"""
Time Series Forecasting
=======================

This example demonstrates time series analysis with Nalyst.timeseries.

Topics covered:
- ARIMA models
- SARIMA (seasonal ARIMA)
- Exponential smoothing
- Trend and seasonality decomposition
- Forecasting evaluation
"""

import numpy as np
from nalyst import timeseries

# Example 1: ARIMA Model Basics

# print a banner so each forecasting topic is easy to find
print("=" * 60)
print("Example 1: ARIMA Model Basics")
print("=" * 60)

np.random.seed(42)

# generate a plain ar(1) process to recover parameters
n = 200
ar_coef = 0.7
y = np.zeros(n)
for t in range(1, n):
    y[t] = ar_coef * y[t-1] + np.random.randn()

# fit a simple arima that matches the data generating process
model = timeseries.ARIMA(order=(1, 0, 0))
model.train(y)

print(f"True AR coefficient: {ar_coef}")
print(f"Estimated AR coefficient: {model.ar_params_[0]:.4f}")

# produce a short horizon forecast
forecast = model.infer(steps=10)
print(f"\n10-step forecast: {forecast.round(3)}")

# Example 2: ARIMA with Differencing

# demonstrate differencing to remove a trend component
print("\n" + "=" * 60)
print("Example 2: ARIMA with Trend (Differencing)")
print("=" * 60)

# generate a simple trend plus random walk noise
t = np.arange(100)
trend = 0.5 * t
noise = np.cumsum(np.random.randn(100))  # Random walk
y = trend + noise

# fit an arima with first differencing to capture the trend
model = timeseries.ARIMA(order=(1, 1, 1))
model.train(y)

print("ARIMA(1,1,1) Model:")
print(f"AR coefficients: {model.ar_params_}")
print(f"MA coefficients: {model.ma_params_}")

# Forecast
forecast = model.infer(steps=10)
print(f"\nForecast for next 10 periods:")
print(forecast.round(2))

# Example 3: SARIMA (Seasonal ARIMA)

# switch to seasonal data and highlight sarima settings
print("\n" + "=" * 60)
print("Example 3: SARIMA (Seasonal ARIMA)")
print("=" * 60)

# create monthly data with a clear yearly wave plus trend
n = 120  # 10 years of monthly data
t = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * t / 12)  # Yearly cycle
trend = 0.1 * t
noise = np.random.randn(n) * 2
y = trend + seasonal + noise

# Fit SARIMA(1,0,1)(1,0,1,12)
model = timeseries.SARIMA(
    order=(1, 0, 1),           # Non-seasonal (p, d, q)
    seasonal_order=(1, 0, 1, 12)  # Seasonal (P, D, Q, s)
)
model.train(y)

print("SARIMA(1,0,1)(1,0,1)[12] Model fitted")
print(f"Seasonal period: 12 months")

# Forecast next 24 months
forecast = model.infer(steps=24)
print(f"\n24-month forecast (first 12):")
print(forecast[:12].round(2))

# Example 4: Exponential Smoothing

# banner for the smoothing family of models
print("\n" + "=" * 60)
print("Example 4: Exponential Smoothing Methods")
print("=" * 60)

# craft sales data showing both trend and seasonal swings
np.random.seed(123)
n = 48  # 4 years quarterly
t = np.arange(n)
trend = 100 + 2 * t
seasonal = 20 * np.sin(2 * np.pi * t / 4)
noise = np.random.randn(n) * 5
y = trend + seasonal + noise

# compare a few smoothing flavors on the same series
ses = timeseries.SimpleExpSmoothing(alpha=0.3)
ses.train(y)
forecast_ses = ses.infer(steps=4)
print("Simple Exponential Smoothing:")
print(f"  Forecast: {forecast_ses.round(2)}")

# Holt's Linear Trend
holt = timeseries.HoltLinear(alpha=0.3, beta=0.1)
holt.train(y)
forecast_holt = holt.infer(steps=4)
print("\nHolt's Linear Trend Method:")
print(f"  Forecast: {forecast_holt.round(2)}")

# Holt-Winters (with seasonality)
hw = timeseries.HoltWinters(alpha=0.3, beta=0.1, gamma=0.2,
                             seasonal_periods=4, seasonal='additive')
hw.train(y)
forecast_hw = hw.infer(steps=4)
print("\nHolt-Winters (additive seasonality):")
print(f"  Forecast: {forecast_hw.round(2)}")

# Example 5: Time Series Decomposition

# decompose the prior series into trend, seasonality, and residuals
print("\n" + "=" * 60)
print("Example 5: Time Series Decomposition")
print("=" * 60)

# Decompose seasonal data
result = timeseries.seasonal_decompose(y, period=4, model='additive')

print("Decomposition Results:")
print(f"  Trend component range:    [{result['trend'].min():.1f}, {result['trend'].max():.1f}]")
print(f"  Seasonal component range: [{result['seasonal'].min():.1f}, {result['seasonal'].max():.1f}]")
print(f"  Residual std:             {result['residual'].std():.2f}")

# Print seasonal pattern
seasonal_pattern = result['seasonal'][:4]
print(f"\nQuarterly seasonal pattern: {seasonal_pattern.round(2)}")

# Example 6: Stationarity Tests

# compare stationary vs nonstationary series using adf and kpss
print("\n" + "=" * 60)
print("Example 6: Stationarity Tests")
print("=" * 60)

# Stationary series
stationary = np.random.randn(100)

# Non-stationary series (random walk)
nonstationary = np.cumsum(np.random.randn(100))

# Augmented Dickey-Fuller test
adf_stationary = timeseries.adf_test(stationary)
adf_nonstationary = timeseries.adf_test(nonstationary)

print("Augmented Dickey-Fuller Test:")
print(f"\nStationary series:")
print(f"  ADF statistic: {adf_stationary['statistic']:.4f}")
print(f"  p-value: {adf_stationary['pvalue']:.4f}")
print(f"  Stationary: {'Yes' if adf_stationary['pvalue'] < 0.05 else 'No'}")

print(f"\nNon-stationary series (random walk):")
print(f"  ADF statistic: {adf_nonstationary['statistic']:.4f}")
print(f"  p-value: {adf_nonstationary['pvalue']:.4f}")
print(f"  Stationary: {'Yes' if adf_nonstationary['pvalue'] < 0.05 else 'No'}")

# KPSS test (null is stationary)
kpss_stationary = timeseries.kpss_test(stationary)
kpss_nonstationary = timeseries.kpss_test(nonstationary)

print("\nKPSS Test (H0: stationary):")
print(f"  Stationary series:     p = {kpss_stationary['pvalue']:.4f}")
print(f"  Non-stationary series: p = {kpss_nonstationary['pvalue']:.4f}")

# Example 7: ACF and PACF

# inspect autocorrelation patterns for an ar(2) process
print("\n" + "=" * 60)
print("Example 7: ACF and PACF Analysis")
print("=" * 60)

# AR(2) process
ar2 = np.zeros(200)
for t in range(2, 200):
    ar2[t] = 0.5 * ar2[t-1] + 0.3 * ar2[t-2] + np.random.randn()

# Compute ACF
acf_values, acf_conf = timeseries.acf(ar2, nlags=15)
print("ACF values (lags 0-5):")
print(f"  {acf_values[:6].round(3)}")

# Compute PACF
pacf_values, pacf_conf = timeseries.pacf(ar2, nlags=15)
print("\nPACF values (lags 1-5):")
print(f"  {pacf_values[1:6].round(3)}")
print("\n PACF cuts off after lag 2 (AR(2) signature)")

# Example 8: Forecast Evaluation

# evaluate a couple of models on a holdout slice
print("\n" + "=" * 60)
print("Example 8: Forecast Evaluation")
print("=" * 60)

# Split data into train/test
y = trend + seasonal + noise  # From Example 4
train, test = y[:-8], y[-8:]

# Fit models and forecast
models = {
    'ARIMA(1,1,1)': timeseries.ARIMA(order=(1, 1, 1)),
    'HoltWinters': timeseries.HoltWinters(alpha=0.3, beta=0.1, gamma=0.2,
                                           seasonal_periods=4)
}

print("Forecast Accuracy Metrics:")
print("-" * 60)
print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
print("-" * 60)

for name, model in models.items():
    model.train(train)
    forecast = model.infer(steps=len(test))

    # Compute metrics
    mae = np.abs(test - forecast).mean()
    rmse = np.sqrt(np.mean((test - forecast) ** 2))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    print(f"{name:<20} {mae:>10.2f} {rmse:>10.2f} {mape:>9.1f}%")

# Example 9: VAR (Vector Autoregression)

# finish with a multivariate var example
print("\n" + "=" * 60)
print("Example 9: VAR (Multivariate Time Series)")
print("=" * 60)

# Generate two related time series
n = 100
x1 = np.zeros(n)
x2 = np.zeros(n)
for t in range(1, n):
    x1[t] = 0.5 * x1[t-1] + 0.3 * x2[t-1] + np.random.randn()
    x2[t] = 0.2 * x1[t-1] + 0.4 * x2[t-1] + np.random.randn()

data = np.column_stack([x1, x2])

# Fit VAR model
var_model = timeseries.VAR(lags=2)
var_model.train(data)

print(f"VAR(2) model with 2 variables")
print(f"Coefficient matrix shapes:")
for i, coef in enumerate(var_model.coefficients_):
    print(f"  Lag {i+1}: {coef.shape}")

# Forecast
forecast = var_model.infer(steps=5)
print(f"\n5-step forecast:")
print(f"  Variable 1: {forecast[:, 0].round(3)}")
print(f"  Variable 2: {forecast[:, 1].round(3)}")

print("\n Time series examples completed!")
