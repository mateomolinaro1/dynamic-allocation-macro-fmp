# Dynamic Allocation Macro Factor-Mimicking Portfolios

This repository implements a complete research pipeline combining **macro factor-mimicking portfolios (FMPs)**, **machine-learning-based macroeconomic forecasting**, and **dynamic asset allocation**.  
The project focuses on inflation as a representative macroeconomic variable but is fully extensible to any macro factor from the FRED-MD dataset.

---

## 📌 Project Overview

Macroeconomic variables such as inflation, interest rates, and business cycle indicators play a central role in asset pricing and portfolio allocation. However, they are not directly tradable and are difficult to incorporate in a systematic and operational way into portfolio construction.

This project addresses this challenge by:

1. **Constructing tradable macro factor-mimicking portfolios (FMPs)** based on stock-level macro exposures.
2. **Forecasting macroeconomic dynamics** using statistical and machine-learning models.
3. **Designing dynamic asset allocation strategies** that adjust portfolio exposure based on macro forecasts.

The methodology follows the framework proposed in  
*Esakia & Goltz (2022), “Targeting Macroeconomic Exposures in Equity Portfolios: A Firm-Level Measurement Approach for Out-of-Sample Robustness”*.

---

## 📂 Repository Structure
To be completed.

---

## 🔧 Methodology

### 1. Factor Mimicking Portfolios (FMP)

- Monthly CRSP equity returns (NYSE, AMEX, NASDAQ)
- Top 1,000 stocks by market capitalization
- Stock-level regressions controlling for market exposure
- Inflation innovations extracted from FRED-MD
- Exponentially weighted regressions (half-life = 60 months)
- Bayesian shrinkage (Vasicek-style) for beta stabilization
- Long–short portfolios formed using top/bottom deciles of estimated betas
- Cross-sectional winsorization of betas

### 2. Inflation Forecasting

- Dataset: **FRED-MD (125 macro variables)**
- Expanding walk-forward cross-validation
- Forecast horizon: 1 month
- Models:
  - OLS
  - Lasso, Ridge, Elastic Net
  - Random Forest, Gradient Boosting
  - XGBoost, LightGBM
  - Support Vector Regression (SVR)
  - Feedforward Neural Network (MLP)
- Optional dimensionality reduction via PCA

### 3. Dynamic Allocation Strategy

At each rebalancing date:
- Go long the **high-beta inflation FMP** if forecasted inflation is positive
- Go long the **low-beta inflation FMP** if forecasted inflation is negative
- Allocate to the **benchmark (long-only equally weighted portfolio)** when the forecast signal is neutral
- Transaction costs are explicitly accounted for in the allocation phase

---

## 📊 Outputs

The pipeline automatically generates:

- Distribution of Bayesian macro betas
- Time-series dynamics of cross-sectional beta statistics
- Significance and stability diagnostics (Newey–West corrected)
- Forecasting performance metrics:
  - Out-of-sample RMSE
  - Sign accuracy
- Equity curves for FMP legs and benchmarks
- Performance tables:
  - Annualized return
  - Annualized volatility
  - Sharpe ratio
  - Maximum drawdown
- Dynamic allocation strategy results

