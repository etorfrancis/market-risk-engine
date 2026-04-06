# 📊 Market Risk Engine
### Basel III/IV VaR, Stressed VaR & FRTB-Aligned Backtesting

> A production-grade market risk modeling pipeline demonstrating quantitative skills for banking and financial risk roles.

---

## 🎯 Project Overview

This project builds an end-to-end **market risk measurement and validation system** for a multi-asset trading portfolio. It implements all methods required under **Basel III** and introduces **FRTB (Basel IV)** concepts — the regulatory framework governing market risk capital at major banks globally.

**Portfolio:** US Equities (SPY), Gold (GLD), Emerging Markets (EEM), EUR/USD FX, Crude Oil  
**Data Source:** Yahoo Finance (daily prices, 2005–2024) + FRED macro factors  
**Stack:** Python · yfinance · arch · XGBoost · SHAP · Streamlit

---

## 📁 Repository Structure

```
market_risk_engine/
├── notebooks/
│   ├── 01_data_acquisition_eda.ipynb          # Multi-asset data, log returns, correlation
│   ├── 02_classical_var_methods.ipynb         # HS, Parametric, Monte Carlo VaR
│   ├── 03_stressed_var_scenarios.ipynb        # Basel SVaR, capital charge, named scenarios
│   ├── 04_garch_ml_var.ipynb                  # GARCH(1,1), GJR-GARCH, XGBoost exceedance
│   ├── 05_backtesting_validation.ipynb        # Kupiec, Christoffersen, Basel traffic-light
│   └── 06_regulatory_capital_report.ipynb     # FRTB ES, capital summary, risk attribution
├── data/                                       # Generated CSVs (created by notebooks)
├── models/                                     # Saved XGBoost model (pkl)
├── reports/                                    # Output charts (PNG)
├── app.py                                      # Streamlit dashboard
└── requirements.txt
```

---

## 🔬 Methodology

### VaR Methods
| Method | Assumptions | Regulatory Use |
|--------|-------------|----------------|
| Historical Simulation | Non-parametric, fat tails captured | Basel III standard |
| Parametric (Normal) | Normal distribution, analytically tractable | Basel III simple approach |
| Monte Carlo | Correlated GBM paths, flexible | Internal models |
| GARCH(1,1) Dynamic | Volatility clustering (Engle 1982) | Advanced internal models |

### Statistical Backtesting
- **Kupiec POF Test:** Validates exceedance frequency (χ² with df=1)
- **Christoffersen Independence Test:** Detects exception clustering
- **Conditional Coverage:** Joint Kupiec + Christoffersen
- **Basel Traffic-Light:** Green (≤4) / Yellow (5–9) / Red (≥10) exception zones

### Regulatory Framework
- **Basel III (2010):** 99% VaR, 10-day holding, Stressed VaR, 60-day multiplier
- **Basel 2.5 (2011):** Stressed VaR requirement — worst 12-month window
- **FRTB / Basel IV (2025):** Expected Shortfall at 97.5%, liquidity horizon scaling

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Portfolio Annualized Volatility | ~11–14% |
| 1-Day VaR (99%, HS) | ~1.5–2.0% |
| 10-Day VaR (99%, HS) | ~4.5–6.5% |
| GFC Stress Multiplier | ~2.5–3.5× normal |
| XGBoost AUC (exceedance prediction) | ~0.72–0.80 |

---

## 🚀 Running Locally

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Run notebooks in order (1 → 6) in Jupyter

# 3. Launch dashboard
streamlit run app.py
```

**Windows path note:** All notebooks define `BASE_DIR` at the top. Set this to your local project folder if needed.

---

## 📊 Dashboard Features

| Tab | Content |
|-----|---------|
| VaR Dashboard | Gauge charts, rolling VaR, method comparison table |
| Stressed VaR | Crisis period SVaR, Basel capital charge calculator |
| Backtesting | Traffic-light zone, Kupiec & Christoffersen tests |
| Scenario Analysis | Named macro scenarios + custom scenario builder |
| Model Info | Methodology documentation |

---

## 🧠 Skills Demonstrated

- **Quantitative Finance:** VaR, ES, GARCH, risk attribution, drawdown analysis
- **Regulatory Knowledge:** Basel III/IV, FRTB, capital charge calculation, backtesting framework
- **Machine Learning:** XGBoost for exceedance prediction, SHAP explainability, time-series CV
- **Statistical Testing:** Likelihood ratio tests, conditional coverage, independence testing
- **Engineering:** Modular pipeline, reproducible notebooks, Streamlit deployment

---

## 👤 Author

**Etor B.** | Data Science & Financial Risk  
*Portfolio project for banking / risk modeler roles*
