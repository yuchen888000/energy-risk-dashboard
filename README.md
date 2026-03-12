# European Energy Risk Dashboard

A real-time risk monitoring tool for European energy markets, built with Python and Streamlit.

**[Live Demo →](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app)**

## What It Does

This dashboard tracks the relationship between TTF natural gas futures (the European benchmark) and EU carbon allowance prices (via KEUA ETF, which directly tracks EU ETS futures), combining quantitative risk metrics with AI-driven market regime detection and NLP sentiment analysis.

### Features

- **Risk Signal** — Current volatility vs historical average, with color-coded alert levels
- **Value at Risk (VaR)** — 95% and 99% historical VaR with return distribution visualization and 60-day rolling VaR
- **GARCH Volatility Forecast** — Forward-looking 10-day volatility prediction using GARCH(1,1) with model parameter transparency
- **Price Trends** — Dual-axis chart of TTF gas vs EU carbon allowance with annotated EU policy events (Fit for 55, Nord Stream, EU ETS 2, CBAM)
- **Market Regime Detection** — K-Means clustering on (volatility, correlation) feature space identifies 3 regimes: Calm, Volatile, Crisis
- **NLP Sentiment** — Real-time VADER sentiment analysis on live energy news from multiple RSS sources
- **Data Export** — Download risk and sentiment data as CSV

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data | yfinance (TTF=F, KEUA, ICLN) |
| Framework | Streamlit |
| Risk Analytics | NumPy, Pandas |
| Machine Learning | scikit-learn (KMeans, StandardScaler) |
| Volatility Modeling | arch (GARCH) |
| NLP | NLTK (VADER SentimentIntensityAnalyzer) |
| News Feeds | feedparser (BBC, OilPrice, Google News RSS) |
| Visualization | Matplotlib |

## Methodology

**Why KEUA?** KEUA is a KraneShares ETF that directly tracks EU ETS carbon allowance (EUA) futures — this is the actual carbon price, not a proxy. When KEUA data is unavailable for the selected date range (it launched in late 2020), the dashboard falls back to ICLN (iShares Global Clean Energy ETF) as a secondary proxy capturing the fossil-to-renewable substitution dynamic.

**VaR approach:** Historical simulation using the full sample of daily TTF returns. The 5th percentile of the return distribution gives the 95% VaR — the maximum expected daily loss not exceeded 95% of the time.

**GARCH(1,1) forecast:** Generalized Autoregressive Conditional Heteroskedasticity — the standard volatility model on energy trading desks. Unlike rolling historical volatility (backward-looking), GARCH produces forward-looking forecasts by modeling how today's volatility depends on recent return shocks (α) and yesterday's volatility (β). When α + β approaches 1, volatility shocks are highly persistent — critical information for risk sizing.

**Regime clustering:** Hybrid approach — KMeans (k=3) on standardized volatility and rolling correlation features for pattern detection, combined with absolute volatility thresholds for regime labeling (Calm < 6%, Volatile 6–12%, Crisis > 12%). This avoids the pure-relative problem where moderate volatility gets mislabeled as Crisis simply because it's the highest in the current sample.

**Sentiment model:** VADER is a rule-based model designed for social media and news text. It scores headlines on a compound scale from -1 to +1. Headlines are filtered by energy-related keywords from live RSS feeds.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Context

Built as a FinTech demo project during my Master's in International Economics at the Geneva Graduate Institute (IHEID). The dashboard demonstrates applied skills in financial data analysis, risk modeling, machine learning, and NLP in the context of European energy and climate markets.

## License

MIT
