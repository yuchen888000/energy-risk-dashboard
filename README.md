# European Energy Risk Dashboard

A real-time risk monitoring tool for European energy markets, built with Python and Streamlit.

**[Live Demo →](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app)**

## What It Does

This dashboard tracks the relationship between TTF natural gas futures (the European benchmark) and the clean energy transition trade (via ICLN ETF), combining quantitative risk metrics with AI-driven market regime detection and NLP sentiment analysis.

### Features

- **Risk Signal** — Current volatility vs historical average, with color-coded alert levels
- **Value at Risk (VaR)** — 95% and 99% historical VaR with return distribution visualization and 60-day rolling VaR
- **Price Trends** — Dual-axis chart of TTF gas vs ICLN with annotated EU policy events (Fit for 55, Nord Stream, EU ETS 2, CBAM)
- **Market Regime Detection** — K-Means clustering on (volatility, correlation) feature space identifies 3 regimes: Calm, Volatile, Crisis
- **NLP Sentiment** — Real-time VADER sentiment analysis on live energy news from multiple RSS sources
- **Data Export** — Download risk and sentiment data as CSV

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data | yfinance (TTF=F, ICLN) |
| Framework | Streamlit |
| Risk Analytics | NumPy, Pandas |
| Machine Learning | scikit-learn (KMeans, StandardScaler) |
| NLP | NLTK (VADER SentimentIntensityAnalyzer) |
| News Feeds | feedparser (BBC, OilPrice, Google News RSS) |
| Visualization | Matplotlib |

## Methodology

**Why ICLN as a proxy?** ICLN is not a direct carbon price — it's a clean energy equity ETF. It captures the fossil-to-renewable substitution dynamic: when gas prices spike, clean energy equities often move inversely, reflecting market repricing of the energy transition. This divergence/convergence pattern is itself a risk signal.

**VaR approach:** Historical simulation using the full sample of daily TTF returns. The 5th percentile of the return distribution gives the 95% VaR — the maximum expected daily loss not exceeded 95% of the time.

**Regime clustering:** KMeans (k=3) on standardized volatility and rolling correlation features. Clusters are labeled by ascending mean volatility: Calm → Volatile → Crisis. This provides an unsupervised, data-driven view of market states without requiring predefined thresholds.

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
