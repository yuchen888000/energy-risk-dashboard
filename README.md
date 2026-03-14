# European Energy & Commodity Risk Dashboard

A real-time risk monitoring tool for European energy and commodity markets, built with Python and Streamlit.

**[Live Demo →](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app)**

## What It Does

Multi-commodity risk analytics platform supporting TTF Natural Gas, WTI Crude Oil, Brent Crude, and EU Carbon Allowances. Combines quantitative risk metrics with AI-driven market regime detection and FinBERT-powered sentiment analysis.

### Features

- **Multi-Commodity Support** — Sidebar selector: Natural Gas, WTI, Brent, EU Carbon. All modules adapt automatically.
- **Risk Signal** — Current volatility vs historical average, with color-coded alert levels
- **Value at Risk (VaR)** — 95% and 99% historical VaR with return distribution and 60-day rolling VaR
- **GARCH Volatility Forecast** — Forward-looking 10-day volatility prediction using GARCH(1,1)
- **Price Trends** — Dual-axis chart with annotated EU policy events (Fit for 55, Nord Stream, EU ETS 2, CBAM)
- **Market Regime Detection** — K-Means clustering + absolute volatility thresholds: Calm, Volatile, Crisis
- **FinBERT Sentiment** — Transformer-based financial sentiment analysis on live European energy news (fallback to VADER)
- **Data Export** — Download risk and sentiment data as CSV

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data | yfinance (TTF=F, CL=F, BZ=F, KEUA, ICLN) |
| Framework | Streamlit |
| Risk Analytics | NumPy, Pandas |
| Volatility Modeling | arch (GARCH) |
| Machine Learning | scikit-learn (KMeans, StandardScaler) |
| NLP | FinBERT via HuggingFace Inference API (VADER fallback) |
| News Feeds | feedparser (BBC, OilPrice, Google News RSS) |
| Visualization | Matplotlib |

## Methodology

**Multi-commodity architecture:** One platform, one URL. Select any commodity and all risk modules — VaR, GARCH, regime detection, sentiment — automatically adapt. This demonstrates the system is generalizable, not hardcoded for a single asset.

**Why KEUA?** KEUA directly tracks EU ETS carbon allowance futures. When unavailable, ICLN serves as fallback.

**VaR approach:** Historical simulation — 5th percentile of daily returns = 95% VaR.

**GARCH(1,1) forecast:** Forward-looking volatility model. α captures reaction to recent shocks, β captures persistence. When α + β ≈ 1, volatility shocks are highly persistent.

**Regime clustering:** Hybrid — KMeans for pattern detection + absolute thresholds (Calm < 6%, Volatile 6–12%, Crisis > 12%).

**FinBERT sentiment:** ProsusAI/finbert transformer model called via HuggingFace Inference API. Unlike rule-based VADER, FinBERT understands financial context — e.g., "despite the crisis, markets recovered" is scored as positive.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Context

Built as a FinTech portfolio project during my Master's in International Economics at the Geneva Graduate Institute (IHEID). Demonstrates applied skills in financial data analysis, risk modeling, machine learning, and NLP for European energy and commodity markets.

## License

MIT
