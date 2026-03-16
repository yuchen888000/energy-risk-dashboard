# European Energy & Commodity Risk Dashboard

A real-time, multi-commodity risk analytics platform for European energy markets. Built with Python and Streamlit.

**[Live Demo →](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app)**

## What It Does

Select any commodity — TTF Natural Gas, WTI Crude Oil, Brent Crude, or EU Carbon Allowances — and every risk module adapts automatically: pricing, volatility, VaR, GARCH forecasting, market regime detection, sentiment analysis, and country-level risk exposure.

### Features

**Price & Risk Analytics**
- **Multi-Commodity Support** — Sidebar selector with 4 commodities. All modules adapt automatically.
- **Risk Signal** — Real-time volatility vs historical average, color-coded alert (High/Medium/Low).
- **Value at Risk (VaR)** — 95% and 99% historical VaR with return distribution and 60-day rolling VaR.
- **GARCH(1,1) Volatility Forecast** — Forward-looking 10-day prediction with confidence bands and model parameter transparency.
- **Price Trends** — Dual-axis chart with annotated EU policy events (Fit for 55, Nord Stream, EU ETS 2, CBAM).
- **Market Regime Detection** — K-Means clustering + absolute volatility thresholds: Calm, Volatile, Crisis.

**Cross-Commodity & Stress Testing**
- **Correlation Matrix** — 4×4 heatmap showing how TTF Gas, WTI, Brent, and EU Carbon move relative to each other. Full-period vs last 30 days to detect regime shifts.
- **Stress Test Scenario** — Slider to simulate price shocks (-50% to +100%). Shows stressed volatility, VaR, regime shift, and top 10 most impacted countries.
- **Portfolio VaR** — Set custom weights across 4 commodities. Calculates combined portfolio risk accounting for cross-commodity correlations, with diversification benefit analysis.

**Country Risk**
- **31 European Countries** — EU-27 + Switzerland, UK, Norway, Turkey.
- **Commodity-Aware** — Selecting gas shows gas dependency; selecting oil shows oil dependency.
- **Dynamic Risk Scoring** — Structural vulnerability (Eurostat) × real-time commodity volatility, weighted by each country's dependency. High-dependency countries feel market shocks more.
- **Per-Country Risk Signal** — Adjusted volatility and VaR for each country, live.
- **Dependency-Weighted Volatility Curve** — Real-time chart showing how market volatility translates to country-specific risk.
- **Per-Country News Sentiment** — Live energy news filtered by selected country.
- **Year Slider (2020–2024)** — Track how energy risk shifted through the 2022 crisis and recovery.

**Sentiment Analysis (NLP)**
- **FinBERT** — Main sentiment module. Transformer model fine-tuned on financial text (ProsusAI/finbert via HuggingFace Inference API). Applied to 10 live headlines from BBC Business and OilPrice. Falls back to FinVADER if API unavailable.
- **FinVADER** — Fallback sentiment model. VADER enhanced with financial domain lexicons (SentiBigomics + Henry), significantly more accurate than standard VADER for financial news.
- **30-Day Sentiment Trend** — Daily average sentiment over past month via Google News RSS, visualized as bar chart with trend line. Scored with FinVADER.
- **Headline Analysis** — Most positive and most negative headlines with source links and sentiment scores.
- **European Focus** — News sourced from BBC Business, OilPrice, and Google News (commodity-specific + EU energy queries).
- **News Sources** — BBC Business, OilPrice, Google News (EU Energy, EU Carbon, commodity-specific). US-domestic headlines filtered out; geopolitically relevant global news retained.

**Data Export**
- Download commodity risk data (CSV)
- Download sentiment data (CSV)
- Download country risk data with dynamic scores (CSV)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data | yfinance (TTF=F, CL=F, BZ=F, KEUA, ICLN) |
| Framework | Streamlit |
| Risk Analytics | NumPy, Pandas |
| Volatility Modeling | arch (GARCH) |
| Machine Learning | scikit-learn (KMeans, StandardScaler) |
| NLP — Main Sentiment | FinBERT via HuggingFace Inference API (ProsusAI/finbert) |
| NLP — Fallback | FinVADER (VADER + SentiBigomics + Henry financial lexicons) |
| News Feeds | feedparser + requests (BBC, OilPrice, Google News RSS) |
| Country Data | Eurostat (nrg_ind_id, sdg_07_50, nrg_ind_ren, nrg_ind_ei), EEA, IEA |
| Visualization | Matplotlib |

## Methodology

**Multi-commodity architecture:** One platform, one URL. Select any commodity and all modules adapt. Demonstrates the system is generalizable, not hardcoded for a single asset.

**Why KEUA?** KEUA directly tracks EU ETS carbon allowance futures. ICLN serves as fallback when KEUA data is unavailable.

**VaR:** Historical simulation — 5th percentile of daily returns = 95% VaR.

**GARCH(1,1):** Forward-looking volatility. α captures shock reaction, β captures persistence. α + β near 1 = highly persistent volatility.

**Regime detection:** Hybrid — KMeans for pattern detection + absolute thresholds (Calm < 6%, Volatile 6–12%, Crisis > 12%).

**Country risk:** Composite structural score from 6 factors (commodity dependency, carbon intensity, total energy dependency, renewable share, price sensitivity, dependency rank), multiplied by country-specific volatility multiplier. Countries with higher dependency feel the same market shock more intensely.

**NLP architecture:** Two-layer sentiment design. FinBERT (ProsusAI/finbert) handles current headlines — unlike rule-based methods, it understands financial context and domain-specific language. FinVADER (VADER enhanced with SentiBigomics and Henry financial lexicons) serves as fallback, providing significantly more accurate financial sentiment than standard VADER when FinBERT API is unavailable.

**Portfolio VaR:** Weighted portfolio returns computed from individual commodity returns. VaR is calculated on the combined return series, automatically capturing cross-commodity correlations. Diversification benefit = sum of individual weighted VaRs minus portfolio VaR.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### FinBERT Setup (Optional)

The dashboard uses FinBERT for financial sentiment analysis. Without a token, it falls back to FinVADER (still accurate for financial text).

To enable FinBERT:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. For Streamlit Cloud: add `HF_TOKEN = "hf_yourtoken"` in Settings → Secrets (never put tokens in code)
4. For local: set environment variable `HF_TOKEN=hf_yourtoken`

## Project Context

Built as a FinTech portfolio project during my Master's in International Economics at the Geneva Graduate Institute (IHEID), with iterative feedback from Professor Joëlle Noailly. Demonstrates applied skills in financial data analysis, risk modeling, machine learning, and NLP for European energy and commodity markets.

## License

MIT
