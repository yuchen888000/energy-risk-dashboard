# European Energy & Commodity Risk Dashboard

A multi-page risk analytics app for European energy markets, built with Python and Streamlit. One app, three pages, each designed with a different reader in mind.

**[Live Demo →](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app)**

| Page | Link | Designed with … in mind | Question it answers |
|------|------|-------------------------|---------------------|
| **Risk** | [/risk](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app/risk) | a trading-house middle office (risk control) | How much can we lose, are we within limits, and does the VaR model hold up? |
| **Power** | [/power](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app/power) | power trading desks and utilities | Does a German gas plant make money running today, and how much can that margin move? |
| **Market** | [/market](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app/market) | data and analytics firms | Which countries are exposed, what is the news mood, and is anything unusual? |

The sidebar holds the shared settings (commodity: TTF Natural Gas, WTI, Brent or EU Carbon; date range), and the selection carries across pages.

## Risk page

**Market risk of the selected commodity**
- **Risk Signal** — the current regime under another name: Calm = Low, Volatile = Medium, Crisis = High risk.
- **Price Trends** — dual-axis chart with EU policy events (Fit for 55, Nord Stream, EU ETS 2, CBAM).
- **Forward Curve** — monthly NYMEX contracts settled within the last 5 trading days. Oil curves are labelled contango or backwardation; TTF is shown as a seasonal curve comparing the next winter (Oct–Mar) with the following summer (Apr–Sep).
- **Volatility & Correlation** — 30-day rolling volatility and rolling correlation of returns.
- **Cross-Commodity Correlation Matrix** — TTF, WTI, Brent and carbon, full period vs last 30 days.
- **Value at Risk** — historical VaR 95%, VaR 99% and Expected Shortfall 97.5% on the last 250 trading days, with the full-period figures alongside; return distribution, 60-day rolling VaR, and a 250-day backtest of the commodity's own VaR (exceptions, Kupiec, Basel traffic light).
- **GARCH(1,1) Forecast** — Student-t GARCH, 10-day volatility forecast with a bootstrap 90% band.
- **Market Regime** — per-commodity volatility thresholds: Calm, Volatile, Crisis.
- **Stress Test** — price-shock slider; stressed volatility, VaR and regime from an illustrative heuristic (not a calibrated model) in which a crash raises percentage risk more than a spike of the same size.
- **Portfolio VaR** — custom weights across the four commodities, with the diversification benefit.

**Positions & Limits**
- Enter positions in €m (long or short) in TTF, WTI, Brent and carbon.
- **€ VaR 95% / 99% and Expected Shortfall 97.5%** from the last 250 trading days.
- **Limit usage** against a user-set VaR limit (green < 80%, amber 80–100%, red = breach), with standalone VaR per position.
- **VaR backtest** over the last 250 days: exceptions at 95% and 99%, **Kupiec** proportion-of-failures test, and the **Basel traffic light** (0–4 green, 5–9 yellow, 10+ red).

## Power page

- **German day-ahead power** (bidding zone DE-LU) from the Energy-Charts API, averaged per Berlin calendar day (baseload). Requests are spaced out and retried with backoff on HTTP 429; completed years are cached for 30 days and the current year for an hour, and years that still fail are skipped with a warning.
- **Clean spark spread** = power − TTF / 0.5 − (0.202 / 0.5) × EUA, where 0.5 is the gas plant efficiency and 0.202 tCO2/MWh the natural gas emission factor.
- Headline: current spread, green when running a gas plant is profitable and red when it isn't.
- Charts: the spread over time, and its decomposition into power price, fuel cost and carbon cost.
- **Spread risk** in €/MWh (not %, since power prices and the spread can be negative): 30-day volatility and 95% historical VaR of daily changes, plus the daily € VaR for a **400 MW unit running 16 hours a day**.
- **EUA price** from EEX primary-auction results (€/t). If EEX cannot be reached, the carbon ETC is rescaled to a user-entered EUA price and labelled as an approximation.
- **Download CSV** — date, DE power price, TTF, EUA price and its source, fuel cost, carbon cost and clean spark spread per business day.

## Market page

- **Country Risk** — 31 countries (EU-27 + Switzerland, UK, Norway, Turkey), 2020–2024. An illustrative structural score (import dependency, carbon intensity, renewable share) is scaled by the commodity's latest daily volatility (refreshed hourly), weighted by each country's dependency. Includes per-country trend, an exposure-weighted volatility index (illustrative) and country news sentiment (only headlines that name the country or use its adjective).
- **News Sentiment** — FinBERT (ProsusAI/finbert via HuggingFace) on live headlines from BBC Business, OilPrice and Google News; FinVADER as fallback, with the reason FinBERT was not used shown on the page. Every feed keeps only headlines containing one of the selected commodity's keywords (whole words), and drops US-domestic stories with no link to Europe (US LNG export news is kept). The score measures headline tone, not whether the news is bullish or bearish for prices.
- **30-Day Sentiment Trend** — daily average sentiment, scored with FinVADER.
- **Anomaly Detection** — volatility z-score, GARCH 10-day forecast vs long-run GARCH volatility, correlation shift, sentiment–regime divergence, recent tail events.
- **AI Risk Interpretation** — the signals above summarised in three sentences by Claude via the Anthropic API (optional).
- **Data Export** — CSV downloads of commodity risk data, sentiment and country risk.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Market data | yfinance (TTF=F, CL=F, BZ=F, CARB.L → KRBN → ICLN) |
| Power data | Energy-Charts API (Fraunhofer ISE), DE-LU day-ahead |
| EUA prices | EEX primary-auction reports |
| Framework | Streamlit (multi-page with `st.navigation`) |
| Risk analytics | NumPy, Pandas, SciPy |
| Volatility modelling | arch (GARCH) |
| NLP | FinBERT (HuggingFace Inference API), FinVADER fallback |
| News feeds | feedparser + requests (BBC, OilPrice, Google News RSS) |
| Country data | Eurostat API (nrg_ind_id, nrg_ind_ren); carbon intensity and any missing values are hand-entered estimates |
| Visualisation | Matplotlib |

## Methodology Notes

**Carbon benchmark.** KEUA, the original EUA ETF, was liquidated in March 2026. The app now tries `CARB.L` (WisdomTree Carbon ETC, USD line on the LSE), then `KRBN`, then `ICLN`, and labels the charts from whichever resolves. `CARB.L` tracks ICE EUA futures, but its quote is an ETC share price in USD, not €/tCO2, so its returns also carry EUR/USD moves, roll yield and fees. The Power page uses EEX auction prices in €/t instead.

**VaR.** Historical simulation: the 5th percentile of daily returns is the 95% VaR, and Expected Shortfall 97.5% is the average loss on the days beyond the 97.5% VaR. The headline figures use the last 250 trading days; the full-period figures (which include the 2022 crisis on the default range) are shown alongside. On the Risk page's position block, VaR and ES come from the last 250 days of the book's € P&L.

**VaR backtest.** The Risk page backtests the selected commodity's own VaR per unit of exposure. The position block's backtest is hypothetical: today's positions are applied to past returns, and each day's VaR is estimated only from the 250 days before it (no look-ahead). Kupiec's test also rejects a model with too few exceptions (over-conservative).

**GARCH(1,1).** Fitted with Student-t shocks, since daily energy returns have fat tails. α captures reaction to shocks, β persistence; α + β near 1 means volatility shocks fade slowly. The forecast reverts to the long-run volatility √(ω / (1 − α − β)). When α + β ≥ 0.99, or that long-run value is more than three times the sample average, the formula is unstable, so the app uses the sample average 30-day volatility as the reference instead and says so. The anomaly check compares the 10-day forecast with that reference, so a return towards it is not flagged.

**Stale prices.** Runs of 5 or more identical closes (a suspended or illiquid quote) are treated as missing, and no return is computed across the gap. The Risk page lists the excluded periods.

**Stress test.** Illustrative heuristic: stressed volatility and VaR = current values × (1 + |shock| / 50) / (1 + shock / 100). Absolute moves grow with the shock and are measured against the new price level, so a crash raises percentage risk more than a spike. VaR is capped at −100%.

**Regime detection.** Per-commodity thresholds only: Volatile starts at the 50th and Crisis at the 90th percentile of the commodity's own 30-day volatility since 2010 (the selected range if that history is too short), so a lower volatility never gets a higher regime. The Risk Signal uses the same classification. The Risk page caption shows the thresholds used.

**Clean spark spread caveats.** Power is the day-ahead spot price, while TTF is the front-month future, so the tenors don't match; desks use contracts with the same delivery period. Efficiency is assumed at 50%. `TTF=F` is a continuous front-month series that jumps at each monthly roll, which inflates the spread's volatility and VaR. Only business days are used.

**Country risk.** An illustrative index, not an official rating. Structural score: for gas and oil, commodity import dependency 35%, total energy dependency 25%, carbon intensity rank 20%, inverse renewable share 20%; for carbon, total energy dependency 40%, carbon intensity rank 30%, inverse renewable share 30%. Each variable enters once. It is multiplied by a volatility multiplier weighted by each country's dependency, so high-dependency countries feel the same market shock more. Import dependency and renewable share are downloaded from Eurostat (nrg_ind_id, nrg_ind_ren) and cached for a day; carbon intensity, and any value Eurostat does not return, are hand-entered estimates, and the table marks each country's data source. Risk levels (High above 70, Medium above 50) are the same in the ranking table, the detail panel and the chart.

## Project Structure

```
app.py            entry point: shared sidebar, navigation (Streamlit Cloud runs this)
common.py         carbon benchmark, commodities, data loading, cached risk calculations
country_data.py   Eurostat loader for country data, with hand-entered fallback
power_data.py     Energy-Charts power prices and EEX EUA auction prices
views/risk.py     Risk page
views/power.py    Power page
views/market.py   Market page
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Optional API keys

Without keys the app still runs: sentiment falls back to FinVADER and the AI summary is hidden.

- **FinBERT**: create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a fine-grained token needs the "Make calls to Inference Providers" permission) and set `HF_TOKEN`. If FinBERT still is not used, the Market page shows a short reason (for example token missing or invalid, no Inference permission, rate limit).
- **AI Risk Interpretation**: set `ANTHROPIC_API_KEY`.

On Streamlit Cloud, add them under Settings → Secrets (never put tokens in code); locally, set them as environment variables.

## Project Context

Built as a FinTech portfolio project during my Master's in International Economics at the Geneva Graduate Institute (IHEID).

## License

MIT
