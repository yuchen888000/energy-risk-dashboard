# European Energy & Commodity Risk Dashboard

A multi-page risk analytics app for European energy markets, built with Python and Streamlit. One app, three pages, each designed with a different reader in mind.

**[Live Demo →](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app)**

| Page | Link | Designed with … in mind | Question it answers |
|------|------|-------------------------|---------------------|
| **Risk** | [/risk](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app/risk) | a trading-house middle office (risk control) | How much can we lose, are we within limits, and does the VaR model hold up? |
| **Power** | [/power](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app/power) | power trading desks and utilities | How much does a German gas plant earn over its fuel and carbon costs in the hours it runs, and how much can that margin move? |
| **Market** | [/market](https://energy-risk-dashboard-zj3n46fw8txggaj3su3br6.streamlit.app/market) | data and analytics firms | Which countries are exposed, what is the news mood, and is anything unusual? |

The sidebar holds the shared settings (commodity: TTF Natural Gas, WTI, Brent or EU Carbon; date range), and the selection carries across pages.

## Risk page

**Market risk of the selected commodity**
- **Volatility Regime**: current 30-day volatility (daily) against the commodity's own last 3 years: Calm, Volatile or Crisis-level. It measures volatility, not overall risk. Next to it: current and average volatility (daily, annualised in the tooltip), the correlation with the comparison series since the start of the range and over the last 30 days, and the 1-day VaR 95%.
- **Price Trends**: dual-axis chart with key policy and market events (Fit for 55, Russia's invasion of Ukraine, the EU oil embargo and its crude ban, Nord Stream, the EU gas price cap, the ETS reform with ETS 2, ETS extension to shipping, CBAM, the US-Israeli strikes on Iran on 28 February 2026 and the Qatar LNG halt and Hormuz closure on 2 March 2026).
- **Forward Curve**: monthly NYMEX contracts with a daily close within the last 5 trading days, at their latest close. Oil curves are labelled contango or backwardation; TTF is shown as a seasonal curve, with a winter (Oct to Mar) or summer (Apr to Sep) average and the winter-summer spread only when the season has at least 3 monthly contracts. The TTF contracts are NYMEX Dutch TTF; the European benchmark is ICE Endex TTF.
- **Volatility & Correlation**: 30-day rolling volatility (daily) and rolling correlation of returns.
- **Cross-Commodity Correlation Matrix**: TTF, WTI, Brent and carbon, over the full period and the last 30 common trading days, each titled with its start and end date.
- **Value at Risk**: historical VaR 95%, VaR 99% and Expected Shortfall 97.5% on the last 250 trading days, with the full-period figures alongside, and the max daily loss with its date (for Brent, the page says when that date is a roll day and gives the worst day excluding roll days); return distribution, 60-day rolling VaR, and a 250-day backtest of the commodity's own VaR (exceptions, Kupiec, Basel traffic light).
- **GARCH(1,1) Forecast**: Student-t GARCH; the daily volatility forecast for day 5 and day 10 ahead, the 10-day holding-period volatility, and a bootstrap 90% band for each day's forecast.
- **Market Regime**: per-commodity volatility thresholds: Calm, Volatile, Crisis. Past days are labelled with today's thresholds.

**Positions & Limits**
- Enter positions in €m notional (long or short) in TTF, WTI, Brent and carbon.
- **€ VaR 95% / 99% and Expected Shortfall 97.5%** from the last 250 trading days.
- **Limit usage** against a user-set VaR limit (green < 80%, amber 80 to 100%, red = breach), with standalone VaR per position, the book's gross and net notional, and the diversification benefit.
- **VaR backtest** over the last 250 days: exceptions at 95% and 99%, **Kupiec** proportion-of-failures test, and the **Basel traffic light** (0 to 4 green, 5 to 9 yellow, 10 or more red), shown only when the backtest has the full 250 days.
- **Historical scenario replay**: the positions entered above, held unchanged through three past windows (Russia's invasion of Ukraine, the August 2022 gas peak, the first two weeks of the 2026 Iran war). For each window: the cumulative return of each commodity, the € P&L per position and for the book, and the book's worst single day. It replays what happened; it is not a forecast.

## Power page

- **German day-ahead power** (bidding zone DE-LU) from the ENTSO-E Transparency Platform, document A44, kept per hour (per 15 minutes since 1 Oct 2025). ENTSO-E publishes two price series for DE-LU: sequence 1 is the 12:00 Single Day-Ahead Coupling (SDAC) auction, sequence 2 is EXAA's separate 10:15 auction; only sequence 1 is used. If ENTSO-E cannot be reached, or no `ENTSOE_TOKEN` is set, the same prices come from the Energy-Charts API (Fraunhofer ISE, data from Bundesnetzagentur / SMARD); the page states which source each year came from. The two sources were compared price by price (all of 2024, 8,784 hours, and 2026 to date): identical. **Peak** is the time-weighted average over weekdays 08:00 to 20:00 (Berlin time), **baseload** over all 24 hours.
- **Timing.** A delivery day's power is sold in the auction held the day before, so each delivery day is matched with the TTF settlement and the EEX EUA auction of the last trading day **before** it (Friday's prices for Monday). The headline shows the latest delivery day and the dates of the gas and carbon prices used.
- **Clean spark spread** = power − TTF / efficiency − (0.18406 / efficiency) × EUA, at 45%, 50% and 60% efficiency on a higher heating value (HHV) basis, as S&P Global Platts publishes German spreads; 50% is the central case. 0.18406 tCO2 per MWh of gas is Platts' 0.053942 tCO2e/MMBtu (HHV) converted at 3.412142 MMBtu/MWh. TTF is quoted on a gross calorific value basis, so an HHV emission factor is used; the often-quoted 0.202 tCO2/MWh is on a net calorific value basis and would overstate the carbon cost by about 10%. It is a gross margin per MWh, before start-up, other variable and fixed costs.
- Headline: the latest delivery day's peak, baseload and **flexible plant** spreads at 50%, then a table of the peak and baseload spreads at 45%, 50% and 60% and the share of days with a positive peak spread over the last 12 months at each efficiency. The flexible plant runs in any period of a business day (all 24 hours) where the price is above its fuel and carbon cost; start-up costs, minimum run times and part-load operation are ignored, and the page shows how often it would start.
- Charts: the peak and baseload spreads over time (with the share of days on which the peak block averaged below baseload, because solar output lowers midday prices), and their decomposition into power price, fuel cost and carbon cost.
- **What a flexible plant earns**: the 400 MW flexible plant's gross margin per quarter next to the average peak spread of the quarter, and the **gas-weighted price**: the day-ahead price weighted by Germany's actual gas-fired output (Energy-Charts public net generation, fossil gas, from SMARD, CC BY 4.0), compared with the peak and baseload averages over the last 90 days.
- **Spread risk**, as day-on-day changes between consecutive delivery business days (Monday compares with Friday): the 30-day volatility and the 1-in-20 fall (95% historical VaR of the daily change) of the peak spread in €/MWh (not %, because power prices and the spread can be negative), and the 1-in-20 fall in the flexible plant's daily margin in €. These measure how much the margin moves from one day to the next, not a loss on a position: each day's margin is fixed once the auction clears.
- **Hedge next month's gas now, or in a week?** `TTF=F` is matched to the monthly contract it tracks (e.g. November 2026 delivery). A zero-mean GARCH(1,1) with Student-t shocks, fitted on its daily log returns, gives a 90% range for the price in 5 business days; the gas volume is the month's business days × the flexible plant's recent average running hours × 400 MW ÷ 50%. The page shows the cost of the month's gas hedged today, the range if hedged in a week, and the extra cost at the 95th percentile. A spot (day-ahead) gas comparison is not shown, because no freely licensed daily TTF spot series is available.
- **EUA price** from EEX primary-auction results (€/t, spot allowances), carried forward at most two weeks between auctions. Rows whose contract code contains "EAA" or "EUAA" (aviation allowances) are dropped; if two auctions fall on the same day their prices are averaged. If EEX cannot be reached, no spread is shown: no built-in or approximate carbon price is used.
- **Download CSV**: per delivery business day, peak and baseload power, TTF and its settlement date, EUA and its auction date, fuel and carbon cost at 50%, peak and baseload spreads at 45%, 50% and 60%, the flexible plant's hours, starts, average spread and daily margin, the day-on-day changes behind the risk figures, and the gas-weighted price.
- **Not included, because no freely licensed data is available:** coal (clean dark spread, API2 coal has no free source), EEX power futures (the public data is for personal use only) and TTF day-ahead gas.

## Market page

- **Country Exposure**: EU-27 member states, Norway, Switzerland, the UK and Turkey, every year from 2020 for which Eurostat publishes the EU-27 value (2020 to 2024 as of October 2026), all from the Eurostat API with no hand-entered value and no composite score. The question follows the commodity. Gas and oil: "Which countries are most exposed to imported gas (oil)?", ranked by net imports of the fuel as a share of all the country's energy = the fuel's share of gross available energy (nrg_bal_c) × its import dependency (nrg_ind_id), shown next to the raw share, the dependency, total energy import dependency and the renewable share, with the EU-27 value as reference. Carbon: "Which economies emit the most greenhouse gas per unit of output?", ranked by greenhouse-gas emissions per million euro of GDP at 2020 prices, with the renewable share. The page states what is not included (gas storage, supplier concentration, LNG import capacity; for oil, emergency stocks; for carbon, which emissions the EU ETS covers). Includes a bar chart of every ranked country, the Eurostat inputs of one country and their trend over the years, and the tone of recent headlines that name the country or use its adjective.
- **Headline Tone**: FinBERT (ProsusAI/finbert via Hugging Face) on headlines from the last 7 days from BBC Business, OilPrice and Google News (queries limited with `when:7d`), at most 4 per feed, duplicates across feeds removed, and the " - Publisher" suffix Google News appends removed before scoring. FinVADER is the fallback, with the reason FinBERT was not used shown on the page. If no feed returns a relevant headline, no tone is computed (the page never scores placeholder text). Every feed keeps only headlines containing one of the selected commodity's keywords (whole words), and drops US-domestic stories with no link to Europe (US LNG export news is kept). Both models are trained on company news (FinBERT on Financial PhraseBank), so the score is the tone of the wording, not a price direction. FinBERT's weights are licensed CC BY-NC-SA 4.0 (non-commercial).
- **Headline Tone, Last 30 Days**: Google News headlines of the last 30 days, de-duplicated and scored with FinVADER; the 30-day average is over all headlines, and the number of headlines per day is shown under the daily bars.
- **Statistical Checks**: each check is calibrated on the commodity's own last 3 years (the same 750 trading days as the regime thresholds): 30-day volatility of daily returns above its 95th percentile (spike) or below its 5th (unusual calm); GARCH 10-day forecast more than 30% above or below the 3-year average 30-day volatility; 30-day correlation with the comparison asset outside its 5th to 95th percentile range; daily losses in the last 252 trading days beyond 2× the 3-year 99% VaR (on a continuous front-month future a roll day can trigger it). A table shows every check, its threshold, the latest value and on how many of the last 252 trading days it fired, so the base rate of each flag is visible.
- **AI Risk Interpretation** (optional): the statistics on this page summarised in three sentences by Claude via the Anthropic API. It sees only those statistics, not fundamentals such as storage, flows or the forward curve.
- **Data Export**: CSV downloads of commodity risk data, headline tone, and the country table with the raw Eurostat inputs and the EU-27 reference row.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Market data | yfinance (TTF=F, CL=F, BZ=F, CARB.L → KRBN → ICLN) |
| Power data | ENTSO-E Transparency Platform (DE-LU day-ahead, SDAC); Energy-Charts API as fallback and for gas-fired generation |
| EUA prices | EEX primary-auction reports |
| Framework | Streamlit (multi-page with `st.navigation`) |
| Risk analytics | NumPy, Pandas, SciPy |
| Volatility modelling | arch (GARCH) |
| NLP | FinBERT (HuggingFace Inference API; weights CC BY-NC-SA 4.0), FinVADER fallback |
| News feeds | feedparser + requests (BBC, OilPrice, Google News RSS) |
| Country data | Eurostat API (nrg_ind_id, nrg_bal_c, nrg_ind_ren, env_air_gge, nama_10_gdp at 2020 prices); no hand-entered values |
| Visualisation | Matplotlib |

## Methodology Notes

**Carbon benchmark.** KEUA, the original EUA ETF, was liquidated in March 2026. The app now tries `CARB.L` (WisdomTree Carbon ETC, USD line on the LSE), then `KRBN`, then `ICLN`, and labels the charts from whichever resolves. `CARB.L` tracks ICE EUA futures, but its quote is an ETC share price in USD, not €/tCO2, so its returns also carry EUR/USD moves, roll yield and fees. The Power page uses EEX auction prices in €/t instead.

**Volatility.** The 30-day volatility is the equal-weighted standard deviation of simple daily returns over 30 trading days (at least 20), shown as a daily figure; multiply by √252 for an annualised figure (the page gives it in the tooltips). This is a simplification: RiskMetrics uses an exponentially weighted average (λ = 0.94 for daily data) of log returns. The average volatility on the page covers the selected date range, while the regime thresholds use the last 3 years, so the two reference periods differ.

**VaR.** Historical simulation, 1-day horizon. The 95% VaR is the 5th percentile of daily returns, the loss exceeded on 5% of days (about 1 day in 20), and the 99% VaR the loss exceeded on 1% of days; neither is the worst case. Percentiles use linear interpolation between the two nearest days (numpy's default). Expected Shortfall 97.5% is the average of the returns at or below the 2.5% percentile: with 250 days that is the worst 7 days (about 2.8% of the sample, since the interpolated 2.5% point falls between the 7th and 8th worst day). The headline figures use the last 250 trading days; the full-period figures (which include the 2022 crisis on the default range) are shown alongside. On the Risk page's position block, VaR and ES come from the last 250 days of the book's € P&L. Prices run up to and including the end date chosen in the sidebar; Saturday and Sunday rows (Yahoo's Sunday evening session) are dropped, and on a trading day the last value is the latest price, not yet a settlement.

**Roll jumps.** `TTF=F`, `CL=F` and `BZ=F` are Yahoo Finance's continuous front-month futures. On the day the front contract rolls, the series switches to the next contract and that day's return is not a real price move. Example: ICE Brent stops trading on the last business day of the second month before delivery, so the November 2026 contract expired on 30 September 2026. `BZ=F` closed at 103.53 dollars a barrel that day (November) while the December contract closed at 98.03; on 1 October the series followed December, which closed at 102.31. The series showed a fall of 1.18% that day while the December contract rose 4.37%: the 5.50 dollar gap between the two contracts turned a rise into a fall. Roll days are not removed, so they can add to the tails, the max daily loss, VaR and backtest exceptions; the Risk page says so. For Brent the page identifies roll days from the ICE rule (the first trading day of each month, and the last trading day of December, because ICE moves the December expiry one business day earlier before New Year's Day) and says when the max daily loss falls on one. For TTF and WTI it does not try to detect roll days: Yahoo's switch date for those series does not always match the exchange expiry. A desk would use a back-adjusted series built from individual contracts.

**VaR backtest.** The Risk page backtests the selected commodity's own VaR per unit of exposure. The position block's backtest is hypothetical: today's positions are applied to past returns, and each day's VaR is estimated only from the 250 days before it (no look-ahead). Kupiec's test also rejects a model with too few exceptions (over-conservative). The Basel traffic light (green 0 to 4, yellow 5 to 9, red 10 or more exceptions at 99%, BCBS 1996, https://www.bis.org/publ/bcbs22.pdf) is defined for 250 days; with a shorter backtest the page shows the exceptions and the Kupiec test but no traffic light, rather than scaling the thresholds.

**GARCH(1,1).** Fitted with Student-t shocks, since daily energy returns have fat tails. α captures reaction to shocks, β persistence; α + β near 1 means volatility shocks fade slowly. The page shows the daily volatility forecast for day 5 and day 10 ahead (the volatility of a single day's return that far out, not of a 5-day or 10-day period) and the 10-day holding-period volatility, the square root of the sum of the 10 daily forecast variances. The forecast reverts to the long-run volatility √(ω / (1 − α − β)), but for these commodities α + β sits at 0.97 to 0.99, where the formula divides by almost zero and swings from day to day. The reference level is therefore always the sample average 30-day volatility; the GARCH long-run value is shown for information only when α + β < 0.98 and it is within 1.5× the sample average. On the Market page the GARCH check compares the 10-day forecast with the 3-year average 30-day volatility (the regime window), not the sample average.

**Stale prices.** In a run of 5 or more identical closes (a suspended or illiquid quote), the first day is kept and the repeats after it are treated as missing, and no return is computed across the gap. The Risk page lists the excluded periods.

**Historical scenarios.** The stress test replays three past windows with the positions entered in Positions & Limits; it shows what happened, not a forecast or a probability. Each position is entered at the close before the window and held unchanged (no rebalancing, hedging or stop-loss), so its P&L is the position × the instrument's cumulative return, and the book's worst day is the largest one-day change in the value of the positions inside the window. The returns are the page's own daily returns (Yahoo continuous front-month futures for TTF, Brent and WTI, roll days included; for carbon the `CARB.L` ETC in USD, so the carbon return includes the EUR/USD move). An instrument with returns on fewer than 70% of a window's trading days is reported as no data. The windows are fixed dates, independent of the sidebar range:
- **Russia invades Ukraine**: close of 23 February 2022 to close of 8 March 2022. The invasion began on 24 February (European Council conclusions of 24 February 2022); 8 March is the day the US announced a ban on Russian oil imports and Brent's highest close of the episode on Yahoo (127.98 dollars); TTF's highest close was on 7 March.
- **August 2022 gas peak**: close of 29 July 2022 to close of 26 August 2022, the record TTF front-month close (about 339 €/MWh) after Gazprom announced a three-day Nord Stream 1 shutdown (Quantum Commodity Intelligence, 26 August 2022, https://qcintel.com/article/winter-ttf-gas-trades-at-record-352-mwh-goldman-says-market-overshoots-8125.html).
- **2026 Iran war, Qatar LNG halt and Hormuz**: close of 27 February 2026 to close of 13 March 2026, the first two trading weeks after the US-Israeli strikes on Iran on Saturday 28 February. QatarEnergy halted LNG production on 2 March and an IRGC official said that day that the Strait of Hormuz was closed (CRS report R45281, 11 March 2026; QatarEnergy statement reported by Marhaba, 4 March 2026). TTF rose 39% on 2 March and fell 16% on 10 March (Trading Economics, 10 March 2026: "declined more than 15% to below €50/MWh").
Each window contains a Brent roll day (1 March 2022, 1 August 2022, 2 March 2026), when part of the Brent move is the gap between contracts. With the default book (TTF +€10m, Brent −€5m, carbon +€3m) the book P&L is +€11.61m, +€8.56m and +€3.44m, and the worst days are −€4.37m (25 February 2022), −€0.66m (23 August 2022) and −€1.96m (10 March 2026); an independent recalculation from raw Yahoo prices gives the same figures.

**Regime detection.** Per-commodity thresholds only: Volatile starts at the 50th and Crisis at the 90th percentile of the commodity's own 30-day volatility over the last 3 years, 750 trading days (the selected range if that history is too short). A longer history would include the 2021-22 crisis and set the "normal" line too high. A lower volatility never gets a higher regime. The Risk Signal uses the same classification. The Risk page caption shows the thresholds used. The regime chart and statistics label every past day with today's thresholds, so labels for earlier years are applied with hindsight, not what a desk would have seen at the time.

**Event lines.** Each date on the Risk page's price chart was checked against a primary or contemporaneous source: Fit for 55 presented 14 July 2021 (EERA); Russia's invasion 24 February 2022 (European Council conclusions); EU oil embargo (sixth sanctions package) adopted 3 June 2022, with crude by sea banned from 5 December 2022 after the wind-down and pipeline crude exempt (Herbert Smith Freehills Kramer sanctions tracker); Nord Stream explosions 26 September 2022, "serious sabotage" (CNBC, 18 November 2022); EU gas market correction mechanism in force 1 February 2023 and able to apply from 15 February 2023 (ACER, 1 February 2023); European Parliament adopts the ETS reform including ETS 2 on 18 April 2023 (EP press release; the Council adopted it on 25 April); EU ETS covers shipping since January 2024 (European Commission); CBAM definitive regime from 1 January 2026 (European Commission, DG TAXUD); US-Israeli strikes on Iran 28 February 2026 and the IRGC statement that Hormuz was closed on 2 March 2026 (CRS R45281); QatarEnergy LNG halt 2 March 2026 (Marhaba, 4 March 2026; Business Today, 2 March 2026); ETS 2 start postponed to 1 January 2028 (ICAP; formally adopted by the Council on 5 March 2026; outside the data range, so not drawn).

**Clean spark spread caveats.** Power is the day-ahead price, while gas is the TTF front-month future, so the tenors don't match; S&P Global Platts computes its German day-ahead spreads from its own TTF day-ahead assessments, which are not freely available. Efficiencies are on an HHV basis (50% HHV is roughly 55% on a lower heating value basis). Fuel cost excludes gas transport and other variable costs, as in the Platts spreads. The peak block includes public holidays, as exchange peak products do. `TTF=F` is a continuous front-month series that jumps at each monthly roll and follows the NYMEX (US) trading calendar.

**Country exposure.** No composite score and no weights. Gas and oil: net imports of the fuel as a share of all gross available energy = (gross available energy of the fuel ÷ gross available energy of all products, Eurostat nrg_bal_c, nrg_bal GAE, siec G3000 or O4000XBIO and TOTAL, ktoe) × import dependency of the fuel (nrg_ind_id, net imports ÷ gross available energy of the fuel). A country that imports all its gas but uses little gas (Sweden) therefore ranks low. Dependency is shown as published: below 0% means a net exporter (Norway), counted as 0 in the product; above 100% means imports also refilled stocks. The EU-27 values match Eurostat's "Energy statistics, an overview" (2024: natural gas 20.6% of gross available energy, 85.0% imported; oil 37.8%, 96.6% imported). Gross available energy includes international maritime bunkers, which raises the oil share of bunkering hubs such as Malta, the Netherlands and Belgium. Carbon: total greenhouse-gas emissions excluding LULUCF (env_air_gge, TOTX4_MEMO) ÷ GDP in chain-linked volumes (nama_10_gdp, CLV20_MEUR), in tCO2e per million euro at 2020 prices, so a fall over time is not an effect of inflation; levels across countries use 2020 exchange rates, not purchasing power. Renewable share: nrg_ind_ren (REN). Not included: gas storage, supplier concentration, LNG import capacity, emergency oil stocks, and which emissions the EU ETS covers. Countries without a value for an input of the measure in the selected year are listed as not ranked. Data are cached for a day; the page shows the date of each dataset's last Eurostat update.

## Project Structure

```
app.py            entry point: shared sidebar, navigation (Streamlit Cloud runs this)
common.py         carbon benchmark, commodities, data loading, cached risk calculations
country_data.py   Eurostat loader for country data (no hand-entered values)
power_data.py     ENTSO-E / Energy-Charts power prices, gas-fired generation, EEX EUA auction prices
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

Without keys the app still runs: sentiment falls back to FinVADER, the AI summary is hidden and power prices come from Energy-Charts.

- **FinBERT**: create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a fine-grained token needs the "Make calls to Inference Providers" permission) and set `HF_TOKEN`. If FinBERT still is not used, the Market page shows a short reason (for example token missing or invalid, no Inference permission, rate limit).
- **AI Risk Interpretation**: set `ANTHROPIC_API_KEY`.
- **ENTSO-E power prices**: request a security token in your ENTSO-E Transparency Platform account (My Account, Web API Access) and set `ENTSOE_TOKEN`. Without it the Power page uses Energy-Charts, which carries the same day-ahead prices.

On Streamlit Cloud, add them under Settings → Secrets (never put tokens in code); locally, set them as environment variables.

## Project Context

Built as a FinTech portfolio project during my Master's in International Economics at the Geneva Graduate Institute (IHEID).

## License

MIT
