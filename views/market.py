"""Market page: country exposure, news sentiment, anomaly checks and an AI summary.

Designed with data and analytics firms in mind.
"""
import os
import re
import time
from urllib.parse import quote_plus

import feedparser
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests as req
import streamlit as st

import common
from country_data import (COUNTRIES, YEARS, load_country_data, dependency_key,
                          mentions_country)

nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402

ctx = common.context()
selected_commodity = ctx.selected_commodity
commodity = ctx.commodity
compare_label = ctx.compare_label
start_date, end_date = ctx.start_date, ctx.end_date
CARBON_TICKER = ctx.carbon["ticker"]

with st.sidebar.expander("Methodology — Market page"):
    st.markdown("""
    - **Country Risk Scoring**: Illustrative composite index of import dependency,
      greenhouse-gas intensity of GDP and renewable share for EU-27, NO, CH, UK and TR.
      All inputs come from the Eurostat API (nrg_ind_id, nrg_ind_ren, env_air_gge,
      nama_10_gdp); a country missing any input for the year is listed as not scored.
      The structural score is multiplied by a volatility factor: the commodity's 30-day
      volatility divided by its average over the selected period (capped between 0.5 and 3),
      weighted by each country's dependency. Dependency therefore enters twice on purpose:
      as a structural weakness and as exposure to current volatility.
    - **FinBERT**: Main sentiment model. Transformer fine-tuned on financial
      text (ProsusAI/finbert via HuggingFace). Applied to live headlines.
    - **FinVADER**: Fallback model. VADER enhanced with SentiBigNomics + Henry
      financial lexicons — more accurate than standard VADER for financial text.
    - **30-Day Sentiment Trend**: Daily average sentiment via Google News RSS,
      scored with FinVADER. Bar chart with a 30-day average line.
    - **Anomaly Detection**: 5 automated checks: volatility z-score (spike, or unusual calm
      below −1.5σ), GARCH 10-day forecast vs the sample average volatility, correlation shift,
      sentiment-volatility divergence, and tail events (loss beyond 2× VaR99 in the past 252
      days, otherwise beyond 3× VaR99 in the whole period).
    - **AI Risk Interpretation**: Quantitative signals (vol, VaR, GARCH,
      regime, sentiment, anomalies) fed to Claude Sonnet via Anthropic API.
      Generates a 3-sentence risk assessment. Refreshes every 30 min.
    """)

st.title("Market Intelligence")
st.markdown(f"Country exposure, news sentiment and signal monitoring for **{selected_commodity}**")

core = common.compute_core(commodity['ticker'], ctx.compare_ticker, start_date, end_date)
if core is None:
    st.warning("Please select a longer time range (at least 30 days of data required).")
    st.stop()

df_analysis = core['df_analysis']
latest_vol, avg_vol = core['latest_vol'], core['avg_vol']
returns_clean = core['returns_clean']
var_95 = core['var_95']              # last 250 trading days
var_99_full = core['var_99_full']    # full period: the tail check looks back further
risk_level = core['risk_level']
dep_key, dep_label = dependency_key(selected_commodity)
dep_col = {'gas': 'Gas Dep. (%)', 'oil': 'Oil Dep. (%)', 'total': 'Total Energy Dep. (%)'}[dep_key]
country_series, country_complete, country_failed = load_country_data()

garch = common.fit_garch(returns_clean)
garch_forecast_10d = garch['forecast_10d'] if garch is not None else None
garch_long_run = garch['long_run_vol'] if garch is not None else None
# Reference level: the sample average 30-day volatility (see common.fit_garch for why the
# GARCH long-run formula is not used).
garch_ref_name = "sample average 30-day volatility"
regime_thr, current_regime = core['regime_thr'], core['current_regime']
features = common.compute_regimes(df_analysis[['Volatility', 'Rolling Correlation']], regime_thr)

# ─── FinBERT via HuggingFace Inference API ───
# Used by both the country sentiment (Section 5b) and the main NLP section (Section 6).
@st.cache_data(ttl=600, show_spinner="Scoring headlines with FinBERT...")
def finbert_analyze(texts):
    """Score headlines with ProsusAI/finbert on the Hugging Face router.

    Returns (scores, labels, ok, reason). When ok is False, reason says why FinBERT was
    not used (missing token, HTTP status and error text, or no response), so the page
    can show it instead of failing silently. Results are cached for 10 minutes.
    """
    API_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        pass
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
        # The router always answers 401 without a token, so don't call it.
        return None, None, False, "no HF_TOKEN set in Secrets"
    headers = {"Authorization": f"Bearer {hf_token}"}

    def parse_results(results, n_texts):
        scores, labels = [], []
        if not isinstance(results, list):
            return None, None
        # The router returns one of three shapes: a list per headline ([[{..}x3], ...]),
        # one dict per headline ([{..}, ...]), or, for a batch, a single outer list that
        # holds the top label of every headline ([[{top1}, {top1}, ...]]). Unwrap the last.
        if (n_texts > 1 and len(results) == 1 and isinstance(results[0], list)
                and len(results[0]) == n_texts
                and all(isinstance(x, dict) for x in results[0])):
            results = results[0]
        for item in results:
            if isinstance(item, list):
                best = max(item, key=lambda x: x['score'])
            elif isinstance(item, dict) and 'label' in item:
                best = item
            else:
                scores.append(0.0); labels.append('Neutral')
                continue
            lbl = best['label'].lower()
            sc = best['score']
            if lbl == 'negative':
                scores.append(-sc); labels.append('Negative')
            elif lbl == 'positive':
                scores.append(sc); labels.append('Positive')
            else:
                scores.append(0.0); labels.append('Neutral')
        if len(scores) == n_texts:
            return scores, labels
        return None, None

    def error_text(response):
        """One short line about the failure; never the raw HTML of an error page."""
        known = {401: "token missing or invalid", 403: "token lacks the Inference permission",
                 404: "model not served at this endpoint", 410: "model no longer served",
                 429: "rate limited", 503: "model loading"}
        msg = ""
        try:
            body = response.json()
            if isinstance(body, dict) and isinstance(body.get('error'), str):
                msg = body['error']
        except Exception:
            pass
        if not msg or '<' in msg:
            msg = known.get(response.status_code, response.reason or "request failed")
        return " ".join(msg.split())[:120]

    # 503 = model loading, 429 / 5xx = transient: retry. 401 / 403 / 404 / 410 will not
    # change on retry, so stop at once and report them.
    reason = "no attempt made"
    for attempt in range(3):
        try:
            response = req.post(API_URL, headers=headers, json={"inputs": list(texts)}, timeout=60)
        except Exception as e:
            reason = f"no response from {API_URL} ({type(e).__name__})"
            if attempt < 2:
                time.sleep(10)
            continue
        if response.status_code == 200:
            sc, lb = parse_results(response.json(), len(texts))
            if sc is not None:
                return sc, lb, True, ""
            shape = type(response.json()).__name__
            return None, None, False, f"unexpected response format ({shape}, {len(texts)} headlines)"
        reason = f"HTTP {response.status_code}: {error_text(response)}"
        if response.status_code == 503:
            try:
                wait_time = float(response.json().get('estimated_time', 20))
            except Exception:
                wait_time = 20
            time.sleep(min(wait_time + 5, 45))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(5)
            continue
        break
    return None, None, False, reason


def finvader_score(text):
    """FinVADER fallback — VADER + SentiBigNomics + Henry financial lexicons."""
    try:
        from finvader import finvader
        return float(finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound'))
    except Exception:
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)['compound']


# ─── Headline relevance ───
def _word_pattern(words, flags=re.IGNORECASE):
    """Whole-word match for any of `words`, allowing a plural ending ("emission" → "emissions").
    Substring matching would let "ETS" match "markets" and "oil" match "turmoil"."""
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z])(?:{alt})(?:s|es)?(?![A-Za-z])", flags)


# A headline about the United States with no link to Europe (or to the global market) is
# dropped: US federal/state policy, pump prices, US production and inventory data.
_US = [_word_pattern(["US", "U.S.", "USA"], flags=0),
       _word_pattern(["America", "American", "Trump", "Biden", "White House", "Congress", "Senate",
                      "Republican", "Democrat", "EPA", "Interior Department", "Energy Department",
                      "Energy Secretary", "federal", "EIA", "Strategic Petroleum Reserve", "shale",
                      "Permian", "gasoline", "at the pump", "governor", "Texas", "California", "Alaska",
                      "Louisiana", "New York", "Pennsylvania", "North Dakota", "New Mexico", "Oklahoma",
                      "Colorado", "Gulf of Mexico", "Wall Street"])]
_EUROPE_OR_GLOBAL = _word_pattern([
    "Europe", "European", "EU", "Brussels", "eurozone", "euro area", "UK", "Britain", "British",
    "Germany", "German", "France", "French", "Italy", "Italian", "Spain", "Spanish", "Netherlands",
    "Dutch", "Norway", "Norwegian", "Poland", "Polish", "Austria", "Belgium", "Russia", "Russian",
    "Ukraine", "TTF", "ETS", "Nord Stream", "North Sea", "Brent", "OPEC", "Hormuz", "Suez",
    "Red Sea", "global", "world"])
# US LNG exports are Europe's largest source of LNG, so they are kept even without "Europe".
_LNG_EXPORT = (_word_pattern(["LNG"]),
               _word_pattern(["export", "exporter", "cargo", "cargoe", "shipment", "terminal"]))


def headline_text(entry_title, source=""):
    """The headline without the " - Publisher" suffix Google News appends."""
    if source and entry_title.endswith(f" - {source}"):
        return entry_title[:-len(source) - 3]
    return entry_title


def is_us_domestic(headline):
    if not any(p.search(headline) for p in _US):
        return False
    if _EUROPE_OR_GLOBAL.search(headline):
        return False
    return not all(p.search(headline) for p in _LNG_EXPORT)


def is_relevant(headline, keyword_re):
    """Keep a headline only if it has one of the commodity keywords and is not US-domestic."""
    return bool(keyword_re.search(headline)) and not is_us_domestic(headline)


commodity_kw_re = _word_pattern(commodity['keywords'])


# ─── Section 5b: European Country Energy Risk ───
st.subheader("European Country Energy Risk Exposure")
st.write("Which European countries are most vulnerable to energy price shocks?")
st.caption("All inputs from the Eurostat API, cached for a day: import dependency (nrg_ind_id), renewable "
           "share (nrg_ind_ren) and greenhouse-gas intensity of GDP (env_air_gge ÷ nama_10_gdp). No value "
           "is typed in by hand. Dependency is clipped to 0-100%: net exporters such as Norway show 0%. "
           "Illustrative index, not an official risk rating.")

selected_year = st.slider("Select Year", min_value=YEARS[0], max_value=YEARS[-1], value=YEARS[-1], step=1)

_scored = [c for c, ok in zip(COUNTRIES, country_complete[selected_year]) if ok]
_unscored = [c for c, ok in zip(COUNTRIES, country_complete[selected_year]) if not ok]
_idx = [COUNTRIES.index(c) for c in _scored]
cr_df = pd.DataFrame({
    'Country': _scored,
    'Gas Dep. (%)': [country_series['gas'][selected_year][i] for i in _idx],
    'Oil Dep. (%)': [country_series['oil'][selected_year][i] for i in _idx],
    'Total Energy Dep. (%)': [country_series['total'][selected_year][i] for i in _idx],
    'Renewable (%)': [country_series['ren'][selected_year][i] for i in _idx],
    'GHG Int. (tCO2e/M€)': [country_series['carbon'][selected_year][i] for i in _idx],
}, columns=['Country', 'Gas Dep. (%)', 'Oil Dep. (%)', 'Total Energy Dep. (%)', 'Renewable (%)',
            'GHG Int. (tCO2e/M€)'])
if _unscored:
    st.caption(f"Not scored for {selected_year} (Eurostat has no value for at least one input): "
               + ", ".join(_unscored) + ".")
if country_failed:
    st.warning("Eurostat could not be reached for: " + ", ".join(country_failed)
               + ". Countries without those inputs are not scored.")

# Eurostat import dependency = net imports / gross available energy. A net exporter has a
# negative value (Norway's gas was about -2600% in 2024), and stock changes can push an
# importer slightly above 100%. Both are clipped to 0-100 for display and scoring.
for _c in ('Gas Dep. (%)', 'Oil Dep. (%)', 'Total Energy Dep. (%)'):
    cr_df[_c] = cr_df[_c].astype(float).clip(lower=0, upper=100)
if cr_df.empty:
    st.warning("No country has complete Eurostat data for this year, so the country index is not shown.")
else:
    cr_df['Dep Clipped'] = cr_df[dep_col]
    cr_df['Total Clipped'] = cr_df['Total Energy Dep. (%)']

    # Structural Score: each variable enters once (no level + rank of the same variable),
    # weights sum to 100%. No subjective "price sensitivity" score: it had no source.
    # The dynamic score below multiplies it by a volatility factor weighted by dependency, so in the
    # final score dependency counts twice on purpose (structural weakness + exposure to volatility).
    if commodity['ticker'] == CARBON_TICKER:
        # Carbon mode: the relevant dependency IS total energy dependency, so it appears once.
        SCORE_WEIGHTS = [(f"{dep_label}", 0.40), ("GHG intensity rank", 0.30),
                         ("Inverse Renewable share", 0.30)]
        cr_df['Structural Score'] = (
            cr_df['Dep Clipped'] * 0.40 +
            cr_df['GHG Int. (tCO2e/M€)'].rank(pct=True) * 100 * 0.30 +
            (100 - cr_df['Renewable (%)']) * 0.30
        ).round(1)
    else:
        SCORE_WEIGHTS = [(f"{dep_label}", 0.35), ("Total Energy Dependency", 0.25),
                         ("GHG intensity rank", 0.20), ("Inverse Renewable share", 0.20)]
        cr_df['Structural Score'] = (
            cr_df['Dep Clipped'] * 0.35 +
            cr_df['Total Clipped'] * 0.25 +
            cr_df['GHG Int. (tCO2e/M€)'].rank(pct=True) * 100 * 0.20 +
            (100 - cr_df['Renewable (%)']) * 0.20
        ).round(1)

    vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
    vol_ratio_clamped = min(max(vol_ratio, 0.5), 3.0)

    cr_df['Country Vol Multiplier'] = (
        0.5 + 0.5 * vol_ratio_clamped * (cr_df['Dep Clipped'] / 100)
    ).round(2)

    cr_df['Risk Score'] = (cr_df['Structural Score'] * cr_df['Country Vol Multiplier']).round(1)
    cr_df = cr_df.sort_values('Risk Score', ascending=False)

    # One set of thresholds for the ranking table, the detail panel and the bar chart.
    RISK_HIGH, RISK_MEDIUM = 70, 50
    RISK_LEVEL_COLORS = {'🔴 High': 'red', '🟡 Medium': 'orange', '🟢 Low': 'green'}


    def risk_category(score):
        if score > RISK_HIGH:
            return '🔴 High'
        elif score > RISK_MEDIUM:
            return '🟡 Medium'
        else:
            return '🟢 Low'

    cr_df['Risk Level'] = cr_df['Risk Score'].apply(risk_category)

    st.markdown(f"**Volatility factor:** Current {selected_commodity} 30-day volatility is **{latest_vol:.1f}%** "
                f"vs its average over the selected period **{avg_vol:.1f}%** → ratio = **{vol_ratio_clamped:.2f}x** "
                f"(capped between 0.5 and 3)")
    st.caption("Each country's multiplier is weighted by its own dependency: high-dependency countries "
               "feel the same market volatility more. Dependency is therefore counted twice in the dynamic "
               "score on purpose, once as a structural weakness and once as exposure to current volatility.")

    cr_col1, cr_col2 = st.columns([2, 1])

    with cr_col1:
        st.write(f"**Risk Ranking ({selected_year}) — by {dep_label}:**")
        display_cols = ['Country', 'Risk Score', 'Country Vol Multiplier', 'Risk Level', dep_col,
                        'Total Energy Dep. (%)', 'GHG Int. (tCO2e/M€)',
                        'Renewable (%)']
        seen = set()
        display_cols = [c for c in display_cols if not (c in seen or seen.add(c))]
        display_df = cr_df[display_cols].reset_index(drop=True)
        display_df.index = display_df.index + 1
        # Tall enough to show every country without scrolling (35 px per row plus the header).
        st.dataframe(display_df, width="stretch", height=(len(display_df) + 1) * 35 + 3)

    with cr_col2:
        selected_country = st.selectbox("Select Country for Detail", cr_df['Country'].tolist())
        country_data = cr_df[cr_df['Country'] == selected_country].iloc[0]

        country_dep_val = country_data[dep_col] / 100 if country_data[dep_col] > 0 else 0
        country_exposure_idx = latest_vol * country_dep_val

        # Same score and thresholds as the Risk Level column of the ranking table.
        c_risk_level = country_data['Risk Level']
        c_risk_color = RISK_LEVEL_COLORS[c_risk_level]

        st.markdown(f"### {selected_country} ({selected_year})")
        st.markdown(f"<h3 style='color:{c_risk_color}; margin-top:0'>{c_risk_level.upper()} RISK</h3>",
                    unsafe_allow_html=True)
        st.metric("Dynamic Risk Score", f"{country_data['Risk Score']:.1f}")
        st.caption(f"High above {RISK_HIGH}, Medium above {RISK_MEDIUM}, the same thresholds as the ranking table.")

        # A two-column table wraps inside the narrow panel, where side-by-side metrics overlap.
        detail_rows = [
            ("Structural Score", f"{country_data['Structural Score']:.1f}"),
            ("Vol Multiplier (live)", f"{country_data['Country Vol Multiplier']:.2f}x"),
            (dep_label, f"{country_data[dep_col]:.0f}%"),
            ("Total Energy Dep.", f"{country_data['Total Energy Dep. (%)']:.0f}%"),
            ("GHG intensity", f"{country_data['GHG Int. (tCO2e/M€)']:.0f} tCO2e/M€"),
            ("Renewable Share", f"{country_data['Renewable (%)']:.0f}%"),
            ("Exposure-weighted volatility index (illustrative)", f"{country_exposure_idx:.2f}"),
        ]
        st.markdown("| Indicator | Value |\n|---|---:|\n"
                    + "\n".join(f"| {k} | {v} |" for k, v in detail_rows))

    # Bar chart
    fig_cr, ax_cr = plt.subplots(figsize=(14, 6))
    top_n = cr_df.head(20)
    bar_colors_cr = [RISK_LEVEL_COLORS[risk_category(s)] for s in top_n['Risk Score']]
    ax_cr.barh(range(len(top_n)), top_n['Risk Score'], color=bar_colors_cr, height=0.6)
    ax_cr.set_yticks(range(len(top_n)))
    ax_cr.set_yticklabels(top_n['Country'], fontsize=9)
    ax_cr.set_xlabel('Composite Energy Risk Score')
    ax_cr.set_title(f'European Countries — Energy Risk Ranking ({selected_year}, by {dep_label})')
    ax_cr.axvline(x=RISK_HIGH, color='red', linewidth=1, linestyle='--', alpha=0.4, label='High risk')
    ax_cr.axvline(x=RISK_MEDIUM, color='orange', linewidth=1, linestyle='--', alpha=0.4, label='Medium risk')
    ax_cr.legend(fontsize=8)
    ax_cr.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig_cr)

    # Year-over-year trend for selected country
    st.write(f"**{selected_country} — Risk Trend 2020–2024:**")
    trend_data = []
    for yr in YEARS:
        idx = COUNTRIES.index(selected_country)
        _d = country_series[dep_key][yr][idx]
        dep_val = None if _d is None else min(max(_d, 0), 100)
        trend_data.append({
            'Year': yr,
            dep_label + ' (%)': dep_val,
            'Renewable (%)': country_series['ren'][yr][idx],
            'GHG Intensity': country_series['carbon'][yr][idx],
        })
    trend_cr = pd.DataFrame(trend_data).astype(float)

    fig_tcr, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(14, 3.5))
    ax_t1.plot(trend_cr['Year'], trend_cr[dep_label + ' (%)'], 'o-', color='red', label=dep_label)
    ax_t1.plot(trend_cr['Year'], trend_cr['Renewable (%)'], 's-', color='green', label='Renewable Share')
    ax_t1.set_ylabel('Percentage (%)')
    ax_t1.set_title(f'{selected_country} — Dependency vs Renewables')
    ax_t1.legend(fontsize=8)
    ax_t1.set_xticks([2020, 2021, 2022, 2023, 2024])

    ax_t2.bar(trend_cr['Year'], trend_cr['GHG Intensity'], color='gray', alpha=0.7)
    ax_t2.set_ylabel('tCO2e per M€ GDP')
    ax_t2.set_title(f'{selected_country} — Greenhouse-Gas Intensity of GDP (Eurostat)')
    ax_t2.set_xticks([2020, 2021, 2022, 2023, 2024])

    plt.tight_layout()
    st.pyplot(fig_tcr)

    # Per-country exposure-weighted volatility index (illustrative, not a VaR or a volatility forecast)
    st.write(f"**{selected_country} — Exposure-Weighted Volatility Index (illustrative):**")
    country_dep_pct = country_data[dep_col] / 100 if country_data[dep_col] > 0 else 0
    country_vol = df_analysis['Volatility'].dropna() * country_dep_pct

    fig_cvol, ax_cvol = plt.subplots(figsize=(14, 3.5))
    ax_cvol.plot(df_analysis['Volatility'].dropna().index, df_analysis['Volatility'].dropna(),
                 color='gray', linewidth=0.8, alpha=0.4, label=f'{selected_commodity} raw volatility')
    ax_cvol.plot(country_vol.index, country_vol,
                 color='red', linewidth=1.5, label=f'{selected_country} index ({country_data[dep_col]:.0f}% dep.)')
    ax_cvol.set_ylabel('Index (volatility % × dependency share)')
    ax_cvol.set_title(f'{selected_country} — Exposure-Weighted Volatility Index (illustrative)')
    ax_cvol.legend(fontsize=8)
    ax_cvol.fill_between(country_vol.index, country_vol, 0, alpha=0.1, color='red')
    plt.tight_layout()
    st.pyplot(fig_cvol)
    st.caption(f"Index = {selected_commodity} 30-day rolling volatility × {selected_country}'s "
               f"{dep_label.lower()} ({country_data[dep_col]:.0f}%). Current: {country_vol.iloc[-1]:.2f}. "
               "An illustrative exposure indicator: import dependency does not change the price "
               "volatility itself, so this is not a VaR or a volatility estimate for the country.")

    # Per-country news sentiment
    # FIX: now uses FinBERT → FinVADER → VADER fallback chain (consistent with main sentiment section)
    st.write(f"**{selected_country} — Current Energy News Sentiment:**")
    _country_q = quote_plus(f"{selected_country} energy {commodity['news_term']}")
    country_rss_url = f"https://news.google.com/rss/search?q={_country_q}+when:7d&hl=en"
    country_headlines = []
    try:
        country_feed = feedparser.parse(country_rss_url)
        _energy_re = _word_pattern(commodity['keywords'] + ['energy', 'oil', 'gas', 'carbon', 'power',
                                                            'fuel', 'electricity', 'pipeline', 'LNG',
                                                            'emission', 'climate', 'price', 'supply',
                                                            'tanker', 'refinery', 'fossil', 'renewable',
                                                            'heating', 'Hormuz', 'sanction', 'ETS'])
        for entry in country_feed.entries[:50]:
            title = entry.title
            # Google News appends " - Publisher"; match on the headline only, so an outlet
            # name such as "Irish Times" does not count as a mention of the country.
            headline = headline_text(title, entry.get('source', {}).get('title', ''))
            energy_match = bool(_energy_re.search(headline))
            if energy_match and mentions_country(headline, selected_country):
                country_headlines.append(title)
            if len(country_headlines) >= 5:
                break
    except Exception:
        pass

    if country_headlines:
        # FinBERT → FinVADER → VADER
        try:
            c_scores_raw, c_labels_raw, c_ok, _ = finbert_analyze(tuple(country_headlines[:5]))
            if c_ok:
                country_scores = c_scores_raw
                c_model = "FinBERT"
            else:
                raise Exception("FinBERT unavailable")
        except Exception:
            try:
                from finvader import finvader as _fv
                country_scores = [
                    float(_fv(h, use_sentibignomics=True, use_henry=True, indicator='compound'))
                    for h in country_headlines
                ]
                c_model = "FinVADER"
            except Exception:
                sia_country = SentimentIntensityAnalyzer()
                country_scores = [sia_country.polarity_scores(h)['compound'] for h in country_headlines]
                c_model = "VADER"

        country_avg = np.mean(country_scores)
        if country_avg > 0.05:
            c_sent_label = "Positive"; c_sent_color = "green"
        elif country_avg < -0.05:
            c_sent_label = "Negative"; c_sent_color = "red"
        else:
            c_sent_label = "Neutral"; c_sent_color = "orange"

        st.markdown(
            f"<span style='color:{c_sent_color}; font-weight:bold'>"
            f"{c_sent_label} ({country_avg:+.3f})</span> based on {len(country_headlines)} headlines · {c_model}",
            unsafe_allow_html=True
        )
        for i, h in enumerate(country_headlines):
            sc = country_scores[i]
            icon = "🟢" if sc > 0.05 else "🔴" if sc < -0.05 else "🟡"
            st.markdown(f"{icon} **[{sc:+.3f}]** {h}")
    else:
        st.info(f"No energy headline from the last 7 days mentions {selected_country} by name or adjective.")

    # Must match the weights used for 'Structural Score' above, which differ in Carbon mode.
    score_formula = " + ".join(f"{name} ({w:.0%})" for name, w in SCORE_WEIGHTS)
    st.caption(
        f"Structural Score = {score_formula}. "
        f"Dynamic Risk = Structural × volatility multiplier, where the multiplier is weighted by dependency, "
        f"so dependency counts twice in the dynamic score on purpose. "
        f"Import dependency: Eurostat nrg_ind_id (natural gas G3000, oil O4000XBIO, total). "
        f"Renewable share: Eurostat nrg_ind_ren (REN). GHG intensity: Eurostat env_air_gge (total "
        f"excluding LULUCF) divided by nama_10_gdp (GDP at current prices)."
    )

st.write("Sentiment analysis of the latest headlines — FinBERT transformer with FinVADER lexicon fallback "
         "(the model actually used is stated below the chart)")

rss_feeds = {
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "OilPrice": "https://oilprice.com/rss/main",
    f"Google ({selected_commodity})": f"https://news.google.com/rss/search?q={commodity['rss_query']}&hl=en",
    "Google EU Energy": "https://news.google.com/rss/search?q=European+energy+market&hl=en",
    "Google EU Carbon": "https://news.google.com/rss/search?q=EU+carbon+ETS&hl=en",
}
headlines = []
headline_links = []
headline_sources = []
seen_titles = set()

_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}

for source_name, url in rss_feeds.items():
    try:
        resp = req.get(url, headers=_HEADERS, timeout=15)
        feed = feedparser.parse(resp.content)
    except Exception:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
    count = 0
    for entry in feed.entries[:80]:
        if count >= 8:
            break
        try:
            title = entry.title.strip()
            link = entry.get('link', '')
            if title.lower() in seen_titles:
                continue
            # Same test for every feed, BBC Business included: a commodity keyword in the
            # headline itself, and no US-domestic story.
            source = entry.get('source', {}).get('title', '')
            if not is_relevant(headline_text(title, source), commodity_kw_re):
                continue
            headlines.append(title)
            headline_links.append(link)
            headline_sources.append(source_name)
            seen_titles.add(title.lower())
            count += 1
        except Exception:
            continue

# No placeholder headlines: if every feed fails, no sentiment is computed at all.
is_live = bool(headlines)

# Limit to 10 headlines
headlines = headlines[:10]
headline_links = headline_links[:10]
headline_sources = headline_sources[:10]

if not is_live:
    finbert_success, finbert_reason = False, "no headlines"
    nlp_model_name = "none"
    sent_df = pd.DataFrame(columns=['Headline', 'Source', 'Link', 'Score', 'Label'])
    avg_score = None
    st.info("No relevant headline could be retrieved from the news feeds right now, so no sentiment "
            "is shown. The page does not score placeholder text.")
else:

    # FinBERT → FinVADER → VADER
    finbert_scores, finbert_labels, finbert_success, finbert_reason = finbert_analyze(tuple(headlines))

    if finbert_success:
        n = min(len(finbert_scores), len(headlines))
        nlp_model_name = "FinBERT (ProsusAI/finbert)"
        sentiment_data = []
        for i in range(n):
            sentiment_data.append({
                'Headline': headlines[i],
                'Source': headline_sources[i],
                'Link': headline_links[i],
                'Score': finbert_scores[i],
                'Label': finbert_labels[i],
            })
    else:
        # FIX: FinVADER fallback (consistent with README and country sentiment)
        try:
            from finvader import finvader as _fv_main
            nlp_model_name = "FinVADER (fallback — FinBERT unavailable)"
            sentiment_data = []
            for i, h in enumerate(headlines):
                score = float(_fv_main(h, use_sentibignomics=True, use_henry=True, indicator='compound'))
                if score > 0.05:
                    label = 'Positive'
                elif score < -0.05:
                    label = 'Negative'
                else:
                    label = 'Neutral'
                sentiment_data.append({
                    'Headline': h,
                    'Source': headline_sources[i],
                    'Link': headline_links[i],
                    'Score': score,
                    'Label': label,
                })
        except Exception:
            nlp_model_name = "VADER (fallback)"
            sia = SentimentIntensityAnalyzer()
            sentiment_data = []
            for i, h in enumerate(headlines):
                sc = sia.polarity_scores(h)
                score = sc['compound']
                if score > 0.05:
                    label = 'Positive'
                elif score < -0.05:
                    label = 'Negative'
                else:
                    label = 'Neutral'
                sentiment_data.append({
                    'Headline': h,
                    'Source': headline_sources[i],
                    'Link': headline_links[i],
                    'Score': score,
                    'Label': label,
                })

    sent_df = pd.DataFrame(sentiment_data)
    avg_score = sent_df['Score'].mean()
    n_pos = (sent_df['Label'] == 'Positive').sum()
    n_neg = (sent_df['Label'] == 'Negative').sum()
    n_neut = (sent_df['Label'] == 'Neutral').sum()

    if avg_score > 0.05:
        sentiment_label = "Positive"
        sentiment_color = "green"
    elif avg_score < -0.05:
        sentiment_label = "Negative"
        sentiment_color = "red"
    else:
        sentiment_label = "Neutral"
        sentiment_color = "orange"

    st.markdown(f"<h3 style='color:{sentiment_color}'>Market Sentiment: {sentiment_label}</h3>",
                unsafe_allow_html=True)
    st.caption("The score measures the tone of the headlines (positive or negative wording), not whether "
               "the news is bullish or bearish for prices: \"gas prices surge\" can score negative although "
               "it describes a price rise.")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(f"Avg Sentiment ({nlp_model_name.split(' (')[0]})", f"{avg_score:.3f}")
    sc2.metric("Positive", f"{n_pos}")
    sc3.metric("Negative", f"{n_neg}")
    sc4.metric("Neutral", f"{n_neut}")

    st.caption(f"Analyzing {len(sent_df)} live headlines from {len(set(headline_sources))} sources · Model: {nlp_model_name}")
    if not finbert_success:
        st.caption(f"FinBERT not used: {finbert_reason}")

    # Sentiment chart
    fig3, ax4 = plt.subplots(figsize=(12, max(3, len(sent_df) * 0.3)))
    bar_colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray'
                  for s in sent_df['Score']]
    ax4.barh(range(len(sent_df)), sent_df['Score'], color=bar_colors, height=0.6)
    ax4.set_yticks(range(len(sent_df)))
    ax4.set_yticklabels([h[:55] + '...' if len(h) > 55 else h for h in sent_df['Headline']],
                        fontsize=7)
    ax4.axvline(x=0, color='black', linewidth=0.5)
    ax4.axvline(x=0.05, color='green', linewidth=0.5, linestyle='--', alpha=0.4)
    ax4.axvline(x=-0.05, color='red', linewidth=0.5, linestyle='--', alpha=0.4)
    ax4.set_xlabel(f'Sentiment Score ({nlp_model_name.split(" (")[0]})')
    ax4.set_title('Per-Headline Sentiment Distribution')
    ax4.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig3)

    # Top positive & negative headlines
    top_pos = sent_df[sent_df['Score'] > 0.05].nlargest(3, 'Score')
    top_neg = sent_df[sent_df['Score'] < -0.05].nsmallest(3, 'Score')

    if not top_pos.empty:
        st.write("**Most Positive Headlines:**")
        for _, row in top_pos.iterrows():
            score_str = f"{row['Score']:+.3f}"
            if row['Link']:
                st.markdown(f"🟢 **[{score_str}]** [{row['Headline']}]({row['Link']}) — *{row['Source']}*")
            else:
                st.markdown(f"🟢 **[{score_str}]** {row['Headline']} — *{row['Source']}*")

    if not top_neg.empty:
        st.write("**Most Negative Headlines:**")
        for _, row in top_neg.iterrows():
            score_str = f"{row['Score']:+.3f}"
            if row['Link']:
                st.markdown(f"🔴 **[{score_str}]** [{row['Headline']}]({row['Link']}) — *{row['Source']}*")
            else:
                st.markdown(f"🔴 **[{score_str}]** {row['Headline']} — *{row['Source']}*")

    if top_pos.empty and top_neg.empty:
        st.info("All current headlines are neutral — no strong positive or negative signal detected.")

# ─── Section 6b: Sentiment Trend (30-day) ───
st.subheader("Sentiment Trend (30 Days)")
st.write(f"Daily average sentiment for {selected_commodity}-related European energy news over the past 30 days")

# FIX: use FinVADER (not basic VADER) for 30-day trend, consistent with fallback strategy
@st.cache_data(ttl=7200, show_spinner="Fetching 30-day news history...")
def get_sentiment_trend(rss_query, news_term, keywords):
    """Fetch past 30 days of news via Google News RSS and compute daily FinVADER sentiment."""
    from datetime import datetime

    daily_scores = {}
    keyword_re = _word_pattern(keywords)

    def score_text(text):
        try:
            from finvader import finvader as _fv_t
            return float(_fv_t(text, use_sentibignomics=True, use_henry=True, indicator='compound'))
        except Exception:
            sia_t = SentimentIntensityAnalyzer()
            return sia_t.polarity_scores(text)['compound']

    for trend_url in [
        f"https://news.google.com/rss/search?q={rss_query}+when:30d&hl=en",
        f"https://news.google.com/rss/search?q=European+energy+{news_term}+when:30d&hl=en",
    ]:
        try:
            feed = feedparser.parse(trend_url)
            for entry in feed.entries[:100]:
                title = entry.title
                source = entry.get('source', {}).get('title', '')
                if not is_relevant(headline_text(title, source), keyword_re):
                    continue
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:3]).strftime('%Y-%m-%d')
                else:
                    continue
                s = score_text(title)
                if pub_date not in daily_scores:
                    daily_scores[pub_date] = []
                daily_scores[pub_date].append(s)
        except Exception:
            pass

    if not daily_scores:
        return None

    trend_df = pd.DataFrame([
        {'Date': date, 'Avg Sentiment': np.mean(scores), 'Headlines Count': len(scores)}
        for date, scores in daily_scores.items()
    ])
    trend_df['Date'] = pd.to_datetime(trend_df['Date'])
    trend_df = trend_df.sort_values('Date')
    return trend_df

trend_df = get_sentiment_trend(commodity['rss_query'], commodity['news_term'], tuple(commodity['keywords']))
avg_30d = None  # initialized here; set inside conditional below

if trend_df is not None and len(trend_df) > 3:
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    today_sent = trend_df[trend_df['Date'] == today_str]
    avg_30d = trend_df['Avg Sentiment'].mean()

    tc1, tc2, tc3 = st.columns(3)
    if len(today_sent) > 0:
        tc1.metric("Today's Avg Sentiment (FinVADER)", f"{today_sent['Avg Sentiment'].iloc[0]:.3f}")
        tc2.metric("Today's Headlines", f"{int(today_sent['Headlines Count'].iloc[0])}")
    else:
        tc1.metric("Today's Avg Sentiment (FinVADER)", "N/A")
        tc2.metric("Today's Headlines", "0")
    tc3.metric("30-Day Avg Sentiment (FinVADER)", f"{avg_30d:.3f}")

    fig_trend, ax_trend = plt.subplots(figsize=(14, 4))
    colors_trend = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray'
                    for s in trend_df['Avg Sentiment']]
    ax_trend.bar(trend_df['Date'], trend_df['Avg Sentiment'], color=colors_trend, alpha=0.7, width=0.8)
    ax_trend.axhline(y=0, color='black', linewidth=0.5)
    ax_trend.axhline(y=avg_30d, color='blue', linewidth=1, linestyle='--', alpha=0.5,
                    label=f'30-day avg: {avg_30d:.3f}')
    ax_trend.set_ylabel('Daily Avg Sentiment')
    ax_trend.set_title(f'{selected_commodity} — 30-Day Sentiment Trend')
    ax_trend.legend(fontsize=8)
    ax_trend.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_trend)

    st.caption(f"Based on {int(trend_df['Headlines Count'].sum())} headlines over {len(trend_df)} days · Trend scored with FinVADER only. The current reading above may come from FinBERT; the two models use different scales, so the numbers are not compared.")
else:
    st.info("Not enough historical headline data to generate trend. This improves over time as more news is collected.")

# ─── Section 6c: Anomaly Detection ───
st.subheader("🔍 Anomaly Detection")
st.write("Automated signal monitoring — flags statistical outliers and structural divergences (daily prices up to the latest available, refreshed hourly).")

vol_series = df_analysis['Volatility'].dropna()
vol_std = vol_series.std()

# Correlation values for the selected commodity vs comparison
# Use the rolling corr series: full period mean vs last 30 days mean
corr_series = df_analysis['Rolling Correlation'].dropna()
corr_full_val = corr_series.mean() if len(corr_series) > 0 else None
corr_30d_val = corr_series.tail(30).mean() if len(corr_series) >= 30 else None

anomalies = []

# ── 1. Volatility spike / complacency ──
if vol_std > 0:
    z = (latest_vol - avg_vol) / vol_std
    if z > 2.5:
        anomalies.append({
            'level': '🔴 CRITICAL',
            'type': 'Extreme Volatility Spike',
            'detail': (f'Current vol {latest_vol:.1f}% is **{z:.1f}σ** above historical mean '
                       f'({avg_vol:.1f}%). Tail-risk elevated — review VaR limits.'),
        })
    elif z > 1.8:
        anomalies.append({
            'level': '🟡 WARNING',
            'type': 'Elevated Volatility',
            'detail': (f'Current vol {latest_vol:.1f}% is {z:.1f}σ above mean. '
                       f'Approaching stress territory — monitor closely.'),
        })
    elif z < -1.5:
        anomalies.append({
            'level': '🔵 WATCH',
            'type': 'Unusual Calm (Complacency Risk)',
            'detail': (f'Current vol {latest_vol:.1f}% is {abs(z):.1f}σ below mean. '
                       f'Low-vol regimes can precede sharp reversals.'),
        })

# ── 2. GARCH forward signal ──
# The forecast is compared with the sample average 30-day volatility, not with current rolling
# volatility: a forecast rising from a quiet spell back towards the average is normal mean
# reversion, not a warning.
if garch_forecast_10d is not None and garch_long_run:
    garch_ratio = garch_forecast_10d / garch_long_run
    if garch_ratio > 1.30:
        anomalies.append({
            'level': '🟡 WARNING',
            'type': 'GARCH Forecast Above Average Level',
            'detail': (f'GARCH 10-day forecast ({garch_forecast_10d:.2f}%) is {(garch_ratio-1)*100:.0f}% above '
                       f'the {garch_ref_name} ({garch_long_run:.2f}%). The model expects volatility '
                       f'to stay elevated over the next 10 days.'),
        })
    elif garch_ratio < 0.70:
        anomalies.append({
            'level': '🟢 INFO',
            'type': 'GARCH Forecast Below Average Level',
            'detail': (f'GARCH 10-day forecast ({garch_forecast_10d:.2f}%) is {(1-garch_ratio)*100:.0f}% below '
                       f'the {garch_ref_name} ({garch_long_run:.2f}%): quieter than usual.'),
        })

# ── 3. Correlation regime shift ──
if corr_full_val is not None and corr_30d_val is not None:
    corr_shift = abs(corr_30d_val - corr_full_val)
    if corr_shift > 0.4:
        direction = "risen" if corr_30d_val > corr_full_val else "fallen"
        anomalies.append({
            'level': '🟡 WARNING',
            'type': 'Correlation Regime Shift',
            'detail': (f'30-day correlation ({corr_30d_val:.2f}) has {direction} {corr_shift:.2f} '
                       f'from historical baseline ({corr_full_val:.2f}). '
                       f'Cross-commodity dynamics are changing.'),
        })
    elif corr_shift > 0.25:
        anomalies.append({
            'level': '🔵 WATCH',
            'type': 'Correlation Drift',
            'detail': (f'30-day correlation ({corr_30d_val:.2f}) drifting from baseline '
                       f'({corr_full_val:.2f}). Diversification assumptions may be shifting.'),
        })

# ── 4. Sentiment–volatility divergence ──
# Placeholder headlines (feeds unreachable) are not news, so they never trigger a signal.
# The current reading and the 30-day trend may come from different models, so each is
# only compared with its own threshold, never with the other.
live_avg_score = avg_score if is_live else None
if live_avg_score is not None or avg_30d is not None:
    if live_avg_score is not None and current_regime in ['Volatile', 'Crisis'] and live_avg_score > 0.10:
        anomalies.append({
            'level': '🟡 WARNING',
            'type': 'Sentiment–Volatility Divergence',
            'detail': (f'Market regime is **{current_regime}** but current sentiment is positive '
                       f'({live_avg_score:+.3f}, {nlp_model_name.split(" (")[0]}). Possible market complacency — short-term divergence.'),
        })
    if current_regime == 'Calm' and avg_30d is not None and avg_30d < -0.15:
        anomalies.append({
            'level': '🔵 WATCH',
            'type': 'Negative Sentiment Trend vs Calm Vol',
            'detail': (f'30-day sentiment average ({avg_30d:+.3f}, FinVADER) persistently negative '
                       f'despite calm volatility. Sentiment may be a leading indicator.'),
        })

# ── 5. Historical tail — only flag if RECENT extreme loss, not just all-time max ──
# Using all-time max_loss > VaR99*1.5 fires for every asset always (mathematical certainty).
# Instead: check if any daily loss in the last 252 trading days (≈1 year) exceeded VaR99*2.
# This is a genuinely rare event that warrants attention.
recent_returns = returns_clean.iloc[-252:]
recent_min = recent_returns.min() * 100
if abs(recent_min) > abs(var_99_full) * 2.0:
    anomalies.append({
        'level': '🔵 WATCH',
        'type': 'Recent Tail Event Beyond 2× VaR99',
        'detail': (f'A daily loss of {recent_min:.2f}% occurred in the past 12 months — '
                   f'{abs(recent_min)/abs(var_99_full):.1f}x the full-period 99% VaR ({var_99_full:.2f}%). '
                   f'Recent fat-tail risk present; historical VaR may understate exposure.'),
    })
elif abs(returns_clean.min() * 100) > abs(var_99_full) * 3.0:
    # All-time extreme that is genuinely beyond 3x VaR99 (very rare)
    max_loss = returns_clean.min() * 100
    anomalies.append({
        'level': '🔵 WATCH',
        'type': 'Historical Tail Beyond 3× VaR99',
        'detail': (f'Max observed daily loss ({max_loss:.2f}%) is {abs(max_loss)/abs(var_99_full):.1f}x '
                   f'the full-period 99% VaR ({var_99_full:.2f}%). Extreme historical fat-tail present.'),
    })

if anomalies:
    for a in anomalies:
        level = a['level']
        if 'CRITICAL' in level:
            bg = '#fff0f0'; border = '#ff4444'
        elif 'WARNING' in level:
            bg = '#fffbe6'; border = '#ffaa00'
        elif 'WATCH' in level:
            bg = '#f0f4ff'; border = '#4488ff'
        else:
            bg = '#f0fff4'; border = '#44bb44'
        st.markdown(
            f"<div style='background:{bg}; border-left:4px solid {border}; "
            f"padding:10px 14px; margin:6px 0; border-radius:4px;'>"
            f"<strong>{a['level']} — {a['type']}</strong><br>"
            f"<span style='font-size:0.9em'>{a['detail']}</span></div>",
            unsafe_allow_html=True
        )
else:
    st.markdown(
        "<div style='background:#f0fff4; border-left:4px solid #44bb44; "
        "padding:10px 14px; border-radius:4px;'>"
        "✅ <strong>No anomalies detected</strong> — all risk signals within normal historical ranges.</div>",
        unsafe_allow_html=True
    )

st.caption("Thresholds: volatility z-score above 2.5σ (critical), above 1.8σ (warning) or below −1.5σ "
           "(unusual calm) · GARCH 10-day forecast more than 30% above or below the sample average 30-day "
           "volatility · correlation shift above 0.25 (drift) or 0.4 (regime shift) · sentiment-regime "
           "divergence · tail: any loss in the last 252 days beyond 2× the full-period VaR99, otherwise "
           "any loss in the whole period beyond 3× VaR99")

# ─── Section 6d: AI Risk Narrative (LLM) ───
st.subheader("🤖 AI Risk Interpretation")
st.write(f"Synthesizes today's quantitative signals into a plain-language risk assessment for {selected_commodity}.")

@st.cache_data(ttl=1800, show_spinner="Generating AI risk interpretation...")
def generate_risk_narrative(commodity_name, risk_level_str, _latest_vol, _avg_vol,
                            _var_95, _current_regime, _garch_10d, _garch_long_run, _garch_ref_name,
                            _avg_score, _score_model, _avg_30d, anomaly_types_str,
                            top_neg_str, date_str, regime_calm, regime_crisis):
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        return None, "no_key"

    garch_line = (f"GARCH forecast of daily volatility on day 10 ahead: {_garch_10d:.2f}%"
                  if _garch_10d else "GARCH forecast: unavailable")
    if _garch_10d and _garch_long_run:
        garch_line += f" ({_garch_ref_name} {_garch_long_run:.2f}%)"
    anomaly_line = anomaly_types_str if anomaly_types_str else "None"
    headline_line = top_neg_str if top_neg_str else "None retrieved"
    sentiment_30d_line = (f"{_avg_30d:+.3f}" if _avg_30d is not None else "N/A")
    sentiment_now_line = (f"{_avg_score:+.3f} ({_score_model})" if _avg_score is not None
                          else "N/A (news feeds unreachable, no sentiment computed)")

    prompt = f"""You are a senior risk analyst on a European energy trading desk, writing the short risk comment on {commodity_name} for the date shown below.

HOW TO READ THE INPUTS

Volatility
- All volatility figures are DAILY standard deviation of returns, in percent. Not annualised.
- The GARCH figure forecasts the daily volatility on the tenth trading day ahead. It is not a cumulative move over ten days.
- The GARCH anomaly flag compares the forecast with the sample average 30-day volatility, not with current volatility. A forecast moving back towards that average is normalisation, not a build-up of risk, and should be described that way. Also check the forecast against the {regime_calm:.2f}% Volatile boundary.

Risk signal and regime - one classification
- "Regime" compares current 30-day volatility with FIXED thresholds for this commodity, taken from its full volatility history (not the selected window): Calm below {regime_calm:.2f}% (its 50th percentile), Volatile {regime_calm:.2f}-{regime_crisis:.2f}%, Crisis above {regime_crisis:.2f}% (its 90th percentile).
- "Risk signal" is the same classification under another name: Calm = LOW, Volatile = MEDIUM, Crisis = HIGH RISK. Do not present them as two separate pieces of evidence.
- The selected-period average volatility is context only; it does not set the risk signal.

VaR
- VaR 95% is the 5th percentile of daily returns over the last 250 trading days: a loss threshold, given as a negative number.

Sentiment - weak evidence, handle with care
- Computed from at most ten scraped headlines.
- The current reading and the 30-day average come from DIFFERENT models depending on which service was reachable, and are NOT on a common scale. Never compare them numerically and never describe a move from one to the other.
- Anything between -0.05 and +0.05 is NEUTRAL. Do not call it mildly positive or mildly negative.

Anomalies and headlines
- The anomaly field lists only the NAMES of the checks that triggered. You do not have their underlying numbers, so do not invent them.
- The headline field is raw scraped text, truncated and unverified.

MARKET DATA AS OF {date_str}
- Risk signal: {risk_level_str}
- 30-day rolling volatility: {_latest_vol:.2f}% (average over the selected period {_avg_vol:.2f}%)
- VaR 95%, 1-day (historical, last 250 trading days): {_var_95:.2f}%
- {garch_line}
- Regime: {_current_regime}
- Sentiment now: {sentiment_now_line} | 30-day average (FinVADER): {sentiment_30d_line}
- Anomaly checks triggered: {anomaly_line}
- Unverified headlines: {headline_line}

TASK
Write EXACTLY 3 sentences. Keep the whole comment under 100 words. No headers, no bullets, no line breaks. Write as of the data date above, not as of any later date.
1. What this combination of readings means. Do not restate the figures - the reader is looking at them. Use at most one number, and only if it carries the argument.
2. The sharpest tension or divergence in the data, and why it matters for positioning.
3. One concrete, checkable thing to watch over the following ten trading days. Name a level, a threshold or a specific series from the data above. Do not simply say to watch whether the forecast proves correct.

HARD RULES
- If the headline field says feeds were unreachable, treat BOTH sentiment readings as meaningless: do not mention sentiment or news anywhere in your answer, and build all three sentences from the volatility, VaR, GARCH and regime data alone.
- If a field reads N/A, None or None retrieved, pass over it in silence. Do not comment on its absence.
- Never attribute a claim, forecast or warning to a named bank, agency, institution or analyst.
- Never present a headline as established fact. Sentiment tells you about mood, not about the world.
- Do not assert a causal chain the data above cannot support. If a link is speculative, mark it as speculative.
- Your only raw input is a series of daily closing prices. You have no term structure, no futures curve, no inventories, no storage, no positioning, no flows and no fundamentals. Never write about roll risk, contango, backwardation, spreads, curve shape, stockpiles, supply, demand or hedging costs.
- Do not introduce seasonal, geopolitical or fundamental drivers that are not present in the inputs above.
- Avoid: "low-risk equilibrium", "market participants should", "underlying stress", "tranquility", "it is important to note".
- Plain desk English. No padding.

Output only the 3 sentences."""

    try:
        response = req.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        if response.status_code == 200:
            narrative = response.json()['content'][0]['text'].strip()
            return narrative, None
        else:
            return None, f"api_error_{response.status_code}"
    except Exception as e:
        return None, str(e)

# Prepare inputs for LLM call
anomaly_types_for_llm = "; ".join([a['type'] for a in anomalies]) if anomalies else ""

if not is_live:
    top_neg_for_llm = "none (news feeds unreachable, no sentiment computed)"
else:
    top_neg_for_llm = "; ".join([
        row['Headline'][:80] for _, row in
        sent_df[sent_df['Score'] < -0.05].nsmallest(3, 'Score').iterrows()
    ]) if not sent_df.empty else ""

today_date_str = df_analysis.index[-1].strftime('%Y-%m-%d')

narrative, error = generate_risk_narrative(
    commodity_name=selected_commodity,
    risk_level_str=risk_level,
    _latest_vol=latest_vol,
    _avg_vol=avg_vol,
    _var_95=var_95,
    _current_regime=current_regime,
    _garch_10d=garch_forecast_10d,
    _garch_long_run=garch_long_run,
    _garch_ref_name=garch_ref_name,
    _avg_score=live_avg_score,
    _score_model=nlp_model_name.split(' (')[0],
    _avg_30d=avg_30d,
    anomaly_types_str=anomaly_types_for_llm,
    top_neg_str=top_neg_for_llm,
    date_str=today_date_str,
    regime_calm=regime_thr['calm'],
    regime_crisis=regime_thr['crisis'],
)

if narrative:
    st.markdown(
        f"<div style='background:#f8f9ff; border-left:4px solid #6655ee; "
        f"padding:14px 18px; border-radius:6px; font-size:0.97em; line-height:1.7'>"
        f"{narrative}</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Generated by Claude (claude-sonnet) · {today_date_str} · Based on live market data + anomaly signals · Refreshes every 30 min")
elif error == "no_key":
    st.caption(
        "AI narrative module not enabled in this deployment. It synthesises the "
        "volatility, VaR, GARCH, regime and sentiment signals above into a short "
        "written risk assessment."
    )
else:
    st.caption(
        "AI narrative temporarily unavailable — all quantitative signals above are unaffected."
    )


# ─── Section 7: Data Export ───
st.subheader("Export Data")

export_df = df_analysis[['Price', 'Compare', 'Volatility', 'Rolling Correlation']].copy()
export_df.columns = [selected_commodity, compare_label, 'Volatility (%)', 'Rolling Correlation (returns)']
export_df['Daily_Return_%'] = df_analysis['Returns'] * 100

if 'Regime' in features.columns:
    export_df = export_df.join(features[['Regime']], how='left')

csv = export_df.to_csv()
st.download_button(
    label=f"Download {selected_commodity} Risk Data (CSV)",
    data=csv,
    file_name=f"{selected_commodity.lower().replace(' ', '_')}_risk_data.csv",
    mime="text/csv"
)

if is_live:
    sent_csv = sent_df.to_csv(index=False)
    st.download_button(
        label="Download Sentiment Data (CSV)",
        data=sent_csv,
        file_name="sentiment_data.csv",
        mime="text/csv"
    )

if not cr_df.empty:
    export_cols = ['Country', 'Risk Score', 'Structural Score', 'Country Vol Multiplier',
                    'Risk Level', dep_col,
                    'Total Energy Dep. (%)', 'GHG Int. (tCO2e/M€)',
                    'Renewable (%)', 'Data source']
    seen_e = set()
    export_cols = [c for c in export_cols if not (c in seen_e or seen_e.add(c))]
    cr_export = cr_df[export_cols].copy()
    rename_map = {'Country': 'Country', 'Risk Score': 'Dynamic Risk Score',
                  'Structural Score': 'Structural Score', 'Country Vol Multiplier': 'Vol Multiplier',
                  'Risk Level': 'Risk Level', dep_col: dep_label,
                  'Total Energy Dep. (%)': 'Total Energy Dependency (%)',
                  'GHG Int. (tCO2e/M€)': 'GHG intensity (tCO2e per M€ GDP)',
                  'Renewable (%)': 'Renewable Share (%)'}
    cr_export = cr_export.rename(columns=rename_map)
    cr_csv = cr_export.to_csv(index=False)
    st.download_button(
        label=f"Download Country Risk Data ({selected_year}, CSV)",
        data=cr_csv,
        file_name=f"country_risk_{selected_year}.csv",
        mime="text/csv"
    )

