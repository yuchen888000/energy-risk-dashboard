"""Market page: country exposure, news sentiment, anomaly checks and an AI summary.

Designed with data and analytics firms in mind.
"""
import os
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
from country_data import (COUNTRIES, gas_dep, oil_dep, total_dep, ren_share,
                          carbon_int, price_sens, dependency_for, mentions_country)

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
    - **Country Risk Scoring**: Composite index of gas/oil dependency,
      carbon intensity, energy import share, renewable share across 31
      European countries (EU-27 + CH, UK, NO, TR). Structural scores are multiplied by real-time
      market volatility to create dynamic risk assessment.
    - **FinBERT**: Main sentiment model. Transformer fine-tuned on financial
      text (ProsusAI/finbert via HuggingFace). Applied to live headlines.
    - **FinVADER**: Fallback model. VADER enhanced with SentiBigNomics + Henry
      financial lexicons — more accurate than standard VADER for financial text.
    - **30-Day Sentiment Trend**: Daily average sentiment via Google News RSS,
      scored with FinVADER. Visualised as bar chart with trend line.
    - **Anomaly Detection**: 5 automated signal checks — volatility z-score,
      GARCH divergence, correlation regime shift, sentiment-volatility divergence,
      recent tail event (loss > 2× VaR99 in past 252 days).
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
var_95, var_99 = core['var_95'], core['var_99']
risk_level = core['risk_level']
dep_col, dep_label, _ = dependency_for(selected_commodity)

garch = common.fit_garch(returns_clean)
garch_forecast_10d = garch['forecast_10d'] if garch is not None else None
regime_thr, current_regime = core['regime_thr'], core['current_regime']
features = common.compute_regimes(df_analysis[['Volatility', 'Rolling Correlation']], regime_thr)

# ─── FinBERT via HuggingFace Inference API ───
# FIX: defined here so it's available to both country sentiment (Section 5b)
# and main NLP section (Section 6)
def finbert_analyze(texts):
    API_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        pass
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", None)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    def parse_results(results, n_texts):
        scores, labels = [], []
        if not isinstance(results, list):
            return None, None
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

    # Warm up API
    for attempt in range(3):
        try:
            warmup = req.post(API_URL, headers=headers,
                              json={"inputs": texts[0]}, timeout=45)
            if warmup.status_code == 503:
                wait_time = warmup.json().get('estimated_time', 20)
                time.sleep(min(wait_time + 5, 45))
                continue
            if warmup.status_code == 200:
                break
        except Exception:
            if attempt < 2:
                time.sleep(10)
            continue
    else:
        return None, None, False

    # Send full batch
    for attempt in range(3):
        try:
            response = req.post(API_URL, headers=headers,
                                json={"inputs": texts}, timeout=60)
            if response.status_code == 503:
                wait_time = response.json().get('estimated_time', 20)
                time.sleep(min(wait_time + 5, 40))
                continue
            if response.status_code == 200:
                sc, lb = parse_results(response.json(), len(texts))
                if sc is not None:
                    return sc, lb, True
                break
        except Exception:
            if attempt < 2:
                time.sleep(10)
            continue

    return None, None, False

def finvader_score(text):
    """FinVADER fallback — VADER + SentiBigNomics + Henry financial lexicons."""
    try:
        from finvader import finvader
        return float(finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound'))
    except Exception:
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)['compound']


# ─── Section 5b: European Country Energy Risk ───
st.subheader("European Country Energy Risk Exposure")
st.write("Which European countries are most vulnerable to energy price shocks?")
st.caption("Coverage: EU-27 + Switzerland, UK, Norway, Turkey · Source: Eurostat (nrg_ind_id, sdg_07_50, nrg_ind_ren), IEA, EEA")

selected_year = st.slider("Select Year", min_value=2020, max_value=2024, value=2024, step=1)

cr_df = pd.DataFrame({
    'Country': COUNTRIES,
    'Gas Dep. (%)': gas_dep[selected_year],
    'Oil Dep. (%)': oil_dep[selected_year],
    'Total Energy Dep. (%)': total_dep[selected_year],
    'Renewable (%)': ren_share[selected_year],
    'Carbon Int. (tCO2/M€)': carbon_int[selected_year],
    'Price Sensitivity': price_sens[selected_year],
})

cr_df['Dep Clipped'] = cr_df[dep_col].clip(lower=0)
cr_df['Total Clipped'] = cr_df['Total Energy Dep. (%)'].clip(lower=0)

# Norway is a net energy exporter; negative total_dep values are economically meaningful
# (surplus) but would display confusingly and skew any ranking column.
# Clamp display column to 0 — the scoring already uses Total Clipped internally.
cr_df['Total Energy Dep. (%)'] = cr_df['Total Energy Dep. (%)'].clip(lower=0)

# Structural Score formula — commodity-aware to avoid double-weighting:
# For Gas/Oil commodities: dep_col is gas or oil dep (separate from total_dep) → 6 distinct factors
# For EU Carbon: dep_col IS total energy dep → would appear twice if formula is identical.
#   Solution: replace the separate total_dep term with carbon-intensity rank (already in formula)
#   and redistribute weight so factors remain independent.
if commodity['ticker'] == CARBON_TICKER:
    # Carbon mode: 6 factors with no double-count
    # Total Energy Dep 25% | Carbon Int rank 20% | Inverse Renewable 20%
    # Total Energy Dep rank 15% | Price Sensitivity 15% | Renewable rank 5% (residual)
    cr_df['Structural Score'] = (
        cr_df['Dep Clipped'] * 0.25 +
        cr_df['Carbon Int. (tCO2/M€)'].rank(pct=True) * 100 * 0.20 +
        (100 - cr_df['Renewable (%)']) * 0.20 +
        cr_df['Dep Clipped'].rank(pct=True) * 100 * 0.15 +
        cr_df['Price Sensitivity'] * 10 * 0.15 +
        (100 - cr_df['Renewable (%)'].rank(pct=True) * 100) * 0.05
    ).round(1)
else:
    # Gas / Oil mode: dep_col ≠ total_dep → all 6 factors are independent
    cr_df['Structural Score'] = (
        cr_df['Dep Clipped'] * 0.25 +
        cr_df['Carbon Int. (tCO2/M€)'].rank(pct=True) * 100 * 0.15 +
        cr_df['Total Clipped'] * 0.15 +
        (100 - cr_df['Renewable (%)']) * 0.15 +
        cr_df['Dep Clipped'].rank(pct=True) * 100 * 0.15 +
        cr_df['Price Sensitivity'] * 10 * 0.15
    ).round(1)

vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
vol_ratio_clamped = min(max(vol_ratio, 0.5), 3.0)

cr_df['Country Vol Multiplier'] = (
    0.5 + 0.5 * vol_ratio_clamped * (cr_df['Dep Clipped'] / 100)
).round(2)

cr_df['Risk Score'] = (cr_df['Structural Score'] * cr_df['Country Vol Multiplier']).round(1)
cr_df = cr_df.sort_values('Risk Score', ascending=False)

def risk_category(score):
    if score > 70:
        return '🔴 High'
    elif score > 50:
        return '🟡 Medium'
    else:
        return '🟢 Low'

cr_df['Risk Level'] = cr_df['Risk Score'].apply(risk_category)

st.markdown(f"**Real-time risk adjustment:** Current {selected_commodity} volatility is **{latest_vol:.1f}%** "
            f"vs average **{avg_vol:.1f}%** → base volatility ratio = **{vol_ratio_clamped:.2f}x**")
st.caption("Each country's multiplier is weighted by its own dependency — high-dependency countries "
           "feel the same market volatility much more than low-dependency ones.")

cr_col1, cr_col2 = st.columns([1, 1])

with cr_col1:
    st.write(f"**Risk Ranking ({selected_year}) — by {dep_label}:**")
    display_cols = ['Country', 'Risk Score', 'Country Vol Multiplier', 'Risk Level', dep_col,
                    'Total Energy Dep. (%)', 'Carbon Int. (tCO2/M€)',
                    'Price Sensitivity', 'Renewable (%)']
    seen = set()
    display_cols = [c for c in display_cols if not (c in seen or seen.add(c))]
    display_df = cr_df[display_cols].reset_index(drop=True)
    display_df.index = display_df.index + 1
    st.dataframe(display_df, width="stretch", height=400)

with cr_col2:
    selected_country = st.selectbox("Select Country for Detail", cr_df['Country'].tolist())
    country_data = cr_df[cr_df['Country'] == selected_country].iloc[0]

    country_dep_val = country_data[dep_col] / 100 if country_data[dep_col] > 0 else 0
    country_adj_vol = latest_vol * country_dep_val
    country_adj_var = var_95 * country_dep_val

    if country_adj_vol > 6:
        c_risk_level = "🔴 HIGH RISK"
        c_risk_color = "red"
    elif country_adj_vol > 3:
        c_risk_level = "🟡 MEDIUM RISK"
        c_risk_color = "orange"
    else:
        c_risk_level = "🟢 LOW RISK"
        c_risk_color = "green"

    st.markdown(f"### {selected_country} ({selected_year})")
    st.markdown(f"<h3 style='color:{c_risk_color}'>{c_risk_level}</h3>",
                unsafe_allow_html=True)

    cr_m1, cr_m2 = st.columns(2)
    cr_m1.metric("Adjusted Volatility (live)", f"{country_adj_vol:.2f}%")
    cr_m2.metric("Adjusted VaR 95% (live)", f"{country_adj_var:.2f}%")

    cd1, cd2 = st.columns(2)
    cd1.metric(dep_label, f"{country_data[dep_col]:.0f}%")
    cd2.metric("Carbon Intensity", f"{country_data['Carbon Int. (tCO2/M€)']:.0f} tCO2/M€")
    cd3, cd4 = st.columns(2)
    cd3.metric("Total Energy Dep.", f"{country_data['Total Energy Dep. (%)']:.0f}%")
    cd4.metric("Renewable Share", f"{country_data['Renewable (%)']:.0f}%")
    cd5, cd6 = st.columns(2)
    cd5.metric("Price Sensitivity", f"{country_data['Price Sensitivity']:.1f}/10")
    cd6.metric("Structural Score", f"{country_data['Structural Score']:.1f}")
    cd7, cd8 = st.columns(2)
    cd7.metric("Vol Multiplier (live)", f"{country_data['Country Vol Multiplier']:.2f}x")
    cd8.metric("Dynamic Risk Score", f"{country_data['Risk Score']:.1f}")

# Bar chart
fig_cr, ax_cr = plt.subplots(figsize=(14, 6))
top_n = cr_df.head(20)
bar_colors_cr = ['red' if s > 70 else 'orange' if s > 50 else 'green' for s in top_n['Risk Score']]
ax_cr.barh(range(len(top_n)), top_n['Risk Score'], color=bar_colors_cr, height=0.6)
ax_cr.set_yticks(range(len(top_n)))
ax_cr.set_yticklabels(top_n['Country'], fontsize=9)
ax_cr.set_xlabel('Composite Energy Risk Score')
ax_cr.set_title(f'European Countries — Energy Risk Ranking ({selected_year}, by {dep_label})')
ax_cr.axvline(x=70, color='red', linewidth=1, linestyle='--', alpha=0.4, label='High risk')
ax_cr.axvline(x=50, color='orange', linewidth=1, linestyle='--', alpha=0.4, label='Medium risk')
ax_cr.legend(fontsize=8)
ax_cr.invert_yaxis()
plt.tight_layout()
st.pyplot(fig_cr)

# Year-over-year trend for selected country
st.write(f"**{selected_country} — Risk Trend 2020–2024:**")
trend_data = []
for yr in [2020, 2021, 2022, 2023, 2024]:
    idx = COUNTRIES.index(selected_country)
    if selected_commodity in ['TTF Natural Gas']:
        dep_val = gas_dep[yr][idx]
    elif selected_commodity in ['WTI Crude Oil', 'Brent Crude Oil']:
        dep_val = oil_dep[yr][idx]
    else:
        dep_val = max(total_dep[yr][idx], 0)
    trend_data.append({
        'Year': yr,
        dep_label + ' (%)': dep_val,
        'Renewable (%)': ren_share[yr][idx],
        'Carbon Intensity': carbon_int[yr][idx],
    })
trend_cr = pd.DataFrame(trend_data)

fig_tcr, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(14, 3.5))
ax_t1.plot(trend_cr['Year'], trend_cr[dep_label + ' (%)'], 'o-', color='red', label=dep_label)
ax_t1.plot(trend_cr['Year'], trend_cr['Renewable (%)'], 's-', color='green', label='Renewable Share')
ax_t1.set_ylabel('Percentage (%)')
ax_t1.set_title(f'{selected_country} — Dependency vs Renewables')
ax_t1.legend(fontsize=8)
ax_t1.set_xticks([2020, 2021, 2022, 2023, 2024])

ax_t2.bar(trend_cr['Year'], trend_cr['Carbon Intensity'], color='gray', alpha=0.7)
ax_t2.set_ylabel('tCO2/M€ GDP')
ax_t2.set_title(f'{selected_country} — Carbon Intensity Trend')
ax_t2.set_xticks([2020, 2021, 2022, 2023, 2024])

plt.tight_layout()
st.pyplot(fig_tcr)

# Per-country real-time adjusted volatility
st.write(f"**{selected_country} — Real-Time Adjusted Volatility:**")
country_dep_pct = country_data[dep_col] / 100 if country_data[dep_col] > 0 else 0
country_vol = df_analysis['Volatility'].dropna() * country_dep_pct

fig_cvol, ax_cvol = plt.subplots(figsize=(14, 3.5))
ax_cvol.plot(df_analysis['Volatility'].dropna().index, df_analysis['Volatility'].dropna(),
             color='gray', linewidth=0.8, alpha=0.4, label=f'{selected_commodity} raw volatility')
ax_cvol.plot(country_vol.index, country_vol,
             color='red', linewidth=1.5, label=f'{selected_country} adjusted ({country_data[dep_col]:.0f}% dep.)')
ax_cvol.axhline(y=6, color='orange', linewidth=0.8, linestyle='--', alpha=0.4)
ax_cvol.axhline(y=12, color='red', linewidth=0.8, linestyle='--', alpha=0.4)
ax_cvol.set_ylabel('Adjusted Volatility (%)')
ax_cvol.set_title(f'{selected_country} — Dependency-Weighted Volatility (Live)')
ax_cvol.legend(fontsize=8)
ax_cvol.fill_between(country_vol.index, country_vol, 0, alpha=0.1, color='red')
plt.tight_layout()
st.pyplot(fig_cvol)
st.caption(f"Adjusted volatility = {selected_commodity} 30-day rolling volatility × {selected_country}'s "
           f"{dep_label.lower()} ({country_data[dep_col]:.0f}%). Current: {country_vol.iloc[-1]:.2f}%")

# Per-country news sentiment
# FIX: now uses FinBERT → FinVADER → VADER fallback chain (consistent with main sentiment section)
st.write(f"**{selected_country} — Current Energy News Sentiment:**")
_country_q = quote_plus(f"{selected_country} energy {commodity['rss_query'].split('+')[0]}")
country_rss_url = f"https://news.google.com/rss/search?q={_country_q}+when:7d&hl=en"
country_headlines = []
try:
    country_feed = feedparser.parse(country_rss_url)
    _energy_kw = commodity['keywords'] + ['energy', 'oil', 'gas', 'carbon', 'power',
                                          'fuel', 'electricity', 'pipeline', 'LNG',
                                          'emission', 'climate', 'price', 'supply',
                                          'tanker', 'refinery', 'fossil', 'renewable',
                                          'heating', 'Hormuz', 'sanction', 'ETS']
    for entry in country_feed.entries[:50]:
        title = entry.title
        # Google News appends " - Publisher"; match on the headline only, so an outlet
        # name such as "Irish Times" does not count as a mention of the country.
        source = entry.get('source', {}).get('title', '')
        headline = title[:-len(source) - 3] if source and title.endswith(f" - {source}") else title
        energy_match = any(kw.lower() in headline.lower() for kw in _energy_kw)
        if energy_match and mentions_country(headline, selected_country):
            country_headlines.append(title)
        if len(country_headlines) >= 5:
            break
except Exception:
    pass

if country_headlines:
    # FinBERT → FinVADER → VADER
    try:
        c_scores_raw, c_labels_raw, c_ok = finbert_analyze(country_headlines[:5])
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
if commodity['ticker'] == CARBON_TICKER:
    score_formula = (f"{dep_label} (25%) + Carbon Intensity rank (20%) + Inverse Renewable (20%) + "
                     f"{dep_label} rank (15%) + Price Sensitivity (15%) + Inverse Renewable rank (5%)")
else:
    score_formula = (f"{dep_label} (25%) + Carbon Intensity rank (15%) + Total Energy Dep. (15%) + "
                     f"Inverse Renewable (15%) + {dep_label} rank (15%) + Price Sensitivity (15%)")
st.caption(
    f"Structural Score = {score_formula}. "
    f"Dynamic Risk = Structural × Country-specific volatility multiplier (weighted by dependency). "
    f"Source: Eurostat (nrg_ind_id, sdg_07_50, nrg_ind_ren, nrg_ind_ei), EEA, IEA. 2024 = preliminary."
)


# ─── Section 6: NLP Sentiment (FinBERT) ───
st.subheader(f"Energy News Sentiment — {selected_commodity}")
st.write("Real-time sentiment analysis — FinBERT transformer with FinVADER lexicon fallback "
         "(the model actually used is stated below the chart)")

general_keywords = ['energy', 'power', 'electricity', 'renewable', 'climate',
                    'emission', 'fuel', 'Europe', 'European', 'heating',
                    'petrol', 'diesel', 'fossil', 'nuclear', 'pipeline',
                    'price hike', 'energy bill', 'energy cost', 'energy supply',
                    'energy crisis', 'energy market', 'energy shock',
                    'LNG', 'OPEC', 'refinery', 'carbon', 'ETS', 'Hormuz']
nlp_keywords = commodity['keywords'] + general_keywords

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
_google_sources = {k for k in rss_feeds if k.startswith("Google")}

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
            _broad_energy = ['energy', 'gas', 'oil', 'carbon', 'power', 'fuel',
                              'electricity', 'renewable', 'emission', 'climate',
                              'pipeline', 'LNG', 'OPEC', 'ETS', 'EUA',
                              'energy price', 'energy shock', 'energy market',
                              'EU energy', 'European energy', 'energy crisis',
                              'fossil fuel', 'coal', 'nuclear', 'solar', 'wind farm',
                              'refinery', 'barrel', 'Brent', 'WTI', 'TTF',
                              'Hormuz', 'Nord Stream', 'energy transition']
            if source_name not in _google_sources:
                if not any(kw.lower() in title.lower() for kw in nlp_keywords):
                    continue
                # Filter out US-domestic-only headlines that have no European relevance.
                # Headlines mentioning global chokepoints (Hormuz, Suez) or EU/Europe are kept.
                _us_domestic = ['US shale', 'U.S. shale', 'American oil', 'US oil output',
                                'US gas output', 'US production', 'U.S. production',
                                'US inventory', 'U.S. inventory', 'EIA report',
                                'US Strategic Reserve', 'U.S. Strategic Petroleum']
                _european_relevance = ['Europe', 'European', 'EU ', 'Hormuz', 'Suez',
                                       'LNG', 'pipeline', 'Nord Stream', 'TTF', 'ETS',
                                       'UK', 'Germany', 'France', 'Italy', 'Spain',
                                       'Russia', 'OPEC', 'global', 'world']
                is_us_only = (any(kw.lower() in title.lower() for kw in _us_domestic) and
                              not any(kw.lower() in title.lower() for kw in _european_relevance))
                if is_us_only:
                    continue
            else:
                if not any(kw.lower() in title.lower() for kw in _broad_energy):
                    continue
            headlines.append(title)
            headline_links.append(link)
            headline_sources.append(source_name)
            seen_titles.add(title.lower())
            count += 1
        except Exception:
            continue

is_live = True
if not headlines:
    is_live = False
    headlines = [
        "European gas prices surge amid supply concerns",
        "EU carbon market faces regulatory uncertainty",
        "Energy crisis pushes European inflation higher",
        "Renewable energy investment hits record in Europe",
        "Oil prices rise on Middle East tensions",
    ]
    headline_links = [''] * len(headlines)
    headline_sources = ['Sample'] * len(headlines)

# Limit to 10 headlines
headlines = headlines[:10]
headline_links = headline_links[:10]
headline_sources = headline_sources[:10]

# FinBERT → FinVADER → VADER
finbert_scores, finbert_labels, finbert_success = finbert_analyze(headlines)

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
        nlp_model_name = "FinVADER (fallback — FinBERT API unavailable)"
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

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Avg Sentiment", f"{avg_score:.3f}")
sc2.metric("Positive", f"{n_pos}")
sc3.metric("Negative", f"{n_neg}")
sc4.metric("Neutral", f"{n_neut}")

if is_live:
    st.caption(f"Analyzing {len(sent_df)} live headlines from {len(set(headline_sources))} sources · Model: {nlp_model_name}")
else:
    st.caption(f"Live feeds unavailable — showing sample headlines · Model: {nlp_model_name}")

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
def get_sentiment_trend(rss_query, keywords):
    """Fetch past 30 days of news via Google News RSS and compute daily FinVADER sentiment."""
    from datetime import datetime

    daily_scores = {}

    def score_text(text):
        try:
            from finvader import finvader as _fv_t
            return float(_fv_t(text, use_sentibignomics=True, use_henry=True, indicator='compound'))
        except Exception:
            sia_t = SentimentIntensityAnalyzer()
            return sia_t.polarity_scores(text)['compound']

    for trend_url in [
        f"https://news.google.com/rss/search?q={rss_query}+when:30d&hl=en",
        f"https://news.google.com/rss/search?q=European+energy+{rss_query.split('+')[0]}+when:30d&hl=en",
    ]:
        try:
            feed = feedparser.parse(trend_url)
            for entry in feed.entries[:100]:
                title = entry.title
                if not any(kw.lower() in title.lower() for kw in keywords):
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

trend_keywords = commodity['keywords'] + ['energy', 'Europe', 'European']
trend_df = get_sentiment_trend(commodity['rss_query'], trend_keywords)
avg_30d = None  # initialized here; set inside conditional below

if trend_df is not None and len(trend_df) > 3:
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    today_sent = trend_df[trend_df['Date'] == today_str]
    avg_30d = trend_df['Avg Sentiment'].mean()

    tc1, tc2, tc3 = st.columns(3)
    if len(today_sent) > 0:
        tc1.metric("Today's Avg Sentiment", f"{today_sent['Avg Sentiment'].iloc[0]:.3f}")
        tc2.metric("Today's Headlines", f"{int(today_sent['Headlines Count'].iloc[0])}")
    else:
        tc1.metric("Today's Avg Sentiment", "N/A")
        tc2.metric("Today's Headlines", "0")
    tc3.metric("30-Day Avg Sentiment", f"{avg_30d:.3f}")

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

    st.caption(f"Based on {int(trend_df['Headlines Count'].sum())} headlines over {len(trend_df)} days · Scored with FinVADER (trend) + FinBERT (current)")
else:
    st.info("Not enough historical headline data to generate trend. This improves over time as more news is collected.")

# ─── Section 6c: Anomaly Detection ───
st.subheader("🔍 Anomaly Detection")
st.write("Automated signal monitoring — flags statistical outliers and structural divergences in real time.")

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
if garch_forecast_10d is not None and latest_vol > 0:
    garch_ratio = garch_forecast_10d / latest_vol
    if garch_ratio > 1.30:
        anomalies.append({
            'level': '🟡 WARNING',
            'type': 'GARCH Vol Expansion Signal',
            'detail': (f'GARCH 10-day forecast ({garch_forecast_10d:.1f}%) exceeds current rolling vol '
                       f'({latest_vol:.1f}%) by {(garch_ratio-1)*100:.0f}%. '
                       f'Model projects volatility expansion ahead.'),
        })
    elif garch_ratio < 0.70:
        anomalies.append({
            'level': '🟢 INFO',
            'type': 'GARCH Mean Reversion',
            'detail': (f'GARCH forecast ({garch_forecast_10d:.1f}%) well below current vol '
                       f'({latest_vol:.1f}%). Model projects volatility normalization.'),
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
if avg_score is not None and avg_30d is not None:
    if current_regime in ['Volatile', 'Crisis'] and avg_score > 0.10:
        anomalies.append({
            'level': '🟡 WARNING',
            'type': 'Sentiment–Volatility Divergence',
            'detail': (f'Market regime is **{current_regime}** but current sentiment is positive '
                       f'({avg_score:+.3f}). Possible market complacency — short-term divergence.'),
        })
    if current_regime == 'Calm' and avg_30d is not None and avg_30d < -0.15:
        anomalies.append({
            'level': '🔵 WATCH',
            'type': 'Negative Sentiment Trend vs Calm Vol',
            'detail': (f'30-day sentiment average ({avg_30d:+.3f}) persistently negative '
                       f'despite calm volatility. Sentiment may be a leading indicator.'),
        })

# ── 5. Historical tail — only flag if RECENT extreme loss, not just all-time max ──
# Using all-time max_loss > VaR99*1.5 fires for every asset always (mathematical certainty).
# Instead: check if any daily loss in the last 252 trading days (≈1 year) exceeded VaR99*2.
# This is a genuinely rare event that warrants attention.
recent_returns = returns_clean.iloc[-252:]
recent_min = recent_returns.min() * 100
if abs(recent_min) > abs(var_99) * 2.0:
    anomalies.append({
        'level': '🔵 WATCH',
        'type': 'Recent Tail Event Beyond 2× VaR99',
        'detail': (f'A daily loss of {recent_min:.2f}% occurred in the past 12 months — '
                   f'{abs(recent_min)/abs(var_99):.1f}x the 99% VaR ({var_99:.2f}%). '
                   f'Recent fat-tail risk present; historical VaR may understate exposure.'),
    })
elif abs(returns_clean.min() * 100) > abs(var_99) * 3.0:
    # All-time extreme that is genuinely beyond 3x VaR99 (very rare)
    max_loss = returns_clean.min() * 100
    anomalies.append({
        'level': '🔵 WATCH',
        'type': 'Historical Tail Beyond 3× VaR99',
        'detail': (f'Max observed daily loss ({max_loss:.2f}%) is {abs(max_loss)/abs(var_99):.1f}x '
                   f'the 99% VaR ({var_99:.2f}%). Extreme historical fat-tail present.'),
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

st.caption("Thresholds: Volatility z-score > 1.8σ · GARCH divergence > 30% · Correlation shift > 0.25 · Sentiment-regime divergence · Recent tail: any loss in last 252 days > 2× VaR99")

# ─── Section 6d: AI Risk Narrative (LLM) ───
st.subheader("🤖 AI Risk Interpretation")
st.write(f"Synthesizes today's quantitative signals into a plain-language risk assessment for {selected_commodity}.")

@st.cache_data(ttl=1800, show_spinner="Generating AI risk interpretation...")
def generate_risk_narrative(commodity_name, risk_level_str, _latest_vol, _avg_vol,
                            _var_95, _current_regime, _garch_10d,
                            _avg_score, _avg_30d, anomaly_types_str,
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
    anomaly_line = anomaly_types_str if anomaly_types_str else "None"
    headline_line = top_neg_str if top_neg_str else "None retrieved"
    sentiment_30d_line = (f"{_avg_30d:+.3f}" if _avg_30d is not None else "N/A")

    prompt = f"""You are a senior risk analyst on a European energy trading desk, writing the short risk comment on {commodity_name} for the date shown below.

HOW TO READ THE INPUTS

Volatility
- All volatility figures are DAILY standard deviation of returns, in percent. Not annualised.
- The GARCH figure forecasts the daily volatility on the tenth trading day ahead. It is not a cumulative move over ten days.
- A GARCH expansion flag compares the forecast against CURRENT volatility only. Before calling it stress, check the forecast against the long-run average and against the {regime_calm:.2f}% Volatile boundary. A forecast that stays below both is normalisation back to typical levels, not a build-up of risk, and should be described that way.

Risk signal and regime - one classification
- "Regime" compares current 30-day volatility with FIXED thresholds for this commodity, taken from its full volatility history (not the selected window): Calm below {regime_calm:.2f}% (its 50th percentile), Volatile {regime_calm:.2f}-{regime_crisis:.2f}%, Crisis above {regime_crisis:.2f}% (its 90th percentile).
- "Risk signal" is the same classification under another name: Calm = LOW, Volatile = MEDIUM, Crisis = HIGH RISK. Do not present them as two separate pieces of evidence.
- The long-run average volatility is context only; it does not set the risk signal.

VaR
- VaR 95% is the 5th percentile of the daily return distribution: a loss threshold, given as a negative number.

Sentiment - weak evidence, handle with care
- Computed from at most ten scraped headlines.
- The current reading and the 30-day average come from DIFFERENT models depending on which service was reachable, and are NOT on a common scale. Never compare them numerically and never describe a move from one to the other.
- Anything between -0.05 and +0.05 is NEUTRAL. Do not call it mildly positive or mildly negative.

Anomalies and headlines
- The anomaly field lists only the NAMES of the checks that triggered. You do not have their underlying numbers, so do not invent them.
- The headline field is raw scraped text, truncated and unverified.

MARKET DATA AS OF {date_str}
- Risk signal: {risk_level_str}
- 30-day rolling volatility: {_latest_vol:.2f}% (long-run average {_avg_vol:.2f}%)
- VaR 95%, 1-day: {_var_95:.2f}%
- {garch_line}
- Regime: {_current_regime}
- Sentiment now: {_avg_score:+.3f} | 30-day average: {sentiment_30d_line}
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
    top_neg_for_llm = (
        "FEEDS UNREACHABLE - the headlines behind the sentiment scores are "
        "hardcoded placeholder text, not real news."
    )
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
    _avg_score=avg_score,
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

sent_csv = sent_df.to_csv(index=False)
st.download_button(
    label="Download Sentiment Data (CSV)",
    data=sent_csv,
    file_name="sentiment_data.csv",
    mime="text/csv"
)

export_cols = ['Country', 'Risk Score', 'Structural Score', 'Country Vol Multiplier',
                'Risk Level', dep_col,
                'Total Energy Dep. (%)', 'Carbon Int. (tCO2/M€)',
                'Price Sensitivity', 'Renewable (%)']
seen_e = set()
export_cols = [c for c in export_cols if not (c in seen_e or seen_e.add(c))]
cr_export = cr_df[export_cols].copy()
rename_map = {'Country': 'Country', 'Risk Score': 'Dynamic Risk Score',
              'Structural Score': 'Structural Score', 'Country Vol Multiplier': 'Vol Multiplier',
              'Risk Level': 'Risk Level', dep_col: dep_label,
              'Total Energy Dep. (%)': 'Total Energy Dependency (%)',
              'Carbon Int. (tCO2/M€)': 'Carbon Intensity (tCO2/M€ GDP)',
              'Price Sensitivity': 'Price Sensitivity (1-10)',
              'Renewable (%)': 'Renewable Share (%)'}
cr_export = cr_export.rename(columns=rename_map)
cr_csv = cr_export.to_csv(index=False)
st.download_button(
    label=f"Download Country Risk Data ({selected_year}, CSV)",
    data=cr_csv,
    file_name=f"country_risk_{selected_year}.csv",
    mime="text/csv"
)

