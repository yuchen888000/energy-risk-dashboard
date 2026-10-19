"""Market page: country exposure, headline tone, statistical checks and an AI summary.

Designed with data and analytics firms in mind.
"""
import datetime as dt
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
from country_data import (EU27, MODES, available_years, dependency_key, load_country_data,
                          measure_column, mentions_country)

nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402

ctx = common.context()
selected_commodity = ctx.selected_commodity
commodity = ctx.commodity
compare_label = ctx.compare_label
start_date, end_date = ctx.start_date, ctx.end_date
CARBON_TICKER = ctx.carbon["ticker"]

NEWS_DAYS = 7          # headlines older than this are not "current"
PER_FEED_CAP = 4       # at most this many headlines from one feed, so no single feed dominates
CHECK_DAYS = 252       # base rates of the statistical checks: the last 252 trading days
VOL_LO_PCT, VOL_HI_PCT = 5, 95       # volatility and correlation checks: percentiles of the last 3 years
GARCH_HI, GARCH_LO = 1.30, 0.70      # GARCH forecast vs the 3-year average 30-day volatility
TAIL_MULT = 2.0                      # tail check: loss beyond 2x the 3-year 99% VaR
ROLLING_FUTURES = ('TTF=F', 'CL=F', 'BZ=F')
# Tone colours carry no good/bad meaning: positive wording is not good news for a price.
TONE_COLORS = {'Positive': 'steelblue', 'Negative': 'darkorange', 'Neutral': 'lightgray'}

with st.sidebar.expander("Methodology: Market page"):
    st.markdown("""
    - **Country exposure**: Eurostat data only (nrg_ind_id, nrg_bal_c, nrg_ind_ren,
      env_air_gge, nama_10_gdp), no hand-entered value and no composite score. For gas
      and oil, countries are ranked by net imports of that fuel as a share of all their
      energy: the fuel's share of gross available energy times its import dependency.
      For carbon, by greenhouse-gas emissions per euro of GDP at 2020 prices. Storage,
      supplier concentration and LNG capacity are not included.
    - **Headline tone**: FinBERT (ProsusAI/finbert via Hugging Face) on headlines from the
      last 7 days, at most 4 per feed, duplicates removed and the publisher name stripped
      before scoring. FinVADER is the fallback: VADER with the SentiBigNomics and Henry
      financial word lists, which per its authors scores better than VADER on Financial
      PhraseBank. Both are trained on company news: the score is the tone of the wording,
      not a price direction.
    - **30-day tone trend**: Google News headlines from the last 30 days, scored with
      FinVADER; the average is over all headlines, with the daily count shown.
    - **Statistical checks**: volatility above the 95th or below the 5th percentile of the
      last 3 years, GARCH 10-day forecast 30% above or below the 3-year average volatility,
      30-day correlation outside its 3-year 5th to 95th percentile range, and daily losses
      beyond 2x the 3-year 99% VaR. Each check shows on how many of the last 252 trading
      days it fired.
    - **AI interpretation** (optional): the statistics on this page summarised in three
      sentences by Claude via the Anthropic API. It sees no fundamentals.
    """)

st.title("Market Intelligence")
st.markdown(f"Country exposure, headline tone and statistical checks for **{selected_commodity}**")

core = common.compute_core(commodity['ticker'], ctx.compare_ticker, start_date, end_date)
if core is None:
    st.warning("Please select a longer time range (at least 30 days of data required).")
    st.stop()

df_analysis = core['df_analysis']
latest_vol, avg_vol = core['latest_vol'], core['avg_vol']
returns_clean = core['returns_clean']
var_95 = core['var_95']              # last 250 trading days
risk_level = core['risk_level']
regime_thr, current_regime = core['regime_thr'], core['current_regime']
features = common.compute_regimes(df_analysis[['Volatility', 'Rolling Correlation']], regime_thr)

garch = common.fit_garch(returns_clean)
garch_forecast_10d = garch['forecast_10d'] if garch is not None else None
# Reference level for the GARCH check: the 3-year average 30-day volatility, the same window
# as the regime thresholds and the volatility check, so it does not move with the sidebar dates.
garch_ref = regime_thr['mean']
garch_ref_name = "3-year average 30-day volatility"

# ─── FinBERT via HuggingFace Inference API ───
# Used by the country headlines and the headline tone section.
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
    """FinVADER fallback: VADER + SentiBigNomics + Henry financial lexicons."""
    try:
        from finvader import finvader
        return float(finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound'))
    except Exception:
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)['compound']


# ─── Headline relevance ───
def _word_pattern(words, flags=re.IGNORECASE):
    """Whole-word match for any of `words`, allowing a plural ending ("emission" to "emissions").
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



def published_at(entry):
    """Publication time of a feed entry in UTC, or None when the feed gives none."""
    t = entry.get('published_parsed') or entry.get('updated_parsed')
    return pd.Timestamp(dt.datetime(*t[:6]), tz='UTC') if t else None


def dedup_key(text):
    """Headline reduced to lower-case words, so the same story from two feeds counts once."""
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def tone_label(score):
    if score > 0.05:
        return 'Positive'
    if score < -0.05:
        return 'Negative'
    return 'Neutral'


def score_headlines(texts):
    """(scores, labels, model name, FinBERT ok, reason) for a list of headline texts.
    FinBERT first, then FinVADER, then plain VADER."""
    scores, labels, ok, reason = finbert_analyze(tuple(texts))
    if ok:
        return scores, labels, "FinBERT", True, ""
    try:
        from finvader import finvader as _fv
        scores = [float(_fv(h, use_sentibignomics=True, use_henry=True, indicator='compound'))
                  for h in texts]
        model = "FinVADER"
    except Exception:
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(h)['compound'] for h in texts]
        model = "VADER"
    return scores, [tone_label(s) for s in scores], model, False, reason


# ─── Section 1: European country exposure ───
country_mode = dependency_key(selected_commodity)
M = MODES[country_mode]
mcol = measure_column(country_mode)
country_df, country_failed, country_updated = load_country_data()
country_years = available_years(country_df, country_mode)

st.subheader("European Country Exposure")
st.write(f"**{M['question']}**")
if country_failed:
    st.warning("Eurostat could not be reached for: " + ", ".join(country_failed)
               + ". Countries without those inputs are not ranked.")

cr_df = pd.DataFrame()
if not country_years:
    st.warning("Eurostat has no complete year for this measure right now, so the country table is not shown.")
else:
    if len(country_years) > 1:
        selected_year = st.slider("Select Year", min_value=country_years[0], max_value=country_years[-1],
                                  value=country_years[-1], step=1)
    else:
        selected_year = country_years[0]
    st.caption(f"Years {country_years[0]} to {country_years[-1]}: every year from {country_years[0]} for which "
               "Eurostat publishes the EU-27 value of this measure. Last Eurostat updates: "
               + ", ".join(f"{k} {v}" for k, v in country_updated.items() if v) + ".")

    yr_df = country_df[country_df['Year'] == selected_year]
    eu = yr_df[yr_df['Country'] == EU27].iloc[0]
    countries_yr = yr_df[yr_df['Country'] != EU27]
    ranked = countries_yr[countries_yr[mcol].notna()].sort_values(mcol, ascending=False).reset_index(drop=True)
    not_ranked = countries_yr.loc[countries_yr[mcol].isna(), 'Country'].tolist()
    n_ranked = len(ranked)

    if country_mode in ('gas', 'oil'):
        fuel = country_mode
        cols = {
            'Country': ranked['Country'],
            M['measure']: ranked[f'{fuel}_exposure'],
            M['share']: ranked[f'{fuel}_share'],
            M['dep']: ranked[f'dep_{fuel}'],
            'Total energy import dependency (%)': ranked['dep_total'],
            'Renewable share (%)': ranked['ren'],
        }
        st.markdown(
            f"Ranked by **net {fuel} imports as a share of all energy** = {fuel} share of gross available "
            f"energy × {fuel} import dependency. EU-27 {selected_year}: {fuel} is "
            f"**{eu[f'{fuel}_share']:.1f}%** of gross available energy, **{eu[f'dep_{fuel}']:.1f}%** of it "
            f"is imported (net), so net {fuel} imports are **{eu[f'{fuel}_exposure']:.1f}%** of all energy.")
        st.caption(
            f"Gross available energy is all energy a country has available (production + imports - exports "
            f"+ stock changes), from Eurostat nrg_bal_c in ktoe; import dependency is Eurostat nrg_ind_id "
            f"(net imports ÷ gross available {M['fuel']}). The product tells how much of a country's total "
            f"energy supply is {fuel} bought abroad, so a country that imports all its {fuel} but uses little "
            f"of it ranks low. Dependency is shown as published: below 0% means a net exporter, above 100% "
            f"means imports also refilled stocks. In the product a net exporter counts as 0.")
        if fuel == 'gas':
            st.caption("**Not included:** gas storage levels (EU filling targets under Regulation (EU) 2022/1032 as amended), how "
                       "concentrated imports are by supplier, LNG import capacity, pipeline routes, and how "
                       "gas use splits between power, industry and heating. A country can rank high here and "
                       "still be well protected by storage or diverse suppliers.")
        else:
            st.caption("**Not included:** emergency oil stocks (the IEA 90-day obligation), supplier "
                       "concentration and refinery capacity. Gross available energy includes fuel for "
                       "international shipping (maritime bunkers), which raises the oil share of bunkering "
                       "hubs such as Malta, the Netherlands and Belgium.")
    else:
        cols = {
            'Country': ranked['Country'],
            M['measure']: ranked['ghg_int'],
            'Renewable share (%)': ranked['ren'],
        }
        st.markdown(
            f"Ranked by **greenhouse-gas emissions per million euro of GDP at 2020 prices**. "
            f"EU-27 {selected_year}: **{eu['ghg_int']:.0f} tCO2e per M€**.")
        st.caption(
            "What it measures: total greenhouse-gas emissions excluding land use (Eurostat env_air_gge, "
            "TOTX4_MEMO) divided by GDP in chain-linked volumes (nama_10_gdp, CLV20_MEUR), so a fall over "
            "time is a fall in emissions per unit of real output, not an effect of inflation. Levels across "
            "countries are converted at 2020 exchange rates, not purchasing power, which makes lower-income "
            "economies look more intensive. Renewable share of gross final energy consumption: nrg_ind_ren. "
            "**Not included:** which emissions are covered by the EU ETS and so priced by EUAs (power, industry, "
            "aviation, maritime), free allocation and carbon leakage. A broad indicator of how carbon-intensive "
            "an economy is, not the cost of EUAs to it.")

    cr_df = pd.DataFrame(cols)
    cr_df.index = cr_df.index + 1   # rank
    cr_df.index.name = 'Rank'
    if not_ranked:
        st.caption(f"Not ranked for {selected_year} (Eurostat has no value for at least one input of the "
                   "measure): " + ", ".join(not_ranked) + ".")

    if cr_df.empty:
        st.warning("No country has the Eurostat inputs for this year.")
    else:
        cr_col1, cr_col2 = st.columns([2, 1])
        _pct = lambda: st.column_config.NumberColumn(format="%.1f")
        with cr_col1:
            st.write(f"**{M['measure']}, {selected_year}: all {n_ranked} countries with data**")
            col_cfg = {c: _pct() for c in cr_df.columns if c != 'Country'}
            col_cfg[M['measure']] = st.column_config.ProgressColumn(
                M['measure'], format="%.1f", min_value=0.0, max_value=float(cr_df[M['measure']].max()))
            # Tall enough to show every country without scrolling (35 px per row plus the header).
            st.dataframe(cr_df, width="stretch", height=(len(cr_df) + 1) * 35 + 3, column_config=col_cfg)

        with cr_col2:
            # Alphabetical options and a default taken from the latest year, so moving the year
            # slider does not reset the chosen country.
            _options = sorted(cr_df['Country'])
            _latest = country_df[(country_df['Year'] == country_years[-1]) & (country_df['Country'] != EU27)]
            _default = _latest.sort_values(mcol, ascending=False)['Country'].iloc[0]
            selected_country = st.selectbox("Select Country for Detail", _options,
                                            index=_options.index(_default) if _default in _options else 0)
            row = ranked[ranked['Country'] == selected_country].iloc[0]
            rank = int(cr_df.index[cr_df['Country'] == selected_country][0])
            st.markdown(f"### {selected_country} ({selected_year})")
            st.metric(M['measure'], f"{row[mcol]:.1f}", help=f"Rank {rank} of {n_ranked}; EU-27 "
                      f"{eu[mcol]:.1f}.")
            st.caption(f"Rank {rank} of {n_ranked} countries with data.")

            def _fmt(v, f="{:.1f}"):
                return "n/a" if v is None or pd.isna(v) else f.format(v)
            if country_mode in ('gas', 'oil'):
                fuel = country_mode
                detail_rows = [
                    (M['share'], _fmt(row[f'{fuel}_share'])),
                    (M['dep'], _fmt(row[f'dep_{fuel}'])),
                    (f"Gross available {fuel} (ktoe)", _fmt(row[f'gae_{fuel}'], "{:,.0f}")),
                    ("Gross available energy, all (ktoe)", _fmt(row['gae_total'], "{:,.0f}")),
                    ("Total energy import dependency (%)", _fmt(row['dep_total'])),
                    ("Renewable share (%)", _fmt(row['ren'])),
                ]
            else:
                detail_rows = [
                    ("GHG emissions excl. LULUCF (kt CO2e)", _fmt(row['ghg'], "{:,.0f}")),
                    ("GDP, 2020 prices (M€)", _fmt(row['gdp_clv'], "{:,.0f}")),
                    ("Renewable share (%)", _fmt(row['ren'])),
                ]
            # A two-column table wraps inside the narrow panel, where side-by-side metrics overlap.
            st.markdown("| Eurostat input | Value |\n|---|---:|\n"
                        + "\n".join(f"| {k} | {v} |" for k, v in detail_rows))

        # Bar chart of every ranked country, shaded by the measure itself.
        fig_cr, ax_cr = plt.subplots(figsize=(14, max(4, 0.28 * n_ranked + 1)))
        vals = cr_df[M['measure']].astype(float)
        shade = plt.cm.Blues(0.3 + 0.7 * vals / vals.max()) if vals.max() > 0 else 'steelblue'
        ax_cr.barh(range(n_ranked), vals, color=shade, height=0.7)
        ax_cr.set_yticks(range(n_ranked))
        ax_cr.set_yticklabels(cr_df['Country'], fontsize=9)
        ax_cr.axvline(x=eu[mcol], color='black', linewidth=1, linestyle='--', alpha=0.6,
                      label=f'EU-27: {eu[mcol]:.1f}')
        ax_cr.set_xlabel(M['measure'])
        ax_cr.set_title(f'{M["measure"]}, {selected_year} (Eurostat, all {n_ranked} countries with data)')
        ax_cr.legend(fontsize=8, loc='lower right')
        ax_cr.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_cr)

        # The same Eurostat inputs for the selected country over every available year.
        trend = (country_df[(country_df['Country'] == selected_country)
                            & country_df['Year'].isin(country_years)]
                 .sort_values('Year').set_index('Year'))
        st.write(f"**{selected_country}: inputs over time, {country_years[0]} to {country_years[-1]}**")
        fig_tcr, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(14, 3.5))
        if country_mode in ('gas', 'oil'):
            fuel = country_mode
            ax_t1.plot(trend.index, trend[f'{fuel}_share'], 'o-', color='steelblue', label=M['share'])
            ax_t1.plot(trend.index, trend[f'{fuel}_exposure'], 's-', color='navy', label=M['measure'])
            ax_t1.set_ylabel('% of gross available energy')
            ax_t1.set_title(f'{selected_country}: {fuel} in the energy mix')
            ax_t1.legend(fontsize=8)
            ax_t2.plot(trend.index, trend[f'dep_{fuel}'], 'o-', color='gray', label=M['dep'])
            ax_t2.axhline(100, color='black', linewidth=0.5, linestyle=':')
            ax_t2.set_ylabel(f'% of gross available {fuel}')
            ax_t2.set_title(f'{selected_country}: {fuel} import dependency (as published)')
        else:
            ax_t1.bar(trend.index, trend['ghg_int'], color='gray', alpha=0.7)
            ax_t1.set_ylabel('tCO2e per M€ GDP (2020 prices)')
            ax_t1.set_title(f'{selected_country}: greenhouse-gas intensity of GDP')
            ax_t2.plot(trend.index, trend['ren'], 'o-', color='steelblue')
            ax_t2.set_ylabel('% of gross final energy consumption')
            ax_t2.set_title(f'{selected_country}: renewable share')
        for ax in (ax_t1, ax_t2):
            ax.set_xticks(country_years)
        plt.tight_layout()
        st.pyplot(fig_tcr)

        # Per-country news tone: same models and caveats as the headline tone section below.
        st.write(f"**{selected_country}: tone of recent energy headlines**")
        _country_q = quote_plus(f"{selected_country} energy {commodity['news_term']}")
        country_rss_url = f"https://news.google.com/rss/search?q={_country_q}+when:{NEWS_DAYS}d&hl=en"
        country_rows, _seen_c = [], set()
        try:
            country_feed = feedparser.parse(country_rss_url)
            _energy_re = _word_pattern(commodity['keywords'] + ['energy', 'oil', 'gas', 'carbon', 'power',
                                                                'fuel', 'electricity', 'pipeline', 'LNG',
                                                                'emission', 'climate', 'price', 'supply',
                                                                'tanker', 'refinery', 'fossil', 'renewable',
                                                                'heating', 'Hormuz', 'sanction', 'ETS'])
            for entry in country_feed.entries[:50]:
                publisher = entry.get('source', {}).get('title', '')
                # Google News appends " - Publisher": match and score the headline only, so an
                # outlet name such as "Irish Times" neither counts as a mention nor adds tone.
                headline = headline_text(entry.title, publisher)
                if dedup_key(headline) in _seen_c:
                    continue
                if _energy_re.search(headline) and mentions_country(headline, selected_country):
                    country_rows.append((headline, publisher))
                    _seen_c.add(dedup_key(headline))
                if len(country_rows) >= 5:
                    break
        except Exception:
            pass

        if country_rows:
            c_texts = [h for h, _ in country_rows]
            c_scores, _, c_model, _, _ = score_headlines(c_texts)
            c_avg = float(np.mean(c_scores))
            st.markdown(f"Average tone **{tone_label(c_avg).lower()} ({c_avg:+.3f})** over "
                        f"{len(c_texts)} headlines from the last {NEWS_DAYS} days · {c_model}. "
                        "Tone of the wording, not a price direction.")
            for (h, pub), sc in zip(country_rows, c_scores):
                st.markdown(f"**[{sc:+.3f}]** {h}" + (f" · *{pub}*" if pub else ""))
        else:
            st.info(f"No energy headline from the last {NEWS_DAYS} days mentions {selected_country} by name or adjective.")


# ─── Section 2: Headline tone ───
st.subheader("Headline Tone")
st.write(f"How positive or negative the wording of recent {selected_commodity} headlines is.")

_when = f"+when:{NEWS_DAYS}d"
rss_feeds = {
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "OilPrice": "https://oilprice.com/rss/main",
    f"Google ({selected_commodity})": f"https://news.google.com/rss/search?q={commodity['rss_query']}{_when}&hl=en",
    "Google EU Energy": f"https://news.google.com/rss/search?q=European+energy+market{_when}&hl=en",
    "Google EU Carbon": f"https://news.google.com/rss/search?q=EU+carbon+ETS{_when}&hl=en",
}
_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
_cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=NEWS_DAYS)

news_rows, _seen = [], set()
for feed_name, url in rss_feeds.items():
    try:
        resp = req.get(url, headers=_HEADERS, timeout=15)
        feed = feedparser.parse(resp.content)
    except Exception:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
    count = 0
    for entry in feed.entries[:100]:
        if count >= PER_FEED_CAP:
            break
        try:
            publisher = entry.get('source', {}).get('title', '')
            # Score the headline only: the " - Publisher" suffix has a tone of its own.
            text = headline_text(entry.title.strip(), publisher)
            key = dedup_key(text)
            if not key or key in _seen:
                continue
            pub = published_at(entry)
            if pub is None or pub < _cutoff:
                continue
            # Same test for every feed: a commodity keyword in the headline, no US-domestic story.
            if not is_relevant(text, commodity_kw_re):
                continue
            news_rows.append({'Headline': text, 'Publisher': publisher or feed_name.split(' (')[0],
                              'Feed': feed_name, 'Published (UTC)': pub.strftime('%Y-%m-%d %H:%M'),
                              'Link': entry.get('link', '')})
            _seen.add(key)
            count += 1
        except Exception:
            continue

# No placeholder headlines: if every feed fails, no tone is computed at all.
is_live = bool(news_rows)
if not is_live:
    finbert_success, finbert_reason = False, "no headlines"
    nlp_model_name = "none"
    sent_df = pd.DataFrame(columns=['Headline', 'Publisher', 'Feed', 'Published (UTC)', 'Link', 'Score', 'Label'])
    avg_score = None
    st.info(f"No relevant headline from the last {NEWS_DAYS} days could be retrieved right now, so no tone "
            "is shown. The page does not score placeholder text.")
else:
    sent_df = pd.DataFrame(news_rows)
    _scores, _labels, nlp_model_name, finbert_success, finbert_reason = score_headlines(sent_df['Headline'].tolist())
    sent_df['Score'], sent_df['Label'] = _scores, _labels
    sent_df = sent_df.sort_values('Published (UTC)', ascending=False).reset_index(drop=True)
    avg_score = float(sent_df['Score'].mean())
    n_pos, n_neg, n_neut = ((sent_df['Label'] == lab).sum() for lab in ('Positive', 'Negative', 'Neutral'))

    st.markdown(f"#### Average tone: {tone_label(avg_score)} ({avg_score:+.3f})")
    st.caption("The score is the tone of the wording, from models trained on company news (FinBERT on "
               "Financial PhraseBank, where annotators judged news for a company's share price; FinVADER is a "
               "finance word list). It is not a price direction and not bullish or bearish: \"gas prices "
               "surge\" can score negative although it describes a price rise. Between -0.05 and +0.05 is neutral.")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(f"Average tone ({nlp_model_name})", f"{avg_score:+.3f}",
               help="Simple average over all headlines below.")
    sc2.metric("Positive wording", f"{n_pos}")
    sc3.metric("Negative wording", f"{n_neg}")
    sc4.metric("Neutral", f"{n_neut}")

    _per_feed = sent_df['Feed'].value_counts()
    st.caption(f"{len(sent_df)} headlines from the last {NEWS_DAYS} days, at most {PER_FEED_CAP} per feed, "
               "duplicates across feeds removed: " + ", ".join(f"{k} {v}" for k, v in _per_feed.items())
               + f" · Model: {nlp_model_name}")
    if not finbert_success:
        st.caption(f"FinBERT not used: {finbert_reason}")

    fig3, ax4 = plt.subplots(figsize=(12, max(3, len(sent_df) * 0.3)))
    ax4.barh(range(len(sent_df)), sent_df['Score'], color=[TONE_COLORS[l] for l in sent_df['Label']], height=0.6)
    ax4.set_yticks(range(len(sent_df)))
    ax4.set_yticklabels([h[:60] + '...' if len(h) > 60 else h for h in sent_df['Headline']], fontsize=7)
    ax4.axvline(x=0, color='black', linewidth=0.5)
    for x in (0.05, -0.05):
        ax4.axvline(x=x, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax4.set_xlabel(f'Tone score ({nlp_model_name}), newest headline at the top')
    ax4.set_title('Tone of each headline')
    ax4.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig3)

    top_pos = sent_df[sent_df['Score'] > 0.05].nlargest(3, 'Score')
    top_neg = sent_df[sent_df['Score'] < -0.05].nsmallest(3, 'Score')

    def _show(rows):
        for _, r in rows.iterrows():
            head = f"[{r['Headline']}]({r['Link']})" if r['Link'] else r['Headline']
            st.markdown(f"**[{r['Score']:+.3f}]** {head} · *{r['Publisher']}, {r['Published (UTC)'][:10]}*")
    if not top_pos.empty:
        st.write("**Most positive wording:**")
        _show(top_pos)
    if not top_neg.empty:
        st.write("**Most negative wording:**")
        _show(top_neg)
    if top_pos.empty and top_neg.empty:
        st.info("All current headlines score as neutral wording.")

# ─── Section 2b: 30-day tone trend ───
st.subheader("Headline Tone, Last 30 Days")
st.write(f"Daily tone of {selected_commodity}-related European energy headlines over the past 30 days")


@st.cache_data(ttl=7200, show_spinner="Fetching 30-day news history...")
def get_sentiment_trend(rss_query, news_term, keywords):
    """Headlines of the last 30 days from two Google News queries, de-duplicated across the
    queries, publisher suffix removed, each scored with FinVADER. One row per headline."""
    keyword_re = _word_pattern(keywords)
    rows, seen = [], set()
    for trend_url in [
        f"https://news.google.com/rss/search?q={rss_query}+when:30d&hl=en",
        f"https://news.google.com/rss/search?q=European+energy+{news_term}+when:30d&hl=en",
    ]:
        try:
            feed = feedparser.parse(trend_url)
        except Exception:
            continue
        for entry in feed.entries[:100]:
            source = entry.get('source', {}).get('title', '')
            text = headline_text(entry.get('title', ''), source)
            key = dedup_key(text)
            if not key or key in seen or not is_relevant(text, keyword_re):
                continue
            pub = published_at(entry)
            if pub is None:
                continue
            rows.append({'Date': pub.tz_convert(None).normalize(), 'Headline': text,
                         'Publisher': source, 'Score': finvader_score(text)})
            seen.add(key)
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)


trend_hl = get_sentiment_trend(commodity['rss_query'], commodity['news_term'], tuple(commodity['keywords']))
avg_30d = None

if trend_hl is not None and trend_hl['Date'].nunique() > 3:
    daily = trend_hl.groupby('Date')['Score'].agg(['mean', 'size'])
    avg_30d = float(trend_hl['Score'].mean())   # every headline counts once
    today_ts = pd.Timestamp.now(tz='UTC').tz_convert(None).normalize()
    tc1, tc2, tc3 = st.columns(3)
    if today_ts in daily.index:
        tc1.metric("Today's average tone (FinVADER)", f"{daily.loc[today_ts, 'mean']:+.3f}")
        tc2.metric("Today's headlines", f"{int(daily.loc[today_ts, 'size'])}")
    else:
        tc1.metric("Today's average tone (FinVADER)", "N/A")
        tc2.metric("Today's headlines", "0")
    tc3.metric("30-day average tone (FinVADER)", f"{avg_30d:+.3f}",
               help=f"Average over all {len(trend_hl)} headlines, not an average of daily averages.")

    fig_trend, (ax_trend, ax_cnt) = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                                                 gridspec_kw={'height_ratios': [3, 1]})
    ax_trend.bar(daily.index, daily['mean'], color=[TONE_COLORS[tone_label(s)] for s in daily['mean']],
                 width=0.8)
    ax_trend.axhline(y=0, color='black', linewidth=0.5)
    ax_trend.axhline(y=avg_30d, color='navy', linewidth=1, linestyle='--', alpha=0.6,
                     label=f'30-day average over all headlines: {avg_30d:+.3f}')
    ax_trend.set_ylabel('Daily average tone')
    ax_trend.set_title(f'{selected_commodity}: headline tone per day (FinVADER)')
    ax_trend.legend(fontsize=8)
    ax_cnt.bar(daily.index, daily['size'], color='gray', width=0.8)
    ax_cnt.set_ylabel('Headlines')
    ax_cnt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_trend)

    st.caption(f"{len(trend_hl)} distinct headlines over {len(daily)} days. A day with one headline is as noisy "
               "as one headline: read the bars with the count below them. Scored with FinVADER only; the current "
               "reading above may come from FinBERT, which uses a different scale, so the two are not compared.")
else:
    st.info("Not enough headlines from the last 30 days to draw a trend.")


# ─── Section 3: Statistical checks ───
st.subheader("Statistical Checks")
st.write("Is anything statistically unusual today compared with this commodity's own last 3 years? "
         "Each check shows how often it fired over the last year, so you can judge how rare a flag is.")


@st.cache_data(ttl=3600, show_spinner="Computing 3-year reference distributions...")
def recent_reference(ticker, compare_ticker):
    """30-day volatility, 30-day correlation and daily returns over the last 3 years up to today.

    Same window as common.regime_thresholds (common.REGIME_WINDOW trading days) and the same
    definitions as common.compute_core, so the volatility mean matches regime_thr['mean'].
    """
    today = dt.date.today()
    start = today - dt.timedelta(days=int(common.REGIME_WINDOW * 365 / 250) + 90)
    price = common.load_close(ticker, start, today)
    compare = common.load_close(compare_ticker, start, today)
    df = pd.DataFrame({'Price': price, 'R': common.simple_returns(price),
                       'CR': common.simple_returns(compare)}).dropna(subset=['Price'])
    df = df[df['Price'] > 0]
    vol = (df['R'].rolling(30, min_periods=20).std() * 100).dropna().tail(common.REGIME_WINDOW)
    corr = df['R'].rolling(30, min_periods=20).corr(df['CR']).dropna().tail(common.REGIME_WINDOW)
    rets = df['R'].dropna()
    if len(vol):
        rets = rets.loc[vol.index[0]:]
    return dict(vol=vol, corr=corr, rets=rets)


def garch_10d_path(g):
    """The 10-day-ahead GARCH volatility forecast as it stood at the end of each day, from the
    fitted parameters: sigma2(t+10) = omega * (1 + p + ... + p^8) + p^9 * sigma2(t+1), p = alpha + beta.
    sigma(t+1) is the model's conditional volatility for the next day; for the last day it is
    the one-day-ahead forecast. The last value equals the page's forecast."""
    p = g['params']
    pers = p['alpha[1]'] + p['beta[1]']
    nxt = g['conditional_volatility'].shift(-1) ** 2
    nxt.iloc[-1] = float(g['forecast_vol'][0]) ** 2
    const = p['omega'] * sum(pers ** i for i in range(9))
    return np.sqrt(const + pers ** 9 * nxt)


ref = recent_reference(commodity['ticker'], ctx.compare_ticker)
if len(ref['vol']) < 250:
    # Same fallback as the regime thresholds: the selected range when recent history is short.
    ref = dict(vol=df_analysis['Volatility'].dropna(), corr=df_analysis['Rolling Correlation'].dropna(),
               rets=returns_clean)
    ref_source = "selected date range (recent history unavailable)"
else:
    ref_source = f"{ref['vol'].index[0]:%d %b %Y} to {ref['vol'].index[-1]:%d %b %Y}, {len(ref['vol'])} trading days"

vol_lo, vol_hi = np.percentile(ref['vol'], [VOL_LO_PCT, VOL_HI_PCT])
vol_pctile = float((ref['vol'] < latest_vol).mean() * 100)
vol_last = ref['vol'].tail(CHECK_DAYS)
corr_now_s = df_analysis['Rolling Correlation'].dropna()
corr_now = float(corr_now_s.iloc[-1]) if len(corr_now_s) else None
has_corr = corr_now is not None and len(ref['corr']) >= 250
if has_corr:
    corr_lo, corr_med, corr_hi = np.percentile(ref['corr'], [VOL_LO_PCT, 50, VOL_HI_PCT])
    corr_last = ref['corr'].tail(CHECK_DAYS)
var99_3y = float(np.percentile(ref['rets'], 1) * 100)
tail_thr = TAIL_MULT * abs(var99_3y)
recent_rets = returns_clean.iloc[-CHECK_DAYS:] * 100
tail_days = recent_rets[recent_rets < -tail_thr]
roll_note = (" On a continuous front-month future a contract roll day can produce such a move without a "
             "real price change." if commodity['ticker'] in ROLLING_FUTURES else "")

checks = []   # one row per check for the table
anomalies = []

# 1. Volatility spike / unusual calm: percentiles of the last 3 years.
n_hi = int((vol_last > vol_hi).sum())
n_lo = int((vol_last < vol_lo).sum())
checks.append(("Volatility spike", f"30-day vol above {vol_hi:.2f}% ({VOL_HI_PCT}th pct)",
               f"{latest_vol:.2f}%", latest_vol > vol_hi, f"{n_hi} of {len(vol_last)}"))
checks.append(("Unusual calm", f"30-day vol below {vol_lo:.2f}% ({VOL_LO_PCT}th pct)",
               f"{latest_vol:.2f}%", latest_vol < vol_lo, f"{n_lo} of {len(vol_last)}"))
if latest_vol > vol_hi:
    anomalies.append({'level': '🟡 WARNING', 'type': 'Volatility Spike',
                      'detail': (f'30-day volatility of daily returns is {latest_vol:.2f}%, above the '
                                 f'{VOL_HI_PCT}th percentile of the last 3 years ({vol_hi:.2f}%); it sits at '
                                 f'the {vol_pctile:.0f}th percentile. Fired on {n_hi} of the last {len(vol_last)} '
                                 'trading days.')})
elif latest_vol < vol_lo:
    anomalies.append({'level': '🔵 WATCH', 'type': 'Unusual Calm',
                      'detail': (f'30-day volatility of daily returns is {latest_vol:.2f}%, below the '
                                 f'{VOL_LO_PCT}th percentile of the last 3 years ({vol_lo:.2f}%). Low volatility '
                                 'describes recent price moves only; it does not mean low risk. Fired on '
                                 f'{n_lo} of the last {len(vol_last)} trading days.')})

# 2. GARCH 10-day forecast vs the 3-year average 30-day volatility.
if garch is not None and garch_ref:
    g_path = garch_10d_path(garch).tail(CHECK_DAYS)
    g_ratio = garch_forecast_10d / garch_ref
    n_ghi = int((g_path > GARCH_HI * garch_ref).sum())
    n_glo = int((g_path < GARCH_LO * garch_ref).sum())
    checks.append(("GARCH forecast high", f"10-day forecast above {GARCH_HI * garch_ref:.2f}% "
                   f"({GARCH_HI:.1f}x 3-year mean)", f"{garch_forecast_10d:.2f}%", g_ratio > GARCH_HI, f"{n_ghi} of {len(g_path)}"))
    checks.append(("GARCH forecast low", f"10-day forecast below {GARCH_LO * garch_ref:.2f}% "
                   f"({GARCH_LO:.1f}x 3-year mean)", f"{garch_forecast_10d:.2f}%", g_ratio < GARCH_LO, f"{n_glo} of {len(g_path)}"))
    if g_ratio > GARCH_HI:
        anomalies.append({'level': '🟡 WARNING', 'type': 'GARCH Forecast Above 3-Year Average',
                          'detail': (f'GARCH forecast of daily volatility on the 10th trading day ahead is '
                                     f'{garch_forecast_10d:.2f}%, {(g_ratio - 1) * 100:.0f}% above the {garch_ref_name} '
                                     f'({garch_ref:.2f}%). Fired on {n_ghi} of the last {len(g_path)} trading days.')})
    elif g_ratio < GARCH_LO:
        anomalies.append({'level': '🟢 INFO', 'type': 'GARCH Forecast Below 3-Year Average',
                          'detail': (f'GARCH forecast of daily volatility on the 10th trading day ahead is '
                                     f'{garch_forecast_10d:.2f}%, {(1 - g_ratio) * 100:.0f}% below the {garch_ref_name} '
                                     f'({garch_ref:.2f}%). Fired on {n_glo} of the last {len(g_path)} trading days.')})

# 3. Correlation outside its own 3-year range.
if has_corr:
    corr_out = corr_now > corr_hi or corr_now < corr_lo
    n_corr = int(((corr_last > corr_hi) | (corr_last < corr_lo)).sum())
    checks.append(("Correlation shift", f"30-day corr outside {corr_lo:.2f} to {corr_hi:.2f} "
                   f"({VOL_LO_PCT}th to {VOL_HI_PCT}th pct)", f"{corr_now:.2f}", corr_out, f"{n_corr} of {len(corr_last)}"))
    if corr_out:
        anomalies.append({'level': '🔵 WATCH', 'type': 'Correlation Outside 3-Year Range',
                          'detail': (f'30-day correlation of daily returns with {compare_label} is {corr_now:.2f}, '
                                     f'outside its 3-year {VOL_LO_PCT}th to {VOL_HI_PCT}th percentile range '
                                     f'({corr_lo:.2f} to {corr_hi:.2f}; median {corr_med:.2f}). Fired on {n_corr} '
                                     f'of the last {len(corr_last)} trading days.')})

# 4. Large daily losses in the last 252 trading days.
checks.append(("Large daily loss", f"loss beyond {tail_thr:.2f}% ({TAIL_MULT:.0f}x 3-year VaR99) in the last "
               f"{len(recent_rets)} days", f"worst {recent_rets.min():.2f}%", len(tail_days) > 0,
               f"{len(tail_days)} of {len(recent_rets)}"))
if len(tail_days):
    anomalies.append({'level': '🔵 WATCH', 'type': f'Daily Loss Beyond {TAIL_MULT:.0f}x 3-Year VaR99',
                      'detail': (f'{len(tail_days)} daily loss(es) in the last {len(recent_rets)} trading days '
                                 f'beyond {tail_thr:.2f}% ({TAIL_MULT:.0f}x the 3-year 99% VaR of {var99_3y:.2f}%); '
                                 f'the largest was {tail_days.min():.2f}% on {tail_days.idxmin():%d %b %Y}.'
                                 + roll_note)})

check_tbl = pd.DataFrame(checks, columns=['Check', 'Fires when', 'Latest', 'Fires now',
                                          f'Days fired (of the last {CHECK_DAYS} trading days, or fewer if not available)'])
check_tbl['Fires now'] = check_tbl['Fires now'].map({True: 'Yes', False: 'No'})
st.dataframe(check_tbl, hide_index=True, width="stretch")

if anomalies:
    for a in anomalies:
        level = a['level']
        if 'WARNING' in level:
            bg, border = '#fffbe6', '#ffaa00'
        elif 'WATCH' in level:
            bg, border = '#f0f4ff', '#4488ff'
        else:
            bg, border = '#f4f4f4', '#999999'
        st.markdown(
            f"<div style='background:{bg}; border-left:4px solid {border}; color:#222; "
            f"padding:10px 14px; margin:6px 0; border-radius:4px;'>"
            f"<strong>{a['level']}: {a['type']}</strong><br>"
            f"<span style='font-size:0.9em'>{a['detail']}</span></div>",
            unsafe_allow_html=True
        )
else:
    st.info(f"None of the {len(checks)} checks fires on the latest data. They cover only volatility, the GARCH "
            "forecast, one correlation and daily losses; they say nothing about supply, storage, positions or news.")

st.caption(
    f"Reference window: {ref_source}, the same 3 years as the regime thresholds. Thresholds and the counts of "
    "days fired come from the same window, so the counts are in-sample: about 5% of the 3 years lies beyond "
    "each percentile by construction, and the count shows how much of that fell in the last year. Latest "
    f"values are at the last date of the selected range ({df_analysis.index[-1]:%d %b %Y}). The GARCH counts "
    "use today's fitted parameters for every past day. Volatility is the standard deviation of daily returns, "
    "not annualised." + roll_note)


# ─── Section 4: AI Risk Narrative (LLM) ───
st.subheader("🤖 AI Risk Interpretation")
st.write(f"A three-sentence plain-language summary of the statistics on this page for {selected_commodity}.")


# No parameter starts with an underscore: st.cache_data skips such parameters when building the
# cache key, and then a changed number could return an old narrative.
@st.cache_data(ttl=1800, show_spinner="Generating AI risk interpretation...")
def generate_risk_narrative(commodity_name, risk_level_str, latest_vol_v, avg_vol_v,
                            var_95_v, current_regime_v, garch_now, garch_10d, garch_ref_v, garch_ref_label,
                            avg_score_v, score_model, avg_30d_v, anomaly_types_str,
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

    if garch_10d and garch_now:
        _dir = ("rising" if garch_10d > garch_now * 1.02 else
                "falling" if garch_10d < garch_now * 0.98 else "roughly flat")
        garch_line = (f"GARCH daily volatility now {garch_now:.2f}%, forecast for day 10 ahead "
                      f"{garch_10d:.2f}% ({_dir})")
    elif garch_10d:
        garch_line = f"GARCH forecast of daily volatility on day 10 ahead: {garch_10d:.2f}%"
    else:
        garch_line = "GARCH forecast: unavailable"
    if garch_10d and garch_ref_v:
        garch_line += f" ({garch_ref_label} {garch_ref_v:.2f}%)"
    anomaly_line = anomaly_types_str if anomaly_types_str else "None"
    headline_line = top_neg_str if top_neg_str else "None retrieved"
    sentiment_30d_line = (f"{avg_30d_v:+.3f}" if avg_30d_v is not None else "N/A")
    sentiment_now_line = (f"{avg_score_v:+.3f} ({score_model})" if avg_score_v is not None
                          else "N/A (news feeds unreachable, no tone computed)")

    prompt = f"""You are a senior risk analyst on a European energy trading desk, writing the short risk comment on {commodity_name} for the date shown below.

HOW TO READ THE INPUTS

Volatility
- All volatility figures are DAILY standard deviation of returns, in percent. Not annualised.
- The GARCH figure forecasts the daily volatility on the tenth trading day ahead. It is not a cumulative move over ten days.
- The GARCH check compares the forecast with the commodity's 3-year average 30-day volatility, not with current volatility. A forecast moving back towards that average is normalisation, not a build-up of risk, and should be described that way. Also check the forecast against the {regime_calm:.2f}% Volatile boundary.

Risk signal and regime - one classification
- "Regime" compares current 30-day volatility with thresholds for this commodity taken from its last 3 years of volatility (not the selected window): Calm below {regime_calm:.2f}% (its 50th percentile), Volatile {regime_calm:.2f}-{regime_crisis:.2f}%, Crisis above {regime_crisis:.2f}% (its 90th percentile).
- "Risk signal" is the same classification under another name. Do not present them as two separate pieces of evidence.
- The regime measures volatility relative to the commodity's own recent past, not overall risk. Never call the market "low risk" because the regime is Calm: supply disruptions, storage levels and policy events can make a calm-volatility market risky.
- The selected-period average volatility is context only; it does not set the risk signal.

VaR
- VaR 95% is the 5th percentile of daily returns over the last 250 trading days: a loss threshold, given as a negative number.

Headline tone - weak evidence, handle with care
- Computed from at most twenty scraped headlines of the last 7 days, by models trained on company news. It is the tone of the wording, NOT a price direction: a headline about a price surge can score negative.
- The current reading and the 30-day average come from DIFFERENT models depending on which service was reachable, and are NOT on a common scale. Never compare them numerically and never describe a move from one to the other.
- Anything between -0.05 and +0.05 is NEUTRAL. Do not call it mildly positive or mildly negative.

Checks and headlines
- The checks field lists only the NAMES of the statistical checks that fired (percentiles of the last 3 years). You do not have their underlying numbers, so do not invent them.
- The headline field is raw scraped text, truncated and unverified.

MARKET DATA AS OF {date_str}
- Risk signal: {risk_level_str}
- 30-day rolling volatility: {latest_vol_v:.2f}% (average over the selected period {avg_vol_v:.2f}%)
- VaR 95%, 1-day (historical, last 250 trading days): {var_95_v:.2f}%
- {garch_line}
- Regime: {current_regime_v}
- Headline tone now: {sentiment_now_line} | 30-day average (FinVADER): {sentiment_30d_line}
- Checks fired: {anomaly_line}
- Unverified headlines: {headline_line}

TASK
Write EXACTLY 3 sentences. Keep the whole comment under 100 words. No headers, no bullets, no line breaks. Write as of the data date above, not as of any later date.
1. What this combination of readings means. Do not restate the figures - the reader is looking at them. Use at most one number, and only if it carries the argument.
2. The sharpest tension or divergence in the data, and why it matters for positioning.
3. One concrete, checkable thing to watch over the following ten trading days. Name a level, a threshold or a specific series from the data above. Do not simply say to watch whether the forecast proves correct.

HARD RULES
- If the headline field says feeds were unreachable, treat BOTH tone readings as meaningless: do not mention tone or news anywhere in your answer, and build all three sentences from the volatility, VaR, GARCH and regime data alone.
- Never infer complacency, fear or a price direction from headline tone.
- If a field reads N/A, None or None retrieved, pass over it in silence. Do not comment on its absence.
- Never attribute a claim, forecast or warning to a named bank, agency, institution or analyst.
- Never present a headline as established fact.
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


anomaly_types_for_llm = "; ".join(a['type'] for a in anomalies)
if not is_live:
    top_neg_for_llm = "none (news feeds unreachable, no tone computed)"
else:
    top_neg_for_llm = "; ".join(r['Headline'][:80] for _, r in
                                sent_df[sent_df['Score'] < -0.05].nsmallest(3, 'Score').iterrows())

today_date_str = df_analysis.index[-1].strftime('%Y-%m-%d')

narrative, error = generate_risk_narrative(
    commodity_name=selected_commodity,
    risk_level_str=risk_level,
    latest_vol_v=float(latest_vol),
    avg_vol_v=float(avg_vol),
    var_95_v=float(var_95),
    current_regime_v=current_regime,
    garch_now=garch['current_cond_vol'] if garch is not None else None,
    garch_10d=garch_forecast_10d,
    garch_ref_v=float(garch_ref) if garch is not None else None,
    garch_ref_label=garch_ref_name,
    avg_score_v=avg_score,
    score_model=nlp_model_name,
    avg_30d_v=avg_30d,
    anomaly_types_str=anomaly_types_for_llm,
    top_neg_str=top_neg_for_llm,
    date_str=today_date_str,
    regime_calm=float(regime_thr['calm']),
    regime_crisis=float(regime_thr['crisis']),
)

_ai_scope = ("It only summarises the statistics on this page (volatility, VaR, GARCH, regime, headline tone "
             "and the checks above); it does not see fundamentals such as storage, flows, positioning or "
             "the forward curve.")
if narrative:
    st.markdown(
        f"<div style='background:#f8f9ff; border-left:4px solid #6655ee; color:#222; "
        f"padding:14px 18px; border-radius:6px; font-size:0.97em; line-height:1.7'>"
        f"{narrative}</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Generated by Claude (claude-sonnet-5) · data as of {today_date_str} · {_ai_scope} "
               "Cached for 30 minutes for identical inputs.")
elif error == "no_key":
    st.caption("AI summary not enabled in this deployment (no API key). " + _ai_scope)
else:
    st.caption("AI summary temporarily unavailable; the statistics above are unaffected.")


# ─── Section 5: Data Export ───
st.subheader("Export Data")

export_df = df_analysis[['Price', 'Compare', 'Volatility', 'Rolling Correlation']].copy()
export_df.columns = [selected_commodity, compare_label, '30-day volatility of daily returns (%)',
                     '30-day rolling correlation (returns)']
export_df['Daily_Return_%'] = df_analysis['Returns'] * 100
if 'Regime' in features.columns:
    export_df = export_df.join(features[['Regime']], how='left')
st.download_button(
    label=f"Download {selected_commodity} Risk Data (CSV)",
    data=export_df.to_csv(),
    file_name=f"{selected_commodity.lower().replace(' ', '_')}_risk_data.csv",
    mime="text/csv"
)

if is_live:
    st.download_button(
        label="Download Headline Tone Data (CSV)",
        data=sent_df.to_csv(index=False),
        file_name="headline_tone.csv",
        mime="text/csv"
    )

if not cr_df.empty:
    # Ranked table plus the raw Eurostat inputs behind it, so every number can be recomputed.
    raw_cols = {'gas': ['gae_gas', 'gae_total', 'dep_gas', 'dep_total', 'ren'],
                'oil': ['gae_oil', 'gae_total', 'dep_oil', 'dep_total', 'ren'],
                'carbon': ['ghg', 'gdp_clv', 'ren']}[country_mode]
    raw_names = {'gae_gas': 'Gross available gas (ktoe, nrg_bal_c G3000)',
                 'gae_oil': 'Gross available oil (ktoe, nrg_bal_c O4000XBIO)',
                 'gae_total': 'Gross available energy, all products (ktoe, nrg_bal_c TOTAL)',
                 'dep_gas': 'Gas import dependency (%, nrg_ind_id G3000)',
                 'dep_oil': 'Oil import dependency (%, nrg_ind_id O4000XBIO)',
                 'dep_total': 'Total energy import dependency (%, nrg_ind_id TOTAL)',
                 'ren': 'Renewable share (%, nrg_ind_ren REN)',
                 'ghg': 'GHG excl. LULUCF (kt CO2e, env_air_gge TOTX4_MEMO)',
                 'gdp_clv': 'GDP (M€ 2020 prices, nama_10_gdp CLV20_MEUR)'}
    cr_export = ranked[['Country', mcol] + raw_cols].rename(columns={mcol: M['measure'], **raw_names})
    eu_export = pd.DataFrame([eu])[['Country', mcol] + raw_cols].rename(columns={mcol: M['measure'], **raw_names})
    cr_export.insert(0, 'Rank', range(1, len(cr_export) + 1))
    eu_export.insert(0, 'Rank', 'reference')
    cr_export = pd.concat([cr_export, eu_export], ignore_index=True)
    cr_export.insert(1, 'Year', selected_year)
    st.download_button(
        label=f"Download Country Exposure Data ({selected_year}, CSV)",
        data=cr_export.to_csv(index=False),
        file_name=f"country_{country_mode}_{selected_year}.csv",
        mime="text/csv"
    )
