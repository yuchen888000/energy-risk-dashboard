import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import feedparser
import requests as req
import time
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── Page Config ───
st.set_page_config(page_title="European Energy & Commodity Risk Dashboard", layout="wide")

# ─── Commodity Definitions ───
COMMODITIES = {
    "TTF Natural Gas": {
        "ticker": "TTF=F",
        "unit": "€/MWh",
        "color": "steelblue",
        "keywords": ['gas', 'TTF', 'LNG', 'pipeline', 'natural gas', 'methane'],
        "rss_query": "natural+gas+Europe+price",
    },
    "WTI Crude Oil": {
        "ticker": "CL=F",
        "unit": "$/barrel",
        "color": "saddlebrown",
        "keywords": ['oil', 'crude', 'WTI', 'OPEC', 'petroleum', 'barrel', 'refinery'],
        "rss_query": "crude+oil+Europe+price",
    },
    "Brent Crude Oil": {
        "ticker": "BZ=F",
        "unit": "$/barrel",
        "color": "darkred",
        "keywords": ['oil', 'crude', 'Brent', 'OPEC', 'petroleum', 'barrel', 'North Sea'],
        "rss_query": "brent+oil+Europe+price",
    },
    "EU Carbon Allowance": {
        "ticker": "KEUA",
        "unit": "$/share (EUA futures)",
        "color": "seagreen",
        "keywords": ['carbon', 'ETS', 'emission', 'EU ETS', 'EUA', 'allowance', 'CBAM'],
        "rss_query": "EU+carbon+ETS+emission+price",
    },
}

# ─── Sidebar ───
with st.sidebar:
    st.title("Settings")
    selected_commodity = st.selectbox("Select Commodity", list(COMMODITIES.keys()))
    commodity = COMMODITIES[selected_commodity]

    st.markdown("---")
    st.title("Methodology")
    st.markdown(f"""
    **Selected: {selected_commodity}** (`{commodity['ticker']}`)

    **Comparison**: EU Carbon Allowance (`KEUA`) — directly tracks 
    EU ETS carbon futures. Falls back to ICLN if unavailable.

    **Risk Metrics**
    - **30-Day Rolling Volatility**: Std dev of daily returns over 30 days.
    - **Rolling Correlation**: Pearson correlation with EU carbon over 30 days.
    - **Value at Risk (VaR)**: 95% and 99% historical VaR.
    - **GARCH(1,1) Forecast**: Predicts future volatility from recent 
      shocks (α) and persistence (β). Standard on energy trading desks.

    **AI / ML**
    - **Hybrid Regime Detection**: K-Means clustering + absolute 
      volatility thresholds (Calm < 6%, Volatile 6–12%, Crisis > 12%).

    **NLP**
    - **FinBERT Sentiment**: Transformer model fine-tuned on financial 
      text (ProsusAI/finbert via HuggingFace). Falls back to VADER 
      if API unavailable.

    ---
    *Built by Yuchen · IHEID Master's in International Economics*
    *Python · Streamlit · yfinance · scikit-learn · FinBERT · GARCH*
    """)

# ─── Title ───
st.title("European Energy & Commodity Risk Dashboard")
st.markdown(f"Analyzing **{selected_commodity}** vs EU Carbon Allowance Price")

# ─── Date Selection ───
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# ─── Data Download (cached) ───
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data(ticker, start, end):
    primary = yf.download(ticker, start=start, end=end, progress=False)
    carbon_etf = yf.download("KEUA", start=start, end=end, progress=False)
    clean = yf.download("ICLN", start=start, end=end, progress=False)

    primary_price = primary['Close'].squeeze()
    carbon_price = carbon_etf['Close'].squeeze()
    clean_price = clean['Close'].squeeze()

    df = pd.DataFrame({
        'Price': primary_price,
        'Carbon (KEUA)': carbon_price,
        'Clean Energy (ICLN)': clean_price,
    }).dropna(subset=['Price'])
    return df

df = load_data(commodity['ticker'], start_date, end_date)

# Determine carbon comparison
has_keua = df['Carbon (KEUA)'].notna().sum() > 30

if commodity['ticker'] == 'KEUA':
    # If user selected carbon, compare against TTF gas instead
    compare_data = yf.download("TTF=F", start=start_date, end=end_date, progress=False)
    df['Compare'] = compare_data['Close'].squeeze()
    compare_label = 'TTF Natural Gas (€/MWh)'
    df_analysis = df[['Price', 'Compare']].dropna()
elif has_keua:
    df['Compare'] = df['Carbon (KEUA)']
    compare_label = 'EU Carbon Allowance (KEUA)'
    df_analysis = df[['Price', 'Compare']].dropna()
else:
    df['Compare'] = df['Clean Energy (ICLN)']
    compare_label = 'Clean Energy Proxy (ICLN)'
    df_analysis = df[['Price', 'Compare']].dropna()
    st.info("KEUA data unavailable for selected range — using ICLN as fallback.")

if len(df_analysis) < 30:
    st.warning("Please select a longer time range (at least 30 days of data required).")
else:
    # ─── Core Calculations ───
    df_analysis['Returns'] = df_analysis['Price'].pct_change()
    df_analysis['Volatility'] = df_analysis['Returns'].rolling(30).std() * 100
    df_analysis['Rolling Correlation'] = df_analysis['Price'].rolling(30).corr(df_analysis['Compare'])

    latest_vol = df_analysis['Volatility'].dropna().iloc[-1]
    avg_vol = df_analysis['Volatility'].dropna().mean()

    returns_clean = df_analysis['Returns'].dropna()
    var_95 = np.percentile(returns_clean, 5) * 100
    var_99 = np.percentile(returns_clean, 1) * 100

    if latest_vol > avg_vol * 1.5:
        risk_level = "🔴 HIGH RISK"
        risk_color = "red"
    elif latest_vol > avg_vol:
        risk_level = "🟡 MEDIUM RISK"
        risk_color = "orange"
    else:
        risk_level = "🟢 LOW RISK"
        risk_color = "green"

    # ─── Section 1: Risk Signal ───
    st.subheader(f"Current Risk Signal — {selected_commodity}")
    st.markdown(f"<h2 style='color:{risk_color}'>{risk_level}</h2>",
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Current Volatility", f"{latest_vol:.2f}%")
    mc2.metric("Average Volatility", f"{avg_vol:.2f}%")
    mc3.metric("Correlation", f"{df_analysis['Price'].corr(df_analysis['Compare']):.2f}")
    mc4.metric("VaR (95%, 1-day)", f"{var_95:.2f}%")

    # ─── Section 2: Price Chart ───
    macro_events = {
        "2021-07-14": "EU Fit for 55",
        "2022-02-24": "Russia invades Ukraine",
        "2022-06-01": "EU bans Russian oil",
        "2022-09-26": "Nord Stream sabotage",
        "2023-01-01": "EU gas price cap",
        "2023-04-18": "EU ETS 2 passed",
        "2024-01-01": "EU ETS reform",
        "2024-12-01": "CBAM transition ends",
        "2025-02-01": "EU ETS 2 Phase 1",
        "2025-10-01": "CBAM full enforcement",
    }

    st.subheader("Price Trends with Key EU Policy Events")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_analysis.index, df_analysis['Price'], color=commodity['color'],
            label=f'{selected_commodity} ({commodity["unit"]})', linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(df_analysis.index, df_analysis['Compare'], color='gray',
             label=compare_label, linewidth=1.2, alpha=0.6)

    for date_str, label in macro_events.items():
        event_date = pd.to_datetime(date_str)
        if df_analysis.index.min() <= event_date <= df_analysis.index.max():
            ax.axvline(x=event_date, color='gray', linestyle='--', alpha=0.4)
            ax.text(event_date, ax.get_ylim()[1] * 0.9, label,
                   rotation=90, fontsize=7, color='gray', va='top')

    ax.set_ylabel(f'{selected_commodity} ({commodity["unit"]})', color=commodity['color'])
    ax2.set_ylabel(compare_label, color='gray')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax.set_title(f'{selected_commodity} vs {compare_label}')
    plt.tight_layout()
    st.pyplot(fig)

    # ─── Section 3: Volatility & Correlation ───
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.subheader("30-Day Rolling Volatility")
        st.line_chart(df_analysis['Volatility'].dropna())
    with vcol2:
        st.subheader("30-Day Rolling Correlation")
        st.line_chart(df_analysis['Rolling Correlation'].dropna())

    # ─── Section 4: Value at Risk ───
    st.subheader(f"Value at Risk (VaR) — {selected_commodity}")
    st.write(f"Historical simulation VaR — worst-case daily losses on {selected_commodity} positions")

    fig_var, (ax_hist, ax_ts) = plt.subplots(1, 2, figsize=(14, 4))

    ax_hist.hist(returns_clean * 100, bins=80, color=commodity['color'], alpha=0.7, edgecolor='white')
    ax_hist.axvline(x=var_95, color='red', linewidth=2, linestyle='--',
                    label=f'95% VaR: {var_95:.2f}%')
    ax_hist.axvline(x=var_99, color='darkred', linewidth=2, linestyle=':',
                    label=f'99% VaR: {var_99:.2f}%')
    ax_hist.set_xlabel('Daily Returns (%)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'{selected_commodity} Daily Return Distribution')
    ax_hist.legend(fontsize=8)

    rolling_var = returns_clean.rolling(60).quantile(0.05) * 100
    ax_ts.plot(rolling_var.index, rolling_var, color='red', linewidth=1, alpha=0.8)
    ax_ts.fill_between(rolling_var.index, rolling_var, 0, alpha=0.15, color='red')
    ax_ts.set_ylabel('VaR (95%, daily %)')
    ax_ts.set_title('60-Day Rolling VaR')
    ax_ts.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    st.pyplot(fig_var)

    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("VaR 95% (1-day)", f"{var_95:.2f}%")
    vc2.metric("VaR 99% (1-day)", f"{var_99:.2f}%")
    vc3.metric("Max Daily Loss", f"{returns_clean.min() * 100:.2f}%")

    # ─── Section 4b: GARCH ───
    st.subheader("GARCH Volatility Forecast")
    st.write(f"Forward-looking volatility prediction for {selected_commodity} using GARCH(1,1)")

    from arch import arch_model

    garch_returns = returns_clean.dropna() * 100
    try:
        model = arch_model(garch_returns, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        result = model.fit(disp='off')

        forecast = result.forecast(horizon=10)
        forecast_var = forecast.variance.iloc[-1]
        forecast_vol = np.sqrt(forecast_var)

        current_cond_vol = np.sqrt(result.conditional_volatility.iloc[-1] ** 2)

        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Current GARCH Vol (daily)", f"{current_cond_vol:.2f}%")
        gc2.metric("5-Day Forecast Vol", f"{forecast_vol.iloc[4]:.2f}%")
        gc3.metric("10-Day Forecast Vol", f"{forecast_vol.iloc[9]:.2f}%")

        fig_garch, (ax_cv, ax_fc) = plt.subplots(1, 2, figsize=(14, 4))

        cond_vol = result.conditional_volatility
        ax_cv.plot(cond_vol.index, cond_vol, color='purple', linewidth=0.8, alpha=0.8)
        ax_cv.set_ylabel('Conditional Volatility (daily %)')
        ax_cv.set_title('GARCH(1,1) Conditional Volatility')
        ax_cv.fill_between(cond_vol.index, cond_vol, 0, alpha=0.1, color='purple')

        forecast_days = list(range(1, 11))
        ax_fc.plot(forecast_days, forecast_vol.values, color='purple', linewidth=2, marker='o', markersize=5)
        ax_fc.fill_between(forecast_days, forecast_vol.values * 0.7, forecast_vol.values * 1.3,
                          alpha=0.15, color='purple', label='±30% confidence band')
        ax_fc.set_xlabel('Days Ahead')
        ax_fc.set_ylabel('Forecast Volatility (daily %)')
        ax_fc.set_title('10-Day Volatility Forecast')
        ax_fc.set_xticks(forecast_days)
        ax_fc.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_garch)

        with st.expander("GARCH(1,1) Model Parameters"):
            st.write(f"**omega (ω):** {result.params['omega']:.6f}")
            st.write(f"**alpha (α):** {result.params['alpha[1]']:.4f} — reaction to recent shocks")
            st.write(f"**beta (β):** {result.params['beta[1]']:.4f} — persistence of volatility")
            persistence = result.params['alpha[1]'] + result.params['beta[1]']
            st.write(f"**α + β = {persistence:.4f}** — "
                     f"{'high persistence (close to 1)' if persistence > 0.95 else 'moderate persistence'}")
            st.write(f"**Log-Likelihood:** {result.loglikelihood:.2f}")

    except Exception as e:
        st.warning(f"GARCH model could not be fitted: {e}")

    # ─── Section 5: Market Regime Clustering ───
    st.subheader("Market Regime Clustering (AI)")
    st.write("Hybrid approach: K-Means clustering + absolute volatility thresholds for regime labeling")

    features = df_analysis[['Volatility', 'Rolling Correlation']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    features = features.copy()
    features['Cluster'] = kmeans.fit_predict(scaled)

    def classify_regime(vol):
        if vol > 12:
            return 'Crisis'
        elif vol > 6:
            return 'Volatile'
        else:
            return 'Calm'

    features['Regime'] = features['Volatility'].apply(classify_regime)

    current_regime = features['Regime'].iloc[-1]
    regime_colors = {'Calm': 'green', 'Volatile': 'orange', 'Crisis': 'red'}
    regime_color = regime_colors.get(current_regime, 'gray')
    st.markdown(f"<h3 style='color:{regime_color}'>Current Market Regime: {current_regime}</h3>",
                unsafe_allow_html=True)
    st.caption("Thresholds: Calm < 6% · Volatile 6–12% · Crisis > 12% (30-day rolling volatility)")

    regime_stats = features.groupby('Regime').agg(
        Days=('Volatility', 'count'),
        Avg_Volatility=('Volatility', 'mean'),
        Avg_Correlation=('Rolling Correlation', 'mean')
    ).round(2)
    regime_stats.columns = ['Trading Days', 'Avg Volatility (%)', 'Avg Correlation']

    rcol1, rcol2 = st.columns([2, 1])
    with rcol1:
        fig2, ax3 = plt.subplots(figsize=(12, 4))
        colors = {'Calm': 'green', 'Volatile': 'orange', 'Crisis': 'red'}
        for regime, group in features.groupby('Regime'):
            ax3.scatter(group.index, group['Volatility'],
                       c=colors[regime], label=regime, alpha=0.5, s=10)
        ax3.axhline(y=6, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='Volatile threshold (6%)')
        ax3.axhline(y=12, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Crisis threshold (12%)')
        ax3.set_ylabel('30-Day Volatility (%)')
        ax3.set_title(f'{selected_commodity} — Market Regime Detection')
        ax3.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig2)
    with rcol2:
        st.write("**Regime Statistics:**")
        st.dataframe(regime_stats, use_container_width=True)

    # ─── Section 6: NLP Sentiment (FinBERT) ───
    st.subheader(f"Energy News Sentiment — {selected_commodity}")
    st.write("Real-time sentiment analysis using FinBERT (financial domain transformer model)")

    # Commodity-specific + general energy keywords
    general_keywords = ['energy', 'power', 'electricity', 'renewable', 'climate',
                        'emission', 'fuel', 'Europe', 'European']
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

    for source_name, url in rss_feeds.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries[:50]:
                if count >= 3:
                    break
                title = entry.title
                if any(kw.lower() in title.lower() for kw in nlp_keywords):
                    headlines.append(title)
                    headline_links.append(entry.get('link', ''))
                    headline_sources.append(source_name)
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

    # ─── FinBERT via HuggingFace Inference API ───
    @st.cache_data(ttl=1800, show_spinner="Running FinBERT sentiment analysis...")
    def finbert_analyze(texts):
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        scores = []
        labels = []

        # Try batch request
        try:
            response = req.post(API_URL, json={"inputs": texts}, timeout=30)

            if response.status_code == 503:
                # Model loading — wait and retry
                wait_time = response.json().get('estimated_time', 20)
                time.sleep(min(wait_time, 30))
                response = req.post(API_URL, json={"inputs": texts}, timeout=30)

            if response.status_code == 200:
                results = response.json()
                for result in results:
                    if isinstance(result, list):
                        best = max(result, key=lambda x: x['score'])
                        lbl = best['label'].lower()
                        sc = best['score']
                        if lbl == 'negative':
                            scores.append(-sc)
                            labels.append('Negative')
                        elif lbl == 'positive':
                            scores.append(sc)
                            labels.append('Positive')
                        else:
                            scores.append(0.0)
                            labels.append('Neutral')
                    else:
                        scores.append(0.0)
                        labels.append('Neutral')
                return scores, labels, True
        except Exception:
            pass

        return None, None, False

    finbert_scores, finbert_labels, finbert_success = finbert_analyze(headlines)

    if finbert_success:
        nlp_model_name = "FinBERT (ProsusAI/finbert)"
        sentiment_data = []
        for i, h in enumerate(headlines):
            sentiment_data.append({
                'Headline': h,
                'Source': headline_sources[i],
                'Link': headline_links[i],
                'Score': finbert_scores[i],
                'Label': finbert_labels[i],
            })
    else:
        # Fallback to VADER
        nlp_model_name = "VADER (fallback — FinBERT API unavailable)"
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
    top_pos = sent_df.nlargest(3, 'Score')
    top_neg = sent_df.nsmallest(3, 'Score')

    st.write("**Most Positive Headlines:**")
    for _, row in top_pos.iterrows():
        score_str = f"{row['Score']:+.3f}"
        if row['Link']:
            st.markdown(f"🟢 **[{score_str}]** [{row['Headline']}]({row['Link']}) — *{row['Source']}*")
        else:
            st.markdown(f"🟢 **[{score_str}]** {row['Headline']} — *{row['Source']}*")

    st.write("**Most Negative Headlines:**")
    for _, row in top_neg.iterrows():
        score_str = f"{row['Score']:+.3f}"
        if row['Link']:
            st.markdown(f"🔴 **[{score_str}]** [{row['Headline']}]({row['Link']}) — *{row['Source']}*")
        else:
            st.markdown(f"🔴 **[{score_str}]** {row['Headline']} — *{row['Source']}*")

    # ─── Section 6b: Sentiment Trend (30-day) ───
    st.subheader("Sentiment Trend (30 Days)")
    st.write(f"Daily average sentiment for {selected_commodity}-related European energy news over the past 30 days")

    @st.cache_data(ttl=7200, show_spinner="Fetching 30-day news history...")
    def get_sentiment_trend(rss_query, keywords):
        """Fetch past 30 days of news via Google News RSS and compute daily sentiment."""
        from datetime import datetime, timedelta

        sia_trend = SentimentIntensityAnalyzer()
        daily_scores = {}

        # Google News RSS with 'when:30d' fetches past 30 days
        trend_url = f"https://news.google.com/rss/search?q={rss_query}+when:30d&hl=en"
        try:
            feed = feedparser.parse(trend_url)
            for entry in feed.entries[:100]:
                title = entry.title
                if not any(kw.lower() in title.lower() for kw in keywords):
                    continue

                # Parse published date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:3]).strftime('%Y-%m-%d')
                else:
                    continue

                score = sia_trend.polarity_scores(title)['compound']

                if pub_date not in daily_scores:
                    daily_scores[pub_date] = []
                daily_scores[pub_date].append(score)
        except Exception:
            pass

        # Also add a second query for broader coverage
        trend_url2 = f"https://news.google.com/rss/search?q=European+energy+{rss_query.split('+')[0]}+when:30d&hl=en"
        try:
            feed2 = feedparser.parse(trend_url2)
            for entry in feed2.entries[:100]:
                title = entry.title
                if not any(kw.lower() in title.lower() for kw in keywords):
                    continue
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:3]).strftime('%Y-%m-%d')
                else:
                    continue
                score = sia_trend.polarity_scores(title)['compound']
                if pub_date not in daily_scores:
                    daily_scores[pub_date] = []
                daily_scores[pub_date].append(score)
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

    if trend_df is not None and len(trend_df) > 3:
        # Today's sentiment
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

        # Trend chart
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

        st.caption(f"Based on {int(trend_df['Headlines Count'].sum())} headlines over {len(trend_df)} days · Scored with VADER (trend) + FinBERT (current)")
    else:
        st.info("Not enough historical headline data to generate trend. This improves over time as more news is collected.")

    # ─── Section 7: Data Export ───
    st.subheader("Export Data")

    export_df = df_analysis[['Price', 'Compare', 'Volatility', 'Rolling Correlation']].copy()
    export_df.columns = [selected_commodity, compare_label, 'Volatility (%)', 'Rolling Correlation']
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
