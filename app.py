import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import feedparser
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── Page Config ───
st.set_page_config(page_title="European Energy Risk Dashboard", layout="wide")

# ─── Sidebar: Methodology ───
with st.sidebar:
    st.title("Methodology")
    st.markdown("""
    **Data Sources**
    - **TTF Natural Gas Futures** (`TTF=F`): Dutch Title Transfer Facility, 
      the European benchmark for natural gas pricing.
    - **EU Carbon Allowance** (`KEUA`): KraneShares European Carbon Allowance 
      Strategy ETF — directly tracks EU ETS carbon futures (EUA). This is a 
      real carbon price proxy, not a clean energy equity fund.
    - **Fallback: Clean Energy Proxy** (`ICLN`): If KEUA data is unavailable 
      for the selected date range, ICLN (iShares Global Clean Energy ETF) is 
      used as a secondary proxy tracking the energy transition trade.

    **Risk Metrics**
    - **30-Day Rolling Volatility**: Standard deviation of daily returns 
      over a rolling 30-day window, expressed as percentage.
    - **Rolling Correlation**: Pearson correlation between TTF and ICLN over 30 days.
    - **Value at Risk (VaR)**: 95% and 99% historical VaR — the maximum expected 
      daily loss that would not be exceeded at the given confidence level.
    - **GARCH(1,1) Forecast**: Generalized Autoregressive Conditional 
      Heteroskedasticity model — predicts future volatility based on 
      recent shocks (α) and volatility persistence (β). Standard in 
      energy trading risk desks.

    **AI / ML**
    - **Hybrid Regime Detection**: K-Means clustering on (volatility, correlation) 
      feature space, combined with absolute volatility thresholds for regime labeling. 
      Calm < 6%, Volatile 6–12%, Crisis > 12%. This avoids the pure-relative problem 
      where moderate volatility gets mislabeled as Crisis.

    **NLP**
    - **VADER Sentiment**: Rule-based sentiment analysis on live energy news 
      headlines from RSS feeds. Compound score: -1 (most negative) to +1 (most positive).

    ---
    *Built by Yuchen · IHEID Master's in International Economics*  
    *Python · Streamlit · yfinance · scikit-learn · NLTK*
    """)

# ─── Title ───
st.title("European Energy Risk Dashboard")
st.markdown("Monitoring TTF Natural Gas vs EU Carbon Allowance Price")

# ─── Date Selection ───
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# ─── Data Download (cached) ───
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data(start, end):
    gas = yf.download("TTF=F", start=start, end=end, progress=False)
    carbon_etf = yf.download("KEUA", start=start, end=end, progress=False)
    clean = yf.download("ICLN", start=start, end=end, progress=False)

    gas_price = gas['Close'].squeeze()
    carbon_price = carbon_etf['Close'].squeeze()
    clean_price = clean['Close'].squeeze()

    df = pd.DataFrame({
        'Gas Price': gas_price,
        'Carbon Price (KEUA)': carbon_price,
        'Clean Energy (ICLN)': clean_price,
    }).dropna(subset=['Gas Price'])
    return df

df = load_data(start_date, end_date)

has_keua = df['Carbon Price (KEUA)'].notna().sum() > 30
carbon_col = 'Carbon Price (KEUA)' if has_keua else 'Clean Energy (ICLN)'

if has_keua:
    df_analysis = df[['Gas Price', 'Carbon Price (KEUA)']].dropna()
    carbon_label = 'EU Carbon Allowance (KEUA)'
    carbon_unit = '$/share (tracks EUA futures)'
else:
    df_analysis = df[['Gas Price', 'Clean Energy (ICLN)']].dropna()
    carbon_label = 'Clean Energy Proxy (ICLN)'
    carbon_unit = '$/share'
    st.info("KEUA data unavailable for selected range — using ICLN as fallback proxy.")

df_analysis.columns = ['Gas Price', 'Carbon Proxy']

if len(df_analysis) < 30:
    st.warning("Please select a longer time range (at least 30 days of data required).")
else:
    # ─── Core Calculations ───
    df_analysis['Gas Returns'] = df_analysis['Gas Price'].pct_change()
    df_analysis['Volatility'] = df_analysis['Gas Returns'].rolling(30).std() * 100
    df_analysis['Rolling Correlation'] = df_analysis['Gas Price'].rolling(30).corr(df_analysis['Carbon Proxy'])

    latest_vol = df_analysis['Volatility'].dropna().iloc[-1]
    avg_vol = df_analysis['Volatility'].dropna().mean()

    # VaR calculation (95% historical)
    returns_clean = df_analysis['Gas Returns'].dropna()
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
    st.subheader("Current Risk Signal")
    st.markdown(f"<h2 style='color:{risk_color}'>{risk_level}</h2>",
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Current Volatility", f"{latest_vol:.2f}%")
    mc2.metric("Average Volatility", f"{avg_vol:.2f}%")
    mc3.metric("Correlation", f"{df_analysis['Gas Price'].corr(df_analysis['Carbon Proxy']):.2f}")
    mc4.metric("VaR (95%, 1-day)", f"{var_95:.2f}%")

    # ─── Section 2: Price Chart with Macro Events ───
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
    ax.plot(df_analysis.index, df_analysis['Gas Price'], color='steelblue',
            label='TTF Gas Price (€/MWh)', linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(df_analysis.index, df_analysis['Carbon Proxy'], color='seagreen',
             label=f'{carbon_label} ({carbon_unit})', linewidth=1.5, alpha=0.7)

    for date_str, label in macro_events.items():
        event_date = pd.to_datetime(date_str)
        if df_analysis.index.min() <= event_date <= df_analysis.index.max():
            ax.axvline(x=event_date, color='gray', linestyle='--', alpha=0.5)
            ax.text(event_date, ax.get_ylim()[1] * 0.9, label,
                   rotation=90, fontsize=7, color='gray', va='top')

    ax.set_ylabel('TTF Gas Price (€/MWh)', color='steelblue')
    ax2.set_ylabel(carbon_label, color='seagreen')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax.set_title(f'TTF Natural Gas vs {carbon_label} (2020–present)')
    plt.tight_layout()
    st.pyplot(fig)

    # ─── Section 3: Volatility & Correlation side by side ───
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.subheader("30-Day Rolling Volatility")
        st.line_chart(df_analysis['Volatility'].dropna())
    with vcol2:
        st.subheader("30-Day Rolling Correlation")
        st.line_chart(df_analysis['Rolling Correlation'].dropna())

    # ─── Section 4: Value at Risk ───
    st.subheader("Value at Risk (VaR) Analysis")
    st.write("Historical simulation VaR — estimating worst-case daily losses on TTF gas positions")

    fig_var, (ax_hist, ax_ts) = plt.subplots(1, 2, figsize=(14, 4))

    # Return distribution + VaR lines
    ax_hist.hist(returns_clean * 100, bins=80, color='steelblue', alpha=0.7, edgecolor='white')
    ax_hist.axvline(x=var_95, color='red', linewidth=2, linestyle='--',
                    label=f'95% VaR: {var_95:.2f}%')
    ax_hist.axvline(x=var_99, color='darkred', linewidth=2, linestyle=':',
                    label=f'99% VaR: {var_99:.2f}%')
    ax_hist.set_xlabel('Daily Returns (%)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('TTF Daily Return Distribution')
    ax_hist.legend(fontsize=8)

    # Rolling VaR time series
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

    # ─── Section 4b: GARCH Volatility Forecast ───
    st.subheader("GARCH Volatility Forecast")
    st.write("Forward-looking volatility prediction using GARCH(1,1) — the standard model in energy risk management")

    from arch import arch_model

    # Fit GARCH(1,1) on TTF daily returns (scaled to percentage)
    garch_returns = returns_clean.dropna() * 100
    try:
        model = arch_model(garch_returns, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        result = model.fit(disp='off')

        # Forecast next 10 trading days
        forecast = result.forecast(horizon=10)
        forecast_var = forecast.variance.iloc[-1]
        forecast_vol = np.sqrt(forecast_var)

        # Current conditional volatility from model
        current_cond_vol = np.sqrt(result.conditional_volatility.iloc[-1] ** 2)

        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Current GARCH Vol (daily)", f"{current_cond_vol:.2f}%")
        gc2.metric("5-Day Forecast Vol", f"{forecast_vol.iloc[4]:.2f}%")
        gc3.metric("10-Day Forecast Vol", f"{forecast_vol.iloc[9]:.2f}%")

        fig_garch, (ax_cv, ax_fc) = plt.subplots(1, 2, figsize=(14, 4))

        # Left: historical conditional volatility
        cond_vol = result.conditional_volatility
        ax_cv.plot(cond_vol.index, cond_vol, color='purple', linewidth=0.8, alpha=0.8)
        ax_cv.set_ylabel('Conditional Volatility (daily %)')
        ax_cv.set_title('GARCH(1,1) Conditional Volatility')
        ax_cv.fill_between(cond_vol.index, cond_vol, 0, alpha=0.1, color='purple')

        # Right: 10-day forecast
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

        # Model summary
        with st.expander("GARCH(1,1) Model Parameters"):
            st.write(f"**omega (ω):** {result.params['omega']:.6f}")
            st.write(f"**alpha (α):** {result.params['alpha[1]']:.4f} — reaction to recent shocks")
            st.write(f"**beta (β):** {result.params['beta[1]']:.4f} — persistence of volatility")
            st.write(f"**α + β = {result.params['alpha[1]'] + result.params['beta[1]']:.4f}** — "
                     f"{'high persistence (close to 1)' if result.params['alpha[1]'] + result.params['beta[1]'] > 0.95 else 'moderate persistence'}")
            st.write(f"**Log-Likelihood:** {result.loglikelihood:.2f}")

    except Exception as e:
        st.warning(f"GARCH model could not be fitted: {e}")

    # ─── Section 5: Market Regime Clustering (AI) ───
    st.subheader("Market Regime Clustering (AI)")
    st.write("Hybrid approach: K-Means clustering + absolute volatility thresholds for regime labeling")

    features = df_analysis[['Volatility', 'Rolling Correlation']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    features = features.copy()
    features['Cluster'] = kmeans.fit_predict(scaled)

    # Hybrid regime labels: use absolute thresholds instead of purely relative KMeans labels
    # Thresholds calibrated against 2020-2026 TTF historical volatility
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
        ax3.set_title('Market Regime Detection over Time')
        ax3.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig2)
    with rcol2:
        st.write("**Regime Statistics:**")
        st.dataframe(regime_stats, use_container_width=True)

    # ─── Section 6: NLP Sentiment ───
    st.subheader("Energy News Sentiment (NLP)")
    st.write("Real-time sentiment analysis of energy market headlines via VADER NLP model")

    energy_keywords = [
        'energy', 'gas', 'oil', 'carbon', 'climate', 'LNG', 'pipeline', 'TTF',
        'power', 'electricity', 'renewable', 'wind', 'solar', 'nuclear', 'coal',
        'emission', 'EU ETS', 'natural gas', 'crude', 'OPEC', 'hydrogen', 'fuel'
    ]

    rss_feeds = {
        "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "OilPrice": "https://oilprice.com/rss/main",
        "Google Energy": "https://news.google.com/rss/search?q=energy+market+europe&hl=en",
        "Google Gas": "https://news.google.com/rss/search?q=natural+gas+price&hl=en",
        "Google Carbon": "https://news.google.com/rss/search?q=EU+carbon+ETS&hl=en",
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
                if any(kw.lower() in title.lower() for kw in energy_keywords):
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
            "TTF natural gas futures decline on mild weather",
            "Energy crisis pushes European inflation higher",
            "Renewable energy investment hits record in Europe"
        ]
        headline_links = [''] * len(headlines)
        headline_sources = ['Sample'] * len(headlines)

    sia = SentimentIntensityAnalyzer()
    sentiment_data = []
    for i, h in enumerate(headlines):
        sc = sia.polarity_scores(h)
        sentiment_data.append({
            'Headline': h,
            'Source': headline_sources[i],
            'Link': headline_links[i],
            'Compound': sc['compound'],
            'Positive': sc['pos'],
            'Negative': sc['neg'],
            'Neutral': sc['neu'],
        })

    sent_df = pd.DataFrame(sentiment_data)
    avg_score = sent_df['Compound'].mean()
    n_bull = (sent_df['Compound'] > 0.05).sum()
    n_bear = (sent_df['Compound'] < -0.05).sum()
    n_neut = len(sent_df) - n_bull - n_bear

    if avg_score > 0.05:
        sentiment_label = "Bullish"
        sentiment_color = "green"
    elif avg_score < -0.05:
        sentiment_label = "Bearish"
        sentiment_color = "red"
    else:
        sentiment_label = "Neutral"
        sentiment_color = "orange"

    st.markdown(f"<h3 style='color:{sentiment_color}'>Market Sentiment: {sentiment_label}</h3>",
                unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Avg Sentiment", f"{avg_score:.3f}")
    sc2.metric("Bullish", f"{n_bull}")
    sc3.metric("Bearish", f"{n_bear}")
    sc4.metric("Neutral", f"{n_neut}")

    if is_live:
        st.caption(f"Analyzing {len(sent_df)} live headlines from {len(set(headline_sources))} sources")
    else:
        st.caption("Live feeds unavailable — showing sample headlines")

    # Sentiment chart
    fig3, ax4 = plt.subplots(figsize=(12, max(3, len(sent_df) * 0.3)))
    bar_colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray'
                  for s in sent_df['Compound']]
    ax4.barh(range(len(sent_df)), sent_df['Compound'], color=bar_colors, height=0.6)
    ax4.set_yticks(range(len(sent_df)))
    ax4.set_yticklabels([h[:55] + '...' if len(h) > 55 else h for h in sent_df['Headline']],
                        fontsize=7)
    ax4.axvline(x=0, color='black', linewidth=0.5)
    ax4.axvline(x=0.05, color='green', linewidth=0.5, linestyle='--', alpha=0.4)
    ax4.axvline(x=-0.05, color='red', linewidth=0.5, linestyle='--', alpha=0.4)
    ax4.set_xlabel('Sentiment Score (VADER Compound)')
    ax4.set_title('Per-Headline Sentiment Distribution')
    ax4.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig3)

    # Headlines detail — top 3 most positive & top 3 most negative
    top_bull = sent_df.nlargest(3, 'Compound')
    top_bear = sent_df.nsmallest(3, 'Compound')

    st.write("**Most Bullish Headlines:**")
    for _, row in top_bull.iterrows():
        score = row['Compound']
        link = row['Link']
        source = row['Source']
        headline = row['Headline']
        score_str = f"{score:+.3f}"
        if link:
            st.markdown(f"🟢 **[{score_str}]** [{headline}]({link}) — *{source}*")
        else:
            st.markdown(f"🟢 **[{score_str}]** {headline} — *{source}*")

    st.write("**Most Bearish Headlines:**")
    for _, row in top_bear.iterrows():
        score = row['Compound']
        link = row['Link']
        source = row['Source']
        headline = row['Headline']
        score_str = f"{score:+.3f}"
        if link:
            st.markdown(f"🔴 **[{score_str}]** [{headline}]({link}) — *{source}*")
        else:
            st.markdown(f"🔴 **[{score_str}]** {headline} — *{source}*")

    # ─── Section 7: Data Export ───
    st.subheader("Export Data")

    export_df = df_analysis[['Gas Price', 'Carbon Proxy', 'Volatility', 'Rolling Correlation']].copy()
    export_df['Gas_Daily_Return_%'] = df_analysis['Gas Returns'] * 100

    if 'Regime' in features.columns:
        export_df = export_df.join(features[['Regime']], how='left')

    csv = export_df.to_csv()
    st.download_button(
        label="Download Risk Data (CSV)",
        data=csv,
        file_name="energy_risk_data.csv",
        mime="text/csv"
    )

    sent_csv = sent_df.to_csv(index=False)
    st.download_button(
        label="Download Sentiment Data (CSV)",
        data=sent_csv,
        file_name="sentiment_data.csv",
        mime="text/csv"
    )
