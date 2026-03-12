import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import feedparser
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("European Energy Risk Dashboard")
st.write("Monitoring TTF Natural Gas vs Carbon Price")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

gas = yf.download("TTF=F", start=start_date, end=end_date, progress=False)
carbon = yf.download("ICLN", start=start_date, end=end_date, progress=False)

gas_price = gas['Close'].squeeze()
carbon_price = carbon['Close'].squeeze()

df = pd.DataFrame({
    'Gas Price': gas_price,
    'Carbon Proxy': carbon_price
}).dropna()

if len(df) < 30:
    st.warning("Please select a longer time range (at least 30 days)")
else:
    df['Volatility'] = df['Gas Price'].pct_change().rolling(30).std() * 100
    df['Rolling Correlation'] = df['Gas Price'].rolling(30).corr(df['Carbon Proxy'])

    latest_vol = df['Volatility'].dropna().iloc[-1]
    avg_vol = df['Volatility'].dropna().mean()

    if latest_vol > avg_vol * 1.5:
        risk_level = "🔴 HIGH RISK"
        risk_color = "red"
    elif latest_vol > avg_vol:
        risk_level = "🟡 MEDIUM RISK"
        risk_color = "orange"
    else:
        risk_level = "🟢 LOW RISK"
        risk_color = "green"

    st.subheader("Current Risk Signal")
    st.markdown(f"<h2 style='color:{risk_color}'>{risk_level}</h2>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Volatility", f"{latest_vol:.2f}%")
    col2.metric("Average Volatility", f"{avg_vol:.2f}%")
    col3.metric("Correlation", f"{df['Gas Price'].corr(df['Carbon Proxy']):.2f}")

    macro_events = {
        "2021-07-14": "EU Fit for 55 Package",
        "2022-02-24": "Russia invades Ukraine",
        "2022-06-01": "EU bans Russian oil",
        "2022-09-26": "Nord Stream sabotage",
        "2023-01-01": "EU gas price cap",
        "2023-04-18": "EU ETS 2 legislation passed",
        "2024-01-01": "EU ETS reform",
    }

    st.subheader("Price Trends with Key Events")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Gas Price'], color='steelblue', label='TTF Gas Price', linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(df.index, df['Carbon Proxy'], color='tomato', label='Carbon Proxy', linewidth=1.5, alpha=0.7)

    for date_str, label in macro_events.items():
        event_date = pd.to_datetime(date_str)
        if df.index.min() <= event_date <= df.index.max():
            ax.axvline(x=event_date, color='gray', linestyle='--', alpha=0.6)
            ax.text(event_date, ax.get_ylim()[1] * 0.85, label,
                   rotation=90, fontsize=7, color='gray', va='top')

    ax.set_ylabel('Gas Price (€/MWh)', color='steelblue')
    ax2.set_ylabel('Carbon Proxy', color='tomato')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("30-Day Rolling Volatility")
    st.line_chart(df['Volatility'].dropna())

    st.subheader("30-Day Rolling Correlation")
    st.line_chart(df['Rolling Correlation'].dropna())

    st.subheader("Market Regime Clustering (AI)")
    st.write("K-Means unsupervised learning identifies 3 market states based on volatility patterns")

    features = df[['Volatility', 'Rolling Correlation']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    features = features.copy()
    features['Cluster'] = kmeans.fit_predict(scaled)

    cluster_vol = features.groupby('Cluster')['Volatility'].mean().sort_values()
    labels = {cluster_vol.index[0]: 'Calm',
              cluster_vol.index[1]: 'Volatile',
              cluster_vol.index[2]: 'Crisis'}
    features['Regime'] = features['Cluster'].map(labels)

    current_regime = features['Regime'].iloc[-1]
    regime_colors = {'Calm': 'green', 'Volatile': 'orange', 'Crisis': 'red'}
    regime_color = regime_colors.get(current_regime, 'gray')
    st.markdown(f"<h3 style='color:{regime_color}'>Current Market Regime: {current_regime}</h3>",
                unsafe_allow_html=True)

    fig2, ax3 = plt.subplots(figsize=(12, 4))
    colors = {'Calm': 'green', 'Volatile': 'orange', 'Crisis': 'red'}
    for regime, group in features.groupby('Regime'):
        ax3.scatter(group.index, group['Volatility'],
                   c=colors[regime], label=regime, alpha=0.5, s=10)
    ax3.set_ylabel('30-Day Volatility (%)')
    ax3.set_title('Market Regime Detection')
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Energy News Sentiment (NLP)")
    st.write("Real-time sentiment analysis of energy market headlines via VADER NLP model")

    energy_keywords = [
        'energy', 'gas', 'oil', 'carbon', 'climate', 'LNG', 'pipeline', 'TTF',
        'power', 'electricity', 'renewable', 'wind', 'solar', 'nuclear', 'coal',
        'emission', 'EU ETS', 'natural gas', 'crude', 'OPEC', 'hydrogen', 'fuel'
    ]

    rss_feeds = {
        "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "Yahoo Energy": "https://finance.yahoo.com/news/sector-energy/?format=rss",
        "OilPrice": "https://oilprice.com/rss/main",
        "Reuters Business": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "CNBC Energy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19836572",
    }

    headlines = []
    headline_links = []
    headline_sources = []

    for source_name, url in rss_feeds.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries[:50]:
                if count >= 5:
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

    # Sentiment distribution chart
    fig3, ax4 = plt.subplots(figsize=(12, 3))
    bar_colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray'
                  for s in sent_df['Compound']]
    ax4.barh(range(len(sent_df)), sent_df['Compound'], color=bar_colors, height=0.6)
    ax4.set_yticks(range(len(sent_df)))
    ax4.set_yticklabels([h[:60] + '...' if len(h) > 60 else h for h in sent_df['Headline']],
                        fontsize=7)
    ax4.axvline(x=0, color='black', linewidth=0.5)
    ax4.axvline(x=0.05, color='green', linewidth=0.5, linestyle='--', alpha=0.4)
    ax4.axvline(x=-0.05, color='red', linewidth=0.5, linestyle='--', alpha=0.4)
    ax4.set_xlabel('Sentiment Score (VADER Compound)')
    ax4.set_title('Per-Headline Sentiment Distribution')
    ax4.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig3)

    # Headline table
    st.write("**Headlines Detail:**")
    for _, row in sent_df.iterrows():
        score = row['Compound']
        if score > 0.05:
            icon = "🟢"
        elif score < -0.05:
            icon = "🔴"
        else:
            icon = "🟡"
        link = row['Link']
        source = row['Source']
        headline = row['Headline']
        score_str = f"{score:+.3f}"
        if link:
            st.markdown(f"{icon} **[{score_str}]** [{headline}]({link}) — *{source}*")
        else:
            st.markdown(f"{icon} **[{score_str}]** {headline} — *{source}*")