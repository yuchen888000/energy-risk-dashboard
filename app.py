import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

    # Macro events
    macro_events = {
        "2021-07-14": "EU Fit for 55 Package",
        "2022-02-24": "Russia invades Ukraine",
        "2022-06-01": "EU bans Russian oil",
        "2022-09-26": "Nord Stream sabotage",
        "2023-01-01": "EU gas price cap",
        "2024-01-01": "EU ETS reform",
        "2023-04-18": "EU ETS 2 legislation passed",
    }

    # Price chart with macro events
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
    from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

st.subheader("Market Regime Clustering (AI)")
st.write("K-Means unsupervised learning identifies 3 market states based on volatility patterns")

# Prepare features
features = df[['Volatility', 'Rolling Correlation']].dropna()
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
features = features.copy()
features['Cluster'] = kmeans.fit_predict(scaled)

# Label clusters by average volatility
cluster_vol = features.groupby('Cluster')['Volatility'].mean().sort_values()
labels = {cluster_vol.index[0]: '🟢 Calm', 
          cluster_vol.index[1]: '🟡 Volatile', 
          cluster_vol.index[2]: '🔴 Crisis'}
features['Regime'] = features['Cluster'].map(labels)

# Current regime
current_regime = features['Regime'].iloc[-1]
st.markdown(f"### Current Market Regime: {current_regime}")

# Plot
fig2, ax3 = plt.subplots(figsize=(12, 4))
colors = {'🟢 Calm': 'green', '🟡 Volatile': 'orange', '🔴 Crisis': 'red'}
for regime, group in features.groupby('Regime'):
    ax3.scatter(group.index, group['Volatility'], 
               c=colors[regime], label=regime, alpha=0.5, s=10)
ax3.set_ylabel('30-Day Volatility (%)')
ax3.set_title('Market Regime Detection')
ax3.legend()
plt.tight_layout()
st.pyplot(fig2)