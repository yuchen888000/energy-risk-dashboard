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
    - **Cross-Commodity Matrix**: 4×4 correlation heatmap (full period vs 30-day).
    - **Value at Risk (VaR)**: 95% and 99% historical VaR.
    - **GARCH(1,1) Forecast**: Predicts future volatility from recent 
      shocks (α) and persistence (β). Standard on energy trading desks.
    - **Stress Test**: Simulate price shocks and see impact on volatility, 
      VaR, regime, and country risk.
    - **Portfolio VaR**: Combined risk of holding multiple commodities, 
      accounting for cross-commodity correlations. Shows diversification benefit.

    **AI / ML**
    - **Hybrid Regime Detection**: K-Means clustering + absolute 
      volatility thresholds (Calm < 6%, Volatile 6–12%, Crisis > 12%).
    - **Country Risk Scoring**: Composite index of gas/oil dependency, 
      carbon intensity, energy import share, renewable share across 29 
      European countries. Structural scores are multiplied by real-time 
      market volatility to create dynamic risk assessment.

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

    # ─── Country Data (used by Stress Test and Country Risk sections) ───
    COUNTRIES = ['Germany', 'France', 'Italy', 'Spain', 'Netherlands',
                 'Poland', 'Belgium', 'Austria', 'Greece', 'Czech Republic',
                 'Hungary', 'Romania', 'Bulgaria', 'Finland', 'Sweden',
                 'Denmark', 'Ireland', 'Portugal', 'Lithuania', 'Latvia',
                 'Estonia', 'Slovakia', 'Croatia', 'Slovenia', 'Luxembourg',
                 'Switzerland', 'United Kingdom', 'Norway', 'Turkey']

    gas_dep = {
        2020: [89,98,93,99,0,79,100,82,99,98,85,78,96,97,6,67,100,100,100,100,100,86,55,99,100,100,48,0,99],
        2021: [91,98,93,99,5,82,100,81,99,97,85,76,94,96,8,60,100,100,100,100,100,85,57,99,100,100,47,0,99],
        2022: [95,98,93,99,15,78,100,80,99,97,85,75,92,95,10,55,100,100,100,100,100,85,60,99,100,100,47,0,99],
        2023: [95,98,93,99,68,78,100,80,99,97,85,75,92,95,12,55,100,100,100,100,100,85,60,99,100,100,47,0,99],
        2024: [94,98,93,99,70,77,100,80,99,97,84,74,91,94,12,54,100,100,100,100,100,84,58,99,100,100,46,0,99],
    }
    oil_dep = {
        2020: [96,98,92,99,95,97,99,93,100,97,82,45,100,91,100,100,100,100,100,100,60,92,82,100,100,100,92,0,93],
        2021: [96,98,92,99,96,97,99,93,100,97,83,44,100,90,100,100,100,100,100,100,58,92,80,100,100,100,93,0,93],
        2022: [96,98,93,99,96,97,99,94,100,97,84,42,100,90,100,100,100,100,100,100,55,92,78,100,100,100,94,0,92],
        2023: [97,98,93,99,96,97,99,94,100,97,84,40,100,90,100,100,100,100,100,100,52,92,76,100,100,100,95,0,92],
        2024: [97,98,93,99,96,97,99,94,100,97,84,39,100,90,100,100,100,100,100,100,50,92,75,100,100,100,95,0,92],
    }
    total_dep = {
        2020: [64,47,73,73,45,42,78,60,81,37,55,28,37,42,33,44,86,65,74,46,10,54,53,48,95,75,36,-580,72],
        2021: [64,47,74,73,46,41,78,61,81,37,55,28,37,43,31,45,86,65,74,45,8,53,52,48,95,75,35,-600,72],
        2022: [63,47,75,73,46,40,78,62,83,37,55,28,37,45,29,47,86,65,74,45,6,53,52,48,95,75,35,-620,72],
        2023: [63,47,75,73,46,40,78,62,81,37,55,28,37,45,29,47,86,65,74,45,3,53,52,48,95,75,35,-650,72],
        2024: [62,46,74,72,45,39,77,61,80,36,54,27,36,44,28,46,85,64,73,44,3,52,51,47,94,74,34,-660,71],
    }
    ren_share = {
        2020: [19,19,20,21,14,16,13,37,22,17,14,24,23,44,60,42,16,34,26,42,28,17,31,25,11,28,13,78,18],
        2021: [19,19,19,21,13,16,13,36,22,17,14,24,23,44,63,42,12,34,27,42,28,17,31,25,11,29,14,80,18],
        2022: [21,21,19,22,15,17,13,36,22,18,14,24,23,47,60,42,13,34,28,43,30,17,31,25,12,30,15,85,19],
        2023: [22,22,19,24,17,17,14,36,23,18,14,28,24,48,66,44,14,35,30,44,38,18,32,26,12,32,16,98,20],
        2024: [23,23,20,25,18,18,15,37,24,19,15,29,25,49,67,45,15,36,32,45,40,19,33,27,13,33,17,98,21],
    }
    carbon_int = {
        2020: [195,100,150,130,170,400,160,120,220,300,260,340,480,140,65,110,115,130,200,180,370,250,175,170,100,60,135,80,320],
        2021: [190,98,148,125,165,395,158,118,215,295,255,335,470,135,62,108,112,128,198,178,365,245,172,168,98,58,132,78,315],
        2022: [185,96,145,122,162,385,156,116,212,292,252,325,460,132,60,106,110,126,196,176,355,242,170,166,96,56,130,76,312],
        2023: [180,95,145,120,160,380,155,115,210,290,250,320,450,130,58,105,108,125,195,175,350,240,170,165,95,55,128,75,310],
        2024: [176,93,142,118,157,375,152,113,208,287,247,315,445,128,56,103,106,123,192,172,345,237,168,163,93,54,126,73,305],
    }
    price_sens = {
        2020: [9.0,7.2,8.5,7.0,8.2,7.0,7.8,7.5,8.2,7.2,7.5,6.8,7.5,7.5,3.2,5.5,6.8,6.5,8.0,7.2,7.5,7.0,5.8,6.2,6.0,6.8,7.8,2.2,8.5],
        2021: [9.1,7.3,8.6,7.1,8.3,7.0,7.9,7.6,8.3,7.1,7.5,6.7,7.4,7.6,3.3,5.6,6.9,6.6,8.1,7.3,7.6,7.1,5.9,6.3,6.1,6.9,7.9,2.1,8.5],
        2022: [9.5,7.8,9.0,7.5,8.8,7.2,8.2,8.0,8.8,7.3,7.8,6.8,7.5,8.0,3.5,5.8,7.2,7.0,8.5,7.8,8.0,7.3,6.2,6.8,6.5,7.2,8.2,2.0,8.8],
        2023: [9.2,7.5,8.8,7.2,8.5,6.8,8.0,7.8,8.5,7.0,7.5,6.5,7.2,7.8,3.5,5.8,7.0,6.8,8.2,7.5,7.8,7.2,6.0,6.5,6.2,7.0,8.0,2.0,8.5],
        2024: [9.0,7.3,8.6,7.0,8.3,6.6,7.8,7.6,8.3,6.8,7.3,6.3,7.0,7.6,3.4,5.6,6.8,6.6,8.0,7.3,7.6,7.0,5.8,6.3,6.0,6.8,7.8,1.8,8.3],
    }

    # Commodity-aware dependency column
    if selected_commodity in ['TTF Natural Gas']:
        dep_col = 'Gas Dep. (%)'
        dep_label = 'Gas Import Dependency'
    elif selected_commodity in ['WTI Crude Oil', 'Brent Crude Oil']:
        dep_col = 'Oil Dep. (%)'
        dep_label = 'Oil Import Dependency'
    else:
        dep_col = 'Total Energy Dep. (%)'
        dep_label = 'Total Energy Dependency'

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

    # ─── Section 3b: Cross-Commodity Correlation Matrix ───
    st.subheader("Cross-Commodity Correlation Matrix")
    st.write("How are European energy commodities moving relative to each other right now?")

    @st.cache_data(ttl=3600, show_spinner="Computing cross-commodity correlations...")
    def get_correlation_matrix(start, end):
        tickers = {"TTF Gas": "TTF=F", "WTI Oil": "CL=F", "Brent Oil": "BZ=F", "EU Carbon": "KEUA"}
        prices = {}
        for name, ticker in tickers.items():
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if len(data) > 30:
                    prices[name] = data['Close'].squeeze().pct_change()
            except Exception:
                continue
        if len(prices) < 2:
            return None, None
        returns_df = pd.DataFrame(prices).dropna()
        corr_full = returns_df.corr().round(3)
        corr_30d = returns_df.tail(30).corr().round(3)
        return corr_full, corr_30d

    corr_full, corr_30d = get_correlation_matrix(start_date, end_date)

    if corr_full is not None:
        cm1, cm2 = st.columns(2)

        with cm1:
            st.write("**Full Period Correlation:**")
            fig_corr1, ax_corr1 = plt.subplots(figsize=(5, 4))
            im1 = ax_corr1.imshow(corr_full, cmap='RdYlGn', vmin=-1, vmax=1)
            ax_corr1.set_xticks(range(len(corr_full.columns)))
            ax_corr1.set_yticks(range(len(corr_full.columns)))
            ax_corr1.set_xticklabels(corr_full.columns, fontsize=9, rotation=45, ha='right')
            ax_corr1.set_yticklabels(corr_full.columns, fontsize=9)
            for i in range(len(corr_full)):
                for j in range(len(corr_full)):
                    ax_corr1.text(j, i, f"{corr_full.iloc[i, j]:.2f}",
                                  ha='center', va='center', fontsize=10, fontweight='bold',
                                  color='white' if abs(corr_full.iloc[i, j]) > 0.5 else 'black')
            plt.colorbar(im1, ax=ax_corr1, shrink=0.8)
            ax_corr1.set_title('Full Period')
            plt.tight_layout()
            st.pyplot(fig_corr1)

        with cm2:
            st.write("**Last 30 Days Correlation:**")
            fig_corr2, ax_corr2 = plt.subplots(figsize=(5, 4))
            im2 = ax_corr2.imshow(corr_30d, cmap='RdYlGn', vmin=-1, vmax=1)
            ax_corr2.set_xticks(range(len(corr_30d.columns)))
            ax_corr2.set_yticks(range(len(corr_30d.columns)))
            ax_corr2.set_xticklabels(corr_30d.columns, fontsize=9, rotation=45, ha='right')
            ax_corr2.set_yticklabels(corr_30d.columns, fontsize=9)
            for i in range(len(corr_30d)):
                for j in range(len(corr_30d)):
                    ax_corr2.text(j, i, f"{corr_30d.iloc[i, j]:.2f}",
                                  ha='center', va='center', fontsize=10, fontweight='bold',
                                  color='white' if abs(corr_30d.iloc[i, j]) > 0.5 else 'black')
            plt.colorbar(im2, ax=ax_corr2, shrink=0.8)
            ax_corr2.set_title('Last 30 Days')
            plt.tight_layout()
            st.pyplot(fig_corr2)

        st.caption("Green = positive correlation (move together). Red = negative (move inversely). "
                   "Compare full-period vs 30-day to detect regime shifts in cross-commodity relationships.")
    else:
        st.info("Not enough data to compute cross-commodity correlations.")

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

    # ─── Section 5a: Stress Test Scenario ───
    st.subheader("Stress Test Scenario")
    st.write(f"What happens if {selected_commodity} prices spike? Simulate the impact on volatility, VaR, and country risk.")

    stress_pct = st.slider("Simulate price shock (%)", min_value=-50, max_value=100, value=30, step=5,
                            help="Positive = price spike, Negative = price crash")

    # Estimate stressed volatility: current vol * shock multiplier
    # Empirical relationship: a 50% price shock roughly doubles short-term volatility
    shock_vol_multiplier = 1 + abs(stress_pct) / 50
    stressed_vol = latest_vol * shock_vol_multiplier
    stressed_var_95 = var_95 * shock_vol_multiplier
    stressed_var_99 = var_99 * shock_vol_multiplier

    st1, st2, st3, st4 = st.columns(4)
    st1.metric("Current Volatility", f"{latest_vol:.2f}%")
    st2.metric("Stressed Volatility", f"{stressed_vol:.2f}%",
               delta=f"+{stressed_vol - latest_vol:.2f}%")
    st3.metric("Stressed VaR 95%", f"{stressed_var_95:.2f}%",
               delta=f"{stressed_var_95 - var_95:.2f}%")
    st4.metric("Stressed VaR 99%", f"{stressed_var_99:.2f}%",
               delta=f"{stressed_var_99 - var_99:.2f}%")

    # Stressed regime
    if stressed_vol > 12:
        stressed_regime = "🔴 Crisis"
    elif stressed_vol > 6:
        stressed_regime = "🟡 Volatile"
    else:
        stressed_regime = "🟢 Calm"

    st.markdown(f"**Under a {stress_pct:+d}% price shock:** Regime shifts to **{stressed_regime}** "
                f"(from {current_regime})")

    # Top 5 most impacted countries under stress
    st.write("**Most impacted countries under this scenario:**")

    if selected_commodity in ['TTF Natural Gas']:
        dep_vals = gas_dep.get(2024, gas_dep[2023])
    elif selected_commodity in ['WTI Crude Oil', 'Brent Crude Oil']:
        dep_vals = oil_dep.get(2024, oil_dep[2023])
    else:
        dep_vals = [max(x, 0) for x in total_dep.get(2024, total_dep[2023])]

    stress_impact = []
    for i, country in enumerate(COUNTRIES):
        dep_pct = max(dep_vals[i], 0) / 100
        country_stressed_vol = stressed_vol * dep_pct
        normal_vol = latest_vol * dep_pct
        stress_impact.append({
            'Country': country,
            'Normal Adj. Vol': f"{normal_vol:.1f}%",
            'Stressed Adj. Vol': f"{country_stressed_vol:.1f}%",
            'Vol Increase': f"+{country_stressed_vol - normal_vol:.1f}%",
            dep_label: f"{dep_vals[i]:.0f}%",
        })
    stress_df = pd.DataFrame(stress_impact)
    stress_df['sort_key'] = [float(x.replace('%', '').replace('+', '')) for x in stress_df['Vol Increase']]
    stress_df = stress_df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    st.dataframe(stress_df.head(10).reset_index(drop=True), use_container_width=True)
    st.caption(f"Stressed volatility = current volatility × shock multiplier ({shock_vol_multiplier:.2f}x), "
               f"then weighted by each country's {dep_label.lower()}.")

    # ─── Section 5ab: Portfolio VaR ───
    st.subheader("Portfolio Value at Risk")
    st.write("If you hold multiple energy commodities, what is the combined portfolio risk?")

    @st.cache_data(ttl=3600, show_spinner="Computing portfolio returns...")
    def get_portfolio_returns(start, end):
        tickers = {"TTF Gas": "TTF=F", "WTI Oil": "CL=F", "Brent Oil": "BZ=F", "EU Carbon": "KEUA"}
        returns = {}
        for name, ticker in tickers.items():
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if len(data) > 30:
                    returns[name] = data['Close'].squeeze().pct_change()
            except Exception:
                continue
        if len(returns) < 2:
            return None
        return pd.DataFrame(returns).dropna()

    port_returns = get_portfolio_returns(start_date, end_date)

    if port_returns is not None and len(port_returns.columns) >= 2:
        st.write("**Set Portfolio Weights (must sum to 100%):**")
        pw1, pw2, pw3, pw4 = st.columns(4)
        w_gas = pw1.number_input("TTF Gas %", min_value=0, max_value=100, value=40, step=5)
        w_wti = pw2.number_input("WTI Oil %", min_value=0, max_value=100, value=30, step=5)
        w_brent = pw3.number_input("Brent Oil %", min_value=0, max_value=100, value=20, step=5)
        w_carbon = pw4.number_input("EU Carbon %", min_value=0, max_value=100, value=10, step=5)

        total_weight = w_gas + w_wti + w_brent + w_carbon

        if total_weight != 100:
            st.warning(f"Weights sum to {total_weight}% — please adjust to 100%.")
        else:
            weights = {}
            if 'TTF Gas' in port_returns.columns:
                weights['TTF Gas'] = w_gas / 100
            if 'WTI Oil' in port_returns.columns:
                weights['WTI Oil'] = w_wti / 100
            if 'Brent Oil' in port_returns.columns:
                weights['Brent Oil'] = w_brent / 100
            if 'EU Carbon' in port_returns.columns:
                weights['EU Carbon'] = w_carbon / 100

            # Normalize weights for available commodities
            available = [k for k in weights if k in port_returns.columns]
            w_array = np.array([weights[k] for k in available])
            if w_array.sum() > 0:
                w_array = w_array / w_array.sum()  # re-normalize

            # Portfolio returns (weighted sum)
            port_ret = (port_returns[available] * w_array).sum(axis=1)

            # Portfolio metrics
            port_vol = port_ret.rolling(30).std().dropna().iloc[-1] * 100
            port_var_95 = np.percentile(port_ret.dropna(), 5) * 100
            port_var_99 = np.percentile(port_ret.dropna(), 1) * 100
            port_max_loss = port_ret.dropna().min() * 100

            # Individual VaRs for comparison
            individual_vars = {}
            for col in available:
                individual_vars[col] = np.percentile(port_returns[col].dropna(), 5) * 100

            # Diversification benefit
            undiversified_var = sum(abs(individual_vars[k]) * weights[k] for k in available)
            diversification_benefit = undiversified_var - abs(port_var_95)

            pv1, pv2, pv3, pv4 = st.columns(4)
            pv1.metric("Portfolio Volatility", f"{port_vol:.2f}%")
            pv2.metric("Portfolio VaR 95%", f"{port_var_95:.2f}%")
            pv3.metric("Portfolio VaR 99%", f"{port_var_99:.2f}%")
            pv4.metric("Diversification Benefit", f"{diversification_benefit:.2f}%",
                       help="Risk reduction from holding multiple commodities vs single")

            # Portfolio return distribution
            fig_pvar, (ax_pd, ax_pc) = plt.subplots(1, 2, figsize=(14, 4))

            ax_pd.hist(port_ret.dropna() * 100, bins=60, color='navy', alpha=0.7, edgecolor='white')
            ax_pd.axvline(x=port_var_95, color='red', linewidth=2, linestyle='--',
                          label=f'95% VaR: {port_var_95:.2f}%')
            ax_pd.axvline(x=port_var_99, color='darkred', linewidth=2, linestyle=':',
                          label=f'99% VaR: {port_var_99:.2f}%')
            ax_pd.set_xlabel('Daily Portfolio Returns (%)')
            ax_pd.set_ylabel('Frequency')
            ax_pd.set_title('Portfolio Return Distribution')
            ax_pd.legend(fontsize=8)

            # Compare individual vs portfolio VaR
            compare_names = available + ['Portfolio']
            compare_vars = [individual_vars[k] for k in available] + [port_var_95]
            compare_colors = ['steelblue', 'saddlebrown', 'darkred', 'seagreen'][:len(available)] + ['navy']
            ax_pc.barh(range(len(compare_names)), [abs(v) for v in compare_vars],
                       color=compare_colors, height=0.5)
            ax_pc.set_yticks(range(len(compare_names)))
            ax_pc.set_yticklabels(compare_names, fontsize=9)
            ax_pc.set_xlabel('VaR 95% (absolute %)')
            ax_pc.set_title('Individual vs Portfolio VaR')
            ax_pc.invert_yaxis()

            plt.tight_layout()
            st.pyplot(fig_pvar)

            # Weight composition pie chart
            st.write("**Portfolio Composition:**")
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            pie_labels = [f"{k}\n({weights[k]*100:.0f}%)" for k in available]
            pie_colors = ['steelblue', 'saddlebrown', 'darkred', 'seagreen'][:len(available)]
            ax_pie.pie([weights[k] for k in available], labels=pie_labels, colors=pie_colors,
                      autopct='', startangle=90)
            ax_pie.set_title('Portfolio Weight Allocation')
            plt.tight_layout()
            st.pyplot(fig_pie)

            st.caption(f"Portfolio VaR accounts for cross-commodity correlations — "
                       f"diversification reduces risk by {diversification_benefit:.2f}% compared to "
                       f"holding each commodity independently. Weights are user-adjustable.")
    else:
        st.info("Not enough multi-commodity data to compute Portfolio VaR.")


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

    # Static structural risk score
    cr_df['Structural Score'] = (
        cr_df['Dep Clipped'] * 0.25 +
        cr_df['Carbon Int. (tCO2/M€)'].rank(pct=True) * 100 * 0.15 +
        cr_df['Total Clipped'] * 0.15 +
        (100 - cr_df['Renewable (%)']) * 0.15 +
        cr_df['Dep Clipped'].rank(pct=True) * 100 * 0.15 +
        cr_df['Price Sensitivity'] * 10 * 0.15
    ).round(1)

    # Dynamic risk = structural vulnerability × country-specific volatility impact
    # Countries with higher dependency feel the same market volatility much more
    vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
    vol_ratio_clamped = min(max(vol_ratio, 0.5), 3.0)

    # Country-specific multiplier: base vol_ratio weighted by dependency
    # High dependency country (99%) gets full multiplier, low dependency (10%) gets much less
    cr_df['Country Vol Multiplier'] = (
        0.5 + 0.5 * vol_ratio_clamped * (cr_df['Dep Clipped'] / 100)
    ).round(2)
    # Floor at 0.5 so no country goes to zero risk

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
        display_df = cr_df[['Country', 'Risk Score', 'Country Vol Multiplier', 'Risk Level', dep_col,
                            'Total Energy Dep. (%)', 'Carbon Int. (tCO2/M€)',
                            'Price Sensitivity', 'Renewable (%)']].reset_index(drop=True)
        display_df.index = display_df.index + 1
        st.dataframe(display_df, use_container_width=True, height=400)

    with cr_col2:
        selected_country = st.selectbox("Select Country for Detail", cr_df['Country'].tolist())
        country_data = cr_df[cr_df['Country'] == selected_country].iloc[0]

        # Country-specific real-time risk signal (same format as top section)
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
    st.caption(f"Adjusted volatility = {selected_commodity} 30-day rolling volatility × {selected_country}'s {dep_label.lower()} ({country_data[dep_col]:.0f}%). "
               f"Current: {country_vol.iloc[-1]:.2f}%")

    # Per-country news sentiment
    st.write(f"**{selected_country} — Current Energy News Sentiment:**")
    country_rss_url = f"https://news.google.com/rss/search?q={selected_country}+energy+{commodity['rss_query'].split('+')[0]}+when:7d&hl=en"
    try:
        country_feed = feedparser.parse(country_rss_url)
        country_headlines = []
        for entry in country_feed.entries[:20]:
            title = entry.title
            if selected_country.lower() in title.lower() or any(kw.lower() in title.lower() for kw in commodity['keywords'][:3]):
                country_headlines.append(title)
            if len(country_headlines) >= 5:
                break

        if country_headlines:
            sia_country = SentimentIntensityAnalyzer()
            country_scores = [sia_country.polarity_scores(h)['compound'] for h in country_headlines]
            country_avg = np.mean(country_scores)

            if country_avg > 0.05:
                c_sent_label = "Positive"
                c_sent_color = "green"
            elif country_avg < -0.05:
                c_sent_label = "Negative"
                c_sent_color = "red"
            else:
                c_sent_label = "Neutral"
                c_sent_color = "orange"

            st.markdown(f"<span style='color:{c_sent_color}; font-weight:bold'>{c_sent_label} ({country_avg:+.3f})</span> based on {len(country_headlines)} recent headlines",
                        unsafe_allow_html=True)
            for i, h in enumerate(country_headlines[:3]):
                sc = country_scores[i]
                icon = "🟢" if sc > 0.05 else "🔴" if sc < -0.05 else "🟡"
                st.markdown(f"{icon} **[{sc:+.3f}]** {h}")
        else:
            st.info(f"No recent energy news found specifically for {selected_country}.")
    except Exception:
        st.info(f"Could not fetch news for {selected_country}.")

    st.caption(f"Structural Score = {dep_label} (25%) + Carbon Intensity rank (15%) + Total Energy Dep. (15%) + Inverse Renewable (15%) + {dep_label} rank (15%) + Price Sensitivity (15%). Dynamic Risk = Structural × Country-specific volatility multiplier (weighted by dependency). Source: Eurostat (nrg_ind_id, sdg_07_50, nrg_ind_ren, nrg_ind_ei), EEA, IEA. 2024 = preliminary.")

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

    cr_export = cr_df[['Country', 'Risk Score', 'Structural Score', 'Country Vol Multiplier',
                        'Risk Level', dep_col,
                        'Total Energy Dep. (%)', 'Carbon Int. (tCO2/M€)',
                        'Price Sensitivity', 'Renewable (%)']].copy()
    cr_export.columns = ['Country', 'Dynamic Risk Score', 'Structural Score', 'Vol Multiplier',
                         'Risk Level', dep_label,
                         'Total Energy Dependency (%)', 'Carbon Intensity (tCO2/M€ GDP)',
                         'Price Sensitivity (1-10)', 'Renewable Share (%)']
    cr_csv = cr_export.to_csv(index=False)
    st.download_button(
        label=f"Download Country Risk Data ({selected_year}, CSV)",
        data=cr_csv,
        file_name=f"country_risk_{selected_year}.csv",
        mime="text/csv"
    )
