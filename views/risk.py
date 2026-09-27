"""Risk page: market risk of the selected commodity and of a multi-commodity book.

Designed with a trading-house middle office (risk control) in mind.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import common
from country_data import COUNTRIES, dependency_for

ctx = common.context()
selected_commodity = ctx.selected_commodity
commodity = ctx.commodity
compare_label = ctx.compare_label
start_date, end_date = ctx.start_date, ctx.end_date
CARBON_TICKER = ctx.carbon["ticker"]
CARBON_LABEL = ctx.carbon["label"]
CARBON_SHORT = ctx.carbon["short"]
CARBON_NOTE = ctx.carbon["note"]

with st.sidebar.expander("Methodology — Risk page"):
    st.markdown("""
    - **30-Day Rolling Volatility**: Std dev of daily returns over 30 days.
    - **Rolling Correlation**: Pearson correlation of daily returns with the comparison asset over 30 days.
    - **Cross-Commodity Matrix**: 4×4 correlation heatmap (full period vs 30-day).
    - **Value at Risk (VaR)**: 95% and 99% historical VaR.
    - **GARCH(1,1) Forecast**: Predicts future volatility from recent
      shocks (α) and persistence (β).
    - **Hybrid Regime Detection**: K-Means (3 clusters on vol + correlation) + absolute
      thresholds. Agreement → unanimous label. Disagreement in boundary zone (4–9% vol)
      → K-Means wins. Disagreement outside boundary → threshold wins.
    - **Stress Test**: Simulate price shocks and see impact on volatility,
      VaR, regime, and country exposure.
    - **Portfolio VaR**: Combined risk of holding multiple commodities,
      accounting for cross-commodity correlations. Shows diversification benefit.
    """)

st.title("Market Risk")
st.markdown(f"Analyzing **{selected_commodity}** vs {compare_label}")
if not ctx.is_carbon:
    st.caption(f"Carbon benchmark: **{CARBON_LABEL}**. {CARBON_NOTE}")

core = common.compute_core(commodity['ticker'], ctx.compare_ticker, start_date, end_date)
if core is None:
    st.warning("Please select a longer time range (at least 30 days of data required).")
    st.stop()
if not ctx.is_carbon and not core['has_compare']:
    st.caption(f"{CARBON_LABEL} has limited coverage over the selected range — "
               "widen the date window for a fuller comparison.")

df_analysis = core['df_analysis']
latest_vol, avg_vol = core['latest_vol'], core['avg_vol']
returns_clean = core['returns_clean']
var_95, var_99 = core['var_95'], core['var_99']
overall_corr = core['overall_corr']
risk_level, risk_color = core['risk_level'], core['risk_color']
dep_col, dep_label, dep_table = dependency_for(selected_commodity)

# ─── Section 1: Risk Signal ───
st.subheader(f"Current Risk Signal — {selected_commodity}")
st.markdown(f"<h2 style='color:{risk_color}'>{risk_level}</h2>",
            unsafe_allow_html=True)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Current Volatility", f"{latest_vol:.2f}%")
mc2.metric("Average Volatility", f"{avg_vol:.2f}%")
mc3.metric("Correlation", f"{overall_corr:.2f}")
mc4.metric("VaR (95%, 1-day)", f"{var_95:.2f}%")

# ─── Section 2: Price Chart ───
# FIX: corrected policy event dates
macro_events = {
    "2021-07-14": "EU Fit for 55",
    "2022-02-24": "Russia invades Ukraine",
    "2022-06-01": "EU bans Russian oil",
    "2022-09-26": "Nord Stream sabotage",
    "2023-01-01": "EU gas price cap",
    "2023-04-18": "EU ETS 2 passed",
    "2024-01-01": "EU ETS reform",
    "2025-12-31": "CBAM transition ends",
    "2026-01-01": "CBAM full enforcement",
    "2027-01-01": "EU ETS 2 starts",
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

_panel = common.returns_panel(start_date, end_date, CARBON_SHORT, CARBON_TICKER)
if _panel is not None and _panel.dropna().shape[0] > 30:
    _returns_df = _panel.dropna()
    corr_full = _returns_df.corr().round(3)
    corr_30d = _returns_df.tail(30).corr().round(3)
else:
    corr_full, corr_30d = None, None

if corr_full is not None:
    cm1, cm2 = st.columns(2)

    with cm1:
        st.write("**Full Period Correlation:**")
        fig_corr1, ax_corr1 = plt.subplots(figsize=(5, 5))
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
        fig_corr1.subplots_adjust(bottom=0.22)
        plt.tight_layout()
        st.pyplot(fig_corr1)

    with cm2:
        st.write("**Last 30 Days Correlation:**")
        fig_corr2, ax_corr2 = plt.subplots(figsize=(5, 5))
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
        fig_corr2.subplots_adjust(bottom=0.22)
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

garch = common.fit_garch(returns_clean)
if garch is not None:
    forecast_vol = pd.Series(garch['forecast_vol'])
    current_cond_vol = garch['current_cond_vol']
    params = garch['params']

    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Current GARCH Vol (daily)", f"{current_cond_vol:.2f}%")
    gc2.metric("5-Day Forecast Vol", f"{forecast_vol.iloc[4]:.2f}%")
    gc3.metric("10-Day Forecast Vol", f"{forecast_vol.iloc[9]:.2f}%")

    fig_garch, (ax_cv, ax_fc) = plt.subplots(1, 2, figsize=(14, 4))

    cond_vol = garch['conditional_volatility']
    ax_cv.plot(cond_vol.index, cond_vol, color='purple', linewidth=0.8, alpha=0.8)
    ax_cv.set_ylabel('Conditional Volatility (daily %)')
    ax_cv.set_title('GARCH(1,1) Conditional Volatility')
    ax_cv.fill_between(cond_vol.index, cond_vol, 0, alpha=0.1, color='purple')

    _ci_lo, _ci_hi = garch['ci_lo'], garch['ci_hi']

    forecast_days = list(range(1, 11))
    ax_fc.plot(forecast_days, forecast_vol.values, color='purple', linewidth=2, marker='o', markersize=5)
    ax_fc.fill_between(forecast_days, _ci_lo, _ci_hi,
                       alpha=0.18, color='purple', label='Bootstrap 90% CI (500 draws)')
    ax_fc.set_xlabel('Days Ahead')
    ax_fc.set_ylabel('Forecast Volatility (daily %)')
    ax_fc.set_title('10-Day Volatility Forecast')
    ax_fc.set_xticks(forecast_days)
    ax_fc.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig_garch)

    with st.expander("GARCH(1,1) Model Parameters"):
        st.write(f"**omega (ω):** {params['omega']:.6f}")
        st.write(f"**alpha (α):** {params['alpha[1]']:.4f} — reaction to recent shocks")
        st.write(f"**beta (β):** {params['beta[1]']:.4f} — persistence of volatility")
        persistence = params['alpha[1]'] + params['beta[1]']
        st.write(f"**α + β = {persistence:.4f}** — "
                 f"{'high persistence (close to 1)' if persistence > 0.95 else 'moderate persistence'}")
        st.write(f"**Log-Likelihood:** {garch['loglikelihood']:.2f}")
        st.caption("Confidence band: bootstrap residual resampling — 500 draws of standardised "
                   "innovations propagated through the GARCH recursion; 5th–95th percentile shown.")

else:
    st.caption("GARCH forecast unavailable for this date range — try a longer window.")


# ─── Section 5: Market Regime Clustering ───
st.subheader("Market Regime Clustering (AI)")
st.write("Hybrid approach: K-Means clustering + absolute volatility thresholds for regime labeling")

features = common.compute_regimes(df_analysis[['Volatility', 'Rolling Correlation']])

current_regime = features['Regime'].iloc[-1]
regime_colors = {'Calm': 'green', 'Volatile': 'orange', 'Crisis': 'red'}
regime_color = regime_colors.get(current_regime, 'gray')
st.markdown(f"<h3 style='color:{regime_color}'>Current Market Regime: {current_regime}</h3>",
            unsafe_allow_html=True)
st.caption("Thresholds: Calm < 6% · Volatile 6–12% · Crisis > 12% (30-day rolling volatility) · "
           "K-Means overrides threshold in 4–9% boundary zone using volatility + correlation")

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
    st.dataframe(regime_stats, width="stretch")


# ─── Section 5a: Stress Test Scenario ───
st.subheader("Stress Test Scenario")
st.write(f"What happens if {selected_commodity} prices spike? Simulate the impact on volatility, VaR, and country risk.")

stress_pct = st.slider("Simulate price shock (%)", min_value=-50, max_value=100, value=30, step=5,
                        help="Positive = price spike, Negative = price crash")

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

# FIX: avoid uninformative "Regime shifts to Calm (from Calm)"
if stressed_vol > 12:
    stressed_regime = "🔴 Crisis"
    stressed_regime_name = "Crisis"
elif stressed_vol > 6:
    stressed_regime = "🟡 Volatile"
    stressed_regime_name = "Volatile"
else:
    stressed_regime = "🟢 Calm"
    stressed_regime_name = "Calm"

if stressed_regime_name == current_regime:
    st.markdown(f"**Under a {stress_pct:+d}% price shock:** Regime remains **{stressed_regime}** "
                f"— volatility stays within {current_regime} threshold ({stressed_vol:.1f}%)")
else:
    st.markdown(f"**Under a {stress_pct:+d}% price shock:** Regime shifts to **{stressed_regime}** "
                f"(from {current_regime})")

st.write("**Most impacted countries under this scenario:**")

dep_vals = [max(x, 0) for x in dep_table[2024]]

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
st.dataframe(stress_df.head(10).reset_index(drop=True), width="stretch")
st.caption(f"Stressed volatility = current volatility × shock multiplier ({shock_vol_multiplier:.2f}x), "
           f"then weighted by each country's {dep_label.lower()}.")


# ─── Section 5ab: Portfolio VaR ───
st.subheader("Portfolio Value at Risk")
st.write("If you hold multiple energy commodities, what is the combined portfolio risk?")

port_returns = common.returns_panel(start_date, end_date, CARBON_SHORT, CARBON_TICKER)
if port_returns is not None:
    port_returns = port_returns.dropna()

if port_returns is not None and len(port_returns.columns) >= 2:
    st.write("**Set Portfolio Weights:**")
    pw1, pw2, pw3, pw4 = st.columns(4)
    w_gas = pw1.number_input("TTF Gas %", min_value=0, max_value=100, value=40, step=5)
    w_wti = pw2.number_input("WTI Oil %", min_value=0, max_value=100, value=30, step=5)
    w_brent = pw3.number_input("Brent Oil %", min_value=0, max_value=100, value=20, step=5)
    w_carbon = pw4.number_input(f"{CARBON_SHORT} %", min_value=0, max_value=100, value=10, step=5)

    total_weight = w_gas + w_wti + w_brent + w_carbon

    if total_weight == 0:
        st.warning("Please set at least one weight above 0%.")
    else:
        if total_weight != 100:
            st.caption(f"Weights sum to {total_weight}% — auto-normalized to 100% for calculation.")

        raw_weights = {}
        if 'TTF Gas' in port_returns.columns:
            raw_weights['TTF Gas'] = w_gas
        if 'WTI Oil' in port_returns.columns:
            raw_weights['WTI Oil'] = w_wti
        if 'Brent Oil' in port_returns.columns:
            raw_weights['Brent Oil'] = w_brent
        if CARBON_SHORT in port_returns.columns:
            raw_weights[CARBON_SHORT] = w_carbon

        available = [k for k in raw_weights if k in port_returns.columns]
        w_array = np.array([raw_weights[k] for k in available], dtype=float)
        if w_array.sum() > 0:
            w_array = w_array / w_array.sum()  # normalized weights (sum to 1)

        # Portfolio returns
        port_ret = (port_returns[available] * w_array).sum(axis=1)

        # Portfolio metrics
        port_vol = port_ret.rolling(30).std().dropna().iloc[-1] * 100
        port_var_95 = np.percentile(port_ret.dropna(), 5) * 100
        port_var_99 = np.percentile(port_ret.dropna(), 1) * 100

        # Individual VaRs
        individual_vars = {}
        for col in available:
            individual_vars[col] = np.percentile(port_returns[col].dropna(), 5) * 100

        # FIX: diversification benefit uses normalized w_array, not raw weights
        undiversified_var = sum(abs(individual_vars[k]) * w_array[i] for i, k in enumerate(available))
        diversification_benefit = undiversified_var - abs(port_var_95)

        pv1, pv2, pv3, pv4 = st.columns(4)
        pv1.metric("Portfolio Volatility", f"{port_vol:.2f}%")
        pv2.metric("Portfolio VaR 95%", f"{port_var_95:.2f}%")
        pv3.metric("Portfolio VaR 99%", f"{port_var_99:.2f}%")
        pv4.metric("Diversification Benefit", f"{diversification_benefit:.2f}%",
                   help="Risk reduction from holding multiple commodities vs single")

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

        st.write("**Portfolio Composition:**")
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        pie_labels = [f"{k}\n({w_array[i]*100:.0f}%)" for i, k in enumerate(available)]
        pie_colors = ['steelblue', 'saddlebrown', 'darkred', 'seagreen'][:len(available)]
        ax_pie.pie(w_array, labels=pie_labels, colors=pie_colors,
                  autopct='', startangle=90)
        ax_pie.set_title('Portfolio Weight Allocation')
        plt.tight_layout()
        st.pyplot(fig_pie)

        st.caption(f"Portfolio VaR accounts for cross-commodity correlations — "
                   f"diversification reduces risk by {diversification_benefit:.2f}% compared to "
                   f"holding each commodity independently. Weights are user-adjustable.")
else:
    st.info("Not enough multi-commodity data to compute Portfolio VaR.")


# ─── Section 5ac: Positions & Limits ───
st.subheader("Positions & Limits")
st.write("Risk of a book of positions in euros: VaR, Expected Shortfall, limit usage and a VaR backtest.")

book_panel = common.returns_panel(start_date, end_date, CARBON_SHORT, CARBON_TICKER)
_LOOKBACK = 250   # days of history behind each VaR estimate
_TEST_DAYS = 250  # days in the backtest

if book_panel is None:
    st.info("Not enough multi-commodity data to compute position risk.")
else:
    st.write("**Positions** (€m market value; positive = long, negative = short):")
    _defaults = {"TTF Gas": 10.0, "WTI Oil": 0.0, "Brent Oil": -5.0, CARBON_SHORT: 3.0}
    _pos_cols = st.columns(len(book_panel.columns) + 1)
    positions_m = {}
    for _col, _name in zip(_pos_cols, book_panel.columns):
        positions_m[_name] = _col.number_input(f"{_name} (€m)", value=_defaults.get(_name, 0.0),
                                               step=0.5, format="%.1f", key=f"pos_{_name}")
    var_limit_m = _pos_cols[-1].number_input("VaR 95% limit (€m)", min_value=0.1, value=1.5,
                                             step=0.1, format="%.1f", key="var_limit")

    positions = {k: v * 1e6 for k, v in positions_m.items() if v != 0}
    book_returns = book_panel[list(positions)].dropna() if positions else None

    if not positions:
        st.info("Enter at least one non-zero position.")
    elif len(book_returns) < _LOOKBACK + 50:
        st.info(f"Position risk needs at least {_LOOKBACK + 50} days of overlapping history for "
                "the instruments held — widen the date range.")
    else:
        pnl = common.book_pnl(book_returns, positions)
        recent = pnl.tail(_LOOKBACK)
        book_var95 = common.hist_var(recent, 0.95)
        book_var99 = common.hist_var(recent, 0.99)
        book_es975 = common.hist_es(recent, 0.975)
        usage = book_var95 / (var_limit_m * 1e6)

        if usage > 1:
            usage_status, usage_color = "🔴 LIMIT BREACH", "red"
        elif usage >= 0.8:
            usage_status, usage_color = "🟡 NEAR LIMIT", "orange"
        else:
            usage_status, usage_color = "🟢 WITHIN LIMIT", "green"

        pl1, pl2, pl3, pl4 = st.columns(4)
        pl1.metric("VaR 95% (1-day)", f"€{book_var95 / 1e6:,.2f}m")
        pl2.metric("VaR 99% (1-day)", f"€{book_var99 / 1e6:,.2f}m")
        pl3.metric("Expected Shortfall 97.5%", f"€{book_es975 / 1e6:,.2f}m",
                   help="Average loss on the worst 2.5% of days — the Basel FRTB measure.")
        pl4.metric("Limit usage (VaR 95%)", f"{usage:.0%}",
                   help=f"VaR 95% of €{book_var95 / 1e6:,.2f}m against a limit of €{var_limit_m:,.1f}m.")
        st.markdown(f"<span style='color:{usage_color}; font-weight:bold'>{usage_status}</span> — "
                    f"€{book_var95 / 1e6:,.2f}m of €{var_limit_m:,.1f}m used",
                    unsafe_allow_html=True)
        st.progress(min(usage, 1.0))

        # Standalone VaR per position vs the book: the gap is the diversification benefit
        standalone = {k: common.hist_var(book_returns[k].tail(_LOOKBACK) * v, 0.95)
                      for k, v in positions.items()}
        contrib_df = pd.DataFrame({
            'Position (€m)': [positions[k] / 1e6 for k in positions],
            'Standalone VaR 95% (€m)': [standalone[k] / 1e6 for k in positions],
        }, index=list(positions)).round(2)
        contrib_df.loc['Book (net)'] = [sum(positions.values()) / 1e6, round(book_var95 / 1e6, 2)]
        st.dataframe(contrib_df, width="stretch")
        st.caption(f"Diversification benefit: €{(sum(standalone.values()) - book_var95) / 1e6:,.2f}m "
                   "(sum of standalone VaRs minus book VaR).")

        # ── VaR backtest ──
        st.write(f"**VaR Backtest — last {_TEST_DAYS} trading days**")
        bt = common.var_backtest(pnl, window=_LOOKBACK, test_days=_TEST_DAYS)
        n_bt = len(bt)
        rows = []
        for conf, col in [(0.95, 'Exc95'), (0.99, 'Exc99')]:
            x = int(bt[col].sum())
            lr, p_val = common.kupiec_pof(n_bt, x, 1 - conf)
            rows.append({
                'VaR level': f"{conf:.0%}",
                'Exceptions': x,
                'Expected': round(n_bt * (1 - conf), 1),
                'Kupiec LR': round(lr, 2),
                'p-value': round(p_val, 3),
                'Kupiec result': 'Reject (miscalibrated)' if p_val < 0.05 else 'Accept',
            })
        exc99 = rows[1]['Exceptions']
        zone = common.basel_zone(exc99, n_bt)
        zone_color = {'Green': 'green', 'Yellow': 'orange', 'Red': 'red'}[zone]
        st.markdown(f"<h4 style='color:{zone_color}'>Basel traffic light: {zone} "
                    f"({exc99} exceptions at 99% over {n_bt} days)</h4>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows).set_index('VaR level'), width="stretch")

        fig_bt, ax_bt = plt.subplots(figsize=(14, 4))
        bar_colors_bt = np.where(bt['Exc99'], 'red', np.where(bt['Exc95'], 'orange', 'lightgray'))
        ax_bt.bar(bt.index, bt['PnL'] / 1e6, color=bar_colors_bt, width=1.0)
        ax_bt.plot(bt.index, -bt['VaR95'] / 1e6, color='orange', linewidth=1, linestyle='--', label='−VaR 95%')
        ax_bt.plot(bt.index, -bt['VaR99'] / 1e6, color='red', linewidth=1.2, label='−VaR 99%')
        ax_bt.axhline(0, color='black', linewidth=0.5)
        ax_bt.set_ylabel('Daily P&L (€m)')
        ax_bt.set_title('Hypothetical daily P&L vs prior-day VaR (orange = beyond VaR 95%, red = beyond VaR 99%)')
        ax_bt.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_bt)

        st.caption(
            f"Historical simulation on the last {_LOOKBACK} trading days, 1-day horizon. "
            "Backtest is hypothetical: today's positions applied to past returns, each day's VaR "
            f"estimated only from the {_LOOKBACK} days before it. Positions are held static and "
            "returns are in each instrument's quote currency (i.e. FX-hedged). Kupiec also "
            "rejects too few exceptions (an over-conservative model). Basel zones: green 0–4, "
            "yellow 5–9, red 10+ exceptions at 99% over 250 days."
        )
