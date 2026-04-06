"""
Market Risk Engine — Streamlit Dashboard
==========================================
Interactive dashboard for VaR analytics, stressed VaR, backtesting, and scenario analysis.
Deploy on Hugging Face Spaces or run locally: streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2329;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #00d4ff;
        margin: 6px 0;
    }
    .risk-green  { border-left-color: #00c851; }
    .risk-yellow { border-left-color: #ffbb33; }
    .risk-red    { border-left-color: #ff4444; }
    .stMetric > div { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/000000/bank-building.png", width=60)
st.sidebar.title("⚙️ Risk Parameters")

PORTFOLIO_VALUE = st.sidebar.number_input(
    "Portfolio Value ($)", min_value=1_000_000, max_value=1_000_000_000,
    value=10_000_000, step=1_000_000, format="%d"
)

CONFIDENCE_LEVEL = st.sidebar.selectbox(
    "VaR Confidence Level", [0.99, 0.975, 0.95], index=0,
    format_func=lambda x: f"{x:.1%}"
)

LOOKBACK = st.sidebar.slider("Lookback Window (days)", 125, 500, 250, 25)
HOLDING  = st.sidebar.selectbox("Holding Period (days)", [1, 10], index=1)

ASSETS_SELECTED = st.sidebar.multiselect(
    "Portfolio Assets",
    ['SPY', 'GLD', 'EEM', 'EURUSD=X', 'CL=F'],
    default=['SPY', 'GLD', 'EEM', 'EURUSD=X', 'CL=F']
)

ASSET_LABELS = {
    'SPY': 'US Equities', 'GLD': 'Gold', 'EEM': 'EM Equities',
    'EURUSD=X': 'EUR/USD', 'CL=F': 'Crude Oil'
}
ASSET_CLEAN = {
    'SPY': 'SPY', 'GLD': 'GLD', 'EEM': 'EEM',
    'EURUSD=X': 'EURUSD', 'CL=F': 'Crude_Oil'
}

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Downloading market data...")
def load_data(assets, start='2005-01-01'):
    end = datetime.today().strftime('%Y-%m-%d')
    try:
        raw = yf.download(assets, start=start, end=end, auto_adjust=True, progress=False)['Close']
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=assets[0])
        raw.columns = [ASSET_CLEAN.get(c, c) for c in raw.columns]
        raw.ffill(inplace=True)
        raw.dropna(how='all', inplace=True)
        log_ret = np.log(raw / raw.shift(1)).dropna()
        return raw, log_ret
    except Exception as e:
        st.error(f"Data download error: {e}")
        return None, None

if not ASSETS_SELECTED:
    st.warning("Please select at least one asset.")
    st.stop()

prices, log_returns = load_data(ASSETS_SELECTED)
if prices is None:
    st.stop()

# Equal weights
n = len(log_returns.columns)
weights = pd.Series({c: 1.0/n for c in log_returns.columns})
portfolio_returns = (log_returns * weights).sum(axis=1)

ALPHA = 1 - CONFIDENCE_LEVEL

# ── Helper Functions ──────────────────────────────────────────────────────────
def historical_var(returns, conf, hold, lookback):
    w = returns.iloc[-lookback:]
    var1 = np.percentile(w, (1-conf)*100)
    varN = var1 * np.sqrt(hold)
    es1  = w[w <= var1].mean()
    esN  = es1 * np.sqrt(hold)
    return var1, varN, es1, esN

def parametric_var(returns, conf, hold, lookback):
    w  = returns.iloc[-lookback:]
    mu, sig = w.mean(), w.std()
    z  = norm.ppf(1-conf)
    v1 = mu + z * sig
    vN = mu*hold + z*sig*np.sqrt(hold)
    e1 = mu - sig * (norm.pdf(norm.ppf(1-conf)) / (1-conf))
    eN = e1 * np.sqrt(hold)
    return v1, vN, e1, eN

def make_gauge(value, title, ref_var):
    pct = abs(value) / abs(ref_var) * 100 if ref_var != 0 else 50
    color = "#00c851" if pct < 80 else ("#ffbb33" if pct < 120 else "#ff4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=abs(value) * 100,
        delta={'reference': abs(ref_var) * 100, 'valueformat': '.4f'},
        number={'suffix': '%', 'valueformat': '.4f'},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, abs(ref_var)*200]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, abs(ref_var)*100], 'color': '#2d3436'},
                {'range': [abs(ref_var)*100, abs(ref_var)*200], 'color': '#636e72'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.75,
                'value': abs(ref_var)*100
            }
        }
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    return fig

# ── Compute VaR ───────────────────────────────────────────────────────────────
hs_v1, hs_vN, hs_e1, hs_eN  = historical_var(portfolio_returns, CONFIDENCE_LEVEL, HOLDING, LOOKBACK)
pm_v1, pm_vN, pm_e1, pm_eN  = parametric_var(portfolio_returns, CONFIDENCE_LEVEL, HOLDING, LOOKBACK)

rolling_hs_var = portfolio_returns.rolling(LOOKBACK).quantile(ALPHA)
exceptions_250  = (portfolio_returns.iloc[-250:] < rolling_hs_var.iloc[-250:]).sum()
exc_zone = 'GREEN' if exceptions_250 <= 4 else ('YELLOW' if exceptions_250 <= 9 else 'RED')
exc_color = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴'}[exc_zone]

# ── Page Title ────────────────────────────────────────────────────────────────
st.title("📊 Market Risk Engine")
st.caption(f"Multi-Asset Portfolio | Basel III/IV Framework | Last updated: {datetime.today().strftime('%Y-%m-%d %H:%M')}")

tabs = st.tabs(["📈 VaR Dashboard", "🔥 Stressed VaR", "✅ Backtesting", "🎭 Scenario Analysis", "ℹ️ Model Info"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: VaR Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1-Day HS VaR", f"{abs(hs_v1):.2%}", f"${abs(hs_v1)*PORTFOLIO_VALUE:,.0f}")
    col2.metric(f"{HOLDING}-Day HS VaR", f"{abs(hs_vN):.2%}", f"${abs(hs_vN)*PORTFOLIO_VALUE:,.0f}")
    col3.metric(f"{HOLDING}-Day HS ES", f"{abs(hs_eN):.2%}", f"${abs(hs_eN)*PORTFOLIO_VALUE:,.0f}")
    col4.metric("Basel Zone", f"{exc_color} {exc_zone}", f"{exceptions_250} exceptions / 250d")

    st.divider()

    # Gauges
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_gauge(hs_vN, f"HS VaR {HOLDING}D ({CONFIDENCE_LEVEL:.0%})", hs_vN),
                        use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(pm_vN, f"Parametric VaR {HOLDING}D", hs_vN),
                        use_container_width=True)
    with g3:
        st.plotly_chart(make_gauge(hs_eN, f"Expected Shortfall {HOLDING}D", hs_vN),
                        use_container_width=True)

    # Rolling VaR chart
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns.values,
                                   name='Portfolio Return', line=dict(color='rgba(100,149,237,0.5)', width=1),
                                   fill='tozeroy', fillcolor='rgba(100,149,237,0.1)'))
    fig_roll.add_trace(go.Scatter(x=rolling_hs_var.index, y=rolling_hs_var.values,
                                   name=f'Rolling HS VaR ({CONFIDENCE_LEVEL:.0%})',
                                   line=dict(color='#ff4444', width=1.5)))
    rolling_pm_var = portfolio_returns.rolling(LOOKBACK).mean() + norm.ppf(ALPHA) * portfolio_returns.rolling(LOOKBACK).std()
    fig_roll.add_trace(go.Scatter(x=rolling_pm_var.index, y=rolling_pm_var.values,
                                   name='Rolling Parametric VaR',
                                   line=dict(color='#00c851', width=1.5, dash='dash')))
    fig_roll.update_layout(title='Portfolio Returns vs Rolling VaR Forecasts',
                            template='plotly_dark', height=350,
                            legend=dict(orientation='h', y=-0.2))
    st.plotly_chart(fig_roll, use_container_width=True)

    # VaR comparison table
    st.subheader("VaR Comparison Table")
    comp_df = pd.DataFrame({
        'Method': ['Historical Simulation', 'Parametric (Normal)', 'Ratio HS/Param'],
        'VaR 1D': [f'{abs(hs_v1):.4f}', f'{abs(pm_v1):.4f}', f'{hs_v1/pm_v1:.2f}×'],
        f'VaR {HOLDING}D': [f'{abs(hs_vN):.4f}', f'{abs(pm_vN):.4f}', f'{hs_vN/pm_vN:.2f}×'],
        f'ES {HOLDING}D': [f'{abs(hs_eN):.4f}', f'{abs(pm_eN):.4f}', '-'],
        f'VaR {HOLDING}D ($)': [f'${abs(hs_vN)*PORTFOLIO_VALUE:,.0f}',
                                 f'${abs(pm_vN)*PORTFOLIO_VALUE:,.0f}', '-'],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Stressed VaR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🔥 Basel III Stressed VaR Analysis")
    st.info("Basel 2.5 requires banks to identify the worst 12-month window for the current portfolio and compute Stressed VaR using that period's returns.")

    STRESS_PERIODS = {
        'GFC 2008–2009':    ('2008-09-01', '2009-09-01'),
        'COVID-19 2020':    ('2020-02-01', '2021-02-01'),
        'Rate Shock 2022':  ('2022-01-01', '2022-12-31'),
    }

    # Algo-detected stress window
    rolling_worst = portfolio_returns.rolling(252).apply(lambda x: np.percentile(x, ALPHA*100))
    if not rolling_worst.dropna().empty:
        worst_date = rolling_worst.idxmin()
        algo_start = worst_date - pd.DateOffset(years=1)
        STRESS_PERIODS['Algo-Selected'] = (str(algo_start.date()), str(worst_date.date()))

    stress_results = []
    normal_var1, normal_varN, _, _ = historical_var(portfolio_returns, CONFIDENCE_LEVEL, HOLDING, LOOKBACK)

    for name, (s, e) in STRESS_PERIODS.items():
        mask = (portfolio_returns.index >= s) & (portfolio_returns.index <= e)
        sr = portfolio_returns[mask]
        if len(sr) < 20: continue
        v1, vN, _, eN = historical_var(sr, CONFIDENCE_LEVEL, HOLDING, len(sr))
        stress_results.append({
            'Stress Period': name, 'Start': s, 'End': e, 'N Days': len(sr),
            'SVaR 1D': abs(v1), f'SVaR {HOLDING}D': abs(vN), f'SES {HOLDING}D': abs(eN),
            f'SVaR {HOLDING}D ($M)': abs(vN)*PORTFOLIO_VALUE/1e6,
            'Stress Multiplier': abs(vN)/abs(normal_varN) if normal_varN != 0 else 0
        })

    sdf = pd.DataFrame(stress_results)

    if not sdf.empty:
        cols = st.columns(len(sdf))
        for i, (_, row) in enumerate(sdf.iterrows()):
            with cols[i]:
                mult = row['Stress Multiplier']
                color = "#ff4444" if mult > 2 else ("#ffbb33" if mult > 1.5 else "#00c851")
                st.markdown(f"""
                <div class='metric-card'>
                    <b>{row['Stress Period']}</b><br>
                    SVaR {HOLDING}D: <span style='color:{color}'><b>{row[f'SVaR {HOLDING}D']:.4f}</b></span><br>
                    ${row[f'SVaR {HOLDING}D ($M)']:.2f}M<br>
                    Stress mult: <b>{mult:.1f}×</b>
                </div>""", unsafe_allow_html=True)

        fig_stress = go.Figure()
        fig_stress.add_trace(go.Bar(name=f'SVaR {HOLDING}D', x=sdf['Stress Period'],
                                     y=sdf[f'SVaR {HOLDING}D'], marker_color='#ff4444', opacity=0.8))
        fig_stress.add_trace(go.Bar(name=f'SES {HOLDING}D', x=sdf['Stress Period'],
                                     y=sdf[f'SES {HOLDING}D'], marker_color='#ffbb33', opacity=0.8))
        fig_stress.add_hline(y=abs(normal_varN), line_dash='dash', line_color='#00c851',
                              annotation_text=f'Normal VaR {HOLDING}D')
        fig_stress.update_layout(title=f'Stressed VaR vs Normal VaR by Crisis Period',
                                   template='plotly_dark', height=350, barmode='group')
        st.plotly_chart(fig_stress, use_container_width=True)
        st.dataframe(sdf.drop(columns=['Start','End']).round(4), hide_index=True, use_container_width=True)

    # Basel III Capital
    st.subheader("📐 Basel III Capital Charge")
    MULTIPLIER = st.slider("Capital Multiplier (3.0 = Green, 4.0 = Red)", 3.0, 4.0, 3.0, 0.1)
    var_comp   = MULTIPLIER * abs(normal_varN) * PORTFOLIO_VALUE
    svar_val   = sdf[f'SVaR {HOLDING}D'].max() if not sdf.empty else abs(normal_varN)*1.5
    svar_comp  = MULTIPLIER * svar_val * PORTFOLIO_VALUE
    total_cap  = var_comp + svar_comp

    c1, c2, c3 = st.columns(3)
    c1.metric("VaR Component", f"${var_comp:,.0f}", f"{var_comp/PORTFOLIO_VALUE:.2%}")
    c2.metric("SVaR Component", f"${svar_comp:,.0f}", f"{svar_comp/PORTFOLIO_VALUE:.2%}")
    c3.metric("Total Capital Charge", f"${total_cap:,.0f}", f"{total_cap/PORTFOLIO_VALUE:.2%}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Backtesting
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("✅ VaR Model Backtesting — Basel Traffic-Light Framework")

    # Exception count last 250 days
    rec_ret = portfolio_returns.iloc[-250:]
    rec_var = rolling_hs_var.iloc[-250:]
    exc_mask = rec_ret < rec_var
    exc_count = exc_mask.sum()
    exp_exc   = 250 * ALPHA
    exc_rate  = exc_count / 250

    # Traffic light
    if exc_count <= 4:   zone, mult, zcolor = 'GREEN',  3.0, '#00c851'
    elif exc_count <= 9: zone, mult, zcolor = 'YELLOW', 3.0+(exc_count-4)*0.08, '#ffbb33'
    else:                zone, mult, zcolor = 'RED',    4.0, '#ff4444'

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exceptions (250d)", int(exc_count))
    c2.metric("Expected Exceptions", f"{exp_exc:.1f}")
    c3.metric("Exceedance Rate", f"{exc_rate:.2%}", f"Target: {ALPHA:.2%}")
    c4.metric("Basel Zone", f"{exc_color} {zone}", f"Multiplier: {mult:.1f}")

    # Rolling exception count
    rolling_exc = (portfolio_returns < rolling_hs_var).rolling(250).sum()
    fig_exc = go.Figure()
    fig_exc.add_trace(go.Scatter(x=rolling_exc.index, y=rolling_exc.values,
                                  fill='tozeroy', name='Rolling 250-day Exceptions',
                                  fillcolor='rgba(100,149,237,0.3)',
                                  line=dict(color='steelblue', width=1.5)))
    fig_exc.add_hline(y=4,  line_dash='dash', line_color='#00c851', annotation_text='Green (4)')
    fig_exc.add_hline(y=9,  line_dash='dash', line_color='#ffbb33', annotation_text='Yellow (9)')
    fig_exc.add_hline(y=10, line_dash='dash', line_color='#ff4444', annotation_text='Red (10)')
    fig_exc.update_layout(title='Rolling 250-Day VaR Exception Count',
                           template='plotly_dark', height=320)
    st.plotly_chart(fig_exc, use_container_width=True)

    # Kupiec POF
    from scipy.stats import chi2
    T = len(rec_ret)
    n = exc_count
    if 0 < n < T:
        p_hat = n / T
        LR = -2 * (n*np.log(ALPHA/p_hat) + (T-n)*np.log((1-ALPHA)/(1-p_hat)))
        p_val = 1 - chi2.cdf(LR, df=1)
        kupiec_pass = p_val > 0.05
    else:
        LR, p_val, kupiec_pass = 0, 1, True

    st.subheader("Kupiec POF & Christoffersen Tests")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **Kupiec Proportion of Failures Test**
        - H₀: Exceedance rate = {ALPHA:.2%}
        - LR statistic: {LR:.4f}
        - p-value: {p_val:.4f}
        - Result: {'✅ PASS — Model correctly calibrated' if kupiec_pass else '❌ REJECT — Model miscalibrated'}
        """)
    with c2:
        exc_series = (rec_ret < rec_var).astype(int).values
        n00 = sum((exc_series[:-1]==0) & (exc_series[1:]==0))
        n01 = sum((exc_series[:-1]==0) & (exc_series[1:]==1))
        n10 = sum((exc_series[:-1]==1) & (exc_series[1:]==0))
        n11 = sum((exc_series[:-1]==1) & (exc_series[1:]==1))
        pi01 = n01/(n00+n01+1e-8)
        pi11 = n11/(n10+n11+1e-8)
        clustering = pi11 > pi01*2
        st.markdown(f"""
        **Christoffersen Independence Test**
        - P(exception | no exception yesterday): {pi01:.4f}
        - P(exception | exception yesterday): {pi11:.4f}
        - Clustering detected: {'⚠️ YES — GARCH model recommended' if clustering else '✅ NO — Exceptions are independent'}
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Scenario Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🎭 Named Macro Scenario Analysis")

    SCENARIOS = {
        'Equity Crash -30%':    {'SPY': -0.30, 'EEM': -0.35, 'GLD':  0.05, 'Crude_Oil': -0.25, 'EURUSD': -0.03},
        'Yield Shock +200bps':  {'SPY': -0.12, 'EEM': -0.08, 'GLD': -0.05, 'Crude_Oil':  0.02, 'EURUSD':  0.01},
        'USD Surge +15%':       {'SPY': -0.05, 'EEM': -0.20, 'GLD': -0.08, 'Crude_Oil': -0.10, 'EURUSD': -0.15},
        'Oil Spike +50%':       {'SPY': -0.04, 'EEM': -0.02, 'GLD':  0.03, 'Crude_Oil':  0.50, 'EURUSD':  0.02},
        'Stagflation':          {'SPY': -0.20, 'EEM': -0.15, 'GLD':  0.10, 'Crude_Oil':  0.30, 'EURUSD': -0.05},
        'Soft Landing':         {'SPY':  0.10, 'EEM':  0.08, 'GLD': -0.02, 'Crude_Oil': -0.05, 'EURUSD':  0.02},
        'Global Recession':     {'SPY': -0.40, 'EEM': -0.45, 'GLD':  0.15, 'Crude_Oil': -0.40, 'EURUSD': -0.08},
    }

    # Custom scenario
    st.write("#### 🎛️ Custom Scenario")
    custom_cols = st.columns(5)
    custom_shocks = {}
    asset_names = ['SPY', 'GLD', 'EEM', 'EURUSD', 'Crude_Oil']
    for i, a in enumerate(asset_names):
        with custom_cols[i]:
            custom_shocks[a] = st.slider(a, -0.5, 0.5, 0.0, 0.01, key=f"custom_{a}")
    SCENARIOS['Custom'] = custom_shocks

    w = weights.reindex([ASSET_CLEAN.get(a,a) for a in asset_names]).values
    scen_results = []
    for name, shocks in SCENARIOS.items():
        sv = np.array([shocks.get(a, 0.0) for a in asset_names])
        pnl = float(sv @ w)
        scen_results.append({'Scenario': name, 'P&L (%)': pnl, 'P&L ($)': pnl*PORTFOLIO_VALUE})

    scen_df = pd.DataFrame(scen_results).sort_values('P&L ($)')
    colors = ['#ff4444' if v < 0 else '#00c851' for v in scen_df['P&L ($)']]

    fig_scen = go.Figure(go.Bar(
        x=scen_df['P&L ($)']/1e6, y=scen_df['Scenario'],
        orientation='h', marker_color=colors, opacity=0.85,
        text=[f'${v/1e6:+.2f}M ({p:.2%})' for v,p in zip(scen_df['P&L ($)'], scen_df['P&L (%)'])],
        textposition='outside'
    ))
    fig_scen.add_vline(x=0, line_color='white', line_width=1)
    fig_scen.update_layout(title='Scenario P&L Analysis', template='plotly_dark',
                            xaxis_title='P&L ($M)', height=420)
    st.plotly_chart(fig_scen, use_container_width=True)

    # Heatmap: asset contribution per scenario
    contrib = {}
    for name, shocks in SCENARIOS.items():
        sv = np.array([shocks.get(a,0.0) for a in asset_names])
        contrib[name] = sv * w
    contrib_df = pd.DataFrame(contrib, index=asset_names).T

    fig_heat = px.imshow(contrib_df, color_continuous_scale='RdBu_r', aspect='auto',
                          title='Scenario P&L Attribution by Asset',
                          labels={'color': 'P&L Contribution'})
    fig_heat.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Model Info
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("ℹ️ Model Documentation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📐 Methodology
        **VaR Methods Implemented:**
        - Historical Simulation (non-parametric)
        - Parametric Normal (variance-covariance)
        - Monte Carlo (correlated Gaussian paths)
        - GARCH(1,1) dynamic VaR (see notebooks)

        **Regulatory Alignment:**
        - Basel III: 99% VaR, 10-day holding period, 250-day lookback
        - Stressed VaR: worst 12-month window (Basel 2.5)
        - FRTB awareness: 97.5% ES by liquidity horizon class
        - Traffic-light backtesting: green/yellow/red zones

        **Statistical Validation:**
        - Kupiec POF test (exceedance frequency)
        - Christoffersen independence test (clustering)
        - Conditional coverage (joint test)
        """)
    with col2:
        st.markdown("""
        ### 📊 Portfolio Details
        **Assets:** US Equities (SPY), Gold (GLD), EM Equity (EEM), EUR/USD, Crude Oil

        **Data source:** Yahoo Finance (daily adjusted close)

        **FRTB Liquidity Horizons:**
        | Asset | Horizon |
        |-------|---------|
        | SPY (Equity) | 10 days |
        | EURUSD (FX) | 10 days |
        | GLD (Gold) | 20 days |
        | EEM (EM) | 20 days |
        | Crude Oil | 20 days |

        ### 🔬 ML Layer (Notebook 4)
        XGBoost classifier trained on lagged returns, GARCH volatility, and macro features to predict VaR exceedance probability.
        """)

    st.divider()
    st.markdown("**Built by:** Etor B. | **Framework:** Basel III / FRTB | **Stack:** Python, yfinance, arch, XGBoost, SHAP, Streamlit")
