import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Configuratie ---
LOG_DIR = "./logs"
METRICS_FILE = os.path.join(LOG_DIR, "metrics.csv")

# --- Start dashboard ---
st.set_page_config(page_title="Crypto LSTM Dashboard", layout="wide")
st.title("📊 Crypto LSTM Training Dashboard")

# --- Laad metrics ---
@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_FILE):
        df = pd.read_csv(METRICS_FILE, parse_dates=['timestamp'])
        return df
    else:
        return pd.DataFrame()

df = load_metrics()
if df.empty:
    st.warning("Geen metrics.csv gevonden in logs-map.")
    st.stop()

# --- Sidebar filter ---
coins = df['coin'].unique().tolist()
timeframes = df['timeframe'].unique().tolist()
selected_coin = st.sidebar.selectbox('Selecteer coin', coins)
selected_tf = st.sidebar.selectbox('Selecteer timeframe', timeframes)

filtered = df[(df['coin'] == selected_coin) & (df['timeframe'] == selected_tf)]
selected_runs = st.sidebar.slider('Aantal laatste runs', 1, len(filtered), min(5, len(filtered)))
runs = filtered.tail(selected_runs)

st.subheader(f"Overzicht {selected_coin} [{selected_tf}] - laatste {selected_runs} runs")
st.dataframe(runs)

# --- Line plots van metrics over runs ---
metrics = ['mse','mae','r2','winrate','sharpe']
cols = st.multiselect('Kies metrics om te plotten', metrics, default=metrics)

if cols:
    fig, ax = plt.subplots(len(cols), 1, figsize=(8, 4*len(cols)))
    for i, m in enumerate(cols):
        ax[i].plot(runs['timestamp'], runs[m], marker='o')
        ax[i].set_title(m.upper())
        ax[i].set_xlabel('Timestamp')
        ax[i].grid(True)
    st.pyplot(fig)

# --- Equity curve vergelijking ---
st.subheader("Equity Curve vs Buy & Hold")
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(runs['timestamp'], runs['pnl_usd'].cumsum(), label='Model Equity')
ax2.plot(runs['timestamp'], runs['buyhold'].cumsum(), label='Buy & Hold')
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('Cumulatieve PnL')
ax2.legend()
st.pyplot(fig2)

# --- Feature importances placeholder ---
st.subheader("Feature Importances (TODO)")
st.text("- Implementatie van SHAP of interne feature importance hier")
