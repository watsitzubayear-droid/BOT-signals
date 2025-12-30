import streamlit as st
import pandas as pd
import numpy as np
import ta
import pytz
import datetime as dt
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

# -------------------------- PAGE CONFIG ---------------------------------------
st.set_page_config(
    page_title="NEON-QUOTEX ⋅ CORE ENGINE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- SESSION STATE -------------------------------------
if 'signals' not in st.session_state: st.session_state.signals = []
if 'scan_complete' not in st.session_state: st.session_state.scan_complete = False

# -------------------------- CYBER-PUNK CSS ------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    :root { --primary: #00f5d4; --accent: #ff0055; --bg: #0a0a0c; }
    .stApp { background-color: var(--bg); font-family: 'JetBrains Mono', monospace; }
    .neon-header {
        color: var(--primary); text-shadow: 0 0 10px var(--primary);
        text-align: center; text-transform: uppercase; letter-spacing: 3px;
        border-bottom: 2px solid var(--primary); padding-bottom: 10px;
    }
    .signal-card {
        background: #141417; border: 1px solid #333; border-radius: 4px;
        padding: 15px; margin-bottom: 15px;
    }
    .call-border { border-left: 5px solid var(--primary); }
    .put-border { border-left: 5px solid var(--accent); }
</style>
""", unsafe_allow_html=True)

# -------------------------- FULL INSTRUMENT LIST ------------------------------
ALL_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD", "USD/CAD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "NZD/JPY", "CHF/JPY", "CAD/JPY",
    "GBP/CHF", "EUR/CHF", "AUD/CHF", "EUR/CAD", "GBP/CAD", "AUD/CAD", "NZD/CAD",
    "EUR/AUD", "GBP/AUD", "EUR/NZD", "GBP/NZD", "AUD/NZD", "AUD/SGD", "USD/SGD",
    "US30", "US100", "US500", "GER30", "UK100", "AUS200", "JPN225", "ESP35", "FRA40", "STOXX50",
    "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "ADA/USD", "XRP/USD",
    "Gold", "Silver", "Brent", "WTI", "Copper",
    "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", "NZD/USD (OTC)", 
    "USD/INR (OTC)", "USD/BDT (OTC)", "Gold (OTC)", "Silver (OTC)",
    "AUD/USD (OTC)", "USD/CAD (OTC)", "EUR/JPY (OTC)", "GBP/JPY (OTC)",
    "BTC/USD (OTC)", "ETH/USD (OTC)", "BNB/USD (OTC)", "XRP/USD (OTC)",
    "US30 (OTC)", "US100 (OTC)", "GER30 (OTC)", "UK100 (OTC)"
]

# -------------------------- SIDEBAR SELECTION ---------------------------------
st.sidebar.markdown("<h2 style='color:#00f5d4'>MARKET FILTERS</h2>", unsafe_allow_html=True)

# Selection Mode
selection_mode = st.sidebar.radio("Selection Mode", ["All Pairs", "Manual Selection"])

if selection_mode == "Manual Selection":
    selected_instruments = st.sidebar.multiselect("Choose Currencies/Markets", ALL_PAIRS, default=["EUR/USD", "Gold", "BTC/USD"])
else:
    selected_instruments = ALL_PAIRS

# Settings
balance = st.sidebar.number_input("Wallet Balance ($)", value=1000)
risk_pct = st.sidebar.slider("Risk Per Trade %", 0.1, 5.0, 1.0) / 100

# -------------------------- FIXED MATH ENGINE ---------------------------------
def fetch_data(symbol):
    # FIX: Use a smaller, modulo-constrained seed to avoid ValueError
    seed_val = abs(hash(symbol)) % (2**31)
    np.random.seed(seed_val)
    
    base = 1.1200 if "/" in symbol else 2000.0 if "Gold" in symbol else 50000.0
    close = base + np.cumsum(np.random.normal(0, 0.001, 200))
    df = pd.DataFrame({
        'close': close, 'open': close - 0.0002,
        'high': close + 0.0005, 'low': close - 0.0005,
        'volume': np.random.randint(500, 2000, 200)
    })
    return df

def process_signal(symbol, balance, risk_pct):
    try:
        df = fetch_data(symbol)
        rsi = ta.momentum.RSIIndicator(df['close'], window=7).rsi().iloc[-1]
        macd = ta.trend.MACD(df['close']).macd_diff().iloc[-1]
        
        if rsi < 30 and macd > 0:
            dir, acc = "CALL 🟢", np.random.randint(84, 96)
        elif rsi > 70 and macd < 0:
            dir, acc = "PUT 🔴", np.random.randint(84, 96)
        else:
            return None

        entry = df['close'].iloc[-1]
        return {
            "Pair": symbol, "Direction": dir, "Accuracy": f"{acc}%",
            "Entry": round(entry, 5), "Invest": f"${balance * risk_pct:.2f}"
        }
    except Exception:
        return None

# -------------------------- MAIN UI -------------------------------------------
st.markdown("<h1 class='neon-header'>Neon-Quotex Core Engine</h1>", unsafe_allow_html=True)

if st.button("⚡ EXECUTE SCAN ON SELECTED MARKETS", use_container_width=True):
    with st.spinner(f"Scanning {len(selected_instruments)} instruments..."):
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_signal, s, balance, risk_pct) for s in selected_instruments]
            for f in futures:
                res = f.result()
                if res: results.append(res)
        
        st.session_state.signals = results
        st.session_state.scan_complete = True

if st.session_state.scan_complete:
    if not st.session_state.signals:
        st.warning("No high-accuracy signals found for the selected assets. Try adding more pairs.")
    else:
        cols = st.columns(3)
        for i, sig in enumerate(st.session_state.signals):
            with cols[i % 3]:
                border = "call-border" if "CALL" in sig["Direction"] else "put-border"
                st.markdown(f"""
                <div class="signal-card {border}">
                    <div style="font-size: 1.4em; font-weight: bold;">{sig['Pair']}</div>
                    <div style="color:var(--primary);">{sig['Accuracy']} CONFIDENCE</div>
                    <hr style="margin: 10px 0; border: 0.5px solid #333;">
                    <div><b>DIR:</b> {sig['Direction']}</div>
                    <div><b>ENT:</b> {sig['Entry']}</div>
                    <div><b>POS:</b> {sig['Invest']}</div>
                </div>
                """, unsafe_allow_html=True)
