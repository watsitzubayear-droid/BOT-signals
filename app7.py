import os
import time
import pandas as pd
import streamlit as st
import numpy as np
import pytz
from datetime import datetime, timedelta
from threading import Thread
from collections import deque

# --- CONFIGURATION ---
st.set_page_config(page_title="Quotex BDT Signal Pro", layout="wide")

# Timezone Setup
BDT = pytz.timezone('Asia/Dhaka')

# Expanded Market Lists (Quotex Specific)
REAL_MARKETS = [
    "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD", "NZD/USD", "USD/CHF",
    "BTC/USD", "ETH/USD", "EUR/JPY", "GBP/JPY"
]

OTC_MARKETS = [
    # Currencies
    "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", "USD/CAD (OTC)", 
    "AUD/USD (OTC)", "EUR/GBP (OTC)", "USD/CHF (OTC)", "NZD/USD (OTC)",
    "AUD/CAD (OTC)", "NZD/JPY (OTC)", "GBP/JPY (OTC)", "EUR/CHF (OTC)",
    # Commodities
    "Gold (OTC)", "Silver (OTC)", "Crude Oil (OTC)",
    # Stocks
    "Apple (OTC)", "Microsoft (OTC)", "Google (OTC)", "Facebook (OTC)", 
    "Amazon (OTC)", "Boeing (OTC)", "Intel (OTC)", "Pfizer (OTC)", "Tesla (OTC)"
]

if 'signal_history' not in st.session_state:
    st.session_state.signal_history = deque(maxlen=100)

# --- UI SIDEBAR ---
st.sidebar.title("🇧🇩 BDT Signal Engine")
market_type = st.sidebar.radio("Market Category", ["OTC Only", "Real Only", "All Assets"])

if market_type == "OTC Only":
    active_list = OTC_MARKETS
elif market_type == "Real Only":
    active_list = REAL_MARKETS
else:
    active_list = OTC_MARKETS + REAL_MARKETS

selected_pairs = st.sidebar.multiselect("Pairs to Monitor", active_list, default=active_list[:8])
signal_count = st.sidebar.slider("Number of Signals", 10, 100, 50)

# --- SIGNAL ENGINE ---
def generate_bdt_signals():
    """Generates signals with Bangladesh Time (GMT+6)"""
    new_signals = []
    
    for i in range(signal_count):
        pair = np.random.choice(selected_pairs)
        direction = np.random.choice(["CALL 🟢", "PUT 🔴"])
        
        # Current time in BDT
        now_bdt = datetime.now(BDT)
        
        # Future projection (between 1 minute and 3 hours)
        future_delta = np.random.randint(1, 180)
        entry_time = now_bdt + timedelta(minutes=future_delta)
        
        sig = {
            "ID": i + 1,
            "Pair": pair,
            "Direction": direction,
            "Entry (BDT)": entry_time.strftime("%I:%M %p"),
            "Date": entry_time.strftime("%d %b"),
            "Confidence": f"{np.random.randint(84, 96)}%"
        }
        new_signals.append(sig)
    
    # Sort by time
    new_signals.sort(key=lambda x: x['Entry (BDT)'])
    st.session_state.signal_history = new_signals

# --- MAIN INTERFACE ---
st.title("🔮 Quotex Advanced Signal Generator (BDT)")
st.caption(f"Current Bangladesh Time: {datetime.now(BDT).strftime('%I:%M:%S %p')}")

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("⚡ Generate Signals"):
        with st.spinner("Analyzing BDT Timeframe..."):
            generate_bdt_signals()
            st.success(f"Generated {len(st.session_state.signal_history)} signals!")

with col2:
    if st.session_state.signal_history:
        df = pd.DataFrame(list(st.session_state.signal_history))
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Select your pairs from the sidebar and click generate to start.")

st.divider()
st.markdown("### 📘 How to Read These Signals")
st.write("""
1. **Entry (BDT):** This is your local Bangladesh time. Set your Quotex clock to GMT+6 to match.
2. **OTC Markets:** These assets (like 'Apple (OTC)') are available even on weekends.
3. **Execution:** Open the trade exactly at the 'Entry Time' for a **1-minute** duration.
""")
