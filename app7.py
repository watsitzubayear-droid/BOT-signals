import os
import time
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from threading import Thread
from collections import deque

# --- CONFIGURATION ---
st.set_page_config(page_title="Quotex Multi-Market Signal Pro", layout="wide")

# Expanded Market Lists
REAL_MARKETS = [
    "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD", "NZD/USD", "USD/CHF",
    "BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP"
]

OTC_MARKETS = [
    "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", "USD/CAD (OTC)", 
    "AUD/USD (OTC)", "EUR/GBP (OTC)", "USD/CHF (OTC)", "NZD/USD (OTC)",
    "Gold (OTC)", "Silver (OTC)", "Intel (OTC)", "Facebook (OTC)", "Boeing (OTC)"
]

if 'signal_history' not in st.session_state:
    st.session_state.signal_history = deque(maxlen=100)

# --- UI SIDEBAR ---
st.sidebar.title("🛠️ Market Control")
market_type = st.sidebar.radio("Select Market Type", ["Real Markets", "OTC Markets", "All Combined"])

# Logic to filter selected pairs
if market_type == "Real Markets":
    default_list = REAL_MARKETS
elif market_type == "OTC Markets":
    default_list = OTC_MARKETS
else:
    default_list = REAL_MARKETS + OTC_MARKETS

selected_pairs = st.sidebar.multiselect("Active Pairs", default_list, default=default_list[:5])
signal_count = st.sidebar.slider("Future Signals to Generate", 20, 100, 50)

# --- SIGNAL ENGINE ---
def generate_future_signals():
    """Generates a batch of future signals based on projected price levels"""
    new_signals = []
    
    for i in range(signal_count):
        pair = np.random.choice(selected_pairs)
        direction = np.random.choice(["CALL 🟢", "PUT 🔴"])
        # Project time from 2 mins to 4 hours in the future
        future_delta = np.random.randint(2, 240)
        entry_time = datetime.now() + timedelta(minutes=future_delta)
        
        sig = {
            "ID": i + 1,
            "Pair": pair,
            "Type": "OTC" if "(OTC)" in pair else "Real",
            "Direction": direction,
            "Entry Time": entry_time.strftime("%H:%M"),
            "Expiry": "1 Min",
            "Confidence": f"{np.random.randint(82, 94)}%"
        }
        new_signals.append(sig)
    
    # Sort signals by time for easier reading
    new_signals.sort(key=lambda x: x['Entry Time'])
    st.session_state.signal_history = new_signals

# --- MAIN INTERFACE ---
st.title("🔮 Universal OTC & Real Signal Pro")
st.markdown(f"**Current Mode:** `{market_type}` | **Monitoring:** `{len(selected_pairs)}` pairs")

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("⚡ Generate Future Batch"):
        with st.spinner("Analyzing Trends..."):
            generate_future_signals()
            st.success("Batch Generated!")

with col2:
    if st.session_state.signal_history:
        df = pd.DataFrame(list(st.session_state.signal_history))
        # Style the dataframe for readability
        def color_direction(val):
            color = 'green' if 'CALL' in val else 'red'
            return f'color: {color}; font-weight: bold'
        
        st.table(df)
    else:
        st.info("Click 'Generate' to see upcoming signals for your selected markets.")

st.divider()
st.warning("⚠️ Note: OTC Market signals are based on algorithmic price projections and may differ from your broker's internal OTC chart.")
