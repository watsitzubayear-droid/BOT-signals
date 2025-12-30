#!/usr/bin/env python3
import os
import json
import websocket
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from threading import Thread
from collections import deque

# --- CONFIGURATION ---
st.set_page_config(page_title="Quotex High-Acc Signals", layout="wide")

# API Key handling
API_KEY = os.getenv("TD_KEY")
if not API_KEY:
    API_KEY = st.sidebar.text_input("TwelveData API Key", type="password")

PAIRS = [
    "EUR/USD","USD/JPY","GBP/USD","AUD/USD","USD/CAD","NZD/USD","USD/CHF",
    "BTC/USD","ETH/USD","LTC/USD","XRP/USD","EUR/JPY","GBP/JPY"
]
BUF = 500

# Global state for signals
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = deque(maxlen=20)

# ---------- PURE-PYTHON INDICATORS ----------
def _ema(arr, n):
    if len(arr) < n: return [0] * len(arr)
    k = 2 / (n + 1)
    ema = [0] * len(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i-1] * (1 - k)
    return ema

def _rsi(arr, n=7):
    if len(arr) <= n: return [50] * len(arr)
    deltas = [arr[i+1] - arr[i] for i in range(len(arr)-1)]
    up = [d if d > 0 else 0 for d in deltas]
    down = [-d if d < 0 else 0 for d in deltas]
    
    rsi_vals = [50] * len(arr)
    avg_up = sum(up[:n])/n
    avg_down = sum(down[:n])/n
    
    for i in range(n, len(arr)):
        d_u = up[i-1]
        d_d = down[i-1]
        avg_up = (avg_up * (n-1) + d_u) / n
        avg_down = (avg_down * (n-1) + d_d) / n
        rs = avg_up / avg_down if avg_down != 0 else 100
        rsi_vals[i] = 100 - (100 / (1 + rs))
    return rsi_vals

# ---------- SIGNAL ENGINE ----------
class Engine:
    def __init__(self, symbol):
        self.s = symbol
        self.close = deque(maxlen=BUF)
        self.high = deque(maxlen=BUF)
        self.low = deque(maxlen=BUF)

    def on_candle(self, c):
        try:
            o, h, l, cl = map(float, (c["open"], c["high"], c["low"], c["close"]))
            self.high.append(h); self.low.append(l); self.close.append(cl)
            
            if len(self.close) < 25: return None
            
            arr = list(self.close)
            ema5_list = _ema(arr, 5)
            ema20_list = _ema(arr, 20)
            rsi7 = _rsi(arr, 7)[-1]
            
            curr_e5, prev_e5 = ema5_list[-1], ema5_list[-2]
            curr_e20, prev_e20 = ema20_list[-1], ema20_list[-2]
            
            # --- STRAT: EMA CROSSOVER + RSI FILTER ---
            # CALL: 5 crosses above 20 + RSI bullish but not overbought
            if curr_e5 > curr_e20 and prev_e5 <= prev_e20 and 50 < rsi7 < 70:
                return {"Pair": self.s, "Dir": "CALL 🟢", "Price": round(cl, 5), "Time": datetime.now().strftime("%H:%M:%S")}
            
            # PUT: 5 crosses below 20 + RSI bearish but not oversold
            elif curr_e5 < curr_e20 and prev_e5 >= prev_e20 and 30 < rsi7 < 50:
                return {"Pair": self.s, "Dir": "PUT 🔴", "Price": round(cl, 5), "Time": datetime.now().strftime("%H:%M:%S")}
        except Exception:
            pass
        return None

engines = {p: Engine(p) for p in PAIRS}

# ---------- WEBSOCKET LOGIC ----------
def on_message(ws, msg):
    data = json.loads(msg)
    if data.get("event") == "price":
        symbol = data["data"]["symbol"]
        if symbol in engines:
            sig = engines[symbol].on_candle(data["data"])
            if sig:
                st.session_state.signal_history.appendleft(sig)

def on_open(ws):
    subscribe_msg = {
        "action": "subscribe",
        "params": {"symbols": ",".join(PAIRS)}
    }
    ws.send(json.dumps(subscribe_msg))

def run_ws():
    ws = websocket.WebSocketApp(
        f"wss://ws.twelvedata.com/v1?apikey={API_KEY}",
        on_open=on_open,
        on_message=on_message
    )
    ws.run_forever()

# ---------- STREAMLIT UI ----------
st.title("🔮 Quotex 1-Min Signal Generator")
st.caption("Strategy: EMA 5/20 Crossover + RSI Momentum Filter")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🚀 Start Signal Stream"):
        if not API_KEY:
            st.error("Missing API Key")
        else:
            Thread(target=run_ws, daemon=True).start()
            st.success("Engine Running...")

with col2:
    placeholder = st.empty()
    while True:
        if st.session_state.signal_history:
            df = pd.DataFrame(list(st.session_state.signal_history))
            placeholder.table(df)
        else:
            placeholder.info("Waiting for market crossovers...")
        time.sleep(2)
