#!/usr/bin/env python3
"""
Quotex 1-min OTC Neural-Signal Generator  –  TA-Lib-free edition
"""
import os, json, websocket, numpy as np
from datetime import datetime
from threading import Thread
from collections import deque

API_KEY = os.getenv("TD_KEY") or input("TwelveData API key: ").strip()
PAIRS = [
    "EUR/USD","USD/JPY","GBP/USD","AUD/USD","USD/CAD","NZD/USD","USD/CHF",
    "BTC/USD","ETH/USD","LTC/USD","XRP/USD","EUR/JPY","GBP/JPY","EUR/GBP",
    "EUR/CHF","GBP/CHF","AUD/JPY","CAD/JPY","CHF/JPY","EURAUD","EURCAD",
    "GBPAUD","GBPCAD","NZDJPY","AUDCAD","AUDNZD","USDSGD"
]
BUF = 500
signals = []

# ---------- lightweight indicator helpers ----------
def _ema(arr, n):
    """NumPy-only EMA"""
    k = 2 / (n + 1)
    ema = np.zeros_like(arr, dtype=float)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i-1] * (1 - k)
    return ema

def _rsi(arr, n=7):
    """NumPy-only RSI"""
    deltas = np.diff(arr)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(arr)
    rsi[:n] = 100. - 100. / (1. + rs)
    for i in range(n, len(arr)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def _atr(h, l, c, n=14):
    """NumPy-only ATR"""
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = tr[1]  # first NaN
    return _ema(tr, n)

# ---------- per-pair engine ----------
class Engine:
    def __init__(self, symbol):
        self.s = symbol
        self.close = deque(maxlen=BUF)
        self.high  = deque(maxlen=BUF)
        self.low   = deque(maxlen=BUF)
    def on_candle(self, c):
        o, h, l, cl = map(float, (c["open"], c["high"], c["low"], c["close"]))
        self.high.append(h); self.low.append(l); self.close.append(cl)
        if len(self.close) < 25: return
        arr = np.array(self.close)
        ema5, ema20 = _ema(arr, 5)[-1], _ema(arr, 20)[-1]
        rsi7 = _rsi(arr, 7)[-1]
        atr14 = _atr(np.array(self.high), np.array(self.low), arr, 14)[-1]
        rng = h - l
        if ema5 > ema20 and 50 < rsi7 < 70 and rng > atr14 * 0.2:
            if _ema(arr, 5)[-2] <= _ema(arr, 20)[-2]:
                sig = {"pair": self.s, "dir": "CALL", "price": cl, "time": datetime.utcnow().isoformat()}
                print(json.dumps(sig)); signals.append(sig)
        elif ema5 < ema20 and 30 < rsi7 < 50 and rng > atr14 * 0.2:
            if _ema(arr, 5)[-2] >= _ema(arr, 20)[-2]:
                sig = {"pair": self.s, "dir": "PUT",  "price": cl, "time": datetime.utcnow().isoformat()}
                print(json.dumps(sig)); signals.append(sig)

engines = {p: Engine(p) for p in PAIRS}

def on_msg(_, msg):
    d = json.loads(msg)
    if d.get("event") == "price":
        sym = d["data"]["symbol"]
        if sym in engines: engines[sym].on_candle(d["data"])

def start():
    ws = websocket.WebSocketApp(f"wss://ws.twelvedata.com/v1?apikey={API_KEY}",
                                on_message=on_msg)
    ws.run_forever()

if __name__ == "__main__":
    Thread(target=start, daemon=True).start()
    while True:
        import time
        time.sleep(1)
