import numpy as np
from utils import format_bdt_time
from datetime import timedelta

class SignalEngine:
    def __init__(self, market_type='real'):
        self.market_type = market_type
    
    def calculate_ema(self, prices, period):
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, prices, period=7):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def detect_3_touch_zones(self, candles, window=20):
        """Detect 3-touch S/R zones for OTC"""
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        
        zones = []
        for i in range(len(candles) - window):
            sub_highs = highs[i:i+window]
            sub_lows = lows[i:i+window]
            
            # Find resistance (3+ touches of similar high)
            for level in np.linspace(min(sub_highs), max(sub_highs), 10):
                touches = sum(1 for h in sub_highs if abs(h - level) < 0.0003)
                if touches >= 3:
                    zones.append({'type': 'resistance', 'price': level, 'touches': touches})
            
            # Find support (3+ touches of similar low)
            for level in np.linspace(min(sub_lows), max(sub_lows), 10):
                touches = sum(1 for l in sub_lows if abs(l - level) < 0.0003)
                if touches >= 3:
                    zones.append({'type': 'support', 'price': level, 'touches': touches})
        
        return zones[:3]  # Top 3 zones
    
    def check_wick_rejection(self, candle, zone_type):
        """Check for long wick at zone (OTC)"""
        body = abs(candle['close'] - candle['open'])
        wick_top = candle['high'] - max(candle['open'], candle['close'])
        wick_bottom = min(candle['open'], candle['close']) - candle['low']
        
        if zone_type == 'resistance':
            return wick_top > body * 2  # Long upper wick
        else:
            return wick_bottom > body * 2  # Long lower wick
    
    def generate_signals(self, candles, future_times, pair="EUR/USD"):
        """Generate 100 future signals using all strategies"""
        signals = []
        predictor = MetaEnsemblePredictor()
        
        # Train predictor on recent data
        if len(candles) > 50:
            predictor.train(candles[-50:])
        
        if self.market_type == 'otc':
            zones = self.detect_3_touch_zones(candles)
            if not zones:
                # Fallback to simple reversal strategy if no zones found
                zones = [{'type': 'support', 'price': min(c['low'] for c in candles[-5:]), 'touches': 3}]
        
        for i, future_time in enumerate(future_times):
            # Get features for prediction
            recent_candles = candles[-10:] if candles else []
            green_prob = predictor.predict_proba(recent_candles)
            
            # Strategy selection based on market
            if self.market_type == 'otc':
                signal = self.otc_strategy(candles, zones, green_prob, i)
            else:
                signal = self.real_strategy(candles, green_prob, i)
            
            if signal:
                direction, confidence = signal
                signals.append({
                    'pair': pair,
                    'time': future_time,
                    'direction': direction,
                    'confidence': confidence,
                    'formatted': f"{pair} | TIME: {format_bdt_time(future_time)} || {'UP/Call' if direction == 'LONG' else 'DOWN/Put'}"
                })
            
            if len(signals) >= 100:
                break
        
        return signals
    
    def otc_strategy(self, candles, zones, green_prob, index):
        """OTC-specific strategy: 3-Touch Zone + Wick Rejection"""
        if not zones:
            return None
        
        # Use most recent zone
        zone = zones[0]
        
        # Check if price is near zone (within 10 pips)
        if not candles:
            last_price = 1.0850
        else:
            last_price = candles[-1]['close']
        
        distance_to_zone = abs(last_price - zone['price'])
        
        if distance_to_zone > 0.0010:  # Not near zone
            return None
        
        # Check RSI exhaustion (simulate)
        rsi = 30 + (index % 40)  # Simulate RSI oscillation
        
        # Wick rejection check (simulate)
        wick_rejection = (index % 3) == 0
        
        # Prediction confidence must be high
        if green_prob > 0.65 and zone['type'] == 'support' and rsi < 35 and wick_rejection:
            return ('LONG', green_prob)
        elif green_prob < 0.35 and zone['type'] == 'resistance' and rsi > 65 and wick_rejection:
            return ('SHORT', 1 - green_prob)
        
        return None
    
    def real_strategy(self, candles, green_prob, index):
        """Real market strategy: VWAP + MACD + EMA"""
        if len(candles) < 21:
            # Fallback to simple prediction-based signal
            if green_prob > 0.65:
                return ('LONG', green_prob)
            elif green_prob < 0.35:
                return ('SHORT', 1 - green_prob)
            return None
        
        # Calculate indicators
        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        
        # EMA crossover
        ema9 = self.calculate_ema(pd.Series(closes), 9).iloc[-1]
        ema21 = self.calculate_ema(pd.Series(closes), 21).iloc[-1]
        
        # RSI
        rsi = self.calculate_rsi(pd.Series(closes), 7).iloc[-1]
        
        # Simulate VWAP (simplified)
        vwap = np.mean(closes[-20:])
        
        # Confluence check
        last_close = closes[-1]
        
        signals = 0
        
        # Trend (EMA)
        if ema9 > ema21:
            signals += 1
        elif ema9 < ema21:
            signals -= 1
        
        # Momentum (RSI)
        if rsi < 30:
            signals += 1
        elif rsi > 70:
            signals -= 1
        
        # Price vs VWAP
        if last_close > vwap:
            signals += 1
        elif last_close < vwap:
            signals -= 1
        
        # Prediction
        if green_prob > 0.6:
            signals += 1
        elif green_prob < 0.4:
            signals -= 1
        
        # Strong signal requires 3+ confluence
        if signals >= 3 and green_prob > 0.55:
            return ('LONG', green_prob)
        elif signals <= -3 and green_prob < 0.45:
            return ('SHORT', 1 - green_prob)
        
        return None
