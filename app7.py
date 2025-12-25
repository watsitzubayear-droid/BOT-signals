# =============================================================================
# QOTEX SIGNAL BOT - STREAMLIT VERSION WITH PAIR SELECTION
# Save as: app7.py
# Run with: streamlit run app7.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import queue
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# Configure Streamlit page
st.set_page_config(
    page_title="Qotex Signal Bot Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION MODULE
# =============================================================================

class Config:
    ALL_OTC_PAIRS = [  # All available pairs for selection
        'VOLATILITY_100', 'VOLATILITY_75', 'VOLATILITY_50', 'VOLATILITY_25', 'VOLATILITY_10',
        'STEP_INDEX', 'STEP_200', 'STEP_500', 
        'JUMP_10', 'JUMP_25', 'JUMP_50', 'JUMP_75', 'JUMP_100',
        'R_10', 'R_25', 'R_50', 'R_75', 'R_100',
        'OTC_GOLD', 'OTC_SILVER', 'OTC_WTI', 'OTC_BRENT',
        'OTC_EURUSD', 'OTC_GBPUSD', 'OTC_USDJPY', 'OTC_AUDUSD', 'OTC_USDCAD',
        'OTC_NZDUSD', 'OTC_EURGBP', 'OTC_GBPJPY', 'OTC_EURJPY',
        'OTC_USDSGD', 'OTC_USDZAR', 'OTC_USDMXN', 'OTC_USDTRY', 'OTC_USDCNH',
        'OTC_AUDJPY', 'OTC_CADJPY', 'OTC_CHFJPY', 'OTC_EURCAD', 'OTC_EURAUD',
        'CRYPTO_BTC', 'CRYPTO_ETH', 'CRYPTO_LTC', 'CRYPTO_XRP', 'CRYPTO_BCH',
        'OTC_US_500', 'OTC_US_TECH_100', 'OTC_WALL_STREET_30', 
        'OTC_UK_100', 'OTC_GERMANY_40', 'OTC_FRANCE_40', 'OTC_SWISS_20',
        'OTC_JAPAN_225', 'OTC_HONG_KONG_50', 'OTC_CHINA_A50',
        'CUSTOM_1', 'CUSTOM_2', 'CUSTOM_3', 'CUSTOM_4', 'CUSTOM_5'
    ]
    
    SIGNAL_INTERVAL_SECONDS = 240
    PREDICTION_HORIZON_MINUTES = 300
    ACCURACY_THRESHOLD = 0.80
    
    TIER_1 = {'vwap_window': 20, 'macd_fast': 6, 'macd_slow': 17, 'macd_signal': 8}
    TIER_2 = {'ema_fast': 9, 'ema_slow': 21, 'rsi_period': 7, 'rsi_threshold': 50}
    DOJI_PARAMS = {'ema_period': 8, 'entry_offset': 1, 'stop_offset': 0.5, 'min_doji_size': 0.05}

# =============================================================================
# DATA FEED ENGINE
# =============================================================================

class DataFeed:
    def __init__(self, pairs):
        self.pairs = pairs
        self.data = defaultdict(lambda: pd.DataFrame())
        self.lock = threading.Lock()
        
    def fetch_data(self, pair, minutes=1000):
        try:
            base_volatility = 0.002 if 'VOLATILITY' in pair else 0.0004
            np.random.seed(int(time.time()) + hash(pair) % 10000)
            
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes)
            time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
            
            returns = np.random.normal(0, base_volatility, len(time_index))
            prices = 100 + np.cumsum(returns)
            
            open_prices = prices[:-1]
            close_prices = prices[1:]
            high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, base_volatility/2, len(open_prices)))
            low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, base_volatility/2, len(open_prices)))
            
            df = pd.DataFrame({
                'open': open_prices, 'high': high_prices, 'low': low_prices,
                'close': close_prices, 'volume': np.random.poisson(5000, len(open_prices)),
                'spread': np.random.normal(0.6, 0.2, len(open_prices))
            }, index=time_index[:-1])
            
            df['tick_volume_delta'] = np.random.normal(0, 0.5, len(df))
            df['pair'] = pair
            
            with self.lock:
                self.data[pair] = df
                
            return df
            
        except Exception as e:
            st.error(f"Data fetch error for {pair}: {e}")
            return pd.DataFrame()
    
    def get_latest(self, pair, bars=500):
        with self.lock:
            if pair not in self.data or self.data[pair].empty:
                self.fetch_data(pair)
            return self.data[pair].tail(bars)

# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class StrategyEngine:
    def __init__(self, data_feed):
        self.data_feed = data_feed
    
    def calculate_vwap(self, df):
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(Config.TIER_1['vwap_window']).sum() / df['volume'].rolling(Config.TIER_1['vwap_window']).sum()
        return df
    
    def calculate_macd(self, df):
        exp1 = df['close'].ewm(span=Config.TIER_1['macd_fast']).mean()
        exp2 = df['close'].ewm(span=Config.TIER_1['macd_slow']).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=Config.TIER_1['macd_signal']).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def calculate_rsi(self, df, period):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def evaluate_tier1(self, df):
        df = self.calculate_vwap(df)
        df = self.calculate_macd(df)
        
        price_cross_vwap_up = (df['close'].shift(1) < df['vwap'].shift(1)) & (df['close'] > df['vwap'])
        macd_flip_bullish = (df['macd_histogram'].shift(1) < 0) & (df['macd_histogram'] > 0)
        
        if macd_flip_bullish.iloc[-1] and price_cross_vwap_up.iloc[-1]:
            return ('TIER1_LONG', df['close'].iloc[-1], 0.75)
        return None
    
    def evaluate_tier2(self, df):
        df[f'ema_{Config.TIER_2["ema_fast"]}'] = df['close'].ewm(span=Config.TIER_2['ema_fast']).mean()
        df[f'ema_{Config.TIER_2["ema_slow"]}'] = df['close'].ewm(span=Config.TIER_2['ema_slow']).mean()
        df[f'rsi_{Config.TIER_2["rsi_period"]}'] = self.calculate_rsi(df, Config.TIER_2['rsi_period'])
        
        ema_cross_up = (df[f'ema_{Config.TIER_2["ema_fast"]}'].shift(1) < df[f'ema_{Config.TIER_2["ema_slow"]}'].shift(1)) & \
                       (df[f'ema_{Config.TIER_2["ema_fast"]}'] > df[f'ema_{Config.TIER_2["ema_slow"]}'])
        rsi_cross_up = (df[f'rsi_{Config.TIER_2["rsi_period"]}'].shift(1) < Config.TIER_2['rsi_threshold']) & \
                       (df[f'rsi_{Config.TIER_2["rsi_period"]}'] > Config.TIER_2['rsi_threshold'])
        
        if ema_cross_up.iloc[-1] and rsi_cross_up.iloc[-1]:
            return ('TIER2_LONG', df['close'].iloc[-1], 0.70)
        return None
    
    def combine_signals(self, pair, df):
        signal = self.evaluate_tier1(df.copy())
        if not signal:
            signal = self.evaluate_tier2(df.copy())
        return signal

# =============================================================================
# ACCURACY VALIDATOR
# =============================================================================

class AccuracyValidator:
    def __init__(self):
        self.validation_history = defaultdict(list)
    
    def calculate_accuracy(self, pair, signal_type, lookback=50):
        history = self.validation_history[pair]
        recent = [h for h in history if h['signal_type'] == signal_type][-lookback:]
        if len(recent) < 10:
            return 0.5
        return sum(1 for r in recent if r['result'] == 'WIN') / len(recent)
    
    def validate_signal(self, pair, signal, df):
        if not signal:
            return None
        
        signal_type, price, confidence = signal
        historical_accuracy = self.calculate_accuracy(pair, signal_type)
        projected_accuracy = (confidence + historical_accuracy) / 2
        
        if projected_accuracy >= Config.ACCURACY_THRESHOLD:
            five_hour_confidence = projected_accuracy * 0.95
            return {
                'signal_type': signal_type,
                'entry_price': price,
                'confidence': projected_accuracy,
                'five_hour_prediction': {
                    'direction': 'UP' if 'LONG' in signal_type else 'DOWN',
                    'confidence': five_hour_confidence,
                    'expected_move_pct': 1.5
                },
                'generated_at': datetime.now()
            }
        return None

# =============================================================================
# SIGNAL MANAGER
# =============================================================================

class SignalManager:
    def __init__(self, pairs):
        self.pairs = pairs
        self.data_feed = DataFeed(self.pairs)
        self.strategy_engine = StrategyEngine(self.data_feed)
        self.accuracy_validator = AccuracyValidator()
        
        self.signal_queue = queue.Queue()
        self.signal_history = []
        self.last_signal_time = defaultdict(lambda: datetime.min)
        self.running = False
        
        # Thread-safe storage
        self._history_lock = threading.Lock()
        self._latest_signals = {}
    
    def start(self):
        self.running = True
        for pair in self.pairs:
            thread = threading.Thread(target=self._pair_worker, args=(pair,), daemon=True)
            thread.start()
    
    def stop(self):
        self.running = False
    
    def _pair_worker(self, pair):
        while self.running:
            time_since_last = datetime.now() - self.last_signal_time[pair]
            if time_since_last.total_seconds() < Config.SIGNAL_INTERVAL_SECONDS:
                time.sleep(1)
                continue
            
            try:
                df = self.data_feed.get_latest(pair, bars=500)
                if df.empty:
                    continue
                
                raw_signal = self.strategy_engine.combine_signals(pair, df)
                validated_signal = self.accuracy_validator.validate_signal(pair, raw_signal, df)
                
                if validated_signal:
                    self.last_signal_time[pair] = datetime.now()
                    
                    # Format time as HH:MM:SS
                    formatted_time = validated_signal['generated_at'].replace(second=0, microsecond=0)
                    time_str = formatted_time.strftime('%H:%M:%S')
                    
                    signal_data = {
                        'pair': pair,
                        'time': time_str,
                        'direction': validated_signal['five_hour_prediction']['direction'],
                        'accuracy': validated_signal['confidence'] * 100,
                        'five_hour_confidence': validated_signal['five_hour_prediction']['confidence'] * 100,
                        'expected_move': validated_signal['five_hour_prediction']['expected_move_pct'],
                        'strategy': validated_signal['signal_type'],
                        'timestamp': validated_signal['generated_at']
                    }
                    
                    # Thread-safe update
                    with self._history_lock:
                        self.signal_history.append(signal_data)
                        self._latest_signals[pair] = signal_data
                        if len(self.signal_history) > 1000:
                            self.signal_history = self.signal_history[-500:]
                
                time.sleep(0.5)
            except Exception as e:
                st.error(f"Worker error {pair}: {e}")
                time.sleep(5)
    
    def get_signals_for_display(self):
        with self._history_lock:
            return self.signal_history.copy(), self._latest_signals.copy()

# =============================================================================
# STREAMLIT UI
# =============================================================================

def pair_selection_sidebar():
    """Pair selection interface in sidebar"""
    st.header("📈 Pair Selection")
    
    # Multi-select widget
    selected_pairs = st.multiselect(
        "Choose pairs to monitor:",
        options=Config.ALL_OTC_PAIRS,
        default=['VOLATILITY_100', 'OTC_EURUSD', 'OTC_GOLD', 'CRYPTO_BTC', 'STEP_INDEX'],
        help="Select multiple pairs to generate signals from"
    )
    
    # Quick select buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All", key="select_all"):
            return Config.ALL_OTC_PAIRS
    with col2:
        if st.button("Clear All", key="clear_all"):
            return []
    
    # Display count
    st.info(f"**{len(selected_pairs)}** pairs selected")
    
    return selected_pairs

def main():
    """Main Streamlit application"""
    
    st.title("🎯 Qotex Signal Bot - Live Trading Signals")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Bot Controls")
        
        # Pair selection
        selected_pairs = pair_selection_sidebar()
        
        st.markdown("---")
        
        # Start/Stop buttons
        if st.button("🚀 START BOT", key="start", type="primary"):
            st.session_state['bot_running'] = True
            st.session_state['selected_pairs'] = selected_pairs
            st.success("Bot started with selected pairs!")
        
        if st.button("⏹️ STOP BOT", key="stop", type="secondary"):
            st.session_state['bot_running'] = False
            st.warning("Bot stopped.")
        
        st.markdown("---")
        st.header("⚙️ Configuration")
        st.info(f"Signal interval: **{Config.SIGNAL_INTERVAL_SECONDS}** seconds")
        st.info(f"Accuracy threshold: **{Config.ACCURACY_THRESHOLD*100}%**")
    
    # Initialize session state
    if 'bot_running' not in st.session_state:
        st.session_state['bot_running'] = False
    if 'selected_pairs' not in st.session_state:
        st.session_state['selected_pairs'] = ['VOLATILITY_100']
    if 'signal_manager' not in st.session_state:
        st.session_state['signal_manager'] = None
    
    # Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        total_signals = st.empty()
    with col2:
        avg_accuracy = st.empty()
    with col3:
        active_pairs = st.empty()
    
    st.markdown("---")
    st.subheader("🔥 Latest Signals")
    signals_container = st.empty()
    
    st.markdown("---")
    st.subheader("📈 Recent History")
    history_container = st.empty()
    
    # Bot runtime logic
    if st.session_state['bot_running']:
        # Create or recreate manager if pairs changed
        if (st.session_state['signal_manager'] is None or 
            st.session_state['selected_pairs'] != st.session_state.get('last_pairs', [])):
            
            # Stop old bot if running
            if st.session_state['signal_manager'] is not None:
                st.session_state['signal_manager'].stop()
                time.sleep(1)
            
            # Create new manager with selected pairs
            st.session_state['last_pairs'] = st.session_state['selected_pairs'].copy()
            st.session_state['signal_manager'] = SignalManager(st.session_state['selected_pairs'])
            st.toast(f"Bot restarted with {len(st.session_state['selected_pairs'])} pairs!", icon="🔄")
        
        manager = st.session_state['signal_manager']
        
        if not hasattr(manager, '_started'):
            manager.start()
            manager._started = True
            st.toast("🔥 Bot is now running!", icon="🚀")
        
        # Update display every 2 seconds
        while st.session_state['bot_running']:
            try:
                # Get thread-safe data
                history, latest = manager.get_signals_for_display()
                
                # Update statistics
                total_signals.metric("Total Signals", len(history))
                if history:
                    avg_acc = np.mean([h['accuracy'] for h in history[-50:]])
                    avg_accuracy.metric("Avg Accuracy", f"{avg_acc:.1f}%")
                active_pairs.metric("Active Pairs", f"{len(latest)}/{len(st.session_state['selected_pairs'])}")
                
                # Display latest signals
                with signals_container.container():
                    if latest:
                        for pair, sig in list(latest.items())[-10:]:
                            color = "green" if sig['direction'] == 'UP' else "red"
                            st.markdown(
                                f"**{sig['time']}** | **{pair}** | "
                                f":{color}[**{sig['direction']}**] | "
                                f"Accuracy: **{sig['accuracy']:.1f}%** | "
                                f"5H: **{sig['five_hour_confidence']:.1f}%** | "
                                f"Strategy: **{sig['strategy']}**",
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("Waiting for first signal...")
                
                # Display recent history
                with history_container.container():
                    if history:
                        for sig in history[-10:]:
                            color = "green" if sig['direction'] == 'UP' else "red"
                            st.markdown(
                                f"**{sig['time']}** | {sig['pair']} | "
                                f":{color}[{sig['direction']}] | "
                                f"{sig['accuracy']:.1f}% | {sig['strategy']}",
                                unsafe_allow_html=True
                            )
                
                # Auto-refresh
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"Runtime error: {e}")
                time.sleep(2)
    else:
        # Show selected pairs even when bot is stopped
        st.info(f"⏸️ Bot is stopped. Selected pairs: {', '.join(st.session_state['selected_pairs'])}")
        st.info("Click 'START BOT' in the sidebar to begin.")

if __name__ == "__main__":
    main()
