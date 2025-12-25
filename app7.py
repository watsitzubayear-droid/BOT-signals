# =============================================================================
# QOTEX SIGNAL BOT - STREAMLIT VERSION (FIXED & ENHANCED)
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
    ALL_OTC_PAIRS = [
        'VOLATILITY_100', 'VOLATILITY_75', 'VOLATILITY_50', 'VOLATILITY_25', 'VOLATILITY_10',
        'STEP_INDEX', 'STEP_200', 'STEP_500', 
        'JUMP_10', 'JUMP_25', 'JUMP_50', 'JUMP_75', 'JUMP_100',
        'R_10', 'R_25', 'R_50', 'R_75', 'R_100',
        'OTC_GOLD', 'OTC_SILVER', 'OTC_WTI', 'OTC_BRENT',
        'OTC_EURUSD', 'OTC_GBPUSD', 'OTC_USDJPY', 'OTC_AUDUSD', 'OTC_USDCAD',
        'OTC_NZDUSD', 'OTC_EURGBP', 'OTC_GBPJPY', 'OTC_EURJPY',
        'CRYPTO_BTC', 'CRYPTO_ETH', 'CRYPTO_LTC', 'CRYPTO_XRP', 'CRYPTO_BCH',
        'OTC_US_500', 'OTC_US_TECH_100', 'OTC_WALL_STREET_30', 
        'OTC_UK_100', 'OTC_GERMANY_40', 'VOLATILITY_75'
    ]
    
    SIGNAL_INTERVAL_SECONDS = 240
    DEFAULT_PREDICTION_HOURS = 24
    ACCURACY_THRESHOLD = 0.80
    
    TIER_1 = {'vwap_window': 20, 'macd_fast': 6, 'macd_slow': 17, 'macd_signal': 8}
    TIER_2 = {'ema_fast': 9, 'ema_slow': 21, 'rsi_period': 7, 'rsi_threshold': 50}

# =============================================================================
# DATA FEED ENGINE
# =============================================================================

class DataFeed:
    def __init__(self, pairs):
        self.pairs = pairs
        self.data = defaultdict(lambda: pd.DataFrame())
        self.lock = threading.Lock()
        
    def fetch_data(self, pair, minutes=2000):
        try:
            base_volatility = 0.002 if 'VOLATILITY' in pair else 0.0004
            session_multiplier = 2.0 if 13 <= datetime.utcnow().hour < 16 else 1.0
            
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes)
            time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
            
            returns = np.random.normal(0, base_volatility * session_multiplier, len(time_index))
            prices = 100 + np.cumsum(returns)
            
            open_prices = prices[:-1]
            close_prices = prices[1:]
            high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, base_volatility/2, len(open_prices)))
            low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, base_volatility/2, len(open_prices)))
            
            df = pd.DataFrame({
                'open': open_prices, 'high': high_prices, 'low': low_prices,
                'close': close_prices, 'volume': np.random.poisson(8000 * session_multiplier, len(open_prices)),
                'spread': np.random.normal(0.6, 0.15, len(open_prices))
            }, index=time_index[:-1])
            
            df['tick_volume_delta'] = np.random.normal(0, 0.5, len(df))
            df['pair'] = pair
            
            with self.lock:
                self.data[pair] = df
                
            return df
            
        except Exception as e:
            st.error(f"Data error {pair}: {str(e)[:50]}")
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
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def evaluate_tier1(self, df):
        df = self.calculate_vwap(df)
        df = self.calculate_macd(df)
        
        price_cross_vwap_up = (df['close'].shift(1) < df['vwap'].shift(1)) & (df['close'] > df['vwap'])
        macd_flip_bullish = (df['macd_histogram'].shift(1) < 0) & (df['macd_histogram'] > 0)
        
        if macd_flip_bullish.iloc[-1] and price_cross_vwap_up.iloc[-1]:
            return ('TIER1_LONG', df['close'].iloc[-1], 0.75)
        return None
    
    def combine_signals(self, pair, df):
        return self.evaluate_tier1(df.copy())

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
    
    def validate_signal(self, pair, signal, df, prediction_hours):
        if not signal:
            return None
        
        signal_type, price, confidence = signal
        historical_accuracy = self.calculate_accuracy(pair, signal_type)
        projected_accuracy = (confidence + historical_accuracy) / 2
        
        if projected_accuracy >= Config.ACCURACY_THRESHOLD:
            # Calculate prediction based on user-specified hours
            prediction_minutes = prediction_hours * 60
            five_hour_confidence = projected_accuracy * 0.95
            
            return {
                'signal_type': signal_type,
                'entry_price': price,
                'confidence': projected_accuracy,
                'prediction_horizon': {
                    'hours': prediction_hours,
                    'minutes': prediction_minutes,
                    'direction': 'UP' if 'LONG' in signal_type else 'DOWN',
                    'confidence': five_hour_confidence,
                    'expected_move_pct': np.random.uniform(0.8, 2.5)  # Simulated
                },
                'generated_at': datetime.now()
            }
        return None

# =============================================================================
# SIGNAL MANAGER
# =============================================================================

class SignalManager:
    def __init__(self, pairs, prediction_hours):
        self.pairs = pairs
        self.prediction_hours = prediction_hours
        self.data_feed = DataFeed(self.pairs)
        self.strategy_engine = StrategyEngine(self.data_feed)
        self.accuracy_validator = AccuracyValidator()
        
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
        st.toast(f"✅ Bot started with {len(self.pairs)} pairs", icon="🎯")
    
    def stop(self):
        self.running = False
        st.toast("⏹️ Bot stopped", icon="⚠️")
    
    def _pair_worker(self, pair):
        while self.running:
            time_since_last = datetime.now() - self.last_signal_time[pair]
            if time_since_last.total_seconds() < Config.SIGNAL_INTERVAL_SECONDS:
                time.sleep(1)
                continue
            
            try:
                df = self.data_feed.get_latest(pair, bars=500)
                if df.empty:
                    time.sleep(1)
                    continue
                
                raw_signal = self.strategy_engine.combine_signals(pair, df)
                validated_signal = self.accuracy_validator.validate_signal(
                    pair, raw_signal, df, self.prediction_hours
                )
                
                if validated_signal:
                    self.last_signal_time[pair] = datetime.now()
                    
                    # Format time as HH:MM:SS
                    formatted_time = validated_signal['generated_at'].replace(second=0, microsecond=0)
                    time_str = formatted_time.strftime('%H:%M:%S')
                    
                    signal_data = {
                        'pair': pair,
                        'time': time_str,
                        'direction': validated_signal['prediction_horizon']['direction'],
                        'accuracy': validated_signal['confidence'] * 100,
                        'prediction_hours': validated_signal['prediction_horizon']['hours'],
                        'prediction_confidence': validated_signal['prediction_horizon']['confidence'] * 100,
                        'expected_move': validated_signal['prediction_horizon']['expected_move_pct'],
                        'strategy': validated_signal['signal_type'],
                        'timestamp': validated_signal['generated_at'],
                        'expires_at': validated_signal['generated_at'] + timedelta(hours=self.prediction_hours)
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

@st.cache_resource
def get_manager(pairs, prediction_hours):
    """Singleton manager instance"""
    return SignalManager(pairs, prediction_hours)

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    """Main Streamlit application"""
    
    st.title("🎯 Qotex Signal Bot - Live Trading Signals")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Bot Controls")
        
        # Pair selection
        st.subheader("📈 Pair Selection")
        selected_pairs = st.multiselect(
            "Choose pairs to monitor:",
            options=Config.ALL_OTC_PAIRS,
            default=['VOLATILITY_100', 'OTC_EURUSD', 'OTC_GOLD'],
            help="Select multiple pairs. Ctrl/Cmd+click for multiple on desktop."
        )
        
        # Quick select buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Select All", key="select_all"):
                st.session_state['select_all_pairs'] = True
        with col_btn2:
            if st.button("Clear All", key="clear_all"):
                st.session_state['select_all_pairs'] = False
        
        if st.session_state.get('select_all_pairs', None) is True:
            selected_pairs = Config.ALL_OTC_PAIRS
        elif st.session_state.get('select_all_pairs', None) is False:
            selected_pairs = []
        
        st.info(f"**{len(selected_pairs)}** pairs selected")
        
        st.markdown("---")
        
        # Prediction horizon control
        st.subheader("⏱️ Prediction Horizon")
        prediction_hours = st.slider(
            "Hours to predict:",
            min_value=1,
            max_value=24,
            value=4,
            help="How far ahead to predict (1-24 hours)"
        )
        
        prediction_minutes = st.slider(
            "Additional minutes:",
            min_value=0,
            max_value=59,
            value=0,
            help="Add minutes to prediction time"
        )
        
        total_prediction_hours = prediction_hours + (prediction_minutes / 60)
        
        st.markdown("---")
        
        # Start/Stop buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("🚀 START BOT", key="start", type="primary"):
                if not selected_pairs:
                    st.error("❌ Select at least one pair!")
                else:
                    st.session_state['bot_running'] = True
                    st.session_state['selected_pairs'] = selected_pairs
                    st.session_state['prediction_hours'] = total_prediction_hours
                    st.experimental_rerun()
        
        with col_stop:
            if st.button("⏹️ STOP BOT", key="stop", type="secondary"):
                st.session_state['bot_running'] = False
                if st.session_state.get('signal_manager'):
                    st.session_state['signal_manager'].stop()
                st.experimental_rerun()
        
        st.markdown("---")
        st.header("⚙️ Status")
        st.info(f"Prediction: **{total_prediction_hours:.1f}** hours ahead")
    
    # Initialize session state
    if 'bot_running' not in st.session_state:
        st.session_state['bot_running'] = False
    if 'selected_pairs' not in st.session_state:
        st.session_state['selected_pairs'] = ['VOLATILITY_100']
    if 'prediction_hours' not in st.session_state:
        st.session_state['prediction_hours'] = 4.0
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
        # Create/recreate manager if needed
        if (st.session_state['signal_manager'] is None or 
            st.session_state['selected_pairs'] != st.session_state.get('last_pairs', []) or
            st.session_state['prediction_hours'] != st.session_state.get('last_prediction', 0)):
            
            # Stop old bot
            if st.session_state['signal_manager'] is not None:
                st.session_state['signal_manager'].stop()
                time.sleep(1)
            
            # Create new manager
            st.session_state['last_pairs'] = st.session_state['selected_pairs'].copy()
            st.session_state['last_prediction'] = st.session_state['prediction_hours']
            
            manager = get_manager(
                st.session_state['selected_pairs'], 
                st.session_state['prediction_hours']
            )
            st.session_state['signal_manager'] = manager
            manager.start()
            st.toast(f"✅ Bot started with {len(st.session_state['selected_pairs'])} pairs", icon="🎯")
        
        manager = st.session_state['signal_manager']
        
        # Update display every 2 seconds
        update_interval = st.empty()
        last_update = time.time()
        
        while st.session_state['bot_running']:
            try:
                # Check if we need to stop due to config change
                if (st.session_state['selected_pairs'] != st.session_state.get('last_pairs', []) or
                    st.session_state['prediction_hours'] != st.session_state.get('last_prediction', 0)):
                    st.warning("Configuration changed! Restarting bot...")
                    st.session_state['bot_running'] = False
                    st.experimental_rerun()
                
                # Get data
                history, latest = manager.get_signals_for_display()
                
                # Update stats
                total_signals.metric("Total Signals", len(history))
                if history:
                    avg_acc = np.mean([h['accuracy'] for h in history[-50:]]) if history else 0
                    avg_accuracy.metric("Avg Accuracy", f"{avg_acc:.1f}%")
                active_pairs.metric("Active Pairs", f"{len(latest)}/{len(st.session_state['selected_pairs'])}")
                
                # Display signals
                with signals_container.container():
                    if latest:
                        for pair, sig in list(latest.items())[-15:]:
                            # Check if signal is still valid
                            time_until_expiry = sig['expires_at'] - datetime.now()
                            if time_until_expiry.total_seconds() > 0:
                                color = "green" if sig['direction'] == 'UP' else "red"
                                st.markdown(
                                    f"**{sig['time']}** | **{pair}** | "
                                    f":{color}[**{sig['direction']}**] | "
                                    f"Accuracy: **{sig['accuracy']:.1f}%** | "
                                    f"Predict: **{sig['prediction_hours']:.1f}h** | "
                                    f"Expires: **{time_until_expiry.seconds // 3600}h {(time_until_expiry.seconds % 3600) // 60}m** | "
                                    f"Strategy: **{sig['strategy']}**",
                                    unsafe_allow_html=True
                                )
                        st.caption(f"⏱️ Signals refresh every {Config.SIGNAL_INTERVAL_SECONDS}s")
                    else:
                        st.info("⏳ Waiting for first signal...")
                
                # Display history
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
                
                # Show last update time
                update_interval.info(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
                
                # Sleep and rerun
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"Runtime error: {e}")
                time.sleep(5)
    else:
        # Show selected pairs and prediction when stopped
        if st.session_state.get('selected_pairs'):
            st.info(f"⏸️ Bot stopped. Selected: {len(st.session_state['selected_pairs'])} pairs | Prediction: {st.session_state['prediction_hours']:.1f}h")

if __name__ == "__main__":
    main()
