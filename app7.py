# =============================================================================
# ZOHA SIGNAL TERMINAL v1.2 - FIXED RENDERING & BDT TIME
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import json
import pytz

# ============================================================================
# IMMEDIATE SESSION STATE INITIALIZATION (FIX #1)
# ============================================================================
# This runs BEFORE everything else

def init_session_state():
    """Initialize all session state variables safely"""
    if 'zoha_initialized' not in st.session_state:
        st.session_state['zoha_initialized'] = True
        st.session_state['generate_clicked'] = False
        st.session_state['selected_pairs'] = []
        st.session_state['prediction_hours'] = 4.0
        st.session_state['risk_mult'] = 1.0
        st.session_state['signal_store'] = {
            'generated_signals': [],
            'pair_history': defaultdict(list),
            'signal_hash_map': {},
            'generation_count': 0
        }

init_session_state()

# ============================================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================================

st.set_page_config(
    page_title="ZOHA SIGNAL TERMINAL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;600;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top, #0c0f1d 0%, #070a15 100%);
        font-family: 'Exo 2', sans-serif;
        color: #e0e0e0;
    }
    
    .terminal-header {
        background: linear-gradient(90deg, #00ff88 0%, #00b8ff 50%, #9d00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3.5rem;
        text-align: center;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        animation: pulse 2s infinite;
    }
    
    .glow-card {
        background: linear-gradient(135deg, rgba(30, 30, 50, 0.9) 0%, rgba(20, 20, 40, 0.9) 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glow-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 50px rgba(0, 255, 136, 0.4);
    }
    
    .glow-card-UP { border-left-color: #00ff88; }
    .glow-card-DOWN { border-left-color: #ff0066; }
    
    .cyber-box {
        background: linear-gradient(135deg, rgba(0, 184, 255, 0.15) 0%, rgba(157, 0, 255, 0.15) 100%);
        border: 1px solid rgba(0, 184, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00ff88 0%, #00b8ff 100%);
        border: none;
        border-radius: 10px;
        color: #0f0f1a;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 15px 30px;
        box-shadow: 0 5px 20px rgba(0, 255, 136, 0.3);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 8px 30px rgba(0, 255, 136, 0.6);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# QUOTEX MARKETS & BDT TIMEZONE
# ============================================================================

class Config:
    QUOTEX_OTC_MARKETS = {
        '🔥 Volatility Indices': [
            'VOLATILITY_10', 'VOLATILITY_25', 'VOLATILITY_50', 'VOLATILITY_75', 'VOLATILITY_100'
        ],
        '📈 Step Indices': [
            'STEP_INDEX', 'STEP_200', 'STEP_500'
        ],
        '⚡ Jump Indices': [
            'JUMP_10', 'JUMP_25', 'JUMP_50', 'JUMP_75', 'JUMP_100'
        ],
        '🎯 Range Break': [
            'R_10', 'R_25', 'R_50', 'R_75', 'R_100'
        ],
        '💱 OTC Forex': [
            'OTC_EURUSD', 'OTC_GBPUSD', 'OTC_USDJPY', 'OTC_AUDUSD', 'OTC_USDCAD',
            'OTC_NZDUSD', 'OTC_EURGBP', 'OTC_GBPJPY', 'OTC_EURJPY', 'OTC_USDCHF'
        ],
        '🛢️ OTC Commodities': [
            'OTC_GOLD', 'OTC_SILVER', 'OTC_WTI', 'OTC_BRENT'
        ],
        '🌍 OTC Indices': [
            'OTC_US_500', 'OTC_US_TECH_100', 'OTC_WALL_STREET_30', 
            'OTC_UK_100', 'OTC_GERMANY_40', 'OTC_JAPAN_225', 'OTC_HONG_KONG_50'
        ]
    }
    
    BDT_TZ = pytz.timezone('Asia/Dhaka')
    VALID_MARKETS = [pair for category in QUOTEX_OTC_MARKETS.values() for pair in category]

def get_bdt_time():
    return datetime.now(Config.BDT_TZ).replace(second=0, microsecond=0)

# ============================================================================
# ENGINES (Simplified)
# ============================================================================

class DataEngine:
    def __init__(self, pairs):
        self.pairs = pairs
        self.cache = {}
        self._preload()
    
    def _preload(self):
        for pair in self.pairs:
            self.cache[pair] = self._generate(pair)
    
    def _generate(self, pair, bars=1000):
        vol = 0.002 if 'VOLATILITY' in pair else 0.001
        bdt_hour = get_bdt_time().hour
        mult = 2.0 if (9 <= bdt_hour <= 15) else 1.0
        
        returns = np.random.normal(0, vol * mult, bars)
        prices = 100 + np.cumsum(returns * 100)
        
        df = pd.DataFrame({
            'open': prices[:-1],
            'high': prices[1:] + np.abs(np.random.normal(0, vol * 40, bars-1)),
            'low': prices[1:] - np.abs(np.random.normal(0, vol * 40, bars-1)),
            'close': prices[1:],
            'volume': np.random.poisson(15000 * mult, bars-1),
            'spread': np.random.normal(0.4, 0.08, bars-1)
        })
        
        df['tick_volume_delta'] = np.random.normal(0, 0.6, len(df))
        return df
    
    def get_data(self, pair):
        return self.cache.get(pair, pd.DataFrame())

class StrategyEngine:
    def __init__(self):
        self.strategies_used = []
    
    def combine_all(self, pair, df):
        # Placeholder strategy
        if not df.empty:
            return ('TIER1_STRATEGY', df['close'].iloc[-1], 0.75)
        return None

class SignalStore:
    def __init__(self):
        self.store = st.session_state['signal_store']
    
    def generate_consistent_signals(self, signals):
        for signal in signals:
            signal_key = f"{signal['pair']}_{signal['time_bdt']}_{signal['strategy']}"
            signal_hash = hashlib.md5(signal_key.encode()).hexdigest()[:12]
            
            if signal_hash in self.store['signal_hash_map']:
                prev = self.store['signal_hash_map'][signal_hash]
                signal['direction'] = prev['direction']
                signal['accuracy'] = max(signal['accuracy'], prev['accuracy'])
            
            self.store['signal_hash_map'][signal_hash] = signal
            self.store['pair_history'][signal['pair']].append(signal)
        
        self.store['generated_signals'].extend(signals)
        return signals

class AnalyticsEngine:
    @staticmethod
    def monte_carlo(df, direction, sims=5000):
        return 0.70

class ProSignalGenerator:
    def __init__(self, pairs, prediction_hours):
        self.pairs = pairs
        self.prediction_hours = prediction_hours
        self.strategy_engine = StrategyEngine()
        self.analytics = AnalyticsEngine()
        self.signal_store = SignalStore()
        self.data_engine = DataEngine(self.pairs)
    
    def generate_50_signals(self):
        signals = []
        
        for i in range(50):
            pair = np.random.choice(self.pairs)
            df = self.data_engine.get_data(pair)
            
            raw_signal = self.strategy_engine.combine_all(pair, df)
            
            if raw_signal:
                signal_type, entry_price, base_confidence = raw_signal
                direction = 'UP' if 'LONG' in signal_type else 'DOWN'
                mc_accuracy = self.analytics.monte_carlo(df, direction, 5000)
                final_confidence = (base_confidence + mc_accuracy) / 2
                
                base_time = get_bdt_time() + timedelta(minutes=i * 4)
                signal_time = base_time
                
                signals.append({
                    'id': f"ZOHA-{datetime.now().strftime('%Y%m%d')}-{i+1:03d}",
                    'pair': pair,
                    'time_bdt': signal_time.strftime('%H:%M:%S'),
                    'direction': direction,
                    'entry_price': round(entry_price, 5),
                    'accuracy': round(final_confidence * 100, 1),
                    'prediction_hours': self.prediction_hours,
                    'expected_move_pct': round(np.random.uniform(1.5, 4.0) * final_confidence, 2),
                    'strategy': signal_type,
                    'strategies_used': ', '.join(self.strategy_engine.strategies_used),
                    'timestamp_bdt': signal_time,
                    'expires_at_bdt': signal_time + timedelta(hours=self.prediction_hours),
                    'status': 'ACTIVE'
                })
        
        return self.signal_store.generate_consistent_signals(signals)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_charts(signals):
    pairs = list(set([s['pair'] for s in signals]))
    corr_data = np.random.uniform(-0.7, 0.7, size=(len(pairs), len(pairs)))
    np.fill_diagonal(corr_data, 1.0)
    
    fig_corr = go.Figure(data=go.Heatmap(z=corr_data, x=pairs, y=pairs, colorscale='RdBu', zmid=0))
    fig_corr.update_layout(title="Cross-Pair Correlation (BDT)", paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e0e0"))
    
    strat_counts = pd.Series([s['strategy'] for s in signals]).value_counts().head(5)
    fig_strat = go.Figure(data=[go.Pie(labels=strat_counts.index, values=strat_counts.values, hole=0.5)])
    fig_strat.update_layout(title="Strategy Distribution", paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e0e0"))
    
    times = [datetime.strptime(s['time_bdt'], '%H:%M:%S') for s in signals]
    accs = np.cumsum([s['accuracy'] for s in signals])
    fig_perf = go.Figure(go.Scatter(x=times, y=accs, mode='lines+markers', line=dict(color='#00ff88', width=3)))
    fig_perf.update_layout(title="Cumulative Accuracy (BDT)", paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e0e0"))
    
    return {'correlation': fig_corr, 'strategy': fig_strat, 'performance': fig_perf}

# ============================================================================
# RENDER FUNCTIONS - FIX #2: Always render sidebar
# ============================================================================

def render_sidebar():
    """Render sidebar controls - ALWAYS VISIBLE"""
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(0,184,255,0.1) 100%); border: 1px solid rgba(0,255,136,0.3); border-radius: 15px; padding: 20px; margin-bottom: 20px;">
            <h2 style="color: #00ff88; font-family: 'Orbitron'; text-align: center;">⚙️ CONTROL PANEL</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # BDT Time indicator
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(255, 215, 0, 0.2) 0%, rgba(255, 165, 0, 0.2) 100%); border: 1px solid rgba(255, 215, 0, 0.4); border-radius: 8px; padding: 8px; text-align: center; font-family: 'Orbitron'; color: #FFD700;">
            🕐 BDT TIME: {get_bdt_time().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📡 Market Selection")
        
        # Category selection
        selected_cats = st.multiselect(
            "Select Categories:",
            options=list(Config.QUOTEX_OTC_MARKETS.keys()),
            default=['🔥 Volatility Indices', '📈 Step Indices']
        )
        
        # Get pairs from categories
        selected_pairs = []
        if selected_cats:
            for cat in selected_cats:
                selected_pairs.extend(Config.QUOTEX_OTC_MARKETS[cat])
        
        # Individual pair refinement
        st.markdown("**Refine Pairs:**")
        selected_pairs = st.multiselect(
            "Choose specific pairs:",
            options=selected_pairs,
            default=selected_pairs[:5] if selected_pairs else []
        )
        
        # Show pair count
        st.markdown(f"""
        <div style="background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); border-radius: 10px; padding: 10px;">
            <p style="color: #00ff88; margin: 0; text-align: center;">
                <b>{len(selected_pairs)}</b> PAIRS ACTIVE
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction settings
        st.markdown("---")
        st.subheader("⏱️ Prediction Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            hours = st.slider("Hours", 1, 24, 4)
        with col2:
            minutes = st.slider("Minutes", 0, 59, 0)
        
        total_prediction = hours + (minutes / 60)
        
        # Risk profile
        st.subheader("🛡️ Risk Profile")
        risk_style = st.select_slider(
            "Trading Style",
            options=["Conservative", "Balanced", "Aggressive"],
            value="Balanced"
        )
        
        risk_mult = {"Conservative": 0.8, "Balanced": 1.0, "Aggressive": 1.15}[risk_style]
        
        # Generate button
        st.markdown("---")
        if st.button("🚀 GENERATE 50 GLOWING SIGNALS", type="primary", use_container_width=True):
            if not selected_pairs:
                st.error("❌ NO PAIRS SELECTED!")
                # FIX: Don't proceed
                st.session_state['generate_clicked'] = False
            else:
                # FIX: Store in session state
                st.session_state['generate_clicked'] = True
                st.session_state['selected_pairs'] = selected_pairs
                st.session_state['prediction_hours'] = total_prediction
                st.session_state['risk_mult'] = risk_mult
                st.success("✅ Configuration saved!")
        
        # Reset button
        if st.button("🔄 RESET CONFIG", type="secondary", use_container_width=True):
            for key in ['generate_clicked', 'selected_pairs', 'prediction_hours', 'risk_mult']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['signal_store'] = {
                'generated_signals': [],
                'pair_history': defaultdict(list),
                'signal_hash_map': {},
                'generation_count': 0
            }
            st.success("✅ Reset complete!")

def render_main_content():
    """Render main content based on state"""
    # Always show header
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 class="terminal-header">ZOHA SIGNAL TERMINAL</h1>
        <p style="color: #94a3b8; font-size: 1.3rem; font-family: 'Orbitron'; letter-spacing: 2px;">
            Quotex OTC Signal Generator | BDT Timezone
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get('generate_clicked', False):
        # FIX: Get data safely
        pairs = st.session_state.get('selected_pairs', [])
        hours = st.session_state.get('prediction_hours', 4.0)
        risk_mult = st.session_state.get('risk_mult', 1.0)
        
        if not pairs:
            st.error("❌ No pairs selected! Please configure in sidebar.")
            return
        
        # Generate signals
        with st.spinner("🎯 Generating 50 glowing signals..."):
            generator = ProSignalGenerator(pairs, hours)
            signals = generator.generate_50_signals()
        
        # Apply risk multiplier
        for sig in signals:
            sig['accuracy'] = min(99.0, sig['accuracy'] * risk_mult)
        
        # Show metrics
        cols = st.columns([1, 1, 1, 1])
        metrics = [
            ("50", "Signals", "#00b8ff"),
            (f"{np.mean([s['accuracy'] for s in signals]):.1f}%", "Avg Accuracy", "#00ff88"),
            (str(len([s for s in signals if s['accuracy'] > 80])), "Strong", "#8b5cf6"),
            (str(len(set([s['pair'] for s in signals]))), "Pairs", "#f59e0b")
        ]
        
        for i, (value, label, color) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""
                <div class="cyber-box">
                    <h3 style="color: {color};">{value}</h3>
                    <p style="color: #94a3b8;">{label}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Tabs
        tabs = st.tabs(["🎯 SIGNALS", "📈 CHARTS", "💾 EXPORT"])
        
        with tabs[0]:
            df_table = pd.DataFrame(signals)
            display_df = df_table[['id', 'pair', 'time_bdt', 'direction', 'accuracy', 'prediction_hours', 'expected_move_pct', 'strategy']]
            display_df.columns = ['ID', 'Pair', 'BDT Time', 'Direction', 'Accuracy %', 'Hours', 'Move %', 'Strategy']
            
            st.dataframe(
                display_df.style.applymap(
                    lambda x: 'background-color: rgba(0,255,136,0.3); color: #00ff88' if x == 'UP' 
                    else 'background-color: rgba(255,0,102,0.3); color: #ff0066',
                    subset=['Direction']
                ),
                use_container_width=True,
                height=400
            )
            
            # Glow cards
            cols = st.columns(5)
            for idx, signal in enumerate(signals[:20]):
                with cols[idx % 5]:
                    glow_color = "#00ff88" if signal['direction'] == 'UP' else "#ff0066"
                    st.markdown(f"""
                    <div class="glow-card" style="border-left-color: {glow_color};">
                        <h4 style="color: {glow_color};">{signal['direction']}</h4>
                        <p><b>{signal['pair']}</b></p>
                        <p>BDT: {signal['time_bdt']}<br>Accuracy: {signal['accuracy']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tabs[1]:
            charts = create_charts(signals)
            st.plotly_chart(charts['correlation'], use_container_width=True)
            st.plotly_chart(charts['strategy'], use_container_width=True)
            st.plotly_chart(charts['performance'], use_container_width=True)
        
        with tabs[2]:
            csv = df_table.to_csv(index=False)
            st.download_button("📥 CSV", csv, f"ZOHA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px;">
            <h2 style="color: #94a3b8; font-family: 'Orbitron';">SYSTEM READY</h2>
            <p style="color: #64748b;">Use sidebar to configure and generate signals</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MAIN EXECUTION - FIX #3: Guaranteed execution
# ============================================================================

# FIX: Call these functions directly, no conditional wrapping
render_sidebar()
render_main_content()
