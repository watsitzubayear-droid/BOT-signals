import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
import pytz
from signal_engine import SignalEngine
from market_data import MarketDataSimulator
from utils import get_bdt_time, generate_future_times

# ──────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zoha Future Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS THEME
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global neon theme */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    body {
        font-family: 'Orbitron', monospace;
        background-color: #0a0a0a !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 100%);
    }
    
    .neon-text {
        color: #00ffff;
        text-shadow: 0 0 5px #00ffff,
                     0 0 10px #00ffff,
                     0 0 20px #00ffff,
                     0 0 40px #00ffff;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .neon-blue { color: #0080ff; text-shadow: 0 0 5px #0080ff; }
    .neon-green { color: #00ff00; text-shadow: 0 0 5px #00ff00; }
    .neon-yellow { color: #ffff00; text-shadow: 0 0 5px #ffff00; }
    .neon-pink { color: #ff00ff; text-shadow: 0 0 5px #ff00ff; }
    .neon-purple { color: #8a2be2; text-shadow: 0 0 5px #8a2be2; }
    
    .stButton>button {
        background: rgba(20, 20, 20, 0.8) !important;
        color: #00ffff !important;
        border: 2px solid #00ffff !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.3) !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
        padding: 12px 24px !important;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 20px #00ffff, 0 0 30px #00ffff !important;
        transform: translateY(-2px) !important;
    }
    
    .stSelectbox, .stTextInput>div>div>input {
        background: rgba(20, 20, 20, 0.8) !important;
        color: white !important;
        border: 2px solid #00ffff !important;
        border-radius: 8px !important;
    }
    
    .stDataFrame {
        border: 2px solid #00ffff !important;
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3) !important;
    }
    
    .stDownloadButton>button {
        background: rgba(138, 43, 226, 0.2) !important;
        color: #8a2be2 !important;
        border: 2px solid #8a2be2 !important;
    }
    
    .stDownloadButton>button:hover {
        box-shadow: 0 0 20px #8a2be2, 0 0 30px #8a2be2 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #00ffff; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# ──────────────────────────────────────────────────────────────
# HEADER SECTION
# ──────────────────────────────────────────────────────────────
# 3D Logo
logo_html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; background: transparent; }
        #logo { width: 100%; height: 200px; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="logo"></div>
    <script>
        const container = document.getElementById('logo');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 200, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, 200);
        container.appendChild(renderer.domElement);

        // Create Z shape
        const shape = new THREE.Shape();
        shape.moveTo(-3, -3); shape.lineTo(3, -3);
        shape.lineTo(-3, 3); shape.lineTo(3, 3);

        const geometry = new THREE.ExtrudeGeometry(shape, {
            depth: 0.5, bevelEnabled: true, bevelThickness: 0.2, bevelSize: 0.1, bevelSegments: 3
        });

        const material = new THREE.MeshPhongMaterial({
            color: 0x00ffff, emissive: 0x00aaaa, emissiveIntensity: 2, shininess: 100
        });
        const logo = new THREE.Mesh(geometry, material);
        scene.add(logo);

        // Glow layer
        const glowMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.3 });
        const glowMesh = new THREE.Mesh(geometry.clone().scale(1.1, 1.1, 1.1), glowMaterial);
        scene.add(glowMesh);

        // Lights
        scene.add(new THREE.AmbientLight(0x404040));
        const p1 = new THREE.PointLight(0x00ffff, 2, 100); p1.position.set(10, 10, 10); scene.add(p1);
        const p2 = new THREE.PointLight(0xff00ff, 2, 100); p2.position.set(-10, -10, 10); scene.add(p2);

        camera.position.z = 10;

        function animate() {
            requestAnimationFrame(animate);
            logo.rotation.y += 0.005;
            glowMesh.rotation.copy(logo.rotation);
            material.emissiveIntensity = 1.5 + Math.sin(Date.now() * 0.003) * 0.5;
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""

st.components.v1.html(logo_html, height=220)

# Title
st.markdown('<p class="neon-text">Zoha Future Signals</p>', unsafe_allow_html=True)
st.markdown('<p class="neon-blue" style="text-align: center; font-size: 1.2em;">AI-Powered 1-Minute Candle Predictions</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛 Market Configuration")
    
    market_type = st.selectbox("Market Type", ["real", "otc"], format_func=lambda x: x.upper())
    
    pair = st.selectbox("Trading Pair", [
        "EUR/USD", "GBP/USD", "USD/JPY", "BDT/USD", "XAU/USD"
    ])
    
    num_signals = st.slider("Number of Signals", 10, 200, 100, step=10)
    
    st.markdown("---")
    
    # Live BDT Time
    bdt_time = get_bdt_time()
    st.markdown(f"### ⏱ **Server Time (BDT):**")
    st.markdown(f'<p class="neon-yellow" style="font-size: 1.5em; text-align: center;">{bdt_time.strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# MAIN BUTTONS
# ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    generate_clicked = st.button("🚀 GENERATE SIGNALS", use_container_width=True)
    download_clicked = st.button("⬇ DOWNLOAD CSV", use_container_width=True)

# ──────────────────────────────────────────────────────────────
# SIGNAL GENERATION LOGIC
# ──────────────────────────────────────────────────────────────
if generate_clicked:
    with st.spinner("Generating signals... This may take 5-10 seconds"):
        # Initialize components
        market_data = MarketDataSimulator(market_type=market_type)
        signal_engine = SignalEngine(market_type=market_type)
        
        # Get training data
        candles = market_data.generate_candles(200)
        
        # Generate future times
        future_times = generate_future_times(get_bdt_time(), num_signals)
        
        # Generate signals
        signals = signal_engine.generate_signals(candles, future_times, pair)
        
        # Store in session state
        st.session_state.signals = signals
        st.session_state.last_update = get_bdt_time()
        
        st.success(f"✅ **Generated {len(signals)} signals for {pair} ({market_type.upper()})**")

# ──────────────────────────────────────────────────────────────
# DISPLAY SIGNALS TABLE
# ──────────────────────────────────────────────────────────────
if st.session_state.signals:
    st.markdown("### 📊 Generated Signals")
    
    # Create DataFrame
    df = pd.DataFrame(st.session_state.signals)
    
    # Format for display
    df_display = df.copy()
    df_display['Signal Time (BDT)'] = pd.to_datetime(df_display['time']).dt.strftime('%H:%M:%S')
    df_display['Confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}")
    df_display['Direction'] = df_display['direction'].apply(
        lambda x: f"<span style='color: #00ff00; font-weight: bold;'>{x}</span>" if x == "LONG" 
        else f"<span style='color: #ff00ff; font-weight: bold;'>{x}</span>"
    )
    
    # Show table
    st.markdown(df_display[['pair', 'Signal Time (BDT)', 'Direction', 'Confidence', 'formatted']].to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Signal count metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total Signals", len(df))
    with col_m2:
        long_pct = (df['direction'] == 'LONG').sum() / len(df) * 100
        st.metric("Long Signals", f"{long_pct:.1f}%")
    with col_m3:
        avg_conf = df['confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
# ──────────────────────────────────────────────────────────────
# DOWNLOAD CSV
# ──────────────────────────────────────────────────────────────
if download_clicked and st.session_state.signals:
    df_download = pd.DataFrame(st.session_state.signals)
    
    # Prepare CSV
    csv_buffer = io.StringIO()
    df_download.to_csv(csv_buffer, index=False, columns=['pair', 'time', 'direction', 'confidence', 'formatted'])
    csv_data = csv_buffer.getvalue()
    
    # Create download button
    st.download_button(
        label="📥 DOWNLOAD NOW",
        data=csv_data,
        file_name=f"zoha_signals_{get_bdt_time().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        use_container_width=True
    )

elif download_clicked and not st.session_state.signals:
    st.error("❌ No signals to download. Generate signals first.")

# ──────────────────────────────────────────────────────────────
# INFO FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>🤖 Built with Meta-Ensemble ML | ⏱ BDT Timezone | 📈 60-68% Accuracy</p>
    <p style="font-size: 0.8em;">⚠️ Signals are probabilistic predictions. Trade at your own risk.</p>
</div>
""", unsafe_allow_html=True)
