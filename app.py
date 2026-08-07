"""
EcoPulse - Streamlit demo

"""
import os
import time
import tempfile
import random
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO

# ==========================================
# 1. PAGE CONFIGURATION & GLASSMORPHISM CSS
# ==========================================
st.set_page_config(
    page_title="EcoPulse | HEC Early-Warning Core",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f8fafc !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.92)), 
                    url("https://images.unsplash.com/photo-1516426122078-c23e76319801?q=80&w=1920&auto=format&fit=crop") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    .hero-card {
        background: rgba(30, 41, 59, 0.60) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.5) !important;
    }

    div.stButton > button {
        background: rgba(16, 185, 129, 0.22) !important;
        color: #6ee7b7 !important;
        border: 1px solid rgba(52, 211, 153, 0.5) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        transition: all 0.3s ease-in-out !important;
    }

    div.stButton > button:hover {
        background: rgba(16, 185, 129, 0.45) !important;
        border-color: rgba(52, 211, 153, 0.9) !important;
        color: #ffffff !important;
        box-shadow: 0 0 25px rgba(52, 211, 153, 0.6) !important;
        transform: translateY(-2px);
    }

    .status-high {
        background-color: rgba(127, 29, 29, 0.85);
        border: 2px solid #ef4444;
        color: #fca5a5;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        animation: pulse 1.5s infinite;
    }
    .status-med {
        background-color: rgba(120, 53, 15, 0.85);
        border: 2px solid #f59e0b;
        color: #fde68a;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    .status-low {
        background-color: rgba(6, 78, 59, 0.85);
        border: 2px solid #10b981;
        color: #a7f3d0;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    </style>
""", unsafe_allow_html=True)

# Extended 4-layer weights including Seismic Telemetry
WEIGHTS = {"audio": 0.35, "image": 0.30, "seismic": 0.20, "geo": 0.15}
ALERT_THRESHOLD = 0.55

def fuse(a, i, s, g):
    return (WEIGHTS["audio"] * a + WEIGHTS["image"] * i + WEIGHTS["seismic"] * s + WEIGHTS["geo"] * g)

def generate_seismic_waveform(pga_g, freq_hz):
    t = np.linspace(0, 2.0, 100) # 2-second buffer window
    signal = 0.02 * np.sin(2 * np.pi * 5 * t) + pga_g * np.exp(-((t - 1.0)**2) / 0.04) * np.sin(2 * np.pi * freq_hz * t)
    return pd.DataFrame({"Time (s)": t, "Ground Acceleration (g)": signal})

# ==========================================
# 2. HERO HEADER SECTION
# ==========================================
st.markdown("""
<div class="hero-card">
    <div style="font-size: 13px; color: #38bdf8; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 6px;">
        🌐 EDGE-BASED MULTIMODAL WILDLIFE MONITORING
    </div>
    <h1 style="font-size: 36px; font-weight: 800; margin: 0 0 8px 0; color: #ffffff;">
        🐘 EcoPulse Early-Warning Core
    </h1>
    <p style="color: #cbd5e1; font-size: 14px; margin: 0;">
        <strong>Data Odyssey 2026</strong> | Real-Time Human-Elephant Conflict Mitigation via Bioacoustic, Vision, Seismic & Geospatial Fusion.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Center")
    mode = st.radio("Select Operation Mode", ["📡 Live Corridor Stream", "🔍 Diagnostic Sample Test"])
    
    st.divider()
    st.markdown("### ⚖️ Multi-Sensor Fusion Weights")
    st.caption(f"🔊 Acoustic Weight: **{WEIGHTS['audio']}**")
    st.caption(f"📷 Vision Weight: **{WEIGHTS['image']}**")
    st.caption(f"🐾 Seismic Weight: **{WEIGHTS['seismic']}**")
    st.caption(f"🗺️ Geo Risk Weight: **{WEIGHTS['geo']}**")
    st.caption(f"🚨 Alert Threshold: **{ALERT_THRESHOLD}**")

# ==========================================
# 4. MODE 1: LIVE CORRIDOR STREAM
# ==========================================
if mode == "📡 Live Corridor Stream":
    st.subheader("📡 Real-Time Multi-Sensor Telemetry Stream")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        start_sim = st.button("🚀 Initialize Early-Warning Stream", type="primary", use_container_width=True)
    with col_ctrl2:
        stop_sim = st.button("🛑 Terminate Stream Processing", use_container_width=True)

    if "sim_active" not in st.session_state:
        st.session_state.sim_active = False

    if start_sim:
        st.session_state.sim_active = True
    if stop_sim:
        st.session_state.sim_active = False

    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("#### 👁️ Thermal & Vision Stream (YOLOv8 Processing)")
        img_placeholder = st.empty()
        
        st.markdown("#### 🐾 Real-Time Seismic Ground Waveform Telemetry")
        seismic_chart_placeholder = st.empty()
    
    with col_right:
        st.markdown("#### 🚨 Multimodal Unified Threat Index")
        status_placeholder = st.empty()
        gauge_placeholder = st.empty()
        
        st.markdown("#### 📊 Sensor Confidences")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        audio_m = m_col1.empty()
        vision_m = m_col2.empty()
        seismic_m = m_col3.empty()
        geo_m = m_col4.empty()

        st.markdown("#### 🧮 Edge Fusion Breakdown")
        math_placeholder = st.empty()

    st.divider()
    st.markdown("#### 📜 Incident Broadcast Logs")
    logs_placeholder = st.empty()

    if "telemetry_logs" not in st.session_state:
        st.session_state.telemetry_logs = []

    if st.session_state.sim_active:
        yolo_model = YOLO("yolov8n.pt")
        settlements = ["Anuradhapura", "Vavuniya", "Habarana", "Polonnaruwa", "Trincomalee"]

        while st.session_state.sim_active:
            a_conf = round(random.uniform(0.20, 0.95), 2)
            v_conf = round(random.uniform(0.15, 0.90), 2)
            s_pga = round(random.uniform(0.05, 0.45), 3) # PGA in g
            s_freq = round(random.uniform(10.0, 22.0), 1) # Hz
            s_conf = round(min(s_pga / 0.35, 1.0), 2)     # Normalized seismic confidence
            
            cur_settlement = random.choice(settlements)
            g_risk = round(random.uniform(0.35, 0.85), 2)

            fused_score = fuse(a_conf, v_conf, s_conf, g_risk)

            # 1. Update Seismic Waveform
            df_wave = generate_seismic_waveform(s_pga, s_freq)
            seismic_chart_placeholder.line_chart(df_wave, x="Time (s)", y="Ground Acceleration (g)", height=160)

            # 2. Display Status
            if fused_score >= ALERT_THRESHOLD:
                status_placeholder.markdown('<div class="status-high">⚠️ HIGH THREAT DETECTED</div>', unsafe_allow_html=True)
                st.toast(f"🚨 ALERT: Elephant movement detected near {cur_settlement}! SMS Dispatched.", icon="🚨")
            elif fused_score >= 0.35:
                status_placeholder.markdown('<div class="status-med">⚡ ELEVATED MOVEMENT</div>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown('<div class="status-low">✅ NOMINAL / LOW RISK</div>', unsafe_allow_html=True)

            gauge_placeholder.progress(min(float(fused_score), 1.0), text=f"Fused Risk Score: {fused_score:.1%}")
            audio_m.metric("Acoustic", f"{a_conf:.0%}")
            vision_m.metric("Vision", f"{v_conf:.0%}")
            seismic_m.metric("Seismic", f"{s_pga:.2f}g")
            geo_m.metric("Geo Risk", f"{g_risk:.0%}")

            # 3. Dynamic LaTeX Math Calculation
            math_placeholder.markdown(f"""
            ```text
            Score = (0.35 × {a_conf}) + (0.30 × {v_conf}) + (0.20 × {s_conf}) + (0.15 × {g_risk})
                  = {fused_score:.2f} (Threshold: {ALERT_THRESHOLD})
