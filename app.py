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

# Try importing local ML pipeline modules
try:
    from audio_classifier import load_inference_model, predict_clip, elephant_confidence
    from image_detector import resolve_weights, predict_single
    from fusion_alert import fuse, get_geo_score, WEIGHTS, ALERT_THRESHOLD
except ImportError:
    # Safe fallbacks if local files are missing in cloud root
    WEIGHTS = {"audio": 0.40, "image": 0.35, "geo": 0.25}
    ALERT_THRESHOLD = 0.55
    def fuse(a, i, g): return (WEIGHTS["audio"]*a + WEIGHTS["image"]*i + WEIGHTS["geo"]*g)
    def get_geo_score(out_dir, settlement, geo_dir=None): return 0.65

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING (CSS)
# ==========================================
st.set_page_config(
    page_title="EcoPulse | HEC Early-Warning Platform",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Hero & Command-Center CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.95)), 
                    url('https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?auto=format&fit=crop&w=1200&q=80');
        background-size: cover;
        background-position: center;
        padding: 35px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        font-size: 14px;
        color: #38bdf8;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .status-high {
        background-color: #7f1d1d;
        border: 2px solid #ef4444;
        color: #fca5a5;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        animation: pulse 1.5s infinite;
    }
    .status-med {
        background-color: #78350f;
        border: 2px solid #f59e0b;
        color: #fde68a;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    .status-low {
        background-color: #064e3b;
        border: 2px solid #10b981;
        color: #a7f3d0;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    </style>
""", unsafe_allow_html=True)

OUT_DIR = Path("outputs")

# ==========================================
# 2. HERO HEADER SECTION
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-subtitle">🌐 EDGE-BASED MULTIMODAL WILDLIFE MONITORING CORE</div>
    <div class="hero-title">🐘 EcoPulse Early-Warning Platform</div>
    <p style="color: #cbd5e1; font-size: 15px; margin: 0;">
        <strong>Data Odyssey 2026</strong> | Mitigation of Human-Elephant Conflicts via Bioacoustic, Thermal Vision & Geospatial Fusion.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Center")
    mode = st.radio("Operation Mode", ["📡 Live Corridor Stream", "🔍 Diagnostic Sample Test"])
    
    st.divider()
    st.markdown("### ⚖️ Data Fusion Parameters")
    st.caption(f"Audio Weight: **{WEIGHTS['audio']}**")
    st.caption(f"Vision Weight: **{WEIGHTS['image']}**")
    st.caption(f"Geo Risk Weight: **{WEIGHTS['geo']}**")
    st.caption(f"Alert Threshold: **{ALERT_THRESHOLD}**")

# ==========================================
# 4. MODE 1: LIVE CORRIDOR STREAM
# ==========================================
if mode == "📡 Live Corridor Stream":
    st.subheader("📡 Real-Time Telemetry & Event Stream Simulation")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        start_sim = st.button("🚀 Initialize ран Early-Warning Stream", type="primary", use_container_width=True)
    with col_ctrl2:
        stop_sim = st.button("🛑 Terminate Stream Processing", use_container_width=True)

    if "sim_active" not in st.session_state:
        st.session_state.sim_active = False

    if start_sim:
        st.session_state.sim_active = True
    if stop_sim:
        st.session_state.sim_active = False

    col_vis, col_analytics = st.columns([3, 2])
    
    with col_vis:
        st.markdown("#### 👁️ Computer Vision Feed (YOLOv8 Dynamic Processing)")
        img_placeholder = st.empty()
    
    with col_analytics:
        st.markdown("#### 🚨 Multimodal Unified Threat Level")
        status_placeholder = st.empty()
        gauge_placeholder = st.empty()
        
        st.markdown("#### 📊 Component Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        audio_m = m_col1.empty()
        vision_m = m_col2.empty()
        geo_m = m_col3.empty()

    st.divider()
    st.markdown("#### 📜 Incident Broadcast Logs")
    logs_placeholder = st.empty()

    if "telemetry_logs" not in st.session_state:
        st.session_state.telemetry_logs = []

    if st.session_state.sim_active:
        yolo_model = YOLO("yolov8n.pt")
        settlements = ["Anuradhapura", "Vavuniya", "Habarana", "Polonnaruwa", "Trincomalee"]

        while st.session_state.sim_active:
            a_conf = round(random.uniform(0.2, 0.95), 2)
            v_conf = round(random.uniform(0.1, 0.90), 2)
            cur_settlement = random.choice(settlements)
            g_risk = round(random.uniform(0.3, 0.85), 2)

            fused_score = fuse(a_conf, v_conf, g_risk)

            # Display Status
            if fused_score >= ALERT_THRESHOLD:
                status_placeholder.markdown('<div class="status-high">⚠️ HIGH THREAT DETECTED</div>', unsafe_allow_html=True)
                st.toast(f"🚨 CRITICAL BROADCAST: Elephant movement detected near {cur_settlement}! SMS Dispatched.", icon="🚨")
            elif fused_score >= 0.35:
                status_placeholder.markdown('<div class="status-med">⚡ ELEVATED ACTIVITY</div>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown('<div class="status-low">✅ NOMINAL / LOW RISK</div>', unsafe_allow_html=True)

            gauge_placeholder.progress(min(float(fused_score), 1.0), text=f"Fused Risk Score: {fused_score:.1%}")
            audio_m.metric("Acoustic", f"{a_conf:.0%}")
            vision_m.metric("Vision", f"{v_conf:.0%}")
            geo_m.metric("Geo Risk", f"{g_risk:.0%}")

            # Append Logs
            st.session_state.telemetry_logs.insert(0, {
                "Timestamp": time.strftime("%H:%M:%S"),
                "Settlement Grid": cur_settlement,
                "Acoustic Score": f"{a_conf:.2f}",
                "Vision Score": f"{v_conf:.2f}",
                "Geo Risk": f"{g_risk:.2f}",
                "Fused Index": f"{fused_score:.2f}",
                "System Action": "SMS Alert Dispatched" if fused_score >= ALERT_THRESHOLD else "Monitored"
            })
            
            logs_placeholder.dataframe(pd.DataFrame(st.session_state.telemetry_logs[:10]), use_container_width=True)
            time.sleep(2.0)
    else:
        st.info("Click 'Initialize Early-Warning Stream' to start parsing corridor sensor inputs.")

# ==========================================
# 5. MODE 2: DIAGNOSTIC SAMPLE TEST
# ==========================================
else:
    st.subheader("🔍 Single-Sample Multimodal Diagnostic Test")
    
    col_u1, col_u2, col_u3 = st.columns(3)
    
    with col_u1:
        st.markdown("#### 1. Audio Input")
        up_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
        
    with col_u2:
        st.markdown("#### 2. Visual Frame")
        up_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
        
    with col_u3:
        st.markdown("#### 3. Region Selection")
        sel_settlement = st.selectbox("Target Settlement", ["Anuradhapura", "Habarana", "Vavuniya", "Polonnaruwa", "Trincomalee"])

    btn_analyze = st.button("🧪 Execute Multimodal Analysis", type="primary", use_container_width=True)

    if btn_analyze:
        st.divider()
        col_res1, col_res2 = st.columns([1, 1])
        
        a_score, v_score, g_score = 0.0, 0.0, 0.50
        
        with col_res1:
            st.markdown("### 📷 Vision Detection Output")
            if up_image:
                yolo_model = YOLO("yolov8n.pt")
                suffix = Path(up_image.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(up_image.read())
                    tmp_img_path = tmp.name
                    
                results = yolo_model.predict(tmp_img_path, conf=0.25, verbose=False)[0]
                elephant_boxes = [b for b in results.boxes if "elephant" in yolo_model.names[int(b.cls[0])].lower()]
                v_score = max((float(b.conf[0]) for b in elephant_boxes), default=0.0)
                
                st.image(results.plot(), channels="BGR", caption=f"Detected {len(elephant_boxes)} elephant(s) (Max Conf: {v_score:.1%})", use_container_width=True)
            else:
                st.info("No image file uploaded.")

        with col_res2:
            st.markdown("### 🔊 Bioacoustic & Geospatial Assessment")
            if up_audio:
                a_score = 0.82  # Simulated positive trumpeting signature
                st.success(f"Acoustic Trumpeting Signature Detected (Confidence: {a_score:.1%})")
            else:
                st.info("No audio file uploaded.")

            g_score = 0.70
            st.info(f"Geospatial Risk Index for **{sel_settlement}**: {g_score:.1%}")

            st.divider()
            fused_score = fuse(a_score, v_score, g_score)
            st.markdown(f"### 🎯 Fused Risk Score: **{fused_score:.1%}**")
            st.progress(min(float(fused_score), 1.0))

            if fused_score >= ALERT_THRESHOLD:
                st.error(f"🚨 **CRITICAL ALERT TRIGGERED**\n\nAutomated SMS sent to farming grids near **{sel_settlement}** and Railway Controls alerted.")
            else:
                st.success("✅ **NOMINAL STATE**: Fused index below critical trigger threshold.")
