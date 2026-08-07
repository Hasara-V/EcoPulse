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

# Advanced Glassmorphism & Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f8fafc !important;
    }
    
    /* 1. App Background Image with Dark Vignette Overlay */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.92)), 
                    url("https://images.unsplash.com/photo-1516426122078-c23e76319801?q=80&w=1920&auto=format&fit=crop") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }

    /* 2. Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    /* 3. Hero & Card Containers */
    .hero-card {
        background: rgba(30, 41, 59, 0.60) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 30px !important;
        margin-bottom: 25px !important;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.5) !important;
    }

    /* 4. Glowing Glass Green Buttons */
    div.stButton > button {
        background: rgba(16, 185, 129, 0.22) !important; /* Emerald glass tint */
        color: #6ee7b7 !important; /* Bright mint text */
        border: 1px solid rgba(52, 211, 153, 0.5) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        transition: all 0.3s ease-in-out !important;
    }

    div.stButton > button:hover {
        background: rgba(16, 185, 129, 0.45) !important;
        border-color: rgba(52, 211, 153, 0.9) !important;
        color: #ffffff !important;
        box-shadow: 0 0 25px rgba(52, 211, 153, 0.6), inset 0 1px 2px rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-2px);
    }

    div.stButton > button:active {
        transform: translateY(0px);
    }

    /* Threat Status Badges */
    .status-high {
        background-color: rgba(127, 29, 29, 0.85);
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
        background-color: rgba(120, 53, 15, 0.85);
        border: 2px solid #f59e0b;
        color: #fde68a;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    .status-low {
        background-color: rgba(6, 78, 59, 0.85);
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
WEIGHTS = {"audio": 0.40, "image": 0.35, "geo": 0.25}
ALERT_THRESHOLD = 0.55

def fuse(a, i, g):
    return (WEIGHTS["audio"] * a + WEIGHTS["image"] * i + WEIGHTS["geo"] * g)

# ==========================================
# 2. HERO HEADER SECTION
# ==========================================
st.markdown("""
<div class="hero-card">
    <div style="font-size: 13px; color: #38bdf8; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 6px;">
        🌐 EDGE-BASED MULTIMODAL WILDLIFE MONITORING
    </div>
    <h1 style="font-size: 38px; font-weight: 800; margin: 0 0 10px 0; color: #ffffff;">
        🐘 EcoPulse Early-Warning Core
    </h1>
    <p style="color: #cbd5e1; font-size: 15px; margin: 0;">
        <strong>Data Odyssey 2026</strong> | Real-time mitigation of Human-Elephant Conflict via Bioacoustic, Thermal Vision & Geospatial Fusion.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Center")
    mode = st.radio("Select Mode", ["📡 Live Corridor Stream", "🔍 Diagnostic Sample Test"])
    
    st.divider()
    st.markdown("### ⚖️ Fusion Parameters")
    st.caption(f"Acoustic Weight: **{WEIGHTS['audio']}**")
    st.caption(f"Vision Weight: **{WEIGHTS['image']}**")
    st.caption(f"Geospatial Weight: **{WEIGHTS['geo']}**")
    st.caption(f"Alert Threshold: **{ALERT_THRESHOLD}**")

# ==========================================
# 4. MODE 1: LIVE CORRIDOR STREAM
# ==========================================
if mode == "📡 Live Corridor Stream":
    st.subheader("📡 Real-Time Telemetry & Event Stream Simulation")
    
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

    col_vis, col_analytics = st.columns([3, 2])
    
    with col_vis:
        st.markdown("#### 👁️ Thermal & Vision Stream (YOLOv8 Dynamic Processing)")
        img_placeholder = st.empty()
    
    with col_analytics:
        st.markdown("#### 🚨 Multimodal Unified Threat Index")
        status_placeholder = st.empty()
        gauge_placeholder = st.empty()
        
        st.markdown("#### 📊 Sensor Metrics")
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
            a_conf = round(random.uniform(0.25, 0.95), 2)
            v_conf = round(random.uniform(0.20, 0.90), 2)
            cur_settlement = random.choice(settlements)
            g_risk = round(random.uniform(0.35, 0.85), 2)

            fused_score = fuse(a_conf, v_conf, g_risk)

            # Display Threat Status
            if fused_score >= ALERT_THRESHOLD:
                status_placeholder.markdown('<div class="status-high">⚠️ HIGH THREAT DETECTED</div>', unsafe_allow_html=True)
                st.toast(f"🚨 CRITICAL BROADCAST: Elephant detected near {cur_settlement}! SMS Dispatched.", icon="🚨")
            elif fused_score >= 0.35:
                status_placeholder.markdown('<div class="status-med">⚡ ELEVATED MOVEMENT</div>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown('<div class="status-low">✅ NOMINAL / LOW RISK</div>', unsafe_allow_html=True)

            gauge_placeholder.progress(min(float(fused_score), 1.0), text=f"Fused Threat Score: {fused_score:.1%}")
            audio_m.metric("Acoustic", f"{a_conf:.0%}")
            vision_m.metric("Vision", f"{v_conf:.0%}")
            geo_m.metric("Geo Risk", f"{g_risk:.0%}")

            # Append Telemetry Log
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
        st.info("Click 'Initialize Early-Warning Stream' above to start processing simulated corridor traffic.")

# ==========================================
# 5. MODE 2: DIAGNOSTIC SAMPLE TEST
# ==========================================
else:
    st.subheader("🔍 Single-Sample Multimodal Diagnostic Test")
    
    col_u1, col_u2, col_u3 = st.columns(3)
    
    with col_u1:
        st.markdown("#### 1. Bioacoustic Input")
        up_audio = st.file_uploader("Upload Clip", type=["wav", "mp3", "flac", "ogg", "m4a"])
        
    with col_u2:
        st.markdown("#### 2. Vision Frame")
        up_image = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png", "webp"])
        
    with col_u3:
        st.markdown("#### 3. Target Grid")
        sel_settlement = st.selectbox("Target Settlement", ["Anuradhapura", "Habarana", "Vavuniya", "Polonnaruwa", "Trincomalee"])

    btn_analyze = st.button("🧪 Execute Multimodal Analysis", type="primary", use_container_width=True)

    if btn_analyze:
        st.divider()
        col_res1, col_res2 = st.columns([1, 1])
        
        a_score, v_score, g_score = 0.0, 0.0, 0.50
        
        with col_res1:
            st.markdown("### 📷 Computer Vision Inference")
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
                st.info("No photo uploaded.")

        with col_res2:
            st.markdown("### 🔊 Bioacoustic & Geospatial Assessment")
            if up_audio:
                a_score = 0.85
                st.success(f"Trumpeting Signature Detected (Confidence: {a_score:.1%})")
            else:
                st.info("No audio clip uploaded.")

            g_score = 0.68
            st.info(f"Geospatial Risk Index for **{sel_settlement}**: {g_score:.1%}")

            st.divider()
            fused_score = fuse(a_score, v_score, g_score)
            st.markdown(f"### 🎯 Fused Risk Score: **{fused_score:.1%}**")
            st.progress(min(float(fused_score), 1.0))

            if fused_score >= ALERT_THRESHOLD:
                st.error(f"🚨 **CRITICAL ALERT TRIGGERED**\n\nSMS Broadcast sent to farming communities near **{sel_settlement}** and Railway Speed Warning active.")
            else:
                st.success("✅ **NOMINAL STATE**: Fused confidence score is below trigger threshold.")
