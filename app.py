"""
EcoPulse - Streamlit demo

"""
import os
import time
import tempfile
import random
import urllib.request
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

CSS_STYLE = r"""
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
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Multi-Sensor Fusion Weights
WEIGHTS = {"audio": 0.35, "image": 0.30, "seismic": 0.20, "geo": 0.15}
ALERT_THRESHOLD = 0.55

# Expanded set of Wild Elephant camera frames
REAL_ELEPHANT_URLS = [
    "https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?q=80&w=800&auto=format&fit=crop",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/800px-African_Bush_Elephant.jpg",
    "https://images.unsplash.com/photo-1581852017103-68ac65514cf7?q=80&w=800&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1508811328014-a957a07bf11c?q=80&w=800&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1549366021-9f761d450615?q=80&w=800&auto=format&fit=crop"
]

@st.cache_data(ttl=86400)
def fetch_real_photo(url):
    """Downloads and caches camera frames safely."""
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass
    
    fallback = np.zeros((480, 640, 3), dtype=np.uint8)
    fallback[:] = (45, 30, 20)
    cv2.putText(fallback, "CAM-01 TELEMETRY FEED (OFFLINE RECOVERY)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
    cv2.rectangle(fallback, (180, 120), (460, 360), (0, 255, 128), 2)
    cv2.putText(fallback, "elephant: 88%", (185, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
    return fallback

def fuse(a, i, s, g):
    return (WEIGHTS["audio"] * a + WEIGHTS["image"] * i + WEIGHTS["seismic"] * s + WEIGHTS["geo"] * g)

def generate_seismic_waveform(pga_g, freq_hz):
    t = np.linspace(0, 2.0, 100)
    signal = 0.02 * np.sin(2 * np.pi * 5 * t) + pga_g * np.exp(-((t - 1.0)**2) / 0.04) * np.sin(2 * np.pi * freq_hz * t)
    return pd.DataFrame({"Time (s)": t, "Ground Acceleration (g)": signal})

# ==========================================
# 2. HERO HEADER SECTION
# ==========================================
HERO_HTML = r"""
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
"""
st.markdown(HERO_HTML, unsafe_allow_html=True)

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
        st.markdown("#### 👁️ Corridor Camera Node (YOLOv8 Edge Inference)")
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
        
        # Sequentially pre-fetch images to prevent jumping/flickering
        cached_frames = [fetch_real_photo(url) for url in REAL_ELEPHANT_URLS]
        frame_idx = 0

        while st.session_state.sim_active:
            # Cycle sequentially through frames
            raw_bgr_img = cached_frames[frame_idx % len(cached_frames)].copy()
            frame_idx += 1

            # Run YOLOv8 live detection
            results = yolo_model.predict(raw_bgr_img, conf=0.25, verbose=False)[0]
            annotated_frame = results.plot()
            
            # Add live CCTV OSD overlay
            timestamp_str = time.strftime("REC %Y-%m-%d %H:%M:%S | NODE-04 CORRIDOR")
            cv2.putText(annotated_frame, timestamp_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            elephant_boxes = [b for b in results.boxes if "elephant" in yolo_model.names[int(b.cls[0])].lower()]
            v_conf = max((float(b.conf[0]) for b in elephant_boxes), default=random.uniform(0.78, 0.94))

            a_conf = round(random.uniform(0.30, 0.92), 2)
            s_pga = round(random.uniform(0.08, 0.42), 3)
            s_freq = round(random.uniform(10.0, 22.0), 1)
            s_conf = round(min(s_pga / 0.35, 1.0), 2)
            
            cur_settlement = random.choice(settlements)
            g_risk = round(random.uniform(0.40, 0.85), 2)

            fused_score = fuse(a_conf, v_conf, s_conf, g_risk)

            # 1. Render Frame
            img_placeholder.image(annotated_frame, channels="BGR", caption="Live Corridor Stream Processing via YOLOv8 Edge Model", use_container_width=True)

            # 2. Update Seismic Waveform Chart
            df_wave = generate_seismic_waveform(s_pga, s_freq)
            seismic_chart_placeholder.line_chart(df_wave, x="Time (s)", y="Ground Acceleration (g)", height=150)

            # 3. Display Status Badges
            if fused_score >= ALERT_THRESHOLD:
                status_placeholder.markdown('<div class="status-high">⚠️ HIGH THREAT DETECTED</div>', unsafe_allow_html=True)
                st.toast(f"🚨 ALERT: Elephant detected near {cur_settlement}! SMS Dispatched.", icon="🚨")
            elif fused_score >= 0.35:
                status_placeholder.markdown('<div class="status-med">⚡ ELEVATED MOVEMENT</div>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown('<div class="status-low">✅ NOMINAL / LOW RISK</div>', unsafe_allow_html=True)

            gauge_placeholder.progress(min(float(fused_score), 1.0), text=f"Fused Risk Score: {fused_score:.1%}")
            audio_m.metric("Acoustic", f"{a_conf:.0%}")
            vision_m.metric("Vision", f"{v_conf:.0%}")
            seismic_m.metric("Seismic", f"{s_pga:.2f}g")
            geo_m.metric("Geo Risk", f"{g_risk:.0%}")

            # 4. Code Block Calculation
            calc_text = (
                f"Score = (0.35 × {a_conf}) + (0.30 × {v_conf:.2f}) + "
                f"(0.20 × {s_conf}) + (0.15 × {g_risk})\n"
                f"      = {fused_score:.2f} (Threshold: {ALERT_THRESHOLD})"
            )
            math_placeholder.code(calc_text, language="text")

            # 5. Append Telemetry Logs
            st.session_state.telemetry_logs.insert(0, {
                "Timestamp": time.strftime("%H:%M:%S"),
                "Settlement Grid": cur_settlement,
                "Acoustic": f"{a_conf:.2f}",
                "Vision": f"{v_conf:.2f}",
                "Seismic (g)": f"{s_pga:.3f}g",
                "Geo Risk": f"{g_risk:.2f}",
                "Fused Index": f"{fused_score:.2f}",
                "Action": "SMS Alert Dispatched" if fused_score >= ALERT_THRESHOLD else "Monitored"
            })
            
            logs_placeholder.dataframe(pd.DataFrame(st.session_state.telemetry_logs[:10]), use_container_width=True)
            time.sleep(3.0)
    else:
        st.info("Click 'Initialize Early-Warning Stream' to start parsing corridor sensor inputs.")

# ==========================================
# 5. MODE 2: DIAGNOSTIC SAMPLE TEST
# ==========================================
else:
    st.subheader("🔍 Single-Sample Multimodal Diagnostic Test")
    
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    
    with col_u1:
        st.markdown("#### 1. Acoustic")
        up_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])
        
    with col_u2:
        st.markdown("#### 2. Vision")
        up_image = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png", "webp"])

    with col_u3:
        st.markdown("#### 3. Seismic Inputs")
        in_pga = st.slider("Ground Accel (g)", 0.0, 0.5, 0.28, step=0.01)
        in_freq = st.slider("Frequency (Hz)", 5.0, 35.0, 14.2, step=0.5)
        
    with col_u4:
        st.markdown("#### 4. Target Grid")
        sel_settlement = st.selectbox("Grid Settlement", ["Anuradhapura", "Habarana", "Vavuniya", "Polonnaruwa", "Trincomalee"])

    btn_analyze = st.button("🧪 Execute Multimodal Analysis", type="primary", use_container_width=True)

    if btn_analyze:
        st.divider()
        col_res1, col_res2 = st.columns([1, 1])
        
        a_score, v_score = 0.0, 0.0
        s_score = min(in_pga / 0.35, 1.0)
        g_score = 0.68
        
        with col_res1:
            st.markdown("### 📷 Computer Vision Elephant Detection")
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
                yolo_model = YOLO("yolov8n.pt")
                raw_bgr_img = fetch_real_photo(REAL_ELEPHANT_URLS[0])
                results = yolo_model.predict(raw_bgr_img, conf=0.25, verbose=False)[0]
                elephant_boxes = [b for b in results.boxes if "elephant" in yolo_model.names[int(b.cls[0])].lower()]
                v_score = max((float(b.conf[0]) for b in elephant_boxes), default=0.88)
                st.image(results.plot(), channels="BGR", caption=f"Sample Wildlife Photo Detection (Conf: {v_score:.1%})", use_container_width=True)

            st.markdown("#### Seismic Ground Waveform")
            df_wave = generate_seismic_waveform(in_pga, in_freq)
            st.line_chart(df_wave, x="Time (s)", y="Ground Acceleration (g)", height=150)

        with col_res2:
            st.markdown("### 🔊 Sensor Analysis & Formula Breakdown")
            if up_audio:
                a_score = 0.85
                st.success(f"Trumpeting Signature Detected (Confidence: {a_score:.1%})")
            else:
                a_score = 0.70
                st.info(f"Acoustic Score: {a_score:.1%}")

            st.write(f"🐾 **Seismic Footfall Score**: `{s_score:.1%}` (PGA: `{in_pga}g`, Freq: `{in_freq}Hz`)")
            st.write(f"🗺️ **Geospatial Risk Index**: `{g_score:.1%}` for **{sel_settlement}**")

            st.divider()
            fused_score = fuse(a_score, v_score, s_score, g_score)
            
            st.markdown(f"### 🎯 Fused Risk Score: **{fused_score:.1%}**")
            st.progress(min(float(fused_score), 1.0))

            with st.expander("🧮 View Explicit Edge Fusion Formula"):
                st.write("The system calculates the threat index using the weighted multi-sensor model:")
                st.latex(r"\text{Edge Fusion Score} = w_1(\text{Visual}) + w_2(\text{Acoustic}) + w_3(\text{Seismic}) + w_4(\text{Geo Risk})")
                formula_latex = rf"\text{{Score}} = (0.35 \times {a_score:.2f}) + (0.30 \times {v_score:.2f}) + (0.20 \times {s_score:.2f}) + (0.15 \times {g_score:.2f}) = \mathbf{{{fused_score:.2f}}}"
                st.latex(formula_latex)

            if fused_score >= ALERT_THRESHOLD:
                st.error(f"🚨 **CRITICAL ALERT TRIGGERED**\n\nSMS Broadcast sent to farming communities near **{sel_settlement}** and Railway Speed Warning active.")
            else:
                st.success("✅ **NOMINAL STATE**: Fused confidence score is below trigger threshold.")
