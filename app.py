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


# Custom Dark Glassmorphism Styling with Hero Banners & Accent Colors
st.markdown("""
    <style>
    /* Custom font & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Banner Styling */
    .hero-container {
        background: linear-gradient(rgba(15, 23, 42, 0.75), rgba(15, 23, 42, 0.95)), 
                    url('https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?auto=format&fit=crop&w=1200&q=80');
        background-size: cover;
        background-position: center;
        padding: 40px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        font-size: 16px;
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Feature Badge Cards */
    .badge-card {
        background: #1e293b;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-subtitle">🌐 EDGE-BASED MULTIMODAL WILDLIFE MONITORING</div>
    <div class="hero-title">🐘 EcoPulse Early-Warning Core</div>
    <p style="color: #cbd5e1; font-size: 15px;">
        Protecting rural communities & rail corridors through acoustic, thermal vision, and seismic AI telemetry.
    </p>
</div>
""", unsafe_allow_html=True)