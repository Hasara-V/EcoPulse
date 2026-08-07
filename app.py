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
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap)');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f8fafc !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.92)), 
                    url("[https://images.unsplash.com/photo-1516426122078-c23e76319801?q=80&w=1920&auto=format&fit=crop](https://images.unsplash.com/photo-1516426122078-c23e76319801?q=80&w=1920&auto=format&fit=crop)") !important;
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
        text
