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

# Custom Glassmorphism & Background Styling
st.markdown(
    """
    <style>
    /* 1. App Background Image with Dark Gradient Overlay */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(10, 15, 26, 0.78), rgba(10, 15, 26, 0.88)), 
                    url("https://images.unsplash.com/photo-1516426122078-c23e76319801?q=80&w=1920&auto=format&fit=crop"); /* Wildlife/Jungle theme background */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* 2. Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.60) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* 3. Glass Green Buttons Styling */
    div.stButton > button {
        background: rgba(16, 185, 129, 0.18) !important; /* Semi-transparent emerald green */
        color: #a7f3d0 !important; /* Soft mint text */
        border: 1px solid rgba(52, 211, 153, 0.4) !important; /* Glowing glass border */
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
        font-weight: 600 !important;
        transition: all 0.3s ease-in-out !important;
    }

    /* Hover effect for Glass Green Buttons */
    div.stButton > button:hover {
        background: rgba(16, 185, 129, 0.38) !important;
        border-color: rgba(52, 211, 153, 0.8) !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(52, 211, 153, 0.5), inset 0 1px 2px rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-2px);
    }

    /* Active click state */
    div.stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 4px 10px rgba(52, 211, 153, 0.3) !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)
