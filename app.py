import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import time
import uuid
import os
import gdown
import requests
from io import BytesIO

import os
import streamlit as st
import gdown
from ultralytics import YOLO

# Optional Folium import with fallback
import os
import streamlit as st
import gdown
import numpy as np
from ultralytics import YOLO

# Optional Folium import with fallback
try:
    from streamlit_folium import st_folium
    import folium
    FOLIUM_AVAILABLE = True
except ModuleNotFoundError:
    FOLIUM_AVAILABLE = False

MODEL_PATH = 'weights/last.pt'  # Path relative to your app root folder

def download_weights():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = 'https://drive.google.com/uc?id=18Kh8T9GdwMBEOw6DvNR3BHLrhGqioPWs'
        st.info("Downloading YOLO weights. Please wait...")
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Download completed!")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"YOLO weights file missing at {MODEL_PATH}. Please add it or check your download.")
        st.stop()
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

# Call download_weights before loading the model to ensure weights exist
download_weights()

# Initialize model in session_state once per session
if 'model' not in st.session_state:
    st.session_state.model = load_model()

def run_inference(image_pil):
    img_np = np.array(image_pil.convert("RGB"))
    results = st.session_state.model.predict(img_np, conf=0.1, verbose=False)
    return results





# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- CONSTANTS for Carbon Estimation ---
DEFAULT_GSD_METERS = 10  # Default GSD for satellite images (e.g., Sentinel-2 is 10m)
CARBON_TONNES_PER_HECTARE = 100 # Average estimate for forest carbon (tC/ha) - needs proper citation/disclaimer
ANNUAL_CARBON_SEQUESTRATION_RATE_TC_PER_HA_YR = 5 # Average annual sequestration rate (tC/ha/year)
CARBON_PRICE_USD_PER_TONNE = 50 # Illustrative price for carbon credits ($/tonne of CO2 equivalent)
USD_TO_INR_RATE = 83.5 # Approximate as of mid-2024, subject to change

# Conversion from Carbon (C) to Carbon Dioxide (CO2) for equivalencies
# C + O2 -> CO2 (Atomic weights: C=12, O=16, CO2=44)
# 1 tonne of Carbon (C) = 44/12 = 3.666... tonnes of CO2
TONNE_CARBON_TO_CO2 = 44/12

# Carbon Equivalencies (approximate values for CO2)
# Source: EPA Greenhouse Gas Equivalencies Calculator (adapted for per tonne CO2)
MILES_PER_TONNE_CO2 = 2500 # Approx. miles driven by average passenger car per tonne CO2
HOUSEHOLDS_PER_TONNE_CO2_ELECTRICITY = 1/7.5 # (1 / avg household annual CO2 from electricity in US)

# --- ULTRA SMOOTH CSS & JS (ADJUSTED: Removed text glow, changed background image, UPDATED METRIC LABEL AND VALUE COLOR) ---
st.markdown("""
<style>
/* Enhanced Base Styles with Modern Typography */
* { 
    box-sizing: border-box; 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --neon-gradient: linear-gradient(45deg, #ff006e, #fb5607, #ffbe0b, #8338ec, #3a86ff);
    --glass-bg: rgba(255, 255, 255, 0.08);
    --glass-border: rgba(255, 255, 255, 0.18);
    --dark-glass: rgba(0, 0, 0, 0.15);
    --text-primary: #ffffff;
    --text-secondary: #1E1717;
    --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
    --shadow-heavy: 0 20px 50px rgba(0, 0, 0, 0.5);
}

.stApp {
    background-image: url('https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?q=80&w=2074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
    background-size: 400% 400%;
    animation: gradientShift 120s ease infinite;
    min-height: 100vh;
    color: var(--text-primary);
    overflow-x: hidden;
    padding: 2rem;
    position: relative;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating Particles Background */
.background-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
}

.background-overlay::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%);
    animation: floatingGlow 10s ease-in-out infinite alternate;
}

@keyframes floatingGlow {
    0% { transform: scale(1) rotate(0deg); opacity: 0.7; }
    100% { transform: scale(1.1) rotate(5deg); opacity: 0.9; }
}

/* Enhanced Glassmorphism Container */
.center-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 3rem;
    background: var(--glass-bg);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border-radius: 30px;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-light);
    position: relative;
    overflow: hidden;
}

.center-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
}

/* Modern Typography with Variable Font Support */
h1, .stMarkdown h1 {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: clamp(3rem, 8vw, 6rem);
    font-weight: 800;
    text-align: center;
    background: var(--neon-gradient);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2rem;
    position: relative;
    line-height: 1.1;
    animation: gradientText 3s ease infinite;
    letter-spacing: -0.02em;
}

@keyframes gradientText {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Subtitle with enhanced typography */
p.subtitle, p.feature-desc {
    font-size: clamp(1.2rem, 3vw, 1.8rem);
    color: var(--text-secondary);
    text-align: center;
    margin-bottom: 3rem;
    font-weight: 400;
    line-height: 1.6;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

/* Enhanced Button Styles with Micro-interactions */
.stButton > button {
    background: var(--accent-gradient);
    color: #ffffff;
    font-size: clamp(1.4rem, 4vw, 2.2rem);
    font-weight: 600;
    padding: 1.5rem 3rem;
    border: none;
    border-radius: 20px;
    width: 100%;
    min-height: 120px;
    margin: 1rem auto;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    box-shadow: 0 10px 40px rgba(79, 172, 254, 0.3);
    transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 60px rgba(79, 172, 254, 0.4);
    background: var(--secondary-gradient);
}

.stButton > button:active {
    transform: translateY(-4px) scale(0.98);
    transition: all 0.1s ease;
}

/* Modern File Uploader with Neumorphism */
section[data-testid="stFileUploader"] > div {
    background: var(--glass-bg);
    border: 2px dashed rgba(79, 172, 254, 0.5);
    border-radius: 25px;
    padding: 3.5rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
    backdrop-filter: blur(15px);
    position: relative;
    overflow: hidden;
}

section[data-testid="stFileUploader"] > div::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent, rgba(79, 172, 254, 0.1), transparent);
    animation: rotate 4s linear infinite;
    z-index: -1;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

section[data-testid="stFileUploader"] > div:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(255, 119, 198, 0.7);
    transform: scale(1.02);
    box-shadow: 0 15px 50px rgba(79, 172, 254, 0.2);
}

/* Enhanced Opaque Box with Neumorphism */
.opaque-box {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
    backdrop-filter: blur(20px);
    padding: 2.5rem;
    border-radius: 25px;
    margin: 2rem 0;
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.opaque-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
}

.opaque-box.disclaimer {
    font-size: 1rem;
    font-weight: 400;
    color: var(--text-secondary);
    background: rgba(0, 0, 0, 0.3);
    padding: 1.5rem 2rem;
    margin-top: 1rem;
    text-align: left;
    border-radius: 15px;
}

/* Enhanced Image Styles with Hover Effects */
.stImage img {
    border-radius: 25px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stImage img:hover {
    transform: scale(1.05) rotateY(5deg);
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.4);
}

/* Column Spacing */
.stColumns > div {
    padding: 0.75rem;
}

/* Enhanced Number Input */
.stNumberInput > div > label {
    font-size: 1.3rem;
    color: rgba(79, 172, 254, 0.9);
    margin-bottom: 0.8rem;
    font-weight: 500;
}

.stNumberInput > div > div > input {
    background: var(--glass-bg);
    border: 2px solid rgba(79, 172, 254, 0.3);
    border-radius: 15px;
    color: var(--text-primary);
    padding: 1.2rem;
    font-size: 1.3rem;
    transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1);
    backdrop-filter: blur(10px);
}

.stNumberInput > div > div > input:focus {
    border-color: rgba(79, 172, 254, 0.8);
    box-shadow: 0 0 30px rgba(79, 172, 254, 0.3);
    outline: none;
}

/* Modern Plotly Graph */
.plotly-graph-div {
    border-radius: 25px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Navigation Buttons Container */
.nav-buttons {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2rem;
    margin: 3rem 0;
}

.stColumns {
    width: 100%;
    max-width: 1200px;
}

/* Enhanced Progress Bar */
.stProgress > div > div {
    background: var(--accent-gradient);
    border-radius: 15px;
    height: 12px;
    position: relative;
    overflow: hidden;
}

.stProgress > div > div::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.stProgress > div {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

/* Modern Loading Spinner */
.loading-spinner {
    width: 60px;
    height: 60px;
    border: 4px solid rgba(79, 172, 254, 0.3);
    border-top: 4px solid #4facfe;
    border-radius: 50%;
    animation: modernSpin 1s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
    margin: 2rem auto;
    position: relative;
}

.loading-spinner::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 20px;
    height: 20px;
    background: var(--accent-gradient);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: pulse 1s ease-in-out infinite alternate;
}

@keyframes modernSpin {
    0% { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(180deg) scale(1.1); }
    100% { transform: rotate(360deg) scale(1); }
}

@keyframes pulse {
    0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.7; }
    100% { transform: translate(-50%, -50%) scale(1.2); opacity: 1; }
}

/* Enhanced Metric Styles */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent-gradient);
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"] {
    font-size: 3rem;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
}

[data-testid="stMetricDelta"] {
    font-size: 1.2rem;
    color: #ff9900;
    font-weight: 600;
}

/* Responsive Design Enhancements */
@media (max-width: 768px) {
    .center-container {
        padding: 2rem;
        margin: 1rem;
    }
    
    .stButton > button {
        min-height: 80px;
        font-size: 1.4rem;
        padding: 1rem 2rem;
    }
    
    h1, .stMarkdown h1 {
        font-size: clamp(2rem, 10vw, 4rem);
    }
}

/* Smooth Scrolling */
html {
    scroll-behavior: smooth;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--accent-gradient);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-gradient);
}
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js"></script>

<script>
document.addEventListener("DOMContentLoaded", function() {
    gsap.registerPlugin(ScrollTrigger);

    // Enhanced entrance animations with modern easing
    const tl = gsap.timeline();
    
    tl.from(".center-container", {
        duration: 1.2,
        y: 100,
        opacity: 0,
        scale: 0.95,
        ease: "power4.out",
        rotationX: 10
    })
    .from(".center-container h1", {
        y: 50,
        opacity: 0,
        duration: 1.5,
        ease: "elastic.out(1, 0.3)",
        stagger: 0.1
    }, "-=0.8")
    .from(".center-container p.subtitle", {
        y: 30,
        opacity: 0,
        duration: 1.2,
        ease: "power3.out"
    }, "-=1.0");

    // Staggered button animations
    gsap.from(".stButton > button", {
        duration: 0.8,
        scale: 0.8,
        opacity: 0,
        y: 30,
        stagger: 0.2,
        ease: "back.out(1.7)",
        delay: 1.5
    });

    // Floating animation for metrics
    gsap.to("[data-testid='stMetric']", {
        y: -10,
        duration: 3,
        ease: "power2.inOut",
        yoyo: true,
        repeat: -1,
        stagger: 0.5
    });

    // Parallax effect for background elements
    gsap.to(".background-overlay::before", {
        yPercent: -50,
        ease: "none",
        scrollTrigger: {
            trigger: "body",
            start: "top top",
            end: "bottom top",
            scrub: true
        }
    });

    // Advanced scroll-triggered animations
    gsap.utils.toArray(".opaque-box, .stImage, [data-testid='stMetric']").forEach(element => {
        gsap.fromTo(element, 
            {
                opacity: 0,
                y: 50,
                scale: 0.9
            }, 
            {
                opacity: 1,
                y: 0,
                scale: 1,
                duration: 1,
                ease: "power3.out",
                scrollTrigger: {
                    trigger: element,
                    start: "top 85%",
                    end: "bottom 15%",
                    toggleActions: "play none none reverse"
                }
            }
        );
    });

    // Micro-interactions for buttons
    document.querySelectorAll('.stButton > button').forEach(button => {
        button.addEventListener('mouseenter', () => {
            gsap.to(button, {
                scale: 1.05,
                duration: 0.3,
                ease: "power2.out"
            });
        });
        
        button.addEventListener('mouseleave', () => {
            gsap.to(button, {
                scale: 1,
                duration: 0.3,
                ease: "power2.out"
            });
        });
    });
});
</script>
""", unsafe_allow_html=True)

# SVG icon for loading animations (Unchanged)
st.markdown("""
<svg class="animated-icon" width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2V6M12 18V22M4.93 4.93L7.76 7.76M16.24 16.24L19.07 19.07M2 12H6M18 12H22M4.93 19.07L7.76 16.24M16.24 7.76L19.07 4.93" stroke="#00ffcc" stroke-width="2"/>
</svg>
""", unsafe_allow_html=True)

# Function to generate LaTeX report (no change)
def generate_latex_report(feature, data):
    latex_content = r"""
\documentclass[a4paper,12pt]{article}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{parskip}
\geometry{margin=1in}
\color{white} % Ensures text is white on a dark background in the PDF if compiled with specific packages
\begin{document}
\begin{center}
    \textbf{\LARGE Forest Analysis Report} \\
    \vspace{0.5cm}
    \textbf{Feature: """ + feature + r"""} \\
    \vspace{0.5cm}
    Generated on \today
\end{center}
\section*{Analysis Results}
""" + data + r"""
\end{document}
"""
    return latex_content

# Function to get mask from YOLO result (MODIFIED: @st.cache_data removed)
# @st.cache_data # REMOVED: Temporarily removed to debug mask identical issue in deforestation
def get_mask_from_result(_result, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if _result.masks is not None and _result.masks.xyn is not None:
        for poly in _result.masks.xyn:
            if poly.size > 0:
                pts = np.array(poly * [width, height], dtype=np.int32)
                if pts.ndim == 2:
                    pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
    return mask

# Function to validate image size (no change)
def validate_image_size(file, max_size_mb=10):
    if file.size > max_size_mb * 1024 * 1024:
        st.error(f"File size exceeds {max_size_mb}MB limit.")
        return False
    return True

# --- Page Content Logic (home page - MODIFIED FOR 3-2-1 BUTTON LAYOUT) ---
if st.session_state.page == 'home':
    st.markdown(f"""
    <div class="center-container">
        <h1>🌲 Forest Analysis Dashboard</h1>
        <p class="subtitle">Discover advanced AI-powered tools to analyze tree coverage, monitor deforestation, and visualize forest health with precision.</p>
    </div>
    <div class="nav-buttons">
    """, unsafe_allow_html=True)

    # First row of 3 buttons
    col1, col2, col3 = st.columns(3, gap="large") # Use 3 equal columns with large gap
    with col1:
        if st.button("Tree Coverage %", key="tree_coverage_btn"):
            st.session_state.page = 'tree_coverage'
            st.rerun()
    with col2:
        if st.button("Deforestation Analysis", key="deforestation_btn"):
            st.session_state.page = 'deforestation'
            st.rerun()
    with col3:
        if st.button("Tree Enumeration", key="tree_enumeration_btn"):
            st.session_state.page = 'tree_enumeration'
            st.rerun()

    # Second row of 3 slots (2 buttons + 1 new button)
    col4, col5, col6 = st.columns(3, gap="large") # Use 3 equal columns with large gap
    with col4:
        if st.button("Tree Heatmap", key="tree_heatmap_btn"):
            st.session_state.page = 'tree_heatmap'
            st.rerun()
    with col5:
        if st.button("Coordinate Analysis", key="coordinate_analysis_btn"):
            st.session_state.page = 'coordinate_analysis'
            st.rerun()
    with col6: # New button for Carbon Estimation
        if st.button("Carbon Estimation", key="carbon_estimation_btn"):
            st.session_state.page = 'carbon_estimation'
            st.rerun()


    st.markdown('</div>', unsafe_allow_html=True)


# Feature: Tree Coverage % (no significant change needed here)
elif st.session_state.page == 'tree_coverage':
    st.markdown("""
    <div class="center-container">
        <h1>🌳 Tree Coverage Analysis</h1>
        <p class="feature-desc">Upload a satellite or aerial image to estimate the percentage of tree coverage using advanced AI segmentation.</p>
    </div>
    """, unsafe_allow_html=True)
    file = st.file_uploader("Upload Satellite or Aerial Image", type=["jpg", "jpeg", "png"], key="tree_coverage_uploader")
    if file:
        if validate_image_size(file):
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            try:
                image = Image.open(file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)
                img_np = np.array(image)
                height, width = img_np.shape[:2]

                st.write("Running AI model for tree segmentation...")
                results = model.predict(img_np, conf=0.1, verbose=False)
                result = results[0]

                if result.masks is None or not result.masks.xyn:
                    st.warning("No tree masks were detected in the uploaded image. Tree coverage will be 0%.")
                    coverage = 0.0
                    binary_mask = np.zeros((height, width), dtype=np.uint8)
                else:
                    binary_mask = get_mask_from_result(result, width, height)
                    tree_pixels = np.count_nonzero(binary_mask == 1)
                    total_pixels = height * width
                    coverage = (tree_pixels / total_pixels) * 100
                    coverage = min(coverage, 100.0)

                st.markdown(f"<div class='opaque-box'>🌳 Estimated Tree Coverage: {coverage:.2f}%</div>", unsafe_allow_html=True)
                st.image(binary_mask * 255, caption="Tree Mask (Black & White)", use_container_width=True)

                overlay_img = img_np.copy()
                overlay_img[binary_mask == 1] = [0, 255, 0]
                st.image(overlay_img, caption="Detected Tree Regions (Green Overlay)", use_container_width=True)

                report_data = f"Estimated Tree Coverage: {coverage:.2f}\\%"
                latex_content = generate_latex_report("Tree Coverage Analysis", report_data)
                report_id = str(uuid.uuid4())
                st.download_button(
                    label="Download Report",
                    data=latex_content,
                    file_name=f"tree_coverage_report_{report_id}.tex",
                    mime="text/latex",
                    key="download_tree_coverage"
                )
            except Exception as e:
                st.error(f"Error processing image for tree coverage: {str(e)}")
                st.exception(e)
    if st.button("Back to Home", key="back_tree_coverage"):
        st.session_state.page = 'home'
        st.rerun()



# Feature: Tree Heatmap (no significant change needed here)
elif st.session_state.page == 'tree_heatmap':
    st.markdown("""
    <div class="center-container">
        <h1>🌿 Tree Density Heatmap</h1>
        <p class="feature-desc">Visualize tree density across a satellite image with a heatmap to identify high-density areas.</p>
    </div>
    """, unsafe_allow_html=True)
    file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"], key="tree_heatmap_uploader")
    if file:
        if validate_image_size(file):
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            try:
                image = Image.open(file).convert("RGB")
                img_np = np.array(image)
                h, w = img_np.shape[:2]

                st.image(img_np, caption="Uploaded Image", use_container_width=True)
                st.write("Running AI model to generate heatmap...")

                results = model.predict(img_np, conf=0.1, verbose=False)
                result = results[0]

                if result.masks is None or not result.masks.xyn:
                    st.warning("No tree masks were detected. Heatmap will be empty.")
                    heatmap = np.zeros((h // 20, w // 20))
                else:
                    binary_mask = get_mask_from_result(result, w, h)
                    grid_size = 20
                    
                    if h < grid_size or w < grid_size:
                        st.error(f"Image too small ({w}x{h}) for a grid size of {grid_size}. Please upload a larger image or reduce grid size.")
                        st.stop()

                    heatmap = np.zeros((h // grid_size, w // grid_size))
                    for y in range(0, h, grid_size):
                        for x in range(0, w, grid_size):
                            y_end = min(y + grid_size, h)
                            x_end = min(x + grid_size, w)
                            patch = binary_mask[y:y_end, x:x_end]

                            y_idx = y // grid_size
                            x_idx = x // grid_size

                            if y_idx < heatmap.shape[0] and x_idx < heatmap.shape[1]:
                                heatmap[y_idx, x_idx] = np.count_nonzero(patch)
                
                st.subheader("📊 Tree Density Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(heatmap, cmap="YlGn", cbar=True, ax=ax)
                ax.set_title("Tree Density Heatmap")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error processing image for tree heatmap: {str(e)}")
                st.exception(e)
    if st.button("Back to Home", key="back_tree_heatmap"):
        st.session_state.page = 'home'
        st.rerun()

# --- Deforestation Analysis Page (no change) ---
elif st.session_state.page == 'deforestation':
    st.markdown("""
    <div class="center-container">
        <h1>📉 Deforestation Analysis</h1>
        <p class="feature-desc">Compare two satellite images from different time periods to identify and quantify changes in forest cover, indicating deforestation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Upload Images for Comparison")
    col_before, col_after = st.columns(2)

    image_before = None
    image_after = None

    with col_before:
        st.markdown("*Image from Earlier Date (e.g., 2020)*")
        file_before = st.file_uploader("Upload 'Before' Image", type=["jpg", "jpeg", "png"], key="deforestation_uploader_before")
        if file_before:
            if validate_image_size(file_before):
                image_before = Image.open(file_before).convert("RGB")
                st.image(image_before, caption="Before Image", use_container_width=True)

    with col_after:
        st.markdown("*Image from Later Date (e.g., 2024)*")
        file_after = st.file_uploader("Upload 'After' Image", type=["jpg", "jpeg", "png"], key="deforestation_uploader_after")
        if file_after:
            if validate_image_size(file_after):
                image_after = Image.open(file_after).convert("RGB")
                st.image(image_after, caption="After Image", use_container_width=True)

    if st.button("Analyze Deforestation", key="analyze_deforestation_btn"):
        if image_before and image_after:
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            try:
                # Ensure images have the same dimensions for comparison
                if image_before.size != image_after.size:
                    st.warning(f"Image dimensions do not match: Before {image_before.size}, After {image_after.size}. Resizing 'After' image to match 'Before' image for consistent comparison.")
                    image_after = image_after.resize(image_before.size, Image.Resampling.LANCZOS) # Use LANCZOS for quality resizing

                img_np_before = np.array(image_before)
                img_np_after = np.array(image_after) # Recreate numpy array from potentially resized PIL image

                h, w = img_np_before.shape[:2]

                st.write("Running AI model for 'Before' image...")
                results_before = model.predict(img_np_before, conf=0.1, verbose=False)
                
                st.write("Running AI model for 'After' image...")
                results_after = model.predict(img_np_after, conf=0.1, verbose=False)
                
                mask_before = get_mask_from_result(results_before[0], w, h) if results_before[0].masks and results_before[0].masks.xyn else np.zeros((h, w), dtype=np.uint8)
                mask_after = get_mask_from_result(results_after[0], w, h) if results_after[0].masks and results_after[0].masks.xyn else np.zeros((h, w), dtype=np.uint8)

                total_pixels = h * w
                cov_before = (mask_before.sum() / total_pixels) * 100 if total_pixels > 0 else 0
                cov_after = (mask_after.sum() / total_pixels) * 100 if total_pixels > 0 else 0
                delta = cov_after - cov_before

                st.markdown(f"<div class='opaque-box'>🌳 Tree Coverage (Before): {cov_before:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='opaque-box'>🌳 Tree Coverage (After): {cov_after:.2f}%</div>", unsafe_allow_html=True)
                
                st.subheader("📊 Tree Coverage Comparison")
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Coverage Before", f"{cov_before:.2f}%")
                col_m2.metric("Coverage After", f"{cov_after:.2f}%", delta=f"{delta:.2f}%")

                st.subheader("🖼 Tree Masks")
                st.image([mask_before * 255, mask_after * 255], caption=["Before Mask", "After Mask"], use_column_width=True)

                # --- Visualizing Lost Trees ---
                # Calculate lost_trees: where mask_before had a tree (1) and mask_after does not (0)
                lost_trees = ((mask_before == 1) & (mask_after == 0))
                
                # Create a copy of the after image for the overlay.
                overlay_visual = img_np_after.copy()
                
                # Ensure overlay_visual is a 3-channel (RGB) uint8 image for proper color application
                if overlay_visual.ndim == 2: # If grayscale, convert to BGR then RGB
                    overlay_visual = cv2.cvtColor(overlay_visual, cv2.COLOR_GRAY2BGR)
                elif overlay_visual.ndim == 3 and overlay_visual.shape[2] == 4: # If RGBA, convert to RGB
                    overlay_visual = cv2.cvtColor(overlay_visual, cv2.COLOR_RGBA2RGB)
                
                # Ensure it's uint8
                if overlay_visual.dtype != np.uint8:
                    overlay_visual = overlay_visual.astype(np.uint8)

                # Apply the red overlay if lost trees are detected
                if np.any(lost_trees):
                    # Ensure the dimensions match for boolean indexing
                    if lost_trees.shape == overlay_visual.shape[:2]:
                        # Apply red color where trees were lost
                        overlay_visual[lost_trees] = [255, 0, 0] # Red (RGB)
                    else:
                        st.error(f"Error: Shape mismatch for lost trees overlay. Mask shape: {lost_trees.shape}, Image shape: {overlay_visual.shape}")
                else:
                    st.info("No significant tree loss detected to visualize in red. This could mean no actual deforestation, or the model didn't detect changes.")

                st.subheader("🔥 Tree Loss Visualized (Red = Lost Trees)")
                st.image(overlay_visual, use_column_width=True, channels="RGB") # Display the modified image

                # Optional: Visualize gained trees (no change)
                gained_trees = ((mask_before == 0) & (mask_after == 1))
                if np.any(gained_trees):
                    overlay_gained = img_np_after.copy()
                    if overlay_gained.ndim == 2:
                        overlay_gained = cv2.cvtColor(overlay_gained, cv2.COLOR_GRAY2BGR)
                    elif overlay_gained.ndim == 3 and overlay_gained.shape[2] == 4:
                        overlay_gained = cv2.cvtColor(overlay_gained, cv2.COLOR_RGBA2RGB)
                    if overlay_gained.shape[2] == 3:
                        overlay_gained[gained_trees] = [0, 0, 255] # Blue highlights tree gain
                        st.subheader("🌲 Tree Gain Visualized (Blue = Gained Trees)")
                        st.image(overlay_gained, use_column_width=True, channels="RGB")
                else:
                    st.info("No significant tree gain detected.")

                report_data = f"""
Initial Tree Coverage: {cov_before:.2f}\\%
Final Tree Coverage: {cov_after:.2f}\\%
Change in Coverage: {delta:.2f}\\%
                """
                latex_content = generate_latex_report("Deforestation Analysis", report_data)
                report_id = str(uuid.uuid4())
                st.download_button(
                    label="Download Report",
                    data=latex_content,
                    file_name=f"deforestation_report_{report_id}.tex",
                    mime="text/latex",
                    key="download_deforestation"
                )
            except Exception as e:
                st.error(f"Error processing images for deforestation: {str(e)}")
                st.exception(e)
        else:
            st.info("Please upload both 'Before' and 'After' images to perform deforestation analysis.")

    if st.button("Back to Home", key="back_deforestation"):
        st.session_state.page = 'home' # Corrected typo
        st.rerun()

# --- Coordinate Analysis Page (API Key Hardcoded - no change from last version) ---
elif st.session_state.page == 'coordinate_analysis': # Corrected typo
    st.markdown("""
    <div class="center-container">
        <h1>🗺 Coordinate Analysis</h1>
        <p class="feature-desc">Input geographic coordinates to fetch a satellite image via Google Maps Static API and perform various forest analyses.</p>
    </div>
    """, unsafe_allow_html=True)

    lat = st.number_input("Enter Latitude", format="%.6f", value=12.9716) # Default values for convenience
    lon = st.number_input("Enter Longitude", format="%.6f", value=77.5946) # Default values for convenience

    analysis_type_coord = st.selectbox(
        "Choose Analysis Type for Coordinates:",
        ("Tree Coverage %", "Tree Density Heatmap")
    )

    # --- API Key Hardcoded ---
    api_key = "AIzaSyBCqeySE_Ls5f3pT-6Z0PvP1X-UNW6okfI"
    # --- End of API Key Hardcoded ---

    if st.button("Get Satellite Image and Analyze", key="get_satellite_image_btn"):
        if lat and lon and api_key:
            st.info("📡 Fetching satellite image from Google Maps...")
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            image_url = (
                f"https://maps.googleapis.com/maps/api/staticmap?"
                f"center={lat},{lon}&zoom=17&size=640x640&maptype=satellite&key={api_key}"
            )
            
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    img_np = np.array(image)
                    height, width = img_np.shape[:2]

                    st.image(image, caption=f"Satellite View of ({lat:.4f}, {lon:.4f})", use_container_width=True)

                    st.write(f"Running AI model for {analysis_type_coord}...")
                    results = st.session_state.model.predict(img_np, conf=0.1, verbose=False)

                    result = results[0]
                    
                    if result.masks is not None and result.masks.xyn:
                        binary_mask = get_mask_from_result(result, width, height)

                        if analysis_type_coord == "Tree Coverage %":
                            tree_pixels = np.count_nonzero(binary_mask)
                            total_pixels = height * width
                            coverage = min((tree_pixels / total_pixels) * 100, 100.0)
                            st.markdown(f"<div class='opaque-box'>🌿 Tree Coverage: {coverage:.2f}%</div>", unsafe_allow_html=True)
                            st.image(binary_mask * 255, caption="Binary Tree Mask", use_column_width=True)

                            overlay_img = img_np.copy()
                            overlay_img[binary_mask == 1] = [0, 255, 0]
                            st.image(overlay_img, caption="Detected Tree Regions (Green Overlay)", use_column_width=True)
                            
                            report_data = f"Tree Coverage at ({lat:.4f}, {lon:.4f}): {coverage:.2f}\\%"
                            latex_content = generate_latex_report("Coordinate Tree Coverage", report_data)
                            report_id = str(uuid.uuid4())
                            st.download_button(
                                label="Download Report",
                                data=latex_content,
                                file_name=f"coord_tree_coverage_report_{report_id}.tex",
                                mime="text/latex",
                                key="download_coord_coverage"
                            )

                        elif analysis_type_coord == "Tree Density Heatmap":
                            h_coord, w_coord = img_np.shape[:2]
                            grid_size = 20
                            
                            if h_coord < grid_size or w_coord < grid_size:
                                st.error(f"Image too small ({w_coord}x{h_coord}) for a grid size of {grid_size}. Please upload a larger image or reduce grid size.")
                                st.stop()

                            heatmap = np.zeros((h_coord // grid_size, w_coord // grid_size))
                            for y in range(0, h_coord, grid_size):
                                for x in range(0, w_coord, grid_size):
                                    y_end = min(y + grid_size, h_coord)
                                    x_end = min(x + grid_size, w_coord)
                                    patch = binary_mask[y:y_end, x:x_end]
                                    
                                    y_idx = y // grid_size
                                    x_idx = x // grid_size

                                    if y_idx < heatmap.shape[0] and x_idx < heatmap.shape[1]:
                                        heatmap[y_idx, x_idx] = np.count_nonzero(patch)

                            st.subheader("📊 Tree Density Heatmap")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(heatmap, cmap="YlGn", cbar=True, ax=ax)
                            ax.set_title("Tree Density Heatmap")
                            st.pyplot(fig)
                    else:
                        st.warning("No tree masks were detected for the given coordinates. Results will be minimal.")

                elif response.status_code == 400:
                    st.error(f"Error fetching image: Invalid API key or request parameters. Response: {response.text}")
                else:
                    st.error(f"Failed to fetch satellite image. Status code: {response.status_code}. Response: {response.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please check your internet connection or try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)
        else:
            st.info("Please enter Latitude, Longitude, and ensure API Key is provided to get the satellite image.")

    if st.button("Back to Home", key="back_coord_analysis"):
        st.session_state.page = 'home' # Corrected typo
        st.rerun()

# --- Carbon Sequestration Estimation Page (NEW FEATURE) ---
elif st.session_state.page == 'carbon_estimation': # Corrected typo
    st.markdown("""
    <div class="center-container">
        <h1>🌱 Carbon Sequestration Estimation</h1>
        <p class="feature-desc">Estimate the amount of carbon stored and annually sequestered in the forest, along with its potential monetary value, based on your image and resolution.</p>
    </div>
    """, unsafe_allow_html=True)

    file = st.file_uploader("Upload Satellite or Aerial Image", type=["jpg", "jpeg", "png"], key="carbon_estimation_uploader")

    # Input for Ground Sample Distance (GSD)
    st.subheader("Image Resolution (Ground Sample Distance - GSD)")
    st.info(f"The GSD is the real-world size of one pixel in your image (e.g., 10 meters/pixel). A precise GSD is crucial for accurate area calculation. If unknown, a common value for detailed satellite imagery (like Sentinel-2) is {DEFAULT_GSD_METERS} meters/pixel.")
    gsd_meters = st.number_input(
        "Enter GSD (meters/pixel):",
        min_value=0.1, # Minimum realistic GSD
        max_value=100.0, # Max realistic GSD for this type of analysis
        value=float(DEFAULT_GSD_METERS), # Default value
        step=0.1,
        format="%.2f",
        key="gsd_input"
    )

    if st.button("Calculate Carbon Metrics", key="calculate_carbon_btn"):
        if file:
            if validate_image_size(file):
                if gsd_meters <= 0:
                    st.error("GSD must be a positive value.")
                    st.stop()

                st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                try:
                    image = Image.open(file).convert("RGB")
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    img_np = np.array(image)
                    height, width = img_np.shape[:2]

                    st.write("Running AI model for tree segmentation...")
                    results = model.predict(img_np, conf=0.1, verbose=False)
                    result = results[0]

                    if result.masks is None or not result.masks.xyn:
                        st.warning("No tree masks were detected in the uploaded image. Carbon estimation will be 0.")
                        tree_pixels = 0
                        binary_mask = np.zeros((height, width), dtype=np.uint8)
                    else:
                        binary_mask = get_mask_from_result(result, width, height)
                        tree_pixels = np.count_nonzero(binary_mask == 1)
                    
                    # 1. Calculate area per pixel (in square meters)
                    area_per_pixel_sq_m = gsd_meters * gsd_meters

                    # 2. Calculate total forest area in square meters
                    total_forest_area_sq_m = tree_pixels * area_per_pixel_sq_m

                    # 3. Convert square meters to hectares (1 hectare = 10,000 sq meters)
                    total_forest_area_hectares = total_forest_area_sq_m / 10000.0

                    # 4. Estimate total carbon sequestered (in tonnes of Carbon)
                    estimated_carbon_tonnes = total_forest_area_hectares * CARBON_TONNES_PER_HECTARE

                    # 5. Estimate annual carbon sequestration (in tonnes of Carbon)
                    annual_carbon_sequestration_tonnes = total_forest_area_hectares * ANNUAL_CARBON_SEQUESTRATION_RATE_TC_PER_HA_YR

                    # 6. Calculate monetary value (using CO2 equivalent for carbon price)
                    estimated_carbon_co2e_tonnes = estimated_carbon_tonnes * TONNE_CARBON_TO_CO2
                    annual_carbon_co2e_tonnes = annual_carbon_sequestration_tonnes * TONNE_CARBON_TO_CO2

                    monetary_value_stored_usd = estimated_carbon_co2e_tonnes * CARBON_PRICE_USD_PER_TONNE
                    monetary_value_annual_usd = annual_carbon_co2e_tonnes * CARBON_PRICE_USD_PER_TONNE

                    # Calculate INR values
                    monetary_value_stored_inr = monetary_value_stored_usd * USD_TO_INR_RATE
                    monetary_value_annual_inr = monetary_value_annual_usd * USD_TO_INR_RATE


                    st.subheader("📊 Carbon Metrics Overview")
                    col_carbon1, col_carbon2 = st.columns(2)
                    with col_carbon1:
                        st.metric("Total Carbon Stored (tonnes C)", f"{estimated_carbon_tonnes:.2f}")
                    with col_carbon2:
                        st.metric("Annual Sequestration (tonnes C/yr)", f"{annual_carbon_sequestration_tonnes:.2f}")

                    st.markdown("---") # Separator

                    st.subheader("💰 Monetary Value")
                    col_value1, col_value2 = st.columns(2)
                    with col_value1:
                        st.metric("Value Stored (USD)", f"${monetary_value_stored_usd:,.2f}")
                    with col_value2:
                        st.metric("Value Stored (INR)", f"₹{monetary_value_stored_inr:,.2f}")
                    
                    col_value3, col_value4 = st.columns(2)
                    with col_value3:
                        st.metric("Annual Value (USD/yr)", f"${monetary_value_annual_usd:,.2f}")
                    with col_value4:
                        st.metric("Annual Value (INR/yr)", f"₹{monetary_value_annual_inr:,.2f}")

                    st.markdown(f"""
                        <div class='opaque-box disclaimer'>
                            <p>
                                <strong>Assumptions & Disclaimers:</strong><br>
                                * <strong>GSD:</strong> {gsd_meters:.2f} meters/pixel. Accurate GSD is critical.<br>
                                * <strong>Stored Carbon Density:</strong> Assumes {CARBON_TONNES_PER_HECTARE} tonnes of Carbon per hectare. This is an average and can vary widely.<br>
                                * <strong>Annual Sequestration Rate:</strong> Assumes {ANNUAL_CARBON_SEQUESTRATION_RATE_TC_PER_HA_YR} tonnes of Carbon per hectare per year. This is an average and varies greatly.<br>
                                * <strong>Carbon Price:</strong> Uses an illustrative price of ${CARBON_PRICE_USD_PER_TONNE}/tonne of CO2 equivalent. Actual carbon credit prices fluctuate significantly.<br>
                                * <strong>USD to INR Rate:</strong> Assumed {USD_TO_INR_RATE} INR/USD. This rate is subject to daily fluctuations.<br>
                                * These are estimates for illustrative purposes. Actual values depend on forest type, age, health, region, and market conditions.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.subheader("🖼 Visualizations")
                    st.image(binary_mask * 255, caption="Detected Tree Mask (Black & White)", use_container_width=True)

                    # Carbon Density Heatmap
                    st.subheader("🔥 Carbon Density Heatmap")
                    grid_size = 20 # Can be adjusted
                    
                    if height < grid_size or width < grid_size:
                        st.error(f"Image too small ({width}x{height}) for a grid size of {grid_size}. Please upload a larger image or reduce grid size for heatmap.")
                        # Skipping heatmap generation if image is too small
                    else:
                        heatmap = np.zeros((height // grid_size, width // grid_size))
                        for y in range(0, height, grid_size):
                            for x in range(0, width, grid_size):
                                y_end = min(y + grid_size, height)
                                x_end = min(x + grid_size, width)
                                patch = binary_mask[y:y_end, x:x_end]

                                y_idx = y // grid_size
                                x_idx = x // grid_size

                                if y_idx < heatmap.shape[0] and x_idx < heatmap.shape[1]:
                                    # Density is count of tree pixels in the patch
                                    heatmap[y_idx, x_idx] = np.count_nonzero(patch)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(heatmap, cmap="YlGn", cbar=True, ax=ax) # YlGn (Yellow-Green) is good for growth/density
                        ax.set_title("Carbon Density Heatmap (Denser Green = More Trees/Carbon)")
                        ax.set_xlabel(f"Grid X ({grid_size}m blocks)")
                        ax.set_ylabel(f"Grid Y ({grid_size}m blocks)")
                        st.pyplot(fig)

                    # Contextual Equivalencies
                    st.subheader("🌍 What This Means (Approximate Equivalencies)")
                    total_co2e_tonnes = estimated_carbon_tonnes * TONNE_CARBON_TO_CO2
                    annual_co2e_tonnes = annual_carbon_sequestration_tonnes * TONNE_CARBON_TO_CO2

                    car_miles = total_co2e_tonnes * MILES_PER_TONNE_CO2
                    household_elec = annual_co2e_tonnes * HOUSEHOLDS_PER_TONNE_CO2_ELECTRICITY

                    st.markdown(f"""
                        <div class='opaque-box'>
                            <p style='font-size:1.5rem; text-align: left;'>
                                The *total carbon stored* in this area is equivalent to: <br>
                                🚗 Driving *{car_miles:,.0f} miles* in an average passenger vehicle.<br>
                                <br>
                                The *annual carbon sequestration* of this area is equivalent to:<br>
                                ⚡ The electricity use of *{household_elec:,.1f} average U.S. homes* for one year.
                            </p>
                        </div>
                        <div class='opaque-box disclaimer'>
                            <small>
                                These equivalencies are based on EPA factors for CO2 and are approximate for illustrative purposes.
                                1 tonne Carbon (C) is converted to ~3.67 tonnes Carbon Dioxide (CO2) for these comparisons.
                            </small>
                        </div>
                    """, unsafe_allow_html=True)


                    # Generate LaTeX Report
                    report_data = f"""
Estimated Forest Area: {total_forest_area_hectares:.2f} hectares
Assumed GSD: {gsd_meters:.2f} m/pixel

Total Carbon Stored (tonnes C): {estimated_carbon_tonnes:.2f}
Annual Carbon Sequestration (tonnes C/year): {annual_carbon_sequestration_tonnes:.2f}

Monetary Value Stored:
 - USD: ${monetary_value_stored_usd:,.2f}
 - INR: ₹{monetary_value_stored_inr:,.2f}

Monetary Value Annual:
 - USD/year: ${monetary_value_annual_usd:,.2f}
 - INR/year: ₹{monetary_value_annual_inr:,.2f}

\\subsection*{{Assumptions and Equivalencies}}
\\begin{{itemize}}
    \\item Assumed Stored Carbon Density: {CARBON_TONNES_PER_HECTARE} tC/ha
    \\item Assumed Annual Sequestration Rate: {ANNUAL_CARBON_SEQUESTRATION_RATE_TC_PER_HA_YR} tC/ha/year
    \\item Assumed Carbon Price: ${CARBON_PRICE_USD_PER_TONNE}/tonne CO2eq
    \\item USD to INR Exchange Rate: {USD_TO_INR_RATE} (approximate, subject to change)
    \\item 1 tonne Carbon (C) = {TONNE_CARBON_TO_CO2:.2f} tonnes Carbon Dioxide (CO2)
    \\item Total stored CO2eq equivalent to driving {car_miles:,.0f} miles in average car.
    \\item Annual sequestered CO2eq equivalent to electricity use of {household_elec:,.1f} average U.S. homes.
\\end{{itemize}}
                    """
                    latex_content = generate_latex_report("Carbon Sequestration Estimation", report_data)
                    report_id = str(uuid.uuid4())
                    st.download_button(
                        label="Download Comprehensive Report",
                        data=latex_content,
                        file_name=f"carbon_estimation_report_{report_id}.tex",
                        mime="text/latex",
                        key="download_carbon_estimation"
                    )

                except Exception as e:
                    st.error(f"Error processing image for carbon estimation: {str(e)}")
                    st.exception(e)
            else:
                st.info("Please upload a valid image.")
        else:
            st.info("Please upload an image to calculate carbon sequestration.")

    if st.button("Back to Home", key="back_carbon_estimation"):
        st.session_state.page = 'home' # Corrected typo
        st.rerun()

elif st.session_state.page == 'tree_enumeration': # Corrected typo
    st.markdown("""
    <div class="center-container">
        <h1>🌳 Tree Enumeration (Coming Soon!)</h1>
        <p class="feature-desc">This feature will count individual trees in an image.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("This feature is under development. Please check back later!")
    if st.button("Back to Home", key="back_tree_enumeration_placeholder"):
        st.session_state.page = 'home' # Corrected typo
        st.rerun()
