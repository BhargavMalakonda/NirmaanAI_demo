import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_model import ConcreteStrengthPredictor
import os
import requests

# Page configuration
st.set_page_config(
    page_title="PreCast Cycle Optimizer | L&T CreaTech",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS from your friend's design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    
    * { margin: 0; padding: 0; box-sizing: border-box; }

    :root {
      --bg:      #F4F1EC;
      --surface: #FFFFFF;
      --border:  #DDD8CF;
      --text:    #1C1915;
      --muted:   #857E74;
      --accent:  #2D5A3D;
      --accent2: #7A9E7E;
      --warm:    #C4763A;
      --tag-bg:  #E8F0EA;
      --tag-txt: #2D5A3D;
    }

    .stApp {
        background: var(--bg);
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--text);
        font-size: 13.5px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem 2rem 5rem;
    }
    
    .hdr {
      margin-bottom: 2.5rem;
      padding-bottom: 2rem;
      border-bottom: 1px solid var(--border);
    }
    
    .hdr h1 {
      font-family: 'Fraunces', serif;
      font-size: 2.2rem;
      font-weight: 300;
      color: var(--text);
      letter-spacing: -0.8px;
      line-height: 1.15;
      margin-bottom: 0.5rem;
    }
    
    .hdr h1 i { font-style: italic; color: var(--accent); font-weight: 300; }
    
    .hdr p {
      color: var(--muted);
      font-size: 14px;
      font-weight: 300;
      max-width: 480px;
    }
    
    .sec {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 1.25rem;
      overflow: hidden;
    }
    
    .sec-title {
      padding: 0.9rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }
    
    .sec-title h2 {
      font-size: 11.5px;
      font-weight: 600;
      color: var(--text);
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin: 0;
    }
    
    .sec-title span { font-size: 12px; color: var(--muted); font-weight: 300; }
    
    .sec-body { padding: 1.5rem; }
    
    .g3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.1rem; }
    .g4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1.1rem; }
    .g2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; }
    
    @media(max-width:660px) { .g3, .g4 { grid-template-columns: 1fr 1fr; } }
    @media(max-width:420px) { .g2, .g3, .g4 { grid-template-columns: 1fr; } }
    
    .f { display: flex; flex-direction: column; gap: 5px; }
    .span2 { grid-column: span 2; }
    
    label.lbl {
      font-size: 11px;
      font-weight: 500;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: block;
      margin-bottom: 5px;
    }
    
    .stSelectbox, .stNumberInput, .stSlider {
      font-family: 'IBM Plex Sans', sans-serif;
    }
    
    div[data-baseweb="select"] > div {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 5px;
    }
    
    .stSlider div[data-baseweb="slider"] {
      padding-top: 0.5rem;
    }
    
    .div { height: 1px; background: var(--border); margin: 1.25rem 0; }
    
    .pills { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 0.5rem; }
    
    .stMultiSelect div[data-baseweb="select"] div {
      background: var(--bg);
    }
    
    .w-row { display: flex; align-items: flex-end; gap: 0.8rem; flex-wrap: wrap; }
    
    .fetch-btn {
      height: 38px; padding: 0 14px;
      border: 1px solid var(--border); border-radius: 5px;
      background: var(--bg); color: var(--muted);
      font-family: 'IBM Plex Sans', sans-serif; font-size: 12.5px; font-weight: 500;
      cursor: pointer; transition: all 0.12s; white-space: nowrap;
    }
    
    .fetch-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--tag-bg); }
    
    .w-live { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 0.7rem; }
    
    .w-chip {
      font-size: 12px; padding: 3px 9px;
      background: var(--tag-bg); border: 1px solid #c5d9c8;
      border-radius: 4px; color: var(--accent); font-weight: 500;
    }
    
    .run-area { margin: 1.75rem 0; display: flex; align-items: center; gap: 1.25rem; }
    
    .stButton > button {
      background: var(--accent) !important;
      color: #fff !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 10px 24px !important;
      font-family: 'IBM Plex Sans', sans-serif !important;
      font-size: 13.5px !important;
      font-weight: 500 !important;
    }
    
    .stButton > button:hover {
      opacity: 0.85 !important;
    }
    
    .run-note { font-size: 12.5px; color: var(--muted); font-weight: 300; }
    
    .results { margin-top: 2.5rem; }
    
    .res-hdr {
      margin-bottom: 1.5rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }
    
    .res-hdr h2 {
      font-family: 'Fraunces', serif;
      font-size: 1.6rem; font-weight: 300;
      color: var(--text); letter-spacing: -0.5px; margin-bottom: 0.25rem;
    }
    
    .metrics {
      display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 1px; background: var(--border);
      border: 1px solid var(--border); border-radius: 8px;
      overflow: hidden; margin-bottom: 1.25rem;
    }
    
    .m-cell { background: var(--surface); padding: 1.1rem 1.25rem; }
    
    .m-num {
      font-family: 'Fraunces', serif;
      font-size: 2rem; font-weight: 300;
      color: var(--accent); line-height: 1;
      margin-bottom: 4px; letter-spacing: -1px;
    }
    
    .m-lbl { font-size: 11.5px; color: var(--muted); font-weight: 400; }
    
    .tbl {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; overflow: hidden; margin-bottom: 1.25rem;
    }
    
    .tr {
      display: grid; grid-template-columns: 2.2fr 1fr 1fr 0.9fr;
      border-bottom: 1px solid var(--border); align-items: center;
    }
    
    .tr:last-child { border-bottom: none; }
    .tr.th { background: var(--bg); }
    .tr.rec { background: #f7fbf8; }
    
    .td { padding: 0.85rem 1.1rem; font-size: 13px; }
    
    .th .td {
      font-size: 10.5px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted);
    }
    
    .sc-name { font-weight: 500; color: var(--text); }
    .sc-time { font-weight: 600; color: var(--accent); }
    
    .bdg-rec {
      background: var(--tag-bg); color: var(--accent); 
      padding: 2px 8px; border-radius: 3px; font-size: 10.5px;
      border: 1px solid #c5d9c8;
    }
    
    .bdg-fast {
      background: #fef3eb; color: var(--warm); 
      padding: 2px 8px; border-radius: 3px; font-size: 10.5px;
      border: 1px solid #e8c9a8;
    }
    
    .insights-sec {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; overflow: hidden; margin-bottom: 1.25rem;
    }
    
    .ins-head {
      padding: 0.8rem 1.25rem; border-bottom: 1px solid var(--border);
      background: var(--bg); font-size: 10.5px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.7px; color: var(--muted);
    }
    
    .ins-item {
      padding: 0.85rem 1.25rem; border-bottom: 1px solid var(--border);
      font-size: 13px; font-weight: 300; color: var(--text);
      display: flex; gap: 1rem; align-items: flex-start;
    }
    
    .ins-item:last-child { border-bottom: none; }
    
    .ins-num {
      font-size: 10.5px; font-weight: 600; color: var(--accent);
      min-width: 16px; padding-top: 3px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Load ML model
@st.cache_resource
def load_ml_model():
    predictor = ConcreteStrengthPredictor()
    if predictor.load_model('concrete_strength_model.pkl'):
        return predictor
    return None

# Load model
with st.spinner("Loading ML model..."):
    st.session_state.model = load_ml_model()

# Main content
st.markdown('<div class="wrap">', unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="hdr">
        <h1>Cycle Time <i>Optimisation</i></h1>
        <p>Enter project parameters below to simulate cycle time scenarios and receive cost-aware recommendations.</p>
    </div>
""", unsafe_allow_html=True)

# Project Details Section
st.markdown("""
    <div class="sec">
        <div class="sec-title">
            <h2>Project Details</h2>
            <span>Basic parameters</span>
        </div>
        <div class="sec-body">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    project_type = st.selectbox("Project Type", ["Infrastructure", "Building", "Bridge", "Metro / Rail"], key="project_type")
with col2:
    region = st.selectbox("Region", ["North India", "South India", "East India", "West India", "Central India", "Coastal"], key="region")
with col3:
    element_type = st.selectbox("Element Type", ["Beams / Girders", "Hollow Core Slabs", "Wall Panels", "Columns", "Piles", "Box Culverts"], key="element")

col1, col2, col3 = st.columns(3)
with col1:
    grade = st.selectbox("Concrete Grade", ["M30", "M40", "M50", "M60"], key="grade")
with col2:
    daily_target = st.number_input("Daily Target (elements)", min_value=1, max_value=50, value=8, key="daily_target")
with col3:
    automation = st.selectbox("Automation Level", ["Manual", "Semi-Automated", "Fully Automated"], key="automation")

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    mould_count = st.number_input("Number of Moulds", min_value=1, value=6, key="moulds")
with col2:
    yard_area = st.number_input("Yard Area (m2)", min_value=100, value=2000, key="yard")
with col3:
    strength_target = st.slider("Demoulding Strength Target", min_value=50, max_value=95, value=70, key="strength_pct",
                                help="Percentage of design strength required for demoulding")

st.markdown("</div></div>", unsafe_allow_html=True)

# Mix Design Section
st.markdown("""
    <div class="sec">
        <div class="sec-title">
            <h2>Mix Design & Curing</h2>
            <span>Admixtures and curing strategy</span>
        </div>
        <div class="sec-body">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    cement_type = st.selectbox("Cement Type", ["OPC 53", "OPC 43", "PPC", "GGBS Blend", "Silica Fume Blend"], key="cement")
with col2:
    wc_ratio = st.number_input("w/c Ratio", min_value=0.25, max_value=0.55, value=0.38, step=0.01, key="wc")
with col3:
    priority = st.selectbox("Optimisation Priority", ["Min Cycle Time", "Cost Optimised", "Quality First"], key="priority")

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    admixtures = st.multiselect("Admixtures", ["HRWR", "Accelerator", "VMA", "Retarder", "Silica Fume"],
                               default=["HRWR"], key="admixtures")
with col2:
    curing_methods = st.multiselect("Curing Methods", ["Steam", "Water Spray", "Membrane", "Heat Blanket", "Autoclave"],
                                   default=["Steam"], key="curing")

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

steam_temp = st.slider("Steam Curing Temperature", min_value=40, max_value=90, value=65, key="steam_temp",
                       help="Temperature in degrees Celsius")

st.markdown("</div></div>", unsafe_allow_html=True)

# Climatic Conditions
st.markdown("""
    <div class="sec">
        <div class="sec-title">
            <h2>Climatic Conditions</h2>
            <span>Live fetch or manual entry</span>
        </div>
        <div class="sec-body">
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    city = st.text_input("City", placeholder="e.g. Chennai, Mumbai", key="city", 
                         help="Enter city name to fetch live weather data")
with col2:
    if st.button("Fetch Live Weather", key="fetch_weather"):
        try:
            # Using Open-Meteo API (free, no key required)
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            geo_response = requests.get(geo_url)
            geo_data = geo_response.json()
            
            if geo_data.get('results'):
                lat = geo_data['results'][0]['latitude']
                lon = geo_data['results'][0]['longitude']
                
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&timezone=auto"
                weather_response = requests.get(weather_url)
                weather_data = weather_response.json()
                
                if 'current' in weather_data:
                    current = weather_data['current']
                    st.session_state['temp'] = current['temperature_2m']
                    st.session_state['humidity'] = current['relative_humidity_2m']
                    st.session_state['weather_fetched'] = True
                    st.success(f"Weather fetched for {city}!")
        except Exception as e:
            st.error("Could not fetch weather data")

st.markdown('<div class="w-live">', unsafe_allow_html=True)
if st.session_state.get('weather_fetched'):
    st.markdown(f"""
        <div class="w-chip">{st.session_state['temp']:.0f} C</div>
        <div class="w-chip">{st.session_state['humidity']:.0f}% Humidity</div>
        <div class="w-chip">Live data updated</div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    temp = st.number_input("Avg Temp (C)", value=st.session_state.get('temp', 28), key="temp")
with col2:
    humidity = st.number_input("Humidity (%)", value=st.session_state.get('humidity', 65), min_value=0, max_value=100, key="humidity")
with col3:
    night_temp = st.number_input("Night Temp (C)", value=st.session_state.get('temp', 28)-8, key="night_temp")
with col4:
    season = st.selectbox("Season", ["Summer", "Monsoon", "Winter", "Transition"], key="season")

st.markdown("</div></div>", unsafe_allow_html=True)

# Run Analysis Button
st.markdown('<div class="run-area">', unsafe_allow_html=True)
if st.button("Run Analysis", key="run_analysis", use_container_width=False):
    if st.session_state.model is None:
        st.error("ML Model not loaded. Please check if model file exists.")
    else:
        with st.spinner("Running analysis..."):
            # Map inputs to model parameters
            grade_map = {"M30": 30, "M40": 40, "M50": 50, "M60": 60}
            target_mpa = grade_map[grade] * (strength_target / 100)
            
            # Map curing method to model's curing types
            curing_map = {
                "Steam": "Steam",
                "Water Spray": "Water",
                "Membrane": "Compound",
                "Heat Blanket": "Air",
                "Autoclave": "Steam"
            }
            
            # Get primary curing method
            primary_curing = curing_map[curing_methods[0]] if curing_methods else "Air"
            
            # Find optimal time using ML model
            optimal_hours, predicted_strength = st.session_state.model.find_optimal_time(
                temp, humidity, primary_curing, target_mpa,
                min_hours=24, max_hours=720, step=24
            )
            
            # Calculate costs
            cost_rates = {'Air': 1500, 'Compound': 2000, 'Water': 2500, 'Steam': 5000}
            hourly_rate = cost_rates.get(primary_curing, 2000)
            total_cost = optimal_hours * hourly_rate
            
            # Store results
            st.session_state.results = {
                'optimal_hours': optimal_hours,
                'optimal_days': optimal_hours/24,
                'predicted_strength': predicted_strength,
                'target_strength': target_mpa,
                'total_cost': total_cost,
                'hourly_rate': hourly_rate,
                'curing_method': primary_curing
            }
st.markdown('<span class="run-note">Evaluates 200+ parameter combinations</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Display Results
if st.session_state.results:
    r = st.session_state.results
    
    st.markdown("""
        <div class="results">
            <div class="res-hdr">
                <h2>Analysis Results</h2>
                <p id="res-sub">Showing optimised scenarios based on your inputs</p>
            </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown(f"""
        <div class="metrics">
            <div class="m-cell"><div class="m-num">{r['optimal_hours']:.0f}h</div><div class="m-lbl">Optimal cycle time</div></div>
            <div class="m-cell"><div class="m-num">{72 - r['optimal_hours']:.0f}h</div><div class="m-lbl">Saved vs conservative</div></div>
            <div class="m-cell"><div class="m-num">₹{r['total_cost']:,.0f}</div><div class="m-lbl">Cost per cycle</div></div>
            <div class="m-cell"><div class="m-num">{r['predicted_strength']:.1f} MPa</div><div class="m-lbl">Predicted strength</div></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("""
        <div class="insights-sec">
            <div class="ins-head">Recommendations</div>
    """, unsafe_allow_html=True)
    
    insights = [
        f"Optimal demoulding time: {r['optimal_hours']:.0f} hours ({r['optimal_days']:.1f} days) at {r['predicted_strength']:.1f} MPa strength.",
        f"Current conditions: {temp}°C, {humidity}% humidity. {'Steam curing recommended' if temp < 25 else 'Standard curing sufficient'}.",
        f"With {mould_count} moulds and {r['optimal_hours']:.0f}-hour cycle, maximum output is {int(24 * mould_count / r['optimal_hours'])} elements per day.",
        f"Cost per element: ₹{r['total_cost']/daily_target:,.0f} at {daily_target} elements/day target.",
    ]
    
    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
            <div class="ins-item">
                <span class="ins-num">{i}</span>
                <span>{insight}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Strength Development Graph
    st.markdown('<div class="sec"><div class="sec-title"><h2>Strength Development</h2></div><div class="sec-body">', unsafe_allow_html=True)
    
    # Generate predictions for graph
    hours_range = range(24, 721, 24)
    strengths = []
    for h in hours_range:
        s = st.session_state.model.predict(temp, humidity, primary_curing, h)
        strengths.append(s)
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.plot([h/24 for h in hours_range], strengths, color='#2D5A3D', linewidth=2)
    ax.axhline(y=r['target_strength'], color='#C4763A', linestyle='--', alpha=0.7, label='Target Strength')
    ax.axvline(x=r['optimal_days'], color='#2D5A3D', linestyle='--', alpha=0.7, label='Optimal Time')
    ax.plot(r['optimal_days'], r['predicted_strength'], 'o', color='#2D5A3D', markersize=8)
    
    ax.set_xlabel('Curing Time (days)', fontsize=11)
    ax.set_ylabel('Concrete Strength (MPa)', fontsize=11)
    ax.set_title('Strength Development Over Time', fontsize=12, fontweight='normal')
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    st.pyplot(fig)
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close wrap