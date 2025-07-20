"""
🩺 Clinical Heart Disease AI - Streamlit Helper Functions
All utility functions, API calls, visualizations, and styling for the Streamlit app.

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import base64
import json
import time
import warnings
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ===============================
# 🔧 CONFIGURATION
# ===============================
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 30


# ===============================
# 🎨 CUSTOM CSS & STYLING
# ===============================
def load_custom_css():
    """Load custom CSS for professional medical interface."""
    st.markdown("""
    <style>
    /* Import medical-grade fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styling */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Additional fixes for Streamlit main content */
    div[data-testid="stMainBlockContainer"], section[data-testid="stMain"] {
        padding-top: 0.5rem !important;
    }

    /* Header styling */
    .medical-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .medical-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .medical-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Risk gauge container */
    .risk-gauge-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e7ff;
        margin: 1rem 0;
    }

    /* Risk level styling */
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }

    .risk-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }

    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }

    /* Medical panel styling */
    .medical-panel {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .medical-panel-header {
        color: #1e40af;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ddd6fe;
    }

    /* Feature importance styling */
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .feature-increase {
        border-left-color: #ef4444;
        background: linear-gradient(90deg, #fef2f2 0%, #ffffff 100%);
    }

    .feature-decrease {
        border-left-color: #10b981;
        background: linear-gradient(90deg, #f0fdf4 0%, #ffffff 100%);
    }

    /* Metrics display */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Clinical summary styling */
    .clinical-summary {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .clinical-summary h4 {
        color: #0c4a6e;
        margin-bottom: 1rem;
    }

    /* Warning/insight box */
    .insight-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .insight-box h5 {
        color: #92400e;
        margin-bottom: 0.5rem;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
    }

    /* Footer styling */
    .medical-footer {
        background: #1f2937;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #3b82f6;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #1d4ed8;
    }
    
    /* Target the expander header directly */
    .streamlit-expander .streamlit-expanderHeader {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Alternative method - target by attribute */
    details[open] > summary,
    details > summary {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
        list-style: none !important;
    }
    
    /* Hide default arrow and style custom */
    details > summary::-webkit-details-marker {
        display: none;
    }
    
    /* Content styling */
    details[open] {
        border: 2px solid #3b82f6 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    

/* Mobile-responsive footer centering */
@media (max-width: 768px) {
    /* Center all column content */
    .stColumns > div {
        text-align: center !important;
    }
    
    /* Target Streamlit subheader specifically */
    .stColumns h3,
    div[data-testid="column"] h3,
    .element-container h3 {
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Mobile-responsive footer centering */
    @media (max-width: 768px) {
        /* Center all column content */
        .stColumns > div {
            text-align: center !important;
        }
        
        /* Target Streamlit subheader specifically */
        .stColumns h3,
        div[data-testid="column"] h3,
        .element-container h3 {
            text-align: center !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        
        /* Center all text elements in columns */
        .stColumns p,
        .stColumns div,
        div[data-testid="column"] p,
        div[data-testid="column"] div {
            text-align: center !important;
        }
        
        /* Center the main caption/copyright text */
        .element-container p {
            text-align: center !important;
        }
        
        /* Ensure markdown links are centered too */
        .stColumns .stMarkdown,
        div[data-testid="column"] .stMarkdown {
            text-align: center !important;
        }
        
        /* Force center alignment for any nested content */
        .stColumns * {
            text-align: center !important;
        }
    }
    
    </style>
    """, unsafe_allow_html=True)

# ===============================
# 🔗 API COMMUNICATION FUNCTIONS
# ===============================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_api_health():
    """Check if the API is healthy and model is loaded."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def call_api_endpoint(endpoint: str, data: Dict = None):
    """Make API call with proper error handling and clean pydantic errors."""
    try:
        if data:
            response = requests.post(f"{API_BASE_URL}/{endpoint}",
                                     json=data, timeout=TIMEOUT)
        else:
            response = requests.get(f"{API_BASE_URL}/{endpoint}",
                                    timeout=TIMEOUT)

        if response.status_code == 200:
            return response.json(), None
        else:
            try:
                detail = response.json().get("detail", None)
                if isinstance(detail, list):
                    error_msgs = []
                    for item in detail:
                        loc = item.get("loc", ["field"])[-1]
                        msg = item.get("msg", "")
                        error_msgs.append(f"{loc}: {msg}")
                    clean_msg = " | ".join(error_msgs)
                    return None, f"Validation error: {clean_msg}"
                else:
                    return None, f"API Error: {detail or response.status_code}"
            except:
                return None, f"API Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return None, "API timeout - please try again"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API - please check if the service is running"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ===============================
# 🎨 VISUALIZATION FUNCTIONS
# ===============================
def create_risk_gauge(probability: float, risk_class: str):
    """Create a professional risk assessment gauge."""
    color_map = {
        "Low Risk": "#10b981",
        "Moderate Risk": "#f59e0b",
        "High Risk": "#ef4444"
    }

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{risk_class}</b><br><span style='font-size:0.8em'>Cardiovascular Risk Assessment</span>"},
        delta={'reference': 50, 'position': "top"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1f2937"},
            'bar': {'color': color_map.get(risk_class, "#6b7280")},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 30], 'color': "#dcfce7"},
                {'range': [30, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': "#1f2937", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=400,
        font={'color': "#1f2937", 'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def create_shap_waterfall(shap_data: Dict):
    """Create SHAP waterfall plot for feature contributions."""
    top_features = shap_data['top_features'][:8]

    features = [f['feature'] for f in top_features]
    values = [f['shap_value'] for f in top_features]
    colors = ['#ef4444' if v > 0 else '#10b981' for v in values]

    fig = go.Figure(go.Waterfall(
        name="SHAP Values",
        orientation="v",
        measure=["relative"] * len(features),
        x=features,
        textposition="outside",
        text=[f"{v:+.3f}" for v in values],
        y=values,
        connector={"line": {"color": "#64748b"}},
        increasing={"marker": {"color": "#ef4444"}},
        decreasing={"marker": {"color": "#10b981"}},
    ))

    fig.update_layout(
        title="<b>Feature Contributions to Risk Prediction</b><br><sub>How each factor influences the prediction</sub>",
        xaxis_title="Clinical Features",
        yaxis_title="SHAP Value (Impact on Risk)",
        height=500,
        font={'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={'tickangle': 45}
    )

    return fig


def create_population_comparison(positions_data: Dict):
    """Create population comparison visualization."""
    feature_positions = positions_data['feature_positions']

    features, percentiles = [], []
    for feature, data in feature_positions.items():
        if feature in ['age', 'trestbps', 'chol', 'thalach', 'hr_achievement']:
            features.append(feature.replace('_', ' ').title())
            percentiles.append(data['percentile'])

    fig = go.Figure(data=go.Scatterpolar(
        r=percentiles,
        theta=features,
        fill='toself',
        line=dict(color='#3b82f6', width=3),
        fillcolor='rgba(59, 130, 246, 0.1)',
        marker=dict(size=8, color='#1d4ed8')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickvals=[25, 50, 75, 100],
                            ticktext=['25th', '50th', '75th', '100th'])
        ),
        title="<b>Patient Position vs. Population</b><br><sub>Percentile rankings across key metrics</sub>",
        height=400,
        font={'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ===============================
# 📊 SAMPLE PATIENT DATA
# ===============================
def get_sample_patients():
    """Return sample patient data for demonstration."""
    return {
        "Patient A": {
            "age": 67, "sex": 1, "cp": 0, "trestbps": 160, "chol": 286,
            "fbs": 0, "restecg": 0, "thalach": 108, "exang": 1,
            "oldpeak": 1.5, "slope": 1, "ca": 3, "thal": 2
        },
        "Patient B": {
            "age": 29, "sex": 1, "cp": 1, "trestbps": 130, "chol": 204,
            "fbs": 0, "restecg": 0, "thalach": 202, "exang": 0,
            "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
        },
        "Patient C": {
            "age": 54, "sex": 0, "cp": 2, "trestbps": 108, "chol": 267,
            "fbs": 0, "restecg": 1, "thalach": 167, "exang": 0,
            "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
        }
    }


# ===============================
# 🔧 UTILITY FUNCTIONS
# ===============================
def display_feature_tooltips():
    """Display helpful tooltips for medical features."""
    return {
        "Age": "Cardiovascular risk increases significantly with age, especially after 65",
        "Sex": "Males generally have higher CAD risk before age 65; risk equalizes post-menopause",
        "Chest Pain": "Typical angina suggests CAD, but atypical presentations are common",
        "Blood Pressure": "Hypertension damages arteries and accelerates atherosclerosis",
        "Cholesterol": "High LDL cholesterol contributes to plaque formation in arteries",
        "Exercise ECG": "ST depression during exercise indicates cardiac ischemia",
        "Heart Rate": "Inability to achieve target heart rate may indicate cardiac dysfunction",
        "Coronary Vessels": "Number of significantly blocked arteries (>50% stenosis)",
        "Thalassemia": "Nuclear stress test showing perfusion defects in heart muscle"
    }


def validate_patient_data(data: Dict) -> List[str]:
    """Validate patient data for clinical plausibility."""
    warnings_list = []

    expected_max_hr = 220 - data['age']
    if data['thalach'] > expected_max_hr + 20:
        warnings_list.append(f"Heart rate ({data['thalach']}) unusually high for age {data['age']}")

    if data['trestbps'] > 180:
        warnings_list.append("Severe hypertension detected - consider immediate medical attention")

    if data['chol'] > 400:
        warnings_list.append("Extremely high cholesterol - may indicate familial hypercholesterolemia")

    if data['oldpeak'] > 4:
        warnings_list.append("Severe ST depression - highly suggestive of significant CAD")

    return warnings_list
