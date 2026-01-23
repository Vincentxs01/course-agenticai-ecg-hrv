"""
Driver Drowsiness Detection System - Streamlit Main Application
Using multi-agent architecture for fatigue detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
from pathlib import Path

# Add agents and tools to path
sys.path.append(str(Path(__file__).parent / "agents"))
sys.path.append(str(Path(__file__).parent / "tools"))

# Import our Agents
from agents.agent1_filter import SignalFilterAgent
from agents.agent2_features import FeatureExtractionAgent
from agents.agent3_decision import DecisionAgent

# Page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="car",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        background-color: #ff4b4b;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .success-box {
        background-color: #00cc66;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Agents
@st.cache_resource
def init_agents():
    """Initialize three Agents"""
    agent1 = SignalFilterAgent()
    agent2 = FeatureExtractionAgent()
    agent3 = DecisionAgent()
    return agent1, agent2, agent3

# Main title
st.markdown('<div class="main-header">Driver Drowsiness Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ECG-based Fatigue Analysis with Multi-Agent AI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("System Settings")

    # Upload ECG file
    uploaded_file = st.file_uploader(
        "Upload ECG Data (CSV)",
        type=['csv'],
        help="Please upload a CSV file containing ECG signal"
    )

    sampling_rate = st.number_input(
        "Sampling Rate (Hz)",
        min_value=100,
        max_value=1000,
        value=250,
        step=50,
        help="ECG signal sampling frequency"
    )

    driving_duration = st.slider(
        "Driving Duration (minutes)",
        min_value=0,
        max_value=300,
        value=60,
        step=15,
        help="Time already spent driving"
    )

    st.markdown("---")
    st.header("System Status")

    # Agent status will be updated during processing
    if 'agent1' in st.session_state:
        st.metric("Agent 1", st.session_state.agent1_status)
    if 'agent2' in st.session_state:
        st.metric("Agent 2", st.session_state.agent2_status)
    if 'agent3' in st.session_state:
        st.metric("Agent 3", st.session_state.agent3_status)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Three-Layer Agent Architecture**:
    - Agent 1: Signal Filtering
    - Agent 2: Feature Extraction
    - Agent 3: Decision Analysis + MCP Tools

    **Features**:
    - Multi-modal risk assessment
    - Environmental context integration
    - Personalized baseline learning
    - Real-time visualization
    """)

# Main content area
if uploaded_file is not None:
    try:
        # Read data
        df = pd.read_csv(uploaded_file)

        # Assume first column is ECG signal
        if len(df.columns) > 0:
            raw_signal = df.iloc[:, 0].values
        else:
            st.error("CSV file format error: No data column found")
            st.stop()

        st.success(f"Data loaded successfully: {len(raw_signal)} samples ({len(raw_signal)/sampling_rate:.1f} seconds)")

        # Initialize Agents
        agent1, agent2, agent3 = init_agents()

        # Create three-column layout
        col1, col2, col3 = st.columns(3)

        # === Agent 1: Signal Filtering ===
        with col1:
            st.subheader("Agent 1: Signal Filter")

            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                time.sleep(0.3)
                progress_bar.progress(33)

                cleaned_signal = agent1.filter_ecg(raw_signal, sampling_rate)

                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()

            st.success(agent1.status)
            st.session_state.agent1_status = agent1.status

            # Show processing log
            with st.expander("View Processing Log"):
                for log in agent1.get_log():
                    st.text(log)

            # Detect artifacts
            artifacts = agent1.detect_artifacts(raw_signal)
            if len(artifacts) > 0:
                with st.expander("Detected Artifacts"):
                    for artifact in artifacts:
                        st.write(artifact)

        # === Agent 2: Feature Extraction ===
        with col2:
            st.subheader("Agent 2: Feature Extraction")

            with st.spinner("Analyzing..."):
                progress_bar = st.progress(0)
                time.sleep(0.3)
                progress_bar.progress(50)

                features = agent2.extract_features(cleaned_signal, sampling_rate)

                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()

            st.success(agent2.status)
            st.session_state.agent2_status = agent2.status

            # Display features
            st.metric("Average Heart Rate", f"{features['heart_rate']} bpm",
                     help="Heartbeats per minute")
            st.metric("HRV (SDNN)", f"{features['hrv_sdnn']} ms",
                     help="Standard deviation of RR intervals")
            st.metric("HRV (RMSSD)", f"{features['hrv_rmssd']} ms",
                     help="Root mean square of successive RR interval differences")
            st.metric("Beat Count", features['num_beats'])

            # Feature interpretation
            with st.expander("Feature Interpretation"):
                interpretations = agent2.interpret_features(features)
                for interp in interpretations:
                    st.write(interp)

        # === Agent 3: Decision Analysis ===
        with col3:
            st.subheader("Agent 3: AI Decision")

            with st.spinner("Analyzing (Querying MCP Tools)..."):
                progress_bar = st.progress(0)
                time.sleep(0.3)
                progress_bar.progress(25)

                decision = agent3.analyze(features, driving_duration_minutes=driving_duration)

                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()

            st.success(agent3.status)
            st.session_state.agent3_status = agent3.status

            # Risk level display
            risk_color = {
                "Very High": "RED",
                "High": "ORANGE",
                "Medium": "YELLOW",
                "Low": "GREEN"
            }

            risk_level = decision.get("risk_level", "Unknown")
            risk_score = decision.get("risk_score", 0)

            st.markdown(f"### Risk Level: {risk_level}")
            st.metric("Risk Score", f"{risk_score} / 100")

        # === Signal Visualization ===
        st.markdown("---")
        st.subheader("ECG Signal Visualization")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Raw ECG Signal", "Filtered ECG Signal"),
            vertical_spacing=0.12
        )

        # For visualization, only show first 2000 points
        display_length = min(2000, len(raw_signal))
        time_axis = np.arange(display_length) / sampling_rate

        # Raw signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=raw_signal[:display_length],
                mode='lines',
                name='Raw Signal',
                line=dict(color='lightblue', width=1)
            ),
            row=1, col=1
        )

        # Filtered signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=cleaned_signal[:display_length],
                mode='lines',
                name='Filtered Signal',
                line=dict(color='green', width=1.5)
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # === Risk Gauge ===
        st.markdown("---")
        st.subheader("Fatigue Risk Gauge")

        # Create gauge chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fatigue Risk Index", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "yellow"},
                    {'range': [50, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        gauge_fig.update_layout(height=300)
        st.plotly_chart(gauge_fig, use_container_width=True)

        # === AI Analysis Report ===
        st.markdown("---")
        st.subheader("AI Analysis Report")

        with st.expander("View Full Analysis Report (including MCP Tool Results)", expanded=True):
            st.markdown(decision.get("analysis", "No analysis results"))

        # Show decision log
        with st.expander("View Decision Log"):
            for log in agent3.get_log():
                st.text(log)

        # === Alert System ===
        if decision.get("alert_needed", False):
            st.markdown("---")
            st.markdown("""
            <div class="alert-box">
                WARNING: High Fatigue Detected!
            </div>
            """, unsafe_allow_html=True)

            # Show recommendations
            col_a, col_b = st.columns(2)
            with col_a:
                st.error("### Take Immediate Action:")
                st.write("1. Find a safe place to stop and rest")
                st.write("2. Rest for at least 15-20 minutes")
                st.write("3. Get out and stretch your body")
                st.write("4. Stay hydrated")

            with col_b:
                # Show nearest rest areas
                from tools.mcp_tools import MCPTools
                rest_areas = MCPTools.get_rest_area()
                st.warning("### Nearest Rest Areas:")
                for area in rest_areas[:3]:
                    st.write(f"**{area['name']}**")
                    st.write(f"Distance: {area['distance']} (approx. {area['eta']})")
                    st.write(f"Facilities: {', '.join(area['facilities'])}")
                    st.write("---")

        else:
            if risk_level == "Low":
                st.markdown("""
                <div class="success-box">
                    Current status is good. Continue safe driving!
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        with st.expander("View Detailed Error"):
            st.exception(e)

else:
    # Welcome page
    st.info("Please upload an ECG data file from the sidebar to begin analysis")

    # Create two columns
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("""
        ## System Features

        ### Three-Layer Agentic AI Architecture

        **Agent 1: Signal Filter Agent**
        - Automatically detect and remove motion artifacts
        - Multi-layer filtering technology
        - Baseline wander correction

        **Agent 2: Feature Extraction Agent**
        - Calculate heart rate, HRV and other physiological metrics
        - Automatic R-peak detection
        - RR interval analysis

        **Agent 3: Decision Agent (with MCP Tools)**
        - Multi-factor risk assessment
        - Environmental context integration:
          - Weather query (affects fatigue level)
          - Time risk assessment (late night = high risk)
          - Driving duration analysis
          - Rest area query
          - Medical knowledge base
        """)

    with col_right:
        st.markdown("""
        ## Quick Start

        ### Prepare Test Data

        If you don't have ECG data, use our generator:

        ```bash
        python utils/data_generator.py
        ```

        This will generate in `data/` folder:
        - `ecg_normal.csv` - Normal state
        - `ecg_drowsy.csv` - Drowsy state
        - `ecg_long_drowsy.csv` - Long duration test

        ### Supported Format

        - **File format**: CSV
        - **Content**: First column as ECG signal values
        - **Recommended sampling rate**: 250 Hz
        - **Recommended duration**: 30-60 seconds

        ### Key Features

        - Real-time ECG signal visualization
        - Multi-modal risk assessment
        - Smart alert system
        - Rule-based AI decision support
        - MCP tool integration
        - Completely free (no API required)
        """)

    # Show example data format
    st.markdown("### Data Format Example")
    example_df = pd.DataFrame({
        "ECG": [0.5, 0.6, 0.8, 1.2, 0.9, 0.5, 0.3, 0.2, 0.4, 0.6]
    })
    st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Driver Drowsiness Detection System v1.0</strong> | Multi-Agent AI Architecture + MCP Tool Integration</p>
    <p>Group: 2026-Wei-Wu-Zheng | License: Apache-2.0 (Code), CC-BY-4.0 (Docs)</p>
    <p style='color: #666; font-size: 0.9rem'>This system uses a rule-based decision engine, completely free without API</p>
</div>
""", unsafe_allow_html=True)
