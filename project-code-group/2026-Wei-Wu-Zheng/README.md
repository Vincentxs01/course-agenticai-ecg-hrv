# Driver Drowsiness Detection System

An intelligent driver drowsiness detection system using ECG signals and multi-agent AI architecture.

**Group:** 2026-Wei-Wu-Zheng
**Members:** Wei JungYing, Wu KunChe, Zheng YiKai
**License:** Apache-2.0 (code), CC-BY-4.0 (documentation)

## Overview

This system uses a three-layer Agentic AI architecture combined with MCP (Model Context Protocol) tools for real-time driver fatigue detection and alerting.

### Key Features

- Three-agent pipeline: Signal Processing -> Feature Extraction -> Decision Making
- Multi-modal analysis: Physiological metrics, weather, time, driving duration
- MCP tool integration for contextual awareness
- Real-time visualization with Streamlit
- Personalized baseline monitoring
- Rule-based decision engine (no API key required)

## System Architecture

```
+--------------------------------------------------+
|                  Streamlit UI                     |
+--------------------------------------------------+
                       |
          +------------+------------+
          v            v            v
     +--------+   +--------+   +--------+
     |Agent 1 |-->|Agent 2 |-->|Agent 3 |
     | Signal |   |Feature |   |Decision|
     | Filter |   |Extract |   | + MCP  |
     +--------+   +--------+   +---+----+
                                   |
                  +----------------+----------------+
                  v                v                v
             +---------+     +---------+      +---------+
             | Weather |     |  Time   |      |  Rest   |
             |   API   |     |  Risk   |      |  Areas  |
             +---------+     +---------+      +---------+
```

### Agent Descriptions

1. **Agent 1 - Signal Processing Agent**
   - Highpass filter: Remove baseline wander (0.5 Hz)
   - Lowpass filter: Remove high-frequency noise (40 Hz)
   - Notch filter: Remove power line interference (50/60 Hz)
   - Artifact detection and signal quality assessment

2. **Agent 2 - Feature Extraction Agent**
   - R-peak detection using scipy's find_peaks
   - RR interval calculation
   - Heart Rate (HR) computation
   - HRV metrics: SDNN, RMSSD

3. **Agent 3 - Decision Agent**
   - Multi-factor risk scoring
   - MCP tool queries (weather, time risk, driving duration)
   - Personalized baseline comparison
   - Intelligent recommendation generation

## Installation

### Prerequisites

- Python 3.10+
- Anaconda (recommended)

### Setup

1. Create conda environment:
```bash
conda create -n drowsiness python=3.10 -y
conda activate drowsiness
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate test data:
```bash
python utils/data_generator.py
```

4. Run the application:
```bash
streamlit run app.py
```

The browser will automatically open at `http://localhost:8501`

## Project Structure

```
drowsiness_detection/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── app.py                  # Main Streamlit application
├── agents/
│   ├── __init__.py
│   ├── agent1_filter.py    # Signal Processing Agent
│   ├── agent2_features.py  # Feature Extraction Agent
│   └── agent3_decision.py  # Decision Agent with MCP
├── tools/
│   ├── __init__.py
│   └── mcp_tools.py        # MCP tool implementations
├── utils/
│   ├── __init__.py
│   └── data_generator.py   # Synthetic ECG generator
└── data/
    ├── ecg_normal.csv      # Normal state test data
    ├── ecg_drowsy.csv      # Drowsy state test data
    └── ecg_long_drowsy.csv # Long duration test data
```

## Usage

### Basic Operation

1. Open the web interface in your browser
2. Upload ECG data (CSV format) from the sidebar
3. Set the sampling rate (default: 250 Hz)
4. Set driving duration
5. The system automatically runs all three agents
6. View risk assessment and recommendations

### Test Data

| File | Description | Expected Result |
|------|-------------|-----------------|
| `ecg_normal.csv` | Normal alert state | Risk Level: Low |
| `ecg_drowsy.csv` | Drowsy state | Risk Level: High |
| `ecg_long_drowsy.csv` | Extended drowsy | Risk Level: Very High |

### Data Format

CSV file with ECG column:
```
ECG
0.5
0.6
0.8
1.2
...
```

- Recommended sampling rate: 250 Hz
- Recommended duration: 30-60 seconds (minimum 10 seconds)

## Technical Details

### Signal Processing

- **Bandpass filter:** 0.5-40 Hz (preserves cardiac signal)
- **Notch filter:** 50 Hz (removes power line interference)
- **Normalization:** Z-score standardization

### Feature Calculation

- **Heart Rate (HR):** 60000 / mean(RR intervals)
- **SDNN:** std(RR intervals)
- **RMSSD:** sqrt(mean(diff(RR intervals)^2))

### Risk Scoring System

| Factor | Max Score |
|--------|-----------|
| Low heart rate | 30 |
| High HRV | 35 |
| Baseline deviation | 15 |
| Time risk | 40 |
| Weather factors | 15 |
| Driving duration | 40 |

**Risk Levels:**
- 0-29: Low Risk
- 30-49: Medium Risk
- 50-69: High Risk
- 70+: Very High Risk

## Testing Individual Agents

Each agent can be tested independently:

```bash
# Test Agent 1
python agents/agent1_filter.py

# Test Agent 2
python agents/agent2_features.py

# Test Agent 3
python agents/agent3_decision.py

# Test MCP Tools
python tools/mcp_tools.py
```

## References

1. Heart Rate Variability: Standards of Measurement (Task Force, 1996)
2. Pan-Tompkins Algorithm for QRS Detection
3. Model Context Protocol (MCP) - Anthropic

## License

- **Code:** Apache License 2.0
- **Documentation:** Creative Commons Attribution 4.0 (CC-BY-4.0)

---

**Version:** 1.0.0
**Last Updated:** 2026-01-16
