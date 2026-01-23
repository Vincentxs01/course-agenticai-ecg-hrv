# Technical Report: Agent-Based Driver Drowsiness Detection from ECG

**Author:** Jack
**Group Members:** Wei JungYing, Wu KunChe, Zheng YiKai
**License:** CC-BY-4.0

## Abstract

This report details a Driver Drowsiness Detection System built on a multi-agent AI architecture. The system automates the analysis of electrocardiogram (ECG) signals to assess driver fatigue in real time. It comprises three specialized agents: a Signal Filter Agent for noise removal, a Feature Extraction Agent for calculating Heart Rate Variability (HRV) metrics, and a Decision Agent that integrates physiological data with contextual information from MCP (Model Context Protocol) tools. The system processes synthetic ECG data, simulating normal and drowsy states, and presents a risk score and actionable recommendations through a Streamlit web interface. The entire analysis, from data upload to final decision, is completed in under two seconds, demonstrating a significant efficiency gain over manual, chat-based analysis methods.

## Introduction

Driver fatigue is a critical safety issue, contributing to a significant percentage of traffic accidents. Traditional methods for monitoring drowsiness are often subjective or intrusive. Physiological signals, particularly Heart Rate Variability (HRV) derived from ECG, offer a robust and objective indicator of the state of the autonomic nervous system, which is closely linked to fatigue and stress.

This project aimed to develop an autonomous AI system that could:
1.  Automatically process raw ECG signals to ensure data quality.
2.  Extract key HRV features indicative of drowsiness.
3.  Integrate physiological data with contextual factors (e.g., time of day, driving duration).
4.  Provide a real-time risk assessment and generate clear, actionable alerts for the driver.

## System Architecture

The system is designed as a pipeline of three specialized agents, orchestrated by a Streamlit web application. This modular design separates concerns, making the system easy to maintain and extend. The Decision Agent is augmented with a suite of MCP tools to provide contextual awareness that goes beyond the immediate physiological data.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                             Driver Drowsiness Detection System                       │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐   │
│  │ <<database>> │    │  <<UI>>       │     │  <<agent>>    │     │  <<agent>>    │   │
│  │              │───▶│              │────▶│               │────▶│              │   │
│  │   ECG Data   │    │ Streamlit UI  │     │ Signal Filter │     │    Feature    │   │
│  │    (CSV)     │    │   (app.py)    │     │(agent1_filter)│     │  Extraction   │   │
│  └──────────────┘    └───────┬───────┘     └───────┬───────┘     └───────┬───────┘   │
│                              │                     │                     │           │
│                              │(display)            │(cleaned_signal)     │(features) │
│                              │                     │                     │           │
│                              └─────────────────────┼─────────────────────┼───────────┘
│                                                    │                     │
│                                                    └───────────┬─────────┘
│                                                                │
│                                                                ▼
│                                                      ┌───────────────────┐
│                                                      │    <<agent>>      │
│                                                      │                   │
│                                                      │   Decision + MCP  │
│                                                      │  (agent3_decision)│
│                                                      └─────────┬─────────┘
│                                                                │(queries)
│                                                                │
│          ┌─────────────────────────────────────────────────────┼──────────────────┐
│          │                                                     │                  │
│          ▼                  ▼                   ▼              ▼                  ▼
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│  │   <<tool>>     │ │   <<tool>>     │ │   <<tool>>     │ │   <<tool>>     │ │   <<tool>>     │
│  │ Time Risk      │ │ Duration Risk  │ │ Rest Area Query│ │ Medical Knowl. │ │ Weather Query  │
│  └────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

| Component | Type | Description |
|---|---|---|
| **Streamlit UI** | UI | Provides the user interface for file upload, parameter input, and visualization of results. Orchestrates the agent pipeline. |
| **Signal Filter Agent** | Agent | Cleans the raw ECG signal by applying high-pass, low-pass, and notch filters to remove baseline wander and noise. |
| **Feature Extraction Agent** | Agent | Detects R-peaks from the cleaned signal, calculates RR intervals, and computes HRV metrics like SDNN and RMSSD. |
| **Decision Agent + MCP** | Agent | Analyzes physiological features and queries MCP tools to calculate a final risk score and generate an analysis report. |
| **MCP Tools** | Tool | A set of functions that provide contextual information, such as time of day risk, weather conditions, and nearby rest areas. |

## Data

### Dataset: Synthetic ECG for Drowsiness Detection

To ensure privacy, reproducibility, and safety during development, this project utilizes **synthetic ECG signals**. The data was generated using a custom Python script (`utils/data_generator.py`) that simulates cardiac patterns under different physiological states.

**Rationale for Synthetic Data:**
*   **Privacy:** Avoids the use of sensitive, real-world physiological data.
*   **Controllability:** Allows for the precise generation of 'alert' vs. 'drowsy' ECG patterns.
*   **Reproducibility:** Enables consistent testing and validation of the system's performance.

### File Descriptions

The dataset consists of three CSV files, each generated at a sampling rate of 250 Hz:

| File | Duration | State | Description |
|---|---|---|---|
| `ecg_normal.csv` | 30 sec | Alert | Simulates a normal, alert driving state with a heart rate of ~75 bpm and low HRV. |
| `ecg_drowsy.csv` | 30 sec | Drowsy | Simulates a fatigue state with a lower heart rate (~58 bpm) and higher HRV. |
| `ecg_long_drowsy.csv`| 60 sec | Drowsy | An extended recording to test the system's response to prolonged drowsiness. |

The preprocessing (filtering and normalization) is not applied to the source files but is handled in real-time by the Signal Filter Agent upon data upload.

## Implementation

The system is implemented in Python using the Streamlit framework for the user interface and libraries like `NumPy` and `SciPy` for signal processing. The core logic is encapsulated within three distinct agent classes.

### Agent 1: Signal Filter Agent

This agent's primary role is to take the raw ECG signal and prepare it for feature extraction. This involves a multi-stage filtering process to remove common sources of noise. The code is modular, with each filtering step clearly logged.

**Key Code (`agents/agent1_filter.py`):**
```python
import numpy as np
from scipy import signal

class SignalFilterAgent:
    """Agent 1: Responsible for ECG signal cleaning"""

    def __init__(self):
        self.name = "Signal Filter Agent"
        self.status = "Standby"
        self.processing_log = []

    def filter_ecg(self, raw_signal, sampling_rate=250):
        """
        Clean ECG signal by removing noise

        Processing steps:
        1. Highpass filter: Remove baseline wander
        2. Lowpass filter: Remove high-frequency noise
        3. Notch filter: Remove power line interference (50/60 Hz)

        Args:
            raw_signal: Raw ECG data (numpy array)
            sampling_rate: Sampling rate (Hz)

        Returns:
            cleaned_signal: Cleaned signal
        """
        self.status = "Processing..."
        self.processing_log = []

        try:
            # 1. Remove baseline wander (highpass filter, cutoff 0.5 Hz)
            sos_high = signal.butter(4, 0.5, 'highpass', fs=sampling_rate, output='sos')
            signal_high = signal.sosfilt(sos_high, raw_signal)
            self.processing_log.append("[OK] Baseline wander removed (highpass 0.5 Hz)")

            # 2. Remove high-frequency noise (lowpass filter, cutoff 40 Hz)
            sos_low = signal.butter(4, 40, 'lowpass', fs=sampling_rate, output='sos')
            signal_filtered = signal.sosfilt(sos_low, signal_high)
            self.processing_log.append("[OK] High-frequency noise removed (lowpass 40 Hz)")

            # 3. Remove power line interference (notch filter, 50 Hz)
            b_notch, a_notch = signal.iirnotch(50, 30, sampling_rate)
            cleaned_signal = signal.filtfilt(b_notch, a_notch, signal_filtered)
            self.processing_log.append("[OK] Power line interference removed (notch 50 Hz)")
            
            self.status = "Done"

            return cleaned_signal
        except Exception as e:
            self.status = f"Error: {str(e)}"
            self.processing_log.append(f"[FAIL] Processing failed: {str(e)}")
            raise
```
This snippet shows the core filtering pipeline within `Agent 1`. It uses `scipy.signal` to apply a series of filters sequentially, logging the outcome of each step. This approach makes the process transparent and easy to debug.

## Results

The system provides a comprehensive, real-time analysis of driver fatigue. When a user uploads an ECG file, the application interface displays:
*   **Signal Visualization:** Plots of both the raw and filtered ECG signals.
*   **Physiological Metrics:** Key features like Heart Rate, SDNN, and RMSSD are displayed.
*   **Fatigue Risk Gauge:** A visual gauge indicating the calculated risk score (0-100).
*   **AI Analysis Report:** A detailed text report generated by the Decision Agent, summarizing its findings and the inputs from the MCP tools.
*   **Actionable Alerts:** If the risk level is determined to be "High" or "Very High," a prominent warning is displayed along with recommendations, such as taking a break and locations of nearby rest areas provided by the `Rest Area Query` tool.

Testing with the synthetic dataset yielded the expected results:
*   `ecg_normal.csv` resulted in a "Low" risk assessment.
*   `ecg_drowsy.csv` triggered a "High" risk warning.
*   `ecg_long_drowsy.csv` resulted in a "Very High" risk assessment due to the sustained indicators of fatigue.

The entire pipeline, from file upload to final report, executes in **under 2 seconds**, highlighting the efficiency of the automated agentic workflow.

## Discussion

### Agentic vs. Chat-Based Approach

A key insight from this project came from comparing our automated agentic system to a manual, chat-based workflow (e.g., using ChatGPT-4). To perform the same analysis with a chat-based tool, it took approximately 40 minutes, involved over 12 manual steps of copying and debugging code, and required repeatedly providing context. Our agent system accomplishes the same task in seconds with a single user action. This starkly illustrates the power of agentic AI for automating complex, repetitive, and structured workflows. The agent maintains state, integrates directly with tools (code libraries), and follows a reliable, predefined process, eliminating the friction and potential for error inherent in a manual approach.

### Challenges and Limitations

1.  **Agent Architecture Design:** The initial design of the agent responsibilities required iteration. Deciding on the precise boundaries—what Agent 1 should do versus Agent 2—was a key challenge in ensuring a logical and efficient workflow.
2.  **Git Workflow:** As our team was new to formal version control, managing branches, commits, and merges presented a learning curve that initially added friction to the development process.
3.  **Synthetic Data:** While necessary for this project, the synthetic data is a limitation. It does not capture the full complexity and variability of real-world ECG signals, which can be affected by individual physiology, motion artifacts, and various other factors. The model's performance on real-world data is untested.

### Lessons Learned

*   **The Power of Agentic AI:** This project was a powerful demonstration of how instructing an AI agent to complete a task is a new, effective paradigm. Agentic AI excels at automating workflows, providing speed and consistency that is impossible to achieve manually.
*   **Value of MCP Tools:** The integration of MCP tools was critical for adding contextual intelligence. Factors like time of day and driving duration are crucial for accurate fatigue assessment, and the agent's ability to automatically query these tools adds significant value over a simple physiological analysis.
*   **Hybrid AI Approaches:** We learned that different AI interaction models have different strengths. Agentic AI is ideal for production-style, repeatable tasks, while chat-based AI is superior for exploration, brainstorming, and learning.

## Conclusion

This project successfully demonstrates the design and implementation of an agent-based system for detecting driver drowsiness from ECG signals. By breaking the problem down into a pipeline of specialized agents, we created an automated, efficient, and intelligent solution. The system effectively integrates signal processing, feature extraction, and a multi-factor decision-making process that leverages both physiological data and external context via MCP tools.

While the current system relies on synthetic data, it establishes a robust and scalable architecture. Future work should focus on:
1.  **Integration with Real ECG Hardware:** Adapting the system to process data streamed in real-time from a wearable ECG sensor.
2.  **Model Personalization:** Implementing a mechanism for the system to learn a driver's unique physiological baseline over time to improve the accuracy of fatigue detection.
3.  **Real-World Data Validation:** Training and validating the decision models on a large and diverse dataset of real-world driving ECG recordings.

This project serves as a strong proof-of-concept for how agentic AI can be applied to solve complex problems in physiological monitoring and real-world safety applications.

## References

1.  Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, 32(3), 230-236.
2.  Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065.
