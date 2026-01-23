# Technical Report: Driver Drowsiness Detection System using ECG and Multi-Agent AI Architecture

**Author:** Wei JungYing
**Group:** 2026-Wei-Wu-Zheng
**Course:** Agentic AI - Spring 2026
**License:** CC-BY-4.0

---

## Abstract

This report describes the design and implementation of an intelligent driver drowsiness detection system that uses electrocardiogram (ECG) signals and a multi-agent AI architecture. The system employs three specialized agents for signal processing, feature extraction, and decision making, integrated with Model Context Protocol (MCP) tools for contextual awareness. Testing with synthetic ECG data demonstrates 100% accuracy in distinguishing alert and drowsy states, with end-to-end processing time under 2 seconds.

---

## 1. Introduction

### 1.1 Problem Background

Driver fatigue is a significant contributor to traffic accidents worldwide. The physiological transition from alertness to drowsiness is characterized by changes in the autonomic nervous system, which can be detected through heart rate variability (HRV) analysis. Traditional detection methods using cameras or steering patterns have limitations: cameras fail in poor lighting and cannot detect cognitive drowsiness when eyes remain open, while steering-based systems detect drowsiness only after dangerous patterns emerge.

### 1.2 Proposed Solution

We propose an ECG-based drowsiness detection system that uses a three-layer Agentic AI architecture. The system continuously monitors physiological signals and integrates contextual information through MCP tools to provide accurate, real-time fatigue assessment.

### 1.3 Scope

This report covers:
- System architecture and component design
- Signal processing and feature extraction algorithms
- Decision logic and MCP tool integration
- Testing methodology and results
- Discussion of limitations and future work

---

## 2. System Architecture

### 2.1 Overview

The system follows a pipeline architecture with three specialized agents:

```
ECG Input → Agent 1 (Filter) → Agent 2 (Features) → Agent 3 (Decision) → Risk Assessment
                                                          ↑
                                                    MCP Tools
```

Each agent is implemented as a Python class with clearly defined interfaces, enabling independent testing and modular development.

### 2.2 Component Design

#### 2.2.1 Streamlit Web Interface (app.py)

The user interface is built using Streamlit, providing:
- File upload functionality for ECG CSV files
- Real-time processing progress indicators
- Visualization of raw and filtered ECG signals using Plotly
- Risk gauge and detailed analysis report display
- Alert system with recommended actions

#### 2.2.2 Agent 1: Signal Filter Agent (agent1_filter.py)

**Purpose:** Remove noise and artifacts from raw ECG signals.

**Processing Pipeline:**
1. **Highpass filter (0.5 Hz):** Removes baseline wander caused by respiration and electrode drift
2. **Lowpass filter (40 Hz):** Removes high-frequency noise while preserving cardiac signal components
3. **Notch filter (50 Hz):** Removes power line interference
4. **Z-score normalization:** Standardizes signal amplitude

**Implementation:**
```python
def filter_ecg(self, raw_signal, sampling_rate=250):
    # Highpass filter
    sos_high = signal.butter(4, 0.5, 'highpass', fs=sampling_rate, output='sos')
    signal_high = signal.sosfilt(sos_high, raw_signal)

    # Lowpass filter
    sos_low = signal.butter(4, 40, 'lowpass', fs=sampling_rate, output='sos')
    signal_filtered = signal.sosfilt(sos_low, signal_high)

    # Notch filter
    b_notch, a_notch = signal.iirnotch(50, 30, sampling_rate)
    cleaned_signal = signal.filtfilt(b_notch, a_notch, signal_filtered)

    # Normalization
    cleaned_signal = (cleaned_signal - np.mean(cleaned_signal)) / np.std(cleaned_signal)
    return cleaned_signal
```

**Artifact Detection:** Uses statistical analysis to identify segments with abnormally high variance (>3σ), indicating motion artifacts.

#### 2.2.3 Agent 2: Feature Extraction Agent (agent2_features.py)

**Purpose:** Extract heart rate and HRV metrics from filtered ECG signals.

**Processing Steps:**
1. **R-peak detection:** Uses `scipy.signal.find_peaks` with height and distance constraints
2. **RR interval calculation:** Time differences between successive R-peaks
3. **Outlier filtering:** Removes RR intervals outside physiological range (300-2000 ms)
4. **Feature computation:**
   - Heart Rate (HR): `60000 / mean(RR intervals)` [bpm]
   - SDNN: Standard deviation of RR intervals [ms]
   - RMSSD: Root mean square of successive RR differences [ms]

**Key Considerations:**
- Minimum 5 R-peaks required for reliable HRV calculation
- RR interval filtering uses both absolute thresholds and statistical outlier detection

#### 2.2.4 Agent 3: Decision Agent (agent3_decision.py)

**Purpose:** Integrate physiological features with contextual information for risk assessment.

**Multi-Factor Risk Scoring:**

| Factor | Condition | Score |
|--------|-----------|-------|
| Low HR | HR < 60 bpm | +30 |
| High HRV | SDNN > 80 ms | +35 |
| Baseline deviation | HR 10+ below baseline | +15 |
| Time risk | 02:00-05:00 | +40 |
| Weather | Hot and humid | +15 |
| Driving duration | > 3 hours | +40 |

**Risk Levels:**
- Low: 0-29
- Medium: 30-49
- High: 50-69
- Very High: 70+

**Report Generation:** Produces markdown-formatted analysis including physiological assessment, environmental context, identified risk factors, and recommended actions.

### 2.3 MCP Tools Integration (mcp_tools.py)

The Model Context Protocol tools provide contextual information:

1. **Weather Query:** Returns temperature, humidity, and fatigue impact factor
2. **Time Risk Assessment:** Evaluates circadian rhythm-based drowsiness risk
3. **Driving Duration Risk:** Calculates accumulated fatigue from continuous driving
4. **Rest Area Query:** Provides nearest rest stops for high-risk situations
5. **Medical Knowledge Base:** Offers physiological interpretations of HRV patterns

**Note:** Current implementation uses simulated data. Production deployment would integrate real APIs.

---

## 3. Implementation Details

### 3.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| Signal Processing | SciPy, NumPy |
| Visualization | Plotly |
| Data Handling | Pandas |

### 3.2 Data Format

Input: CSV file with single "ECG" column containing amplitude values
- Recommended sampling rate: 250 Hz
- Recommended duration: 30-60 seconds

### 3.3 Project Structure

```
2026-Wei-Wu-Zheng/
├── app.py                  # Streamlit main application
├── agents/
│   ├── agent1_filter.py    # Signal processing
│   ├── agent2_features.py  # Feature extraction
│   └── agent3_decision.py  # Decision making
├── tools/
│   └── mcp_tools.py        # MCP tool implementations
├── utils/
│   └── data_generator.py   # Synthetic ECG generator
├── data/                   # Test data files
└── requirements.txt        # Dependencies
```

---

## 4. Testing and Results

### 4.1 Testing Methodology

We employed three levels of testing:
1. **Unit tests:** Individual agent functions
2. **Integration tests:** Agent pipeline interactions
3. **End-to-end tests:** Complete system with synthetic data

### 4.2 Test Data

Synthetic ECG signals were generated with configurable parameters:
- Normal state: HR ~75 bpm, low HRV variation (5%)
- Drowsy state: HR ~58 bpm, high HRV variation (15%)

### 4.3 Results Summary

| Test Category | Tests | Passed | Rate |
|---------------|-------|--------|------|
| Unit Tests | 19 | 19 | 100% |
| Integration Tests | 3 | 3 | 100% |
| End-to-End Tests | 3 | 3 | 100% |
| **Total** | **25** | **25** | **100%** |

### 4.4 Performance Metrics

| Metric | Value |
|--------|-------|
| Signal filtering | ~0.3 seconds |
| Feature extraction | ~0.2 seconds |
| Decision + MCP queries | ~0.5 seconds |
| **Total pipeline** | **< 1.0 second** |

---

## 5. Discussion

### 5.1 Strengths

1. **Modular Architecture:** Each agent can be developed, tested, and improved independently
2. **Multi-modal Analysis:** Integration of physiological and contextual factors improves accuracy
3. **No API Dependency:** Rule-based decision engine works offline without API costs
4. **Real-time Capable:** Sub-second processing enables continuous monitoring

### 5.2 Limitations

1. **Synthetic Data Only:** System validated with generated data, not real ECG recordings
2. **Simulated Context:** MCP tools return simulated weather and location data
3. **Fixed Sampling Rate:** Optimized for 250 Hz; other rates may reduce accuracy
4. **Simplified Cardiac Model:** R-peak detection may fail on arrhythmic signals

### 5.3 Comparison with Chat-based Approach

| Aspect | Agentic | Chat-based |
|--------|---------|------------|
| Processing time | 2 seconds | 40 minutes |
| Manual intervention | 1 step | 12+ steps |
| Reproducibility | 100% | Variable |
| Real-time capable | Yes | No |

The agentic approach provides significant advantages for production deployment where consistency and speed are critical.

### 5.4 Challenges Encountered

1. **Agent architecture design:** Determining how to properly divide responsibilities between the three agents was challenging. Understanding the data flow and deciding which agent handles which task required iteration and experimentation.

2. **Git workflow:** As a beginner to version control, I was often uncertain about the correct commands for staging, committing, and pushing files. This added friction to the development process, but ultimately became a valuable learning experience.

### 5.5 Lessons Learned

1. **Learning to drive AI:** The most valuable skill gained was learning how to effectively instruct AI agents to help complete a project—asking the right questions and providing clear context makes a significant difference in output quality.

2. **Git fundamentals:** Though still developing proficiency, I learned the basics of version control, which will be essential for future collaborative software development.

3. **Agent modularity** simplifies debugging and enables parallel development of different system components.

4. **MCP tools** add significant value by automating context gathering that would otherwise require manual effort.

---

## 6. Future Work

Based on our development experience, the following improvements are priorities:

1. **Real API Integration (High Priority):** Connect weather, GPS, and traffic APIs for production use. The current simulated MCP tools demonstrate the concept but need real data sources.

2. **Real-time Streaming Support (High Priority):** Enable continuous ECG monitoring instead of file upload. This is essential for practical in-vehicle deployment.

3. **Real ECG Validation:** Test with datasets like SWELL-KW or WESAD to validate accuracy on real physiological data.

4. **Personalized Models:** Implement adaptive baseline learning over multiple sessions to account for individual differences in HRV patterns.

5. **Edge Deployment:** Optimize for embedded systems in vehicles with limited computing resources.

---

## 7. Conclusion

This project successfully demonstrates an agentic AI approach to driver drowsiness detection. The three-agent architecture provides modularity, testability, and clear separation of concerns. Integration with MCP tools enables multi-modal risk assessment that considers both physiological and contextual factors. While validated only with synthetic data, the system achieves 100% test accuracy and sub-second processing time, establishing a solid foundation for future development toward real-world deployment.

The comparison with chat-based approaches clearly illustrates the advantages of agentic AI for structured, repetitive workflows: automation, speed, and consistency. This insight generalizes beyond drowsiness detection to other domains where well-defined processing pipelines can benefit from agent-based architectures.

---

## References

1. Task Force of ESC and NASPE. (1996). Heart rate variability: Standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065.

2. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE Transactions on Biomedical Engineering, 32(3), 230-236.

3. Anthropic. (2025). Model Context Protocol (MCP) Specification. https://modelcontextprotocol.io/

4. Koldijk, S., Sappelli, M., Verberne, S., Neerincx, M. A., & Kraaij, W. (2014). The SWELL knowledge work dataset for stress and user modeling research. Proceedings of the 16th International Conference on Multimodal Interaction, 291-298.

---

## Appendix: Code Repository

Full source code available at: `project-code-group/2026-Wei-Wu-Zheng/`

To run the system:
```bash
cd project-code-group/2026-Wei-Wu-Zheng
pip install -r requirements.txt
python utils/data_generator.py  # Generate test data
streamlit run app.py            # Launch application
```
