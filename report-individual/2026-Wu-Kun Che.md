# Technical Report: Driver Drowsiness Detection System using ECG and Multi-Agent AI Architecture

**Author:** Wu-Kun,Che  
**Group:** 2026-Wu-Kun,Che  
**Course:** Agentic AI - Spring 2026  
**License:** Documentation: CC-BY-4.0 (except logos). Source code: Apache-2.0.

---

## Abstract

This report presents a lightweight, end-to-end driver drowsiness detection system built on short ECG recordings. The system is delivered as a Streamlit web application and organized as a three-agent pipeline: (1) signal filtering and artifact screening, (2) feature extraction (R-peaks, heart rate, and HRV metrics), and (3) a transparent, multi-factor decision module that combines physiological evidence with contextual cues queried via MCP-style tools (time-of-day risk, weather stressors, driving duration, nearby rest areas, and a small medical knowledge base). Using synthetic ECG test cases, the pipeline achieved a 25/25 test pass rate and produced results within a couple of seconds end-to-end, making the approach practical for frequent re-analysis and as a foundation for future real-time monitoring.

---

## 1. Introduction

### 1.1 Problem Background

Driver fatigue is a major safety problem because the transition from alertness to drowsiness is often subtle and difficult for the driver to self-detect. Existing approaches (camera-based eye tracking, steering behavior, or lane deviation) can fail in poor lighting, detect fatigue too late, or require vehicle-specific calibration. Our goal is to explore a physiological alternative: ECG-based monitoring.

ECG offers three practical advantages for drowsiness monitoring:
1. **Direct physiological relevance:** It reflects autonomic nervous system activity, which can change earlier than visible behavior.
2. **Robustness:** It does not rely on camera lighting or face visibility.
3. **Continuous, low-latency processing:** With efficient signal processing, analysis can run fast enough to support near real-time decisions.

### 1.2 Proposed Solution

We build a modular, “agentic” ECG pipeline:
- **Agent 1 (Signal Filter):** Removes baseline wander, high-frequency noise, and power-line interference; flags likely motion artifacts.
- **Agent 2 (Feature Extraction):** Detects R-peaks and computes HR and HRV time-domain metrics (SDNN, RMSSD).
- **Agent 3 (Decision + MCP):** Converts features into an interpretable risk score and augments the decision with context (circadian risk window, driving duration, weather stress, rest areas, medical guidance). It then generates a structured report and recommended actions.

The user experience is intentionally simple: a single CSV upload triggers automated analysis and produces an actionable output.

### 1.3 Scope

This project focuses on:
- Short ECG segments (recommended ~30–60 seconds) sampled at 250 Hz.
- Time-domain HRV features (SDNN, RMSSD) suitable for quick estimation.
- Rule-based, transparent decisions rather than a black-box classifier.

Out of scope:
- Clinical diagnosis (e.g., arrhythmia detection) and medical-grade claims.
- Full validation on large-scale real driving datasets (future work).

---

## 2. System Architecture

### 2.1 Overview

The system is organized around a deterministic pipeline with clear boundaries:

```
Streamlit UI (CSV upload)
        |
        v
Agent 1: Signal Filter  ----->  artifact flag(s)
        |
        v
Agent 2: Feature Extraction (R-peaks, HR, HRV)
        |
        v
Agent 3: Decision + MCP Tools (risk score + context)
        |
        v
Risk level + recommendations + markdown report
```

This separation makes each stage testable in isolation and allows upgrades without rewriting the entire system.

### 2.2 Component Design

#### 2.2.1 Streamlit Web Interface (app.py)

The Streamlit UI supports a “one-click” workflow:
- Upload a CSV file (single column `"ECG"` with amplitude values).
- Visualize raw vs. filtered ECG traces (and peak markers when available).
- Display extracted metrics (HR, SDNN, RMSSD) and quality indicators (e.g., artifact warnings).
- Show a risk gauge and a human-readable report with recommended actions.

A key design choice is to show both **signals** and **decisions** so the system remains explainable and debuggable.

#### 2.2.2 Agent 1: Signal Filter Agent (agent1_filter.py)

**Purpose:** Improve ECG quality while preserving QRS morphology for peak detection.

**Filtering pipeline (250 Hz default):**
1. **High-pass (0.5 Hz), 4th-order Butterworth:** suppresses baseline wander (respiration/electrode drift).
2. **Low-pass (40 Hz), 4th-order Butterworth:** reduces high-frequency noise while retaining key ECG content.
3. **Notch (50 Hz, Q≈30):** attenuates power-line interference.
4. **Z-score normalization:** stabilizes downstream thresholding across different amplitude scales.

**Artifact handling:** The agent computes a windowed standard deviation and flags windows that exceed a high-variance threshold (typical of motion artifacts). Instead of heavily “repairing” the trace, the pipeline surfaces artifact risk so the decision agent can interpret the output more conservatively when needed.

#### 2.2.3 Agent 2: Feature Extraction Agent (agent2_features.py)

**Purpose:** Convert a cleaned ECG trace into interpretable physiological features.

**Steps:**
1. **R-peak detection:** Uses `scipy.signal.find_peaks` with (a) a dynamic amplitude threshold based on the filtered signal statistics, and (b) a minimum inter-peak distance derived from a plausible maximum HR (to avoid double counting).
2. **RR intervals:** Computed from adjacent R-peak timestamps (in milliseconds).
3. **RR filtering:** Removes implausible RR intervals (physiological bounds) and statistical outliers to prevent HRV inflation due to false peaks.
4. **Features:**
   - **Heart Rate (HR):** derived from mean RR.
   - **SDNN:** standard deviation of RR (time-domain HRV).
   - **RMSSD:** root mean square of successive RR differences.

**Reliability note:** HRV from short windows is naturally noisier than long-term HRV. The agent therefore enforces a minimum number of detected peaks before reporting HRV metrics.

#### 2.2.4 Agent 3: Decision Agent (agent3_decision.py)

**Purpose:** Combine physiology + context into an actionable risk assessment.

**Multi-factor risk scoring:** The decision agent assigns points to several factors and sums them into a total risk score (maximum 175). Key factors include:
- **Low HR:** HR < 60 bpm (up to +30)
- **High HRV:** SDNN > 80 ms (up to +35)
- **Baseline deviation:** HR ≥ 10 bpm below the user’s baseline (+15)
- **Time risk:** late-night circadian window (02:00–05:00) (up to +40)
- **Weather stress:** hot and humid conditions (up to +15)
- **Driving duration:** accumulated fatigue (up to +40; higher after ~2–3 hours continuous driving)

**Risk levels:** Low (0–29), Medium (30–49), High (50–69), Very High (70+).

**Action generation:** For elevated risk, the agent generates concrete steps (e.g., pull over, take a break, hydrate, improve cabin ventilation, and seek a rest area). The output is delivered as a markdown-formatted report so it can be rendered in the UI and reused for logging.

### 2.3 MCP Tools Integration (mcp_tools.py)

To avoid “ECG-only” decisions, Agent 3 queries several MCP-style tools:
1. **Time Risk Assessment:** circadian risk scoring (late night → higher risk).
2. **Driving Duration Risk:** accumulated fatigue based on continuous driving time.
3. **Weather Query:** temperature/humidity stress estimate (hot + humid increases fatigue probability).
4. **Rest Area Query:** a list of nearby rest stops (used when risk is High/Very High).
5. **Medical Knowledge Base:** short guidance on interpreting HR/HRV patterns.

In this prototype, these tools return simulated or rule-based outputs. The interfaces are designed so real APIs (weather, GPS) can be swapped in later with minimal code changes.

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
| Testing | pytest |

### 3.2 Data Format

Input: CSV with a single column named `"ECG"` containing amplitude values (one sample per row).  
Recommended:
- Sampling rate: **250 Hz**
- Duration: **30–60 seconds** (shorter windows reduce HRV reliability)

### 3.3 Execution Flow

1. User uploads ECG CSV via Streamlit.
2. Agent 1 filters the signal and produces a cleaned trace + artifact flags.
3. Agent 2 detects peaks and computes HR/HRV metrics.
4. Agent 3 queries MCP tools, calculates a risk score, and generates a report.
5. UI renders plots, metrics, risk gauge, and recommended actions.

---

## 4. Testing and Results

### 4.1 Testing Methodology

We used layered testing to keep failures local and debuggable:
1. **Unit tests:** validate each agent’s core functions (filters, peak detection, scoring).
2. **Integration tests:** confirm data contracts between agents (signal → peaks → features → decision).
3. **End-to-end tests:** run the full pipeline on synthetic datasets.

### 4.2 Test Data

To validate behavior without relying on private or clinical datasets, we generated synthetic ECG signals that emulate:
- **Normal/alert:** HR around 70–80 bpm with lower variability.
- **Drowsy-like:** lower HR (roughly 55–65 bpm) with higher RR variability (higher SDNN/RMSSD).

We also created a “long drowsy” scenario by combining drowsy physiology with long continuous driving and a high circadian risk window.

### 4.3 Results Summary

Representative synthetic test cases:
- `ecg_normal.csv` → predicted **Low Risk**
- `ecg_drowsy.csv` → predicted **Very High Risk**
- `ecg_long_drowsy.csv` → predicted **Very High Risk**

The overall test suite contains 25 tests and achieved a full pass rate.

### 4.4 Performance Metrics

The pipeline is designed for low latency:
- Signal processing and feature extraction run in well under a second on a typical 30-second recording.
- End-to-end latency (including context queries and report generation) stays within a couple of seconds, supporting frequent re-analysis and providing a foundation for streaming extensions.

---

## 5. Discussion

### 5.1 Strengths

1. **Explainability:** each risk contribution is visible (physiology vs. context), supporting user trust.
2. **Modularity:** the agent boundaries match engineering boundaries (filtering / features / decision).
3. **Context awareness:** MCP tools add practical driving context beyond ECG alone.
4. **Automation & reproducibility:** a single upload triggers the full workflow with consistent outputs.

### 5.2 Limitations

1. **Synthetic validation:** real-world ECG noise and arrhythmias may break peak detection.
2. **Short-window HRV:** SDNN/RMSSD from 30–60 seconds can be unstable; longer windows improve reliability but reduce responsiveness.
3. **Simulated tools:** weather, duration, and rest-area services are mock tools in this prototype.
4. **Population variability:** rule thresholds may misclassify athletes or users with atypical baselines without personalization.

### 5.3 Comparison with Chat-based Approach

A chat-based workflow can explain concepts well, but it is inefficient for repeated multi-step signal processing. In our setting, an agentic pipeline has clear advantages:
- One automated workflow vs. many manual analysis steps
- Consistent rules and repeatable outputs
- Speed suitable for near real-time monitoring

Chat-based AI remains valuable for exploratory questions (e.g., “why is SDNN high?”), while agentic AI excels at structured processing.

### 5.4 Challenges Encountered

- **Motion artifacts:** filtering must reduce noise without suppressing QRS complexes.
- **Peak detection robustness:** thresholds must handle varying amplitude even after normalization.
- **Baseline personalization:** physiological “normal” differs by person; baseline-aware scoring reduces bias.

### 5.5 Lessons Learned

1. A well-defined agent pipeline is easier to test, debug, and extend than a monolithic script.
2. Context signals (time, driving duration, environment) substantially improve practical usefulness.
3. Rule-based decision layers are effective prototypes when transparency is important and API costs should be avoided.

---

## 6. Future Work

1. **Real API integration:** replace simulated MCP tools with real weather and GPS/rest-area queries.
2. **Real-time streaming:** move from file upload to continuous ECG ingestion for in-vehicle monitoring.
3. **Real dataset validation:** evaluate on real driving or stress/fatigue datasets and quantify sensitivity/specificity.
4. **Adaptive personalization:** learn a rolling baseline over multiple sessions and adapt thresholds per user.
5. **Edge deployment:** optimize inference and UI for embedded/vehicle hardware constraints.

---

## 7. Conclusion

We implemented a three-agent ECG analysis pipeline for driver drowsiness detection and wrapped it in a Streamlit dashboard for simple user interaction. The system extracts HR and HRV time-domain features, integrates driving context through MCP-style tools, and outputs an interpretable risk level with actionable recommendations. While current validation uses synthetic ECG data, the modular architecture, automated testing, and low-latency design provide a solid foundation for future work with real data, real context services, and continuous monitoring.

---

## References

1. Task Force of ESC and NASPE. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065.  
2. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, 32(3), 230-236.  
3. Anthropic. (2025). Model Context Protocol (MCP) Specification. https://modelcontextprotocol.io/  
4. Koldijk, S., Sappelli, M., Verberne, S., Neerincx, M. A., & Kraaij, W. (2014). The SWELL knowledge work dataset for stress and user modeling research. *ICMI*, 291-298.  
5. Project slides and architecture diagrams: “2026-Wei-Wu-Zheng.pdf” and “2026-Wei-Wu-Zheng-architecture.pdf”.

---

## Appendix: Code Repository

Repository path (course structure): `project-code-group/2026-Wei-Wu-Zheng/`

Typical local run:
```bash
cd project-code-group/2026-Wei-Wu-Zheng
pip install -r requirements.txt
python utils/data_generator.py   # Generate synthetic test data
streamlit run app.py             # Launch application
```
