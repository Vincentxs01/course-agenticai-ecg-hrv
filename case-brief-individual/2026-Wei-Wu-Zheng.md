# Case Brief: Intelligent Driver Drowsiness Detection System using ECG and Agentic AI

**Authors:** 2026-Wei-Wu-Zheng (Wei JungYing, Wu KunChe, Zheng YiKai)
**License:** CC-BY-4.0

## Problem Statement

Driver fatigue contributes to thousands of traffic accidents globally each year. During long drives, a driver's mental state can subtly shift from alert to drowsy without immediate awareness, significantly increasing accident risk due to delayed reactions and impaired decision-making. The core challenge is the inability to accurately distinguish these states in real-time, complicated by environmental noise and the lack of robust automated monitoring systems.

## Context/Background

Current drowsiness detection systems primarily rely on camera-based eye-tracking or steering pattern analysis, which have significant limitations. Camera systems fail in poor lighting and cannot detect drowsiness when eyes remain open but cognitive alertness decreases. Steering-based systems only detect drowsiness after dangerous patterns emerge.

Our proposed solution utilizes Electrocardiogram (ECG) data from wearable devices integrated into vehicles. ECG-based detection offers continuous, non-invasive monitoring of physiological changes, particularly heart rate (HR) and heart rate variability (HRV). Drowsy states are characterized by decreased heart rate and increased HRV due to heightened parasympathetic activity.

However, the driving environment introduces specific challenges. Motion artifacts contaminate ECG signals from speaking (EMG noise), head movement (baseline wander), and vehicle vibration (mechanical noise). Existing systems struggle to filter these artifacts effectively, leading to high false alarm rates or missed detections.

## Analysis

### Root Causes

1. **Physiological invisibility:** The alert-to-drowsy transition occurs internally through autonomic nervous system changes, invisible to camera-based systems but detectable through ECG.

2. **Signal contamination:** Natural driver behaviors create noise that mimics or masks cardiac features needed for detection.

3. **Individual variability:** Each person has unique baseline cardiac patterns. Population-level thresholds fail to account for this variability.

### Constraints

- **Real-time processing:** Analysis must occur within seconds for timely intervention
- **Non-invasive deployment:** Wearable devices must be comfortable and not interfere with driving
- **Noise resilience:** Must differentiate true drowsiness signals from motion artifacts
- **Personalization:** Must adapt to individual baselines without extensive calibration
- **Fail-safe design:** Minimize false alarms while avoiding missed detections

### Requirements

- Robust signal processing with automatic artifact filtering
- Accurate state classification: Alert (higher HR, lower HRV) vs. Drowsy (lower HR, higher HRV)
- Personalized baseline learning during first 10-15 minutes of driving
- Multi-modal integration of contextual factors (time, weather, driving duration)
- Immediate intervention through automated alert system

## Proposed Approach (Chat-based)

In a traditional chat-based AI workflow, the process is retrospective and manual:

1. **Collect:** Driver records ECG data locally during driving
2. **Upload and Prompt:** User uploads CSV file to chat AI with prompt: "Analyze this ECG data, remove noise from head movement, calculate HRV, and determine if driver appears drowsy"
3. **Analyze:** Chat AI generates Python code to filter signal and calculate metrics
4. **Recommend:** AI outputs text summary of drowsiness assessment

**Limitations:** This approach is unsuitable for driving safety. It requires manual data collection and upload, operates retrospectively, and cannot function as an immediate intervention system.

## Proposed Approach (Agentic)

We propose a multi-agent AI architecture for autonomous real-time decision-making:

**Agent 1: Signal Processing Agent**

Continuously monitors raw ECG stream and employs adaptive filtering: band-pass filtering (0.5-40 Hz), notch filtering (50/60 Hz), wavelet transform for artifact suppression, and baseline wander correction. Outputs clean, normalized ECG signal.

**Agent 2: Feature Extraction Agent**

Performs cardiac analysis: R-peak detection, RR interval calculation, and computation of time-domain HRV metrics (SDNN, RMSSD, mean HR). Outputs structured feature vector with confidence scores.

**Agent 3: Decision Agent with MCP Tool Integration**

Integrates physiological features with contextual information through Model Context Protocol (MCP) tools. Compares metrics against personalized baseline and queries weather conditions, time-of-day risk, and medical knowledge base as needed. Uses Claude Sonnet 4 as reasoning engine, which autonomously decides which MCP tools to invoke.

**Decision Logic:**
```
IF (HRV > personalized_threshold AND HR < personalized_threshold AND contextual_risk_factors_present)
THEN State = Drowsy
```

**Action:** Upon detecting drowsiness while vehicle is in motion, autonomously triggers visual alerts, auditory alarm, haptic feedback, and incident logging.

The key innovation is that this agent uses AI reasoning to weigh multiple factors and make nuanced decisions, rather than following rigid rules.

## Expected Outcomes

- **Detection accuracy:** Reliable classification of alert versus drowsy states
- **False positive reduction:** Minimized false alarm rate through personalized baseline learning
- **Response time:** Detection within 15-30 seconds of drowsiness onset
- **Noise immunity:** Robust performance despite speech, head movement, and vehicle vibration
- **Personalization:** Automatic adaptation to individual cardiac patterns
- **Multi-modal intelligence:** Enhanced accuracy through integration of physiological, environmental, and temporal factors
- **Accident prevention:** Timely warnings to prevent drowsiness-related accidents

## References

1. Sahayadhas A, Sundaraj K, Murugappan M. Detecting driver drowsiness based on sensors: a review. Sensors (Basel). 2012;12(12):16937-53. doi: 10.3390/s121216937. PMID: 23223151.
2. Vicente J, Laguna P, Bartra A, Bailón R. Drowsiness detection using heart rate variability. Med Biol Eng Comput. 2016 Jun;54(6):927-37. doi: 10.1007/s11517-015-1448-7. PMID: 26780463.
3. Anthropic. Model Context Protocol (MCP). url: https://modelcontextprotocol.io/ (Accessed 2026-01-15)
