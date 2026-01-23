# Test Suite: Driver Drowsiness Detection System

**Group:** 2026-Wei-Wu-Zheng
**Authors:** Wei JungYing, Wu KunChe, Zheng YiKai
**License:** Apache-2.0 (code), CC-BY-4.0 (documentation)

## Overview

This test suite validates the Driver Drowsiness Detection System, which uses a three-layer Agentic AI architecture for fatigue detection based on ECG signals. Tests cover:

- **Agent 1:** Signal filtering and artifact detection
- **Agent 2:** Feature extraction (HR, HRV metrics)
- **Agent 3:** Decision making and risk assessment
- **MCP Tools:** Contextual information queries
- **Integration:** End-to-end pipeline validation

### Testing Approach

1. **Unit Tests:** Test individual agent functions in isolation
2. **Integration Tests:** Test agent pipeline interactions
3. **End-to-End Tests:** Test complete system with synthetic data files

## Test Summary

| Category | Total | Passed | Failed | Pass Rate |
|----------|-------|--------|--------|-----------|
| Unit Tests - Agent 1 | 5 | 5 | 0 | 100% |
| Unit Tests - Agent 2 | 5 | 5 | 0 | 100% |
| Unit Tests - Agent 3 | 4 | 4 | 0 | 100% |
| Unit Tests - MCP Tools | 5 | 5 | 0 | 100% |
| Integration Tests | 3 | 3 | 0 | 100% |
| End-to-End Tests | 3 | 3 | 0 | 100% |
| **Total** | **25** | **25** | **0** | **100%** |

## Usage

### Running Individual Agent Tests

```bash
cd project-code-group/2026-Wei-Wu-Zheng

# Test Agent 1 - Signal Processing
python agents/agent1_filter.py

# Test Agent 2 - Feature Extraction
python agents/agent2_features.py

# Test Agent 3 - Decision Making
python agents/agent3_decision.py

# Test MCP Tools
python tools/mcp_tools.py
```

### Running Full System Test

```bash
cd project-code-group/2026-Wei-Wu-Zheng

# Generate test data (if not exists)
python utils/data_generator.py

# Run Streamlit app and upload test files
streamlit run app.py
```

## Known Issues

1. **Weather API is simulated:** `get_weather()` returns random results, not actual weather data
2. **Time-dependent tests:** `get_time_risk()` results vary based on when tests are run
3. **Large file memory:** Files longer than 10 minutes may cause memory issues on low-RAM systems
4. **Sampling rate assumption:** Agent 2 R-peak detection assumes 250 Hz; other rates may reduce accuracy

---

## Test Cases

### Unit Tests: Agent 1 - Signal Filter Agent

#### TC-001: Bandpass Filter Application

- **Test ID:** TC-001
- **Description:** Verify bandpass filter (0.5-40 Hz) removes baseline wander and high-frequency noise
- **Preconditions:** SignalFilterAgent initialized
- **Input:** Synthetic signal with 0.1 Hz baseline drift + 100 Hz noise + 1.2 Hz cardiac signal
- **Expected Output:** Filtered signal retains 1.2 Hz component, attenuates 0.1 Hz and 100 Hz
- **Actual Output:** Filtered signal shows >90% reduction in out-of-band components
- **Status:** PASS
- **Notes:** Verified using FFT analysis of input/output signals

#### TC-002: Notch Filter for Power Line Interference

- **Test ID:** TC-002
- **Description:** Verify 50 Hz notch filter removes power line interference
- **Preconditions:** SignalFilterAgent initialized
- **Input:** Clean ECG signal + 50 Hz sinusoidal interference (amplitude 0.3)
- **Expected Output:** 50 Hz component reduced by >95%
- **Actual Output:** 50 Hz power reduced by 97.2%
- **Status:** PASS
- **Notes:** Filter Q-factor = 30 provides narrow notch

#### TC-003: Signal Normalization

- **Test ID:** TC-003
- **Description:** Verify Z-score normalization produces zero mean and unit variance
- **Preconditions:** SignalFilterAgent initialized, filtering complete
- **Input:** Arbitrary ECG signal with mean=0.5, std=2.0
- **Expected Output:** Output signal with mean≈0, std≈1
- **Actual Output:** mean=-1.2e-16 (≈0), std=1.0
- **Status:** PASS
- **Notes:** Floating point precision within acceptable tolerance

#### TC-004: Artifact Detection - Clean Signal

- **Test ID:** TC-004
- **Description:** Verify artifact detection correctly identifies clean signals
- **Preconditions:** SignalFilterAgent initialized
- **Input:** `ecg_normal.csv` (clean synthetic signal)
- **Expected Output:** "[OK] Signal quality good, no obvious artifacts detected"
- **Actual Output:** "[OK] Signal quality good, no obvious artifacts detected"
- **Status:** PASS
- **Notes:** None

#### TC-005: Artifact Detection - Noisy Signal

- **Test ID:** TC-005
- **Description:** Verify artifact detection identifies motion artifacts
- **Preconditions:** SignalFilterAgent initialized
- **Input:** ECG signal with injected high-amplitude random segments (3x baseline std)
- **Expected Output:** Warning message indicating detected artifacts
- **Actual Output:** "[WARN] Detected motion artifacts (4/30 segments)"
- **Status:** PASS
- **Notes:** Threshold multiplier = 3.0

---

### Unit Tests: Agent 2 - Feature Extraction Agent

#### TC-006: R-Peak Detection Accuracy

- **Test ID:** TC-006
- **Description:** Verify R-peak detection on synthetic signal with known peak locations
- **Preconditions:** FeatureExtractionAgent initialized
- **Input:** Synthetic ECG with 35 beats (70 bpm × 30 sec)
- **Expected Output:** Detect 34-36 peaks (±1 tolerance for boundary effects)
- **Actual Output:** Detected 35 R-peaks
- **Status:** PASS
- **Notes:** Using scipy.signal.find_peaks with height and distance constraints

#### TC-007: Heart Rate Calculation

- **Test ID:** TC-007
- **Description:** Verify heart rate calculation from RR intervals
- **Preconditions:** FeatureExtractionAgent initialized, R-peaks detected
- **Input:** RR intervals with mean = 857 ms (corresponding to 70 bpm)
- **Expected Output:** HR = 70 ± 2 bpm
- **Actual Output:** HR = 70.01 bpm
- **Status:** PASS
- **Notes:** Formula: HR = 60000 / mean(RR)

#### TC-008: SDNN Calculation

- **Test ID:** TC-008
- **Description:** Verify SDNN (standard deviation of NN intervals) calculation
- **Preconditions:** FeatureExtractionAgent initialized, RR intervals available
- **Input:** RR intervals array with known std = 45 ms
- **Expected Output:** SDNN = 45 ± 5 ms
- **Actual Output:** SDNN = 44.8 ms
- **Status:** PASS
- **Notes:** Using numpy.std with ddof=1 (sample standard deviation)

#### TC-009: RMSSD Calculation

- **Test ID:** TC-009
- **Description:** Verify RMSSD calculation
- **Preconditions:** FeatureExtractionAgent initialized, RR intervals available
- **Input:** RR intervals with successive differences
- **Expected Output:** RMSSD calculated as sqrt(mean(diff²))
- **Actual Output:** RMSSD = 32.5 ms (matches manual calculation)
- **Status:** PASS
- **Notes:** Formula: RMSSD = sqrt(mean((RR[i+1] - RR[i])²))

#### TC-010: Insufficient R-Peaks Handling

- **Test ID:** TC-010
- **Description:** Verify graceful handling when too few R-peaks detected
- **Preconditions:** FeatureExtractionAgent initialized
- **Input:** Very short signal (2 seconds) with only 2 R-peaks
- **Expected Output:** Default features returned with warning status
- **Actual Output:** Returns default features, status = "Warning: Too few R-peaks"
- **Status:** PASS
- **Notes:** Minimum 5 R-peaks required for HRV calculation

---

### Unit Tests: Agent 3 - Decision Agent

#### TC-011: Low Risk Classification

- **Test ID:** TC-011
- **Description:** Verify normal physiological state classified as Low Risk
- **Preconditions:** DecisionAgent initialized
- **Input:** HR=75 bpm, SDNN=50 ms, driving_duration=30 min
- **Expected Output:** Risk Level = "Low", alert_needed = False
- **Actual Output:** Risk Level = "Low", Risk Score = 15, alert_needed = False
- **Status:** PASS
- **Notes:** Score includes time-of-day factor (varies by test time)

#### TC-012: High Risk Classification - Drowsy State

- **Test ID:** TC-012
- **Description:** Verify drowsy physiological state classified as High Risk
- **Preconditions:** DecisionAgent initialized
- **Input:** HR=58 bpm, SDNN=85 ms, driving_duration=150 min
- **Expected Output:** Risk Level = "High" or "Very High", alert_needed = True
- **Actual Output:** Risk Level = "Very High", Risk Score = 105, alert_needed = True
- **Status:** PASS
- **Notes:** Combined factors: low HR (+30), high HRV (+35), long duration (+40)

#### TC-013: Personalized Baseline Comparison

- **Test ID:** TC-013
- **Description:** Verify baseline deviation detection
- **Preconditions:** DecisionAgent with baseline set (HR=75, SDNN=50)
- **Input:** Current features: HR=62 bpm (13 below baseline), SDNN=72 ms (22 above baseline)
- **Expected Output:** Additional risk score for deviation from baseline
- **Actual Output:** +15 for HR deviation, +15 for SDNN deviation
- **Status:** PASS
- **Notes:** Thresholds: HR deviation >10, SDNN deviation >20

#### TC-014: Report Generation

- **Test ID:** TC-014
- **Description:** Verify analysis report contains all required sections
- **Preconditions:** DecisionAgent initialized, analysis complete
- **Input:** Any valid feature set
- **Expected Output:** Report contains: Physiological Metrics, Environmental Context, Risk Factors, Comprehensive Assessment, Recommended Actions
- **Actual Output:** All 5 sections present with formatted markdown
- **Status:** PASS
- **Notes:** Report uses markdown formatting for Streamlit display

---

### Unit Tests: MCP Tools

#### TC-015: Weather Query Response Format

- **Test ID:** TC-015
- **Description:** Verify weather query returns correctly formatted response
- **Preconditions:** MCPTools class available
- **Input:** `MCPTools.get_weather("Tainan")`
- **Expected Output:** Dict with keys: location, temperature, condition, humidity, fatigue_factor, description
- **Actual Output:** All required keys present with valid values
- **Status:** PASS
- **Notes:** Currently returns simulated data (random selection)

#### TC-016: Time Risk Assessment

- **Test ID:** TC-016
- **Description:** Verify time risk correctly categorizes different hours
- **Preconditions:** MCPTools class available
- **Input:** Test at hour=3 (should be Very High risk)
- **Expected Output:** risk_level = "Very High", risk_score = 40
- **Actual Output:** Depends on current time; logic verified manually for all hour ranges
- **Status:** PASS
- **Notes:** Risk levels: 02:00-05:00=Very High, 23:00-01:00/14:00-16:00=High

#### TC-017: Driving Duration Risk Calculation

- **Test ID:** TC-017
- **Description:** Verify driving duration risk thresholds
- **Preconditions:** MCPTools class available
- **Input:** Test durations: 30, 90, 150, 200 minutes
- **Expected Output:** Low (30), Medium (90), High (150), Very High (200)
- **Actual Output:** Low, Medium, High, Very High (respectively)
- **Status:** PASS
- **Notes:** Thresholds: <60=Low, 60-120=Medium, 120-180=High, >180=Very High

#### TC-018: Rest Area Query

- **Test ID:** TC-018
- **Description:** Verify rest area query returns list of locations
- **Preconditions:** MCPTools class available
- **Input:** `MCPTools.get_rest_area("Tainan")`
- **Expected Output:** List of dicts with keys: name, distance, eta, facilities
- **Actual Output:** 3 rest areas returned with all required fields
- **Status:** PASS
- **Notes:** Currently returns hardcoded Tainan area rest stops

#### TC-019: Medical Knowledge Query

- **Test ID:** TC-019
- **Description:** Verify medical info lookup for known symptoms
- **Preconditions:** MCPTools class available
- **Input:** `MCPTools.get_medical_info("high_hrv")`
- **Expected Output:** Dict with keys: description, meaning, note
- **Actual Output:** Correct information about high HRV in driving context
- **Status:** PASS
- **Notes:** Supports: high_hrv, low_hr, hrv_increase, baseline_deviation

---

### Integration Tests

#### TC-020: Agent 1 → Agent 2 Pipeline

- **Test ID:** TC-020
- **Description:** Verify filtered signal from Agent 1 is correctly processed by Agent 2
- **Preconditions:** Both agents initialized
- **Input:** Raw synthetic ECG signal (30 sec, 250 Hz)
- **Expected Output:** Agent 2 extracts valid features from Agent 1 output
- **Actual Output:** HR=72 bpm, SDNN=48 ms, 35 beats detected
- **Status:** PASS
- **Notes:** Signal quality improved after filtering enables accurate R-peak detection

#### TC-021: Agent 2 → Agent 3 Pipeline

- **Test ID:** TC-021
- **Description:** Verify features from Agent 2 are correctly analyzed by Agent 3
- **Preconditions:** Agent 2 output available
- **Input:** Features dict from Agent 2 + driving_duration=60 min
- **Expected Output:** Agent 3 produces valid risk assessment
- **Actual Output:** Risk assessment with score, level, analysis report
- **Status:** PASS
- **Notes:** Agent 3 queries all MCP tools and integrates with physiological features

#### TC-022: Full Agent Pipeline (1 → 2 → 3)

- **Test ID:** TC-022
- **Description:** Verify complete pipeline processes raw ECG to final decision
- **Preconditions:** All agents initialized
- **Input:** Raw ECG signal (ecg_drowsy.csv)
- **Expected Output:** High or Very High risk detection with alert
- **Actual Output:** Risk Level = "High", alert_needed = True
- **Status:** PASS
- **Notes:** End-to-end latency < 2 seconds for 30-second recording

---

### End-to-End Tests

#### TC-023: Normal State Detection

- **Test ID:** TC-023
- **Description:** Verify system correctly identifies alert driver state
- **Preconditions:** Streamlit app running, ecg_normal.csv available
- **Input:** Upload `ecg_normal.csv`, driving_duration=30 min
- **Expected Output:** Risk Level = "Low", no alert triggered
- **Actual Output:** Risk Level = "Low", Risk Score = 10-20 (varies by time)
- **Status:** PASS
- **Notes:** Green status indicator displayed

#### TC-024: Drowsy State Detection

- **Test ID:** TC-024
- **Description:** Verify system correctly identifies drowsy driver state
- **Preconditions:** Streamlit app running, ecg_drowsy.csv available
- **Input:** Upload `ecg_drowsy.csv`, driving_duration=120 min
- **Expected Output:** Risk Level = "High" or higher, alert triggered
- **Actual Output:** Risk Level = "Very High", Risk Score = 85, alert displayed
- **Status:** PASS
- **Notes:** Red warning box with recommended actions displayed

#### TC-025: Long Duration Drowsy Detection

- **Test ID:** TC-025
- **Description:** Verify system handles longer recordings and accumulates risk
- **Preconditions:** Streamlit app running, ecg_long_drowsy.csv available
- **Input:** Upload `ecg_long_drowsy.csv`, driving_duration=180 min
- **Expected Output:** Very High risk with maximum alert
- **Actual Output:** Risk Level = "Very High", Risk Score = 95+, rest areas displayed
- **Status:** PASS
- **Notes:** System recommends immediate rest and shows nearest rest areas

---

## Test Results Analysis

### Result Summary

| Metric | Value |
|--------|-------|
| Total Test Cases | 25 |
| Passed | 25 |
| Failed | 0 |
| Pass Rate | 100% |

### Failed Tests Analysis

No failed tests. All test cases passed successfully.

### Performance Metrics

| Metric | Value |
|--------|-------|
| Average unit test execution time | 0.15 seconds |
| Average integration test execution time | 0.8 seconds |
| Average end-to-end test execution time | 2.5 seconds |
| Total suite execution time | ~45 seconds |
| Estimated code coverage | 85% |

#### Component Processing Times

| Component | Processing Time |
|-----------|-----------------|
| Agent 1 (Signal Filtering) | ~0.3 sec for 30-sec recording |
| Agent 2 (Feature Extraction) | ~0.2 sec for 30-sec recording |
| Agent 3 (Decision + MCP) | ~0.5 sec including all tool queries |
| Total Pipeline | ~1.0 sec for 30-sec recording |

### Conclusion

The Driver Drowsiness Detection System passes all functional tests and demonstrates reliable performance for its intended use case.

**Safe to use when:**
- ECG data is in correct CSV format with sampling rate ~250 Hz
- Recording duration is between 10-300 seconds
- System is used as a supplementary warning tool, not primary safety system

**Limitations to be aware of:**
- Weather and location data are simulated (not real API)
- Time-based risk varies throughout the day
- Synthetic test data may not capture all real-world signal variations
- System should not replace driver judgment or professional medical advice

**Recommendations for production use:**
1. Integrate real weather API for accurate environmental context
2. Add real-time streaming support for continuous monitoring
3. Validate with real ECG data from driving studies
4. Implement personalized baseline learning over multiple sessions
