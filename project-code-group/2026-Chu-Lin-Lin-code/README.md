# Agentic ECG HRV Baseline Evaluation System

An **HRV-based stress classifier** that extracts Heart Rate Variability features from raw ECG signals and evaluates stress vs baseline states through dataset analysis.

**Group:** 2026-Chu-Lin-Lin
**Authors:** CHU,YEN-CHIEH, LIN,CHIH-YI ,LIN,WEN-HSIN
**License:** Apache-2.0

---

## Learning Objectives

After studying this example, students will understand:

1. **ECG Signal Processing**  
   Bandpass filtering and R-peak detection from raw ECG signals.

2. **HRV Feature Extraction**  
   Extraction of time-domain, frequency-domain, and non-linear HRV metrics from RR intervals.

3. **Rule-Based Physiological Evaluation**  
   How physiological rules and personalized baselines can be used to evaluate ECG recordings without model training.

4. **Agentic System Design**  
   How an agent orchestrates multiple analysis steps (loading, processing, feature extraction, evaluation) and makes decisions based on intermediate results.

5. **Dataset-Level Analysis**  
   How to process and evaluate entire ECG datasets across multiple subjects and conditions using adaptive window-based analysis.

---

## Description

This project implements an **agentic, rule-based ECG analysis system** for evaluating heart rate variability (HRV) across multiple subjects and physiological conditions.

- **Input**: Raw ECG recordings stored as CSV files, organized by subject and condition (e.g., Rest, Active).
- **Features**: HRV metrics extracted from RR intervals, including time-domain, frequency-domain, and non-linear features.
- **Decision Logic**:  
  Instead of training a machine learning classifier, the system applies **personalized baseline rules** to determine whether each ECG segment is physiologically consistent with the subject’s baseline.
- **Output**:  
  Dataset-level evaluation results, including per-file pass rates, baseline statistics, and detailed per-window evaluations.
- **Windowing Strategy**:  
  A duration-adaptive approach is used:
  - Recordings shorter than the window size are evaluated as a single window.
  - Longer recordings are evaluated using overlapping sliding windows with aggregated pass rates.
---
### Agent Orchestration

The `HRVAnalysisOrchestrator` coordinates the dataset analysis pipeline:

1. **ECG Loading**  
   Loads ECG data from CSV files according to the dataset configuration.

2. **Signal Processing**  
   Applies bandpass filtering and detects R-peaks to compute RR intervals.

3. **Feature Extraction**  
   Computes HRV metrics for each evaluation window.

4. **Baseline Construction**  
   Builds personalized baselines for each subject and physiological condition.

5. **Window and File Evaluation**  
   Evaluates each window against its corresponding baseline and aggregates results at the file level.

6. **Result Generation**  
   Saves baseline statistics, pass rates, and detailed evaluation outputs for further analysis and visualization.

This design emphasizes **agentic decision-making and physiological interpretability**, enabling robust ECG evaluation without requiring model training or labeled stress annotations.

```
┌─────────────────┐
│  HRV Analysis   │
│   Orchestrator  │
│  (coordinates)  │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┬────────────┬─────────┐
    ▼         ▼          ▼          ▼            ▼         ▼
┌───────┐ ┌───────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌───────────┐
│  Load │→│Process│→│Extract  │→│ Build   │→│ Evaluate │→│ Save      │
│  ECG  │ │ Signal│ │ Features│ │ Baseline│ │ Files    │ │ Results   │
└───────┘ └───────┘ └─────────┘ └─────────┘ └──────────┘ └───────────┘
```

---

## Requirements

- Python 3.11+

---

## Installation

### Option 1: Using Standard `venv` + `pip`

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```
---
### Option 2: Using `uv` (Recommended - Faster but less established)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral-sh/uv/install.sh | sh

# Or with Homebrew (macOS)
brew install uv
```

Then set up the project:

```bash
# Create virtual environment with uv
uv venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies with uv (much faster than pip)
uv pip install -r requirements.txt
```

---

## Dataset Setup

1.Expected directory structure:
    data-group/data/
    ├── first_person/
    │ ├── Rest/
    │ │ ├── *.csv
    │ └── Active/
    │ ├── *.csv
    │
    └── second_person/
    ├── Rest/
    │ ├── *.csv
    └── Active/
    ├── *.csv


Each CSV file contains a single ECG recording with associated timestamps and sensor values.  
Recordings are organized by **subject** (`first_person`, `second_person`) and **physiological condition** (`Rest`, `Active`).

For additional details about the dataset format and recording protocol, see  
`../../data-group/data/README.md`.

---

## Usage

### Quick Reference: uv vs venv

| Scenario | Recommendation |
|----------|----------------|
| One-off run / testing | `uv run --with` (ephemeral, no setup) |
| Repeated development | Persistent venv (faster subsequent runs) |
| CI/CD pipelines | `uv run` or lockfile-based venv |

**All scripts require these dependencies** (via `src.tools` package imports):
`numpy`, `scipy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `reportlab`

**Additional dependencies by script:**
| Script | Extra Dependencies | Needs API Key |
|--------|-------------------|---------------|
| `analyze_subjects.py` | `pandas` | No |
| `visualize_ecg_conditions.py` | `numpy`, `scipy`, `matplotlib` | No |
| `visualize_feature_conditions.py` | `numpy`, `scipy`, `matplotlib` | No |
| `calculate_value.py` | `pandas`, `pyyaml` | No |
| `run_analysis.py` | `pandas`, `pyyaml` | No |

Note: All scripts import from `src.tools`, which loads all tool modules including those requiring `matplotlib` and `reportlab`.

---

### Step-by-Step Analysis Guide

#### Step 1 — Calculate values / model-related preparation
```bash
python scripts/calculate_value.py
```
Purpose:
This step computes ECG/HRV-related values and prepares intermediate results required by later stages of the analysis. (This script was previously named train_classifier.py.)

#### Step 2 — Run the main agentic analysis
```bash
python scripts/run_analysis.py
```
Purpose:
This script performs the core agentic ECG analysis, including:
•	ECG signal processing
•	R-peak detection and RR interval extraction
•	HRV feature computation
•	Dataset-level evaluation and baseline-related analysis
The orchestrator coordinates these steps and represents the Agent Decision Module of the system.

#### Step 3 — Visualize ECG signals by condition
```bash
python scripts/visualize_ecg_conditions.py
```
Purpose:
This script visualizes ECG signals under different physiological conditions (e.g., Rest vs. Active) to verify signal quality and observe waveform differences.

#### Step 4 — Visualize HRV feature distributions
```bash
python scripts/visualize_feature_conditions.py
```
Purpose:
This script visualizes extracted HRV features across conditions and individuals, supporting interpretation and comparison of physiological states.

---

### Python API

```python
from src.orchestrator import HRVAnalysisOrchestrator
from src.utils import load_config
from pathlib import Path

# Load configuration
config_path = Path("config/config.yaml")
config = load_config(str(config_path))

# Instantiate and run the orchestrator for dataset analysis
orchestrator = HRVAnalysisOrchestrator()
result = orchestrator.run_dataset(config)

print(f"Dataset analysis complete: {result['status']}")
print(f"Output directory: {result['outdir']}")
print(f"Number of files processed: {result['n_files']}")
```

---

## Feature Extraction and Windowing Strategy

ECG recordings in this project have **variable durations** (e.g., ~120 seconds to ~350 seconds).  
To handle both short and long recordings consistently, the system applies a **duration-adaptive windowing strategy**.

### Windowing Parameters

| Parameter | Value |
|----------|-------|
| Window size | 180 seconds |
| Overlap | 50% (90-second stride) |
| Minimum heartbeats | 30 beats per window |

---

### Behavior Based on Signal Duration

| Signal Duration | System Behavior |
|----------------|-----------------|
| < 180 seconds | Single-window evaluation over the entire signal |
| ≥ 180 seconds | Sliding-window evaluation with aggregation |

---

### For Signals Shorter Than 180 Seconds

- The entire recording is treated as **one evaluation window**
- HRV features are extracted once
- A single **pass/fail decision** is produced based on personalized baseline rules
- If fewer than **30 detected heartbeats** are found, the recording is skipped

---

### For Signals 180 Seconds or Longer

- HRV features are extracted every **90 seconds**, each using a **180-second ECG segment**
- Each window produces:
  - A feature vector (HRV metrics)
  - An independent **pass/fail evaluation** (baseline consistency check)
- Windows with fewer than **30 detected heartbeats** are excluded for reliability
- **Aggregation**:
  - The final `pass_rate` is computed as `n_pass / n_windows` over valid windows
  - Both per-window details and aggregated results are reported

This design allows the agent to evaluate physiological stability over time while remaining robust to transient noise or motion artifacts.



---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run comprehensive tests only (85 tests covering all parameters/classifiers)
python -m pytest tests/test_comprehensive.py -v

# Generate test report (runs pytest and creates markdown report)
python tests/generate_test_report.py
```
---

### Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_tools.py` | ~20 | Unit tests for core tool modules |
| `test_classifier.py` | ~30 | Tests for all 20 classifiers (training + save/load) |
| `test_extended_features.py` | ~15 | Tests for HRV feature extraction functions |
| `test_orchestrator.py` | ~10 | Tests for orchestrator dataset analysis |
| `test_comprehensive.py` | **85** | All parameter variations, all classifiers, signal processor |
| `test_helpers.py` | ~30 | Tests for utility functions (logging, config, validation) |
| `test_report_generator.py` | ~12 | Tests for report generator non-visual functions |

---

### Test Report Generation

The `generate_test_report.py` script:
- Runs pytest and collects results
- Generates documented test cases
- Creates markdown report following the required format
- Output: `reports/2026-Chu-Lin-Lin.md`

```bash
# Generate the test report
python tests/generate_test_report.py

# View the report
cat reports/2026-Chu-Lin-Lin.md
```

---

## Project Structure

```
2026-Chu-Lin-Lin-code/
├── README.md
├── TROUBLESHOOTING.md           # Solutions to common issues
├── requirements.txt
├── env.example
├── config/
│   └── config.yaml              # HRV analysis configuration
├── src/
│   ├── __init__.py
│   ├── orchestrator.py          # HRV analysis dataset orchestrator
│   ├── tools/
│   │   ├── __init__.py          # Exports all tool functions
│   │   ├── ecg_loader.py        # WESAD pickle + text file loading
│   │   ├── signal_processor.py  # Bandpass filter, R-peak detection
│   │   ├── feature_extractor.py # Basic HRV features
│   │   ├── extended_features.py # 20 comprehensive HRV features
│   │   ├── classifier.py        # 20 classifiers with selection API
│   │   └── report_generator.py  # PDF report with visualizations
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Logging, config, validation utilities
├── scripts/
│   ├── run_analysis.py          # CLI for dataset analysis
│   ├── calculate_value.py       # Threshold calibration tool
│   ├── visualize_ecg_conditions.py  # ECG + condition label plots
│   ├── visualize_feature_conditions.py  # HRV features comparison
│   └── analyze_subjects.py      # Summarize pass rates
├── models/                          # Trained models (after training)
│   ├── logistic_regression.joblib   # Example trained model
│   └── ...                          # Other trained models
├── reports/                     # Generated reports and figures (not tracked in git)
│   └── ...                          # Output files from analysis scripts
└── tests/
    ├── __init__.py
    ├── test_tools.py            # Unit tests for core tool modules
    ├── test_classifier.py       # Tests for 20 classifiers
    ├── test_extended_features.py    # Tests for HRV feature extraction
    ├── test_orchestrator.py     # Tests for orchestrator dataset analysis
    ├── test_comprehensive.py    # Comprehensive tests for individual tools
    ├── test_helpers.py          # Tests for utility functions
    ├── test_report_generator.py # Tests for report generator
    └── generate_test_report.py  # Generates markdown test report
```

---

## Known Issues and Limitations

The system may not function as expected under the following conditions:

| Issue | Impact | Workaround |
|------|--------|------------|
| **Short ECG recordings (< 60 seconds)** | Insufficient RR intervals for stable baseline estimation and HRV metrics | Use recordings ≥ 60s (≥ 180s recommended for frequency-domain features) |
| **Severe motion artifacts** | R-peak detection may fail or produce unstable RR intervals | Avoid excessive body movement or apply additional signal preprocessing |
| **Flat or saturated ECG signals** | Zero or near-zero variance leads to invalid HRV calculations | Verify sensor placement and recording hardware |
| **Highly non-stationary signals** | Baseline assumptions may be violated across windows | Use condition-specific baselines (Rest vs Active) |
| **Very long recordings (> 1 hour)** | Increased memory usage and processing time | Split recordings into smaller segments before analysis |
| **Non-ECG physiological signals** | HRV metrics and baseline rules become invalid | Ensure input data corresponds to ECG signals only |

**Minimum Input Requirements**

- **ECG duration**: ≥ 60 seconds (≥ 180 seconds recommended for stable frequency-domain analysis)
- **Sampling rate**: 30–250 Hz (validated at 50 Hz in this project)
- **Signal quality**: Clearly identifiable R-peaks (QRS complexes) across most windows


---

## Evaluation Results

This section reports the dataset-level evaluation results of the proposed agentic ECG analysis system.  
Each ECG recording is segmented into overlapping sliding windows, and each window is evaluated against a personalized baseline.  
The pass rate is defined as the ratio of windows that satisfy physiological and baseline consistency criteria.

### Evaluation Setup

- **Subjects**: 2 (first_person, second_person)
- **Conditions**: Rest, Active
- **Sampling Rate**: 50 Hz
- **Windowing Strategy**: Fixed-length sliding windows with overlap
- **Metric**:  
  - `n_windows`: total number of windows  
  - `n_pass`: number of windows passing baseline checks  
  - `pass_rate = n_pass / n_windows`

---

### First Person — Active State

| Activity | n_windows | n_pass | Pass Rate |
|--------|----------:|-------:|----------:|
| Rotate | 5 | 0 | 0.00 |
| Bike (Level 5) | 19 | 17 | 0.895 |
| Bike (Level 3) | 17 | 16 | 0.941 |
| Bike (Level 1) | 21 | 21 | 1.00 |

**Observation**  
Most cycling activities achieve high pass rates, indicating that the personalized Active baseline generalizes well across different exercise intensities.  
However, the rotation activity completely fails the baseline criteria, suggesting severe motion artifacts or physiological deviations that fall outside the subject’s personalized baseline.

---

### First Person — Rest State

| Activity | n_windows | n_pass | Pass Rate |
|--------|----------:|-------:|----------:|
| Speak | 5 | 5 | 1.00 |
| Static (Level 1) | 5 | 5 | 1.00 |
| Static (Level 3) | 7 | 7 | 1.00 |
| Static (Level 5) | 5 | 5 | 1.00 |

**Observation**  
All resting-related activities achieve perfect pass rates, indicating a stable and well-defined Rest baseline for this subject.

---

### Second Person — Active State

| Activity | n_windows | n_pass | Pass Rate |
|--------|----------:|-------:|----------:|
| Bike (Level 1) | 19 | 18 | 0.947 |
| Bike (Level 5) | 17 | 17 | 1.00 |
| Bike (Level 3) | 19 | 19 | 1.00 |
| Rotate | 7 | 7 | 1.00 |

**Observation**  
The Active baseline for the second subject shows strong robustness across all tested activities, including rotation, highlighting inter-subject differences in physiological response and motion tolerance.

---

### Second Person — Rest State

| Activity | n_windows | n_pass | Pass Rate |
|--------|----------:|-------:|----------:|
| Static (Level 1) | 5 | 3 | 0.60 |
| Speak | 5 | 5 | 1.00 |
| Static (Level 3) | 5 | 5 | 1.00 |
| Static (Level 5) | 5 | 5 | 1.00 |

**Observation**  
One low-intensity static condition shows a reduced pass rate, suggesting that subtle physiological fluctuations or baseline boundary sensitivity may occur even during nominal resting states.

---

### Cross-Subject Insights

- Rest states generally exhibit higher stability and pass rates than Active states.
- The same activity (e.g., rotation) may yield significantly different outcomes across subjects, emphasizing the importance of individualized baselines.
- The agentic decision logic successfully adapts evaluation criteria based on subject-specific and condition-specific baselines rather than relying on global thresholds.

### Discussion

These results demonstrate that the proposed system can effectively construct personalized HRV baselines and evaluate ECG recordings across multiple physiological states.  
The agentic design enables adaptive, rule-based decision-making that captures inter-subject variability and condition-dependent dynamics, supporting long-term and multi-condition physiological monitoring without requiring model retraining.


---

## References

1.Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology.Heart rate variability: standards of measurement, physiological interpretation and clinical use.Circulation. 1996;93(5):1043–1065.

2.Shaffer F, Ginsberg JP.An overview of heart rate variability metrics and norms.Front Public Health. 2017;5:258. doi:10.3389/fpubh.2017.00258.

3.Behar J, Johnson AEW, Clifford GD, Oster J.ECG signal quality during arrhythmia and its application to false alarm reduction.IEEE Trans Biomed Eng. 2013 Jun;60(6):1660–1666. doi:10.1109/TBME.2013.2240452.
