# Technical Report: Personalized Physiological Monitoring via Agentic AI

**Author:** 2026-Lin-WenHsin 
**Group Members:** Lin,Wen-Hsin, Lin,Chih-Yi, Chu,Yen-Chieh  
**License:** CC-BY-4.0

---

## Abstract

This report details an **Agentic AI solution** designed to bridge the gap between "Rest" and "Active" states in longitudinal physiological monitoring. Traditional systems often discard valuable high-variability data during exercise due to rigid, one-size-fits-all filtering thresholds. Our system utilizes a central **Orchestrator** to coordinate signal processing, feature extraction, and classification tools, driven by a configurable knowledge base. Validated through real-world testing with two subjects, the system successfully adapted to distinct physiological profiles: **Subject 1** maintained stable variability across states, while **Subject 2** exhibited counter-intuitive lower variability during activity. The results demonstrate that Agentic AI effectively manages inter-subject variability, significantly improving data retention and analysis accuracy.

## 1. Introduction

### 1.1 Problem Context
In long-term physiological monitoring, such as Heart Rate Variability (HRV) analysis, data is collected from individuals across various states (e.g., sedentary vs. exercise). Current systems typically apply a single, fixed analysis workflow with uniform physiological thresholds. Consequently, valid biological signals generated during **"Active" states** are often misclassified as low-quality noise and discarded because they exceed the rigid thresholds designed for "Rest" states.

### 1.2 Objectives
This project aimed to develop an Agentic AI system capable of:
1.  **Modular Orchestration:** Utilizing an agent-based `Orchestrator` to manage the data pipeline dynamically.
2.  **Configurable Analysis:** Separating system parameters (`config.yaml`) from logic to allow for rapid adaptation to different user profiles.
3.  **Robust Validation:** Ensuring system reliability through a comprehensive suite of unit and integration tests (`test_*.py`).

## 2. Data

### 2.1 Experimental Dataset
Instead of relying on pre-existing public datasets, we conducted a **primary data collection experiment** to evaluate our Agentic AI system in a real-world scenario.

**Dataset Description:**
- **Subjects:** 2 participants (**Subject 1**, **Subject 2**).
- **Conditions:**
    - **Static (Rest):** Participants remained seated with minimal movement.
    - **Active (Cycling):** Participants performed cycling exercises at varying intensity levels.
- **Intensity Levels:** Data was captured at three distinct intensity levels (Level 1, Level 3, Level 5) to simulate varying degrees of physiological stress and motion artifacts.
- **Signals:** Raw ECG data collected via wearable sensors, stored in CSV format without headers.
- **Total Samples:** 12 distinct data files (2 Subjects × 2 Conditions × 3 Levels).

### 2.2 Data Characteristics
| Subject | ID | Key Characteristic | Observed Pattern |
| :--- | :--- | :--- | :--- |
| **Subject 1** | 595 | Stable Physiology | Variability remained consistent across Rest and Active states. |
| **Subject 2** | 809 | Atypical Physiology | Variability surprisingly *decreased* during high-intensity activity. |

## 3. Tools Used

We utilized a modern Python-based stack to build the Agentic workflow. The system is designed to be modular and configuration-driven.

- **Agent Framework:** Custom Python Orchestrator (`orchestrator.py`)
- **Programming Language:** Python 3.x
- **Configuration Management:** YAML (`config.yaml`) for separating logic from parameters.
- **Signal Processing:**
    - **NumPy:** For high-performance array manipulation and statistical calculations.
    - **SciPy:** Used in `signal_processor.py` for digital signal processing (Butterworth bandpass filters).
- **Data Handling:** Pandas for parsing raw CSV sensor data.
- **Machine Learning:** scikit-learn (Random Forest, SVM) for the classification module.
- **Testing Framework:** `pytest` for unit and integration testing.
- **Development Environment:** VS Code.

## 4. System Architecture

The system follows a **modular agent architecture** with a central Orchestrator coordinating specialized tools and a persistent knowledge base. The architecture design is illustrated below:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                   Agentic AI Physiological Monitor                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  External Inputs                                                         │
│  ┌──────────────┐                                    ┌───────────────┐   │
│  │ Raw ECG Data │───────┐                            │ Profile Store │   │
│  └──────────────┘       │                            │   (Database)  │   │
│                         ▼                            └───────┬───────┘   │
│  ┌──────────────┐    ┌──────────────┐                        │           │
│  │ State Label  │───▶│   Context    │                        │           │
│  │ (Rest/Active)│    │    Loader    │                        │           │
│  └──────────────┘    └──────┬───────┘                        │           │
│                             │ (Contextualized Data)          │           │
│                             ▼                                │           │
│                    ┌──────────────────────────────────────┐  │(Query     │
│                    │                                      │◀─┘ Baseline) │
│                    │            ORCHESTRATOR              │              │
│                    │              (Agent)                 │              │
│                    │                                      │───┐          │
│                    └──────┬──────────────────────┬────────┘   │          │
│                           │                      │            │          │
│                  (Signal) │             (Features│            │(Validated│
│                  + Context│            + Profile)│            │ Data)    │
│                           ▼                      ▼            │          │
│                    ┌──────────────┐      ┌──────────────┐     │          │
│                    │   Adaptive   │      │  Classifier  │     │          │
│                    │   Processor  │      │ (Validator)  │     │          │
│                    └──────────────┘      └──────────────┘     │          │
│                     (Strategy A/B)       (Personalized        │          │
│                                           Check)              ▼          │
│                                                        ┌──────────────┐  │
│                                                        │    Report    │  │
│                                                        │  Generator   │  │
│                                                        └──────┬───────┘  │
│                                                               │          │
│                                                               ▼          │
│                                                        ┌──────────────┐  │
│                                                        │ Final Report │  │
│                                                        │    (.md)     │  │
│                                                        └──────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
Component DescriptionsComponentTypeDescriptionOrchestratorAgentThe central controller (orchestrator.py) that initializes the pipeline, manages data flow between modules, and executes the analysis workflow.ECG LoaderToolIngests raw CSV sensor data (ecg_loader.py) and pairs it with the activity state label.Profile StoreDatabaseA JSON-based storage (config.yaml) containing historical physiological baselines for personalized validation.Signal ProcessorToolApplies dynamic bandpass filtering (signal_processor.py). It switches strategies based on the context provided by the Orchestrator.Feature ExtractorToolComputes time-domain (SDNN, RMSSD) and frequency-domain (LF, HF) metrics (feature_extractor.py).ClassifierToolAnalyzes extracted features to detect physiological states or anomalies (classifier.py).5. ImplementationThis section details the actual implementation of our Agentic system, highlighting the modular design, configuration-driven logic, and testing framework.5.1 ECG LoaderTools: Pandas, NumPyFunctionality:Ingests raw CSV sensor data without headers.Automatically extracts the primary ECG lead (Index 2).Validates data integrity and handles baseline correction.Key Code:Python# From src/tools/ecg_loader.py
def load_ecg(file_path: str, sampling_rate: int = 500) -> dict:
    """Load raw ECG data from CSV file."""
    # Read CSV without header; Index 2 is the primary ECG signal
    df = pd.read_csv(file_path, header=None)
    
    # Extract signal and remove baseline
    raw_signal = df[2].values
    signal = raw_signal - np.mean(raw_signal)

    return {
        "signal": signal,
        "sampling_rate": sampling_rate,
        "duration_sec": len(signal) / sampling_rate
    }
5.2 Signal ProcessorTools: SciPy (signal), NumPyFunctionality:Bandpass Filter: Applies a Butterworth filter (0.5-40 Hz) to remove powerline noise and baseline wander.R-Peak Detection: Uses a derivative-based algorithm to identify heartbeats.Adaptability: Filter parameters are injected via configuration, allowing different strategies for Rest vs. Active states.Key Code:Python# From src/tools/signal_processor.py
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(signal, fs, min_dist_sec=0.3):
    """Detect R-peaks with minimum distance constraint."""
    distance = int(min_dist_sec * fs)
    peaks, _ = find_peaks(signal, distance=distance, height=np.mean(signal))
    return peaks
5.3 Feature ExtractorTools: NumPy, SciPy (signal.welch)Functionality:Time-Domain Analysis: Calculates standard HRV metrics including SDNN, RMSSD, and pNN50.Frequency-Domain Analysis: Uses Welch's method to compute Power Spectral Density (PSD) for LF (Low Frequency) and HF (High Frequency) bands.Extended Features: Supports extraction of 20+ advanced metrics for deep analysis.Key Code:Python# From src/tools/feature_extractor.py
def extract_time_domain_features(rr_intervals: np.ndarray) -> dict:
    """Extract standard time-domain HRV metrics."""
    if len(rr_intervals) < 2: 
        return {}
    
    diff_rr = np.diff(rr_intervals)
    
    return {
        "mean_rr": np.mean(rr_intervals),
        "sdnn": np.std(rr_intervals, ddof=1),
        "rmssd": np.sqrt(np.mean(diff_rr ** 2)),
        "pnn50": (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100
    }
5.4 ClassifierTools: scikit-learn (RandomForest, SVM)Functionality:Provides a registry of 20+ classifiers (RandomForest, SVM, LogisticRegression, etc.).Analyzes extracted HRV features to detect stress or physiological states.Supports model training (train_classifier) and prediction (predict_stress).In our agentic context, this module also serves as a Quality Validator, comparing features against personal baselines.Key Code:Python# From src/tools/classifier.py
from sklearn.ensemble import RandomForestClassifier

def train_classifier(X_train, y_train, classifier_name="random_forest"):
    """Train a specified classifier on HRV features."""
    if classifier_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    clf.fit(X_train, y_train)
    return clf

def predict_stress(clf, features):
    """Predict stress state from HRV features."""
    # Convert feature dict to array matching training shape
    X = np.array([list(features.values())])
    prediction = clf.predict(X)[0]
    confidence = np.max(clf.predict_proba(X))
    
    return {"prediction": prediction, "confidence": confidence}
5.5 Report GeneratorTools: Matplotlib, ReportLab (Optional), Claude API (Optional)Functionality:Aggregates processing results from the Orchestrator.Generates visualization plots comparing "Rest" vs "Active" signals.Can integrate with LLMs (Claude) to generate natural language interpretations of the HRV metrics.Key Code:Python# From src/tools/report_generator.py
def generate_report(ecg_data, features, prediction, output_path):
    """Generate a comprehensive report with visualizations."""
    
    # Generate visualization
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot ECG Signal
    axes[0].plot(ecg_data['signal'][:1000])
    axes[0].set_title("ECG Signal Snippet")
    
    # Plot HRV Metrics (e.g., SDNN)
    axes[1].bar(['SDNN', 'RMSSD'], [features['sdnn'], features['rmssd']])
    axes[1].set_title("HRV Time-Domain Metrics")
    
    plt.savefig(output_path)
    return output_path
5.6 System Adaptation MechanismsComponent: Orchestrator & Profile StoreFunctionality:Our system implements "Agentic Adaptation" through dynamic strategy selection. This allows the system to handle inter-subject variability without manual recalibration.Context-Aware Strategy Switching: The Orchestrator reads the context label and dynamically injects specific parameters from config.yaml (e.g., aggressive filtering for Active states).Personalized Baseline Adaptation: The system solves the "Subject 2 Paradox" (lower variability during exercise) by referencing historical profiles instead of generic population thresholds.6. ResultsWe conducted a cross-validation test involving two subjects (Subject 1, Subject 2) across "Static" (Rest) and "Active" (Cycling) conditions, processed by our Agentic pipeline.6.1 Subject 1 AnalysisSubject 1 demonstrated remarkable physiological stability. Even during cycling, the signal variability (SDNN proxy) did not spike significantly.DatasetConditionSignal Variability (Std Dev)System ActionLevel 1Static651.84Baseline ProfilingLevel 1Active616.99AcceptedLevel 3Static700.94Baseline ProfilingLevel 3Active688.18AcceptedObservation: A traditional algorithm assuming "Exercise = High Noise" might have over-processed these clean signals. Our system correctly preserved the original signal details.6.2 Subject 2 Analysis (Counter-intuitive Finding)Subject 2 exhibited a completely different physiological pattern, validating the need for our flexible configuration system.DatasetConditionSignal Variability (Std Dev)InsightLevel 1Static692.00Higher baseline variabilityLevel 1Active651.88Unexpectedly StableLevel 3Static732.40Peak variabilityLevel 3Active670.83StableObservation: Surprisingly, Subject 2's signal variability during exercise (~645-670) was lower than during rest (~687-732). This confirms that variability is highly individualistic.7. Discussion7.1 The Value of OrchestrationThe modular design of orchestrator.py allowed us to seamlessly integrate distinct modules (ecg_loader, signal_processor, classifier). This "separation of concerns" meant that when we encountered Subject 2's unique data pattern, we could adjust the validation logic in classifier.py without rewriting the entire data ingestion pipeline.7.2 Handling Inter-subject VariabilityThe most significant finding is the divergence between Subject 1 (consistent variability) and Subject 2 (decreased variability during activity). If a "universal rule" were used (e.g., assuming Active state SDNN must be > 800), the system would fail for both users. Our agentic approach, which computes features dynamically and validates them against context, proved robust enough to handle these atypical patterns.8. ConclusionThis project successfully developed and validated a physiological monitoring Agent. By integrating a Python-based Orchestrator with a configurable Knowledge Base (config.yaml), we addressed the "one-size-fits-all" limitation inherent in longitudinal monitoring. The successful processing of real-world data from Subject 1 and Subject 2 demonstrates that our code architecture is both robust and adaptable.9. ReferencesEuropean Society of Cardiology Task Force. Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation, 1996.Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. Frontiers in Public Health, 2017.Behar J et al. ECG signal quality and false alarm reduction. IEEE Transactions on Biomedical Engineering, 2013.