# Technical Report: Agentic ECG HRV Baseline Evaluation System

**Author:** 2026-Lin-ChihYi
**Group Members:** Chu YenChieh, Lin ChihYi, Lin WenHsin
**License:** CC-BY-4.0

## Abstract

This report details an agentic, rule-based system designed for comprehensive Heart Rate Variability (HRV) analysis from raw ECG signals. The system orchestrates critical steps including ECG data loading, signal processing, R-peak detection, and advanced HRV feature extraction. Instead of relying on traditional machine learning classification, it employs personalized baseline rules to evaluate ECG segments against subject-specific physiological states. The primary objective is to provide robust, interpretable assessments of physiological consistency across various subjects and conditions, enabling effective dataset-level analysis without the need for model training. The system demonstrates agentic decision-making to adaptively evaluate HRV, offering insights into individual physiological responses.

## Introduction

Heart Rate Variability (HRV), derived from Electrocardiogram (ECG) signals, serves as a crucial non-invasive biomarker reflecting autonomic nervous system activity. Abnormal HRV patterns can indicate various physiological and psychological states, including stress, fatigue, or cardiovascular issues. Traditional methods of ECG and HRV analysis often involve complex processing steps and can be prone to inter-individual variability or require extensive manual interpretation.

This project aimed to develop an AI agent system to address these challenges and streamline the process of HRV analysis. The decision to build an agentic system is driven by the need for autonomous, adaptive, and goal-oriented processing of complex physiological data. An AI agent, by design, can perceive its environment (ECG data), process information through a series of specialized tools, make rule-based decisions, and act by generating comprehensive evaluations, thus reducing manual effort and improving consistency.

Specifically, this project focuses on:
1.  **Automated ECG Signal Processing**: From raw signal to R-peak detection.
2.  **Comprehensive HRV Feature Extraction**: Capturing time-domain, frequency-domain, and non-linear metrics.
3.  **Rule-Based Physiological Evaluation**: Assessing physiological states using personalized baselines rather than trained models, prioritizing interpretability and explainability.
4.  **Agentic Orchestration**: Coordinating complex analysis workflows and decision-making for dataset-level evaluation, showcasing the principles of agentic system design.

The overarching goal is to provide a consistent, interpretable, and safe method for assessing physiological states from ECG recordings, demonstrating the practical application of agentic AI principles in a real-world biomedical context.

## System Architecture

Our system is built around an `HRVAnalysisOrchestrator` that acts as the central agent, coordinating a series of specialized tools to perform a comprehensive ECG and HRV analysis pipeline. The architecture is modular, promoting clarity and maintainability.

The following diagram illustrates the high-level process flow of the system:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ <<component>>                                                                │
│                                                                              │
│                              HRV Analysis Agent                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│──────────────────────────────────────────────────────────────────────────────│
│                                                                              │
│     ┌────────────────────┐        ┌────────────────────┐                     │
│     │ «agent»            │        │ «database»         │                     │
│     │ Orchestrator       │        │ Profile Store      │                     │
│     └─────────┬──────────┘        └─────────┬──────────┘                     │
│               │                             │                                │
│               │                             │                                │
│     ┌─────────▼──────────┐        ┌─────────▼──────────┐        ┌──────────┐ │
│     │ «tool»             │        │ «tool»             │        │ «tool»   │ │
│     │ Context Loader     │─────▶ │ Adaptive Analyzer   │─────▶ │ Quality  │ │
│     └────────────────────┘        └─────────┬──────────┘        │ Validator│ │
│                                             │                   └────┬─────┘ │
│                                             │                        │       │
│                                    ┌────────▼────────┐       ┌───────▼──────┐│
│                                    │ R-Peak Detector │       │ «tool»       ││
│                                    └─────────────────┘       │ Result       ││
│                                                              │ Generator    ││
│                                                              └───────┬──────┘│
│                                                                      │       │
│                                                           ┌──────────▼──────┐│
│                                                           │ «artifact»      ││
│                                                           │ md Report       ││
│                                                           └─────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

```
The core of our solution is an **HRV Analysis Agent** driven by an **Orchestrator**. This agent coordinates various specialized tools and a profile store to perform comprehensive HRV analysis and evaluation. 

The key components and their interactions are as follows:

*   **Orchestrator (`«agent»`)**: This is the central control unit. It manages the workflow, initiates tool calls, and handles the overall decision-making process for the HRV analysis. It receives feedback from components like the `Quality Validator`.

*   **Context Loader (`«tool»`)**: This tool is responsible for loading the necessary data and context for analysis. It provides the `Adaptive Analyzer` with raw ECG signals and relevant parameters.

*   **Adaptive Analyzer (`«tool»`)**: This is a crucial component that performs the core processing and feature extraction for HRV. It takes input from the `Context Loader` and leverages information from the `Profile Store`. A specialized `R-Peak Detector` operates in conjunction with or as part of this analyzer. The `Adaptive Analyzer` passes its results to the `Quality Validator`.

*   **Profile Store (`«database»`)**: This database holds personalized profiles and baselines for subjects. The `Adaptive Analyzer` queries this store to obtain subject-specific data for personalized evaluation.

*   **R-Peak Detector**: Although depicted as a separate entity in the diagram, it is a critical sub-component likely integrated within or closely associated with the `Adaptive Analyzer`, responsible for identifying R-peaks in the ECG signal.

*   **Quality Validator (`«tool»`)**: This tool assesses the quality of the analysis results generated by the `Adaptive Analyzer`. It can identify issues or inconsistencies and provides feedback to the `Orchestrator`, potentially triggering re-analysis or adaptive adjustments. It also feeds into the `Result Generator`.

*   **Result Generator (`«tool»`)**: Based on the validated results, this tool compiles the final output of the analysis.

*   **MD Report (`«artifact»`)**: The ultimate output of the system is a Markdown report, summarizing the findings and evaluations. This report is generated by the `Result Generator`.

The workflow generally flows from the `Context Loader` through the `Adaptive Analyzer` and `Quality Validator` to the `Result Generator`, all under the supervision of the `Orchestrator`, which also interacts with the `Profile Store` and receives feedback from the `Quality Validator`.

## Data

### Dataset Structure

The system is designed to process raw ECG recordings organized in a specific directory structure. This structure facilitates the analysis across different subjects and physiological conditions.

```
data-group/data/
├── first_person/
│   ├── Rest/
│   │   ├── *.csv
│   └── Active/
│       ├── *.csv
│
└── second_person/
    ├── Rest/
    │   ├── *.csv
    └── Active/
        ├── *.csv
```

Each `.csv` file is expected to contain a single ECG recording with associated timestamps and sensor values. Recordings are categorized by **subject** (e.g., `first_person`, `second_person`) and **physiological condition** (e.g., `Rest`, `Active`). This organization allows the system to build and apply personalized baselines effectively.



## Implementation

The system is implemented in Python 3.11+, leveraging standard scientific computing libraries. The `HRVAnalysisOrchestrator` (`src/orchestrator.py`) is the core of the agentic design, orchestrating the entire process.

### Key Components and Methods

1.  **ECG Loading (`src/tools/ecg_loader.py`)**: Utilizes `numpy` and `pandas` to efficiently read CSV files containing single-column ECG data. It handles dataset organization based on subject and physiological condition.

2.  **Signal Processing (`src/tools/signal_processor.py`)**: Employs `scipy.signal` for digital signal processing. This includes:
    *   **Bandpass Filtering**: A Butterworth filter is applied to remove baseline wander and high-frequency noise, isolating the relevant ECG frequency range.
    *   **R-peak Detection**: Robust algorithms (e.g., derivative-based methods) are used to accurately identify R-peaks, which are critical for deriving R-R intervals.

3.  **HRV Feature Extraction (`src/tools/feature_extractor.py`, `src/tools/extended_features.py`)**: `numpy` and `scipy.signal` are used to compute:
    *   **Time-Domain Features**: Such as SDNN (standard deviation of NN intervals), RMSSD (root mean square of successive differences).
    *   **Frequency-Domain Features**: Calculated using Welch's method for Power Spectral Density (PSD), yielding LF (low-frequency) and HF (high-frequency) power, and their ratio (LF/HF).
    *   **Non-Linear Features**: Including Poincare plot descriptors (SD1, SD2) and Sample Entropy. The project focuses on a comprehensive set of 20 HRV features.

4.  **Baseline Construction**: The system builds personalized baselines per subject and condition. This involves analyzing reference ECG recordings to establish typical HRV metric ranges for "Rest" and "Active" states. These baselines are crucial for the rule-based evaluation.

5.  **Window-Based Evaluation Logic**: This is where the core "agentic decision-making" occurs.
    *   **Duration-Adaptive Windowing**: ECG recordings are segmented into overlapping sliding windows. Shorter recordings are treated as a single window.
    *   **Rule-Based Decision**: For each window, the extracted HRV features are compared against the corresponding personalized baseline. These rules are often based on statistical thresholds derived from the baselines, defining acceptable ranges for HRV metrics to classify a window as physiologically consistent. This leads to a "pass" or "fail" decision for that window.
    *   **Aggregation**: Window-level decisions are aggregated to calculate a `pass_rate` for the entire recording (`n_pass / n_windows`), reflecting overall physiological consistency.

6.  **Result Generation (`src/tools/report_generator.py`)**: Outputs include `baselines.json`, `pass_rates.csv`, and detailed per-file JSON outputs. `matplotlib` is used for visualizations, and `reportlab` can be utilized for generating PDF reports.

### Agentic Decision Module

The `HRVAnalysisOrchestrator` centralizes the logic for coordinating these tools, managing data flow between steps, and implementing the rule-based evaluation. This contrasts with systems that might employ a trained machine learning model (e.g., a classifier) for making pass/fail decisions. Our approach prioritizes interpretability and adaptability to individual physiological baselines.

## Results

The agentic ECG HRV Baseline Evaluation System successfully processed and evaluated synthetic and real ECG datasets across multiple subjects and conditions. All core functionalities, from signal loading to feature extraction and rule-based evaluation, passed validation tests.

### Dataset-Level Evaluation Details

*Note: The ECG data used for this evaluation was provided by teammates (first_person and second_person), as I did not personally collect electrocardiogram data.*

The following table summarizes the pass rates, where `n_windows` is the total number of evaluation windows and `n_pass` is the number of windows passing baseline checks.
```
| Person | Activity | n_windows | n_pass | Pass Rate |
|---|---|---:|---:|---:|
| first_person | Rotate | 5 | 0 | 0.00 |
| first_person | Bike (Level 5) | 19 | 17 | 0.895 |
| first_person | Bike (Level 3) | 17 | 16 | 0.941 |
| first_person | Bike (Level 1) | 21 | 21 | 1.00 |
| first_person | Speak | 5 | 5 | 1.00 |
| first_person | Static (Level 1) | 5 | 5 | 1.00 |
| first_person | Static (Level 3) | 7 | 7 | 1.00 |
| first_person | Static (Level 5) | 5 | 5 | 1.00 |
| second_person | Bike (Level 1) | 19 | 18 | 0.947 |
| second_person | Bike (Level 5) | 17 | 17 | 1.00 |
| second_person | Bike (Level 3) | 19 | 19 | 1.00 |
| second_person | Rotate | 7 | 7 | 1.00 |
| second_person | Speak | 5 | 5 | 1.00 |
| second_person | Static (Level 1) | 5 | 3 | 0.60 |
| second_person | Static (Level 3) | 5 | 5 | 1.00 |
| second_person | Static (Level 5) | 5 | 5 | 1.00 |
```
Observations from this evaluation include:
*   **Accurate Physiological Consistency**: Rest states generally exhibit high pass rates, indicating a stable and well-defined baseline.
*   **Deviation Highlighting**: Activities like "Rotate" for the first subject show low pass rates, suggesting significant physiological shifts or motion artifacts that fall outside the personalized baseline.
*   **Inter-Subject Variability**: The system successfully captures differences in physiological response across subjects; for instance, the second subject's "Rotate" activity had a high pass rate, unlike the first.
*   **Condition Adaptability**: The agentic decision logic effectively applies condition-specific (Rest vs. Active) baselines, allowing for nuanced evaluation.

These detailed results validate the system's capacity for robust, interpretable physiological assessment based on its rule-based agentic design.

## Discussion

### Challenges Encountered

One primary challenge revolved around ensuring the robustness of R-peak detection and HRV feature extraction across diverse signal qualities and durations. ECG signals are prone to noise and artifacts (e.g., motion artifacts, baseline wander), which can significantly impact the accuracy of R-peak detection and subsequent HRV calculations. The system's adaptive windowing strategy and signal processing steps (`signal_processor.py`) were critical in mitigating these issues.

### Limitations

The system's current limitations include:
*   **Data Quality Sensitivity**: While robust, severe motion artifacts or flat/saturated ECG signals can still lead to invalid HRV calculations or R-peak detection failures. This necessitates good data acquisition practices.
*   **Minimum Signal Duration**: Accurate HRV analysis and stable baseline estimation require a minimum ECG recording duration. Very short signals may lead to unreliable results or be skipped.
*   **Rule-Based Adaptability**: While personalized and rule-based, the system's "intelligence" is confined to predefined rules. Adapting to highly novel or ambiguous physiological states might require manual adjustment of these rules or integration with more adaptive AI methods.

### Lessons Learned

*   **Value of Agentic Orchestration**: The `HRVAnalysisOrchestrator` proved highly effective in managing the complex multi-step analysis pipeline. It enabled modular development, clear data flow, and systematic error handling, reducing overall development complexity.
*   **Interpretability of Rule-Based Systems**: By explicitly defining physiological rules and baselines, the system offers high interpretability. Users can understand *why* a particular ECG segment passed or failed, which is a significant advantage over "black-box" machine learning models in clinical or physiological assessment contexts.
*   **Importance of Personalized Baselines**: Given the inherent inter-subject variability in physiological responses, personalized baselines are crucial for accurate and meaningful evaluation. A global threshold would be far less effective.

## Conclusion

The developed Agentic ECG HRV Baseline Evaluation System successfully addresses the objectives outlined in the introduction, providing a robust, interpretable, and automated solution for physiological assessment. The system effectively leverages an `HRVAnalysisOrchestrator` to coordinate signal processing, HRV feature extraction, and rule-based evaluation against personalized baselines. All core functionalities have been validated through testing, confirming its reliability.

This project demonstrates the power of agentic design in orchestrating complex analytical tasks and making interpretable decisions based on domain-specific rules. The system is well-suited for dataset-level analysis across various subjects and conditions, offering valuable insights into physiological states without relying on traditional machine learning model training.

### Future Work

*   **Enhanced Rule Adaptability**: Explore methods for dynamic adjustment of rule thresholds based on real-time physiological context or expert feedback.
*   **Integration with Advanced AI**: Investigate the potential for integrating specialized AI models (e.g., for noise reduction or specific artifact detection) while maintaining overall rule-based interpretability.
*   **Real-time Processing**: Optimize the pipeline for lower latency to support real-time monitoring applications.
*   **Broader Validation**: Test the system with a wider range of clinical and research datasets to further validate its generalizability and robustness.

## References

1.  Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065.
2.  Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5, 258. doi:10.3389/fpubh.2017.00258.
3.  Behar, J., Johnson, A. E. W., Clifford, G. D., & Oster, J. (2013). ECG signal quality during arrhythmia and its application to false alarm reduction. *IEEE Transactions on Biomedical Engineering*, 60(6), 1660-1666. doi:10.1109/TBME.2013.2240452.
