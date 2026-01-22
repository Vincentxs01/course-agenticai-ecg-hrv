# Technical Report: Agentic ECG HRV Baseline Evaluation System

**Author:** 2026-Chu-YenChieh
**Group Members:** Chu YenChieh, Lin ChihYi, Lin WenHsin
**License:** CC-BY-4.0

## Abstract

This report details the development of an **Agentic ECG HRV Baseline Evaluation System**, designed to address the "Personalization Gap" in physiological monitoring. Traditional health monitoring systems often rely on universal thresholds that fail to account for individual variances and activity states. Our solution implements a modular agentic architecture where a central **Orchestrator** coordinates specialized tools—including an **Adaptive Analyzer** and a **Quality Validator**—to process signals dynamically. A key innovation is the **Profile Store**, which maintains individualized baselines (stored in `baselines.json`), enabling the system to contextualize signal quality. Validated on a dataset of mixed activities, the system demonstrated the ability to autonomously differentiate valid physiological stress from motion artifacts, significantly reducing the manual burden of data review.

## Introduction

In the field of wearable health monitoring, distinguishing between genuine physiological anomalies and motion artifacts is a persistent challenge.

### The Problem: The Personalization Gap
Most current systems use static, hard-coded thresholds for signal quality and Heart Rate Variability (HRV) metrics. This leads to two critical failures:

> 1.  **False Positives in Active States:** High heart rates during exercise are often misclassified as anomalies.
> 2.  **Individual Incompatibility:** A "normal" baseline for a sedentary individual may trigger alarms for an athlete.

### Objectives
Our group aimed to construct an AI Agent that bridges this gap by:
* **Contextualizing Data:** Recognizing the user's activity state (e.g., Static vs. Biking).
* **Personalizing Evaluation:** Using historical personal data to evaluate new signals.
* **Automating Workflow:** Employing an agentic workflow to handle the end-to-end process.

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

**Key Logic:**
When the **Context Loader** reads a file (e.g., `subject_01_bike.csv`), the agent follows this sequence:

1.  **Identify:** Extract Subject ID and Activity from metadata.
2.  **Query:** Fetch the corresponding baseline object from **Profile Store**.
3.  **Inject:** Pass these personalized parameters to the **Adaptive Analyzer**.

This approach enables the **Quality Validator** to apply dynamic, context-aware logic:

# Pseudo-code logic for Quality Validator
if (Current_RMSSD < Personal_Baseline_RMSSD * 0.5) and (Activity == 'Rest'):
    return "Flag as Low Quality / Potential Anomaly"
else:
    return "Pass"

## Results

We evaluated the system using a real-world dataset comprising different subjects and activity intensities. The key findings are categorized by subject and activity below:

* **First Person - Static (Rest):**
    * **Pass Rate:** 100%
    * **Analysis:** The signal consistently matched the personal resting baseline, validating the Profile Store's accuracy for rest states.

* **First Person - Bike (Active):**
    * **Pass Rate:** 89.5%
    * **Analysis:** The system correctly adapted thresholds for high heart rates. Traditional static thresholds would have rejected these valid exercise signals, but our context-aware agent accepted them.

* **First Person - Rotate (Noise/Artifacts):**
    * **Pass Rate:** 0%
    * **Analysis:** This was a critical success. The Quality Validator correctly identified excessive motion artifacts in every window and rejected them, preventing data pollution.

* **Second Person - Static:**
    * **Pass Rate:** 60-100%
    * **Analysis:** The system successfully detected outliers in specific windows where signal quality dropped, demonstrating sensitivity to individual variations.

## Discussion

### Trade-offs: Reasoning vs. Latency
One challenge encountered was the trade-off between the depth of the agent's reasoning and processing speed. Calling the LLM to interpret every validation step provided excellent explainability but increased latency. We optimized this by caching the **Profile Store** data to reduce redundant lookups.

### Limitations
* **Static Profiles:** Currently, `baselines.json` is static. The system does not yet support "Online Learning" to update baselines automatically.
* **Dependency on Formatting:** The **Context Loader** is sensitive to file naming conventions, requiring strict data governance.


## Conclusion

We successfully developed and validated an Agentic ECG HRV Evaluation System. By decoupling the decision logic into a personalized **Profile Store**, we achieved a system that is flexible and rigorous. The architecture allows for scalable health monitoring that respects individual physiological differences, successfully meeting the project's goal of closing the Personalization Gap.

## References
1.  Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065.
2.  Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5, 258. doi:10.3389/fpubh.2017.00258.
3.  Behar, J., Johnson, A. E. W., Clifford, G. D., & Oster, J. (2013). ECG signal quality during arrhythmia and its application to false alarm reduction. *IEEE Transactions on Biomedical Engineering*, 60(6), 1660-1666. doi:10.1109/TBME.2013.2240452.