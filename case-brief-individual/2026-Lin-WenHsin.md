# Case Brief: Case Brief: Bridging the "Visibility Gap" in Post-Discharge Cardiac Monitoring via Agentic AI

**Author:** 2026-Lin-WenHsin

**License:** CC-BY-4.0

## Problem Statement

**The Loss of Valid Recovery Metrics in Home Settings**

The critical problem addressed in this case is the **"Visibility Gap"** that occurs when cardiac patients transition from a controlled hospital environment to an uncontrolled home setting.

In a clinical setting, ECG monitoring is stable. [cite_start]However, once discharged, patients are subjected to diverse environmental stressors (movement, vibration, improper sensor placement), resulting in significant signal noise[cite: 7]. [cite_start]Current monitoring systems utilize a single, fixed analysis workflow [cite: 9] [cite_start]that indiscriminately rejects these noisy segments[cite: 15].

This leads to a failure in tracking essential **Recovery Metrics**—specifically Heart Rate Variability (HRV) trends and Resting Heart Rate (RHR) improvement. Without valid data, clinicians cannot distinguish between a patient who is "recovering well" (improving HRV) and one who is deteriorating, until a critical readmission event occurs.

## Context/Background

* [cite_start]**Context:** The system is designed to monitor multiple individuals (N=3) continuously, processing 30–50 ECG fragments per person daily[cite: 3, 4].
* **Stakeholders:** Post-discharge patients, cardiologists requiring longitudinal data, and monitoring service providers.
* **Current Solutions:** Traditional Holter monitors or consumer wearables that apply universal noise filters.
* **The Cost:**
    * [cite_start]**Data Loss:** Valuable physiological data is discarded because the algorithms cannot adapt to the "noise profile" of a patient's daily life[cite: 15].
    * [cite_start]**Manual Overload:** Technicians must manually review rejected segments to see if they are salvageable, creating a bottleneck that prevents the system from scaling to more patients[cite: 18, 21].

## Analysis

A systematic analysis reveals that the inability to track Recovery Metrics stems from three root causes:

1.  **Algorithmic Rigidity:** Current systems treat "Home" data with the same strict filters as "Hospital" data. [cite_start]They lack the elasticity to attempt alternative processing strategies when the first attempt fails due to noise.
2.  **Lack of Personalization:** Recovery trajectories vary. [cite_start]An athlete's recovering heart rate differs significantly from an elderly patient's[cite: 5]. [cite_start]Standard thresholds often misclassify normal recovery variances as anomalies[cite: 10, 19].
3.  [cite_start]**Contextual Disconnect:** The analysis engine processes each segment in isolation, failing to query the patient's historical profile (e.g., pre-discharge baseline) to validate whether current metrics represent a positive or negative trend.

## Proposed Approach

To restore visibility into patient recovery, we propose an Agentic AI Solution that prioritizes Adaptive Signal Recovery. This approach shifts from manual intervention to autonomous, context-aware processing.

## Proposed Approach (Chat-based)

A Chat-based AI (e.g., ChatGPT) approach involves users manually pasting CSV data for analysis.


Deficiency: This approach suffers from "Context Loss" , as the model often forgets individual physiological traits between sessions. Furthermore, the analysis strategy is typically fixed, preventing automatic adjustment for different individuals. The requirement for repetitive manual input makes it unfeasible for 24/7 continuous recovery tracking.

## Proposed Approach (Agentic)

We utilize an autonomous AI Agent designed to "hunt" for valid Recovery Metrics within noisy data by implementing a dynamic workflow:


Ingestion & Profiling: Upon receiving ECG data , the Agent queries the Personal Profile Module to retrieve the patient's specific historical heart rate and HRV distribution.
Adaptive Decision Loop: The Agent applies an initial analysis strategy. If the result is deemed unreliable due to noise, it does not discard the data. Instead, it automatically adjusts strategies —such as modifying filter parameters or R-peak detection methods —and retries the analysis.
Metric Validation: Finally, the Agent evaluates if the calculated HRV aligns with the patient's physiological range. Valid results are stored to update the user's profile , while unrecoverable segments are flagged , minimizing the need for human review.

## Proof of Concept: Recovering the "Home" Signal
We planned to use a Python simulation that demonstrates the Agent's ability to maintain data continuity during the hospital-to-home transition.

## Expected Outcomes

1.  **Continuous Recovery Tracking:** Clinicians will have access to unbroken trend lines of HRV and Heart Rate, allowing for early detection of deterioration before readmission is needed.
2.  [cite_start]**Personalized Alerts:** By benchmarking against individual history rather than population averages, false positives are minimized[cite: 30, 36].
3.  [cite_start]**Automated Data Salvage:** The system autonomously recovers noisy data segments, ensuring that "life" (movement/activity) does not interrupt "monitoring"[cite: 31, 32].

## References

1.  [cite_start]*Project Requirements & Problem Definition Document* [cite: 1-75].
2.  *Agentic ECG Analysis & Signal Recovery Simulation* (Python Implementation).