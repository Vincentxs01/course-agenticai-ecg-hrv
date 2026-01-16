# **Case Brief: Real-Time ECG/HRV Driver Drowsiness Detection Under Motion Artifacts**

**Author:** 2026-Wu-Kun,Che
**License:** CC-BY-4.0

## **Problem Definition**

Driver fatigue is a major safety risk. During long drives, a driver’s mental state can gradually shift from **“Alert/Focused”** to **“Relaxed/Drowsy,”** increasing accident likelihood.

### **Why this is hard (ECG in a moving vehicle)**

We analyze **wearable ECG** signals inside a vehicle, but the driving environment introduces motion artifacts that corrupt the signal and break standard ECG/HRV pipelines. The setting contains four representative noise conditions:

- **Speaking** → EMG noise from jaw muscles  
- **Head Moving** → baseline wander and signal fluctuation  
- **Cycling (Simulation)** → vibration-like disturbances  
- **Speaking (Recurring)** → repeated noise to test robustness

**Core issue:** detect drowsiness **in real time**, while distinguishing *true physiological changes* from *artifact-induced changes*.

### **Baseline workflow (Before): manual/offline + fragile rules**

A typical “before” approach in this setting looks like:

1. **Collect ECG** during driving.
2. Apply **simple filters** (or none), then run a **basic R-peak detector**.
3. Compute HR/HRV features and apply fixed thresholds.
4. When the result is questionable (often under artifacts), **humans inspect plots** or re-run with adjusted parameters.

#### **Work-efficiency pain points (Before)**

- **Artifact-driven false alarms:** noisy segments can trigger incorrect “drowsy” detection, causing frequent interruptions.
- **Manual re-check cost:** reviewing ECG segments or tuning thresholds becomes repetitive (especially during development/testing).
- **Non-scalable:** the approach does not generalize well across different drivers (individual baselines differ).

> **How to quantify your “before” inefficiency (fill in your own numbers):**  
> Let :
> - **A** = number of false alarms per hour = 0.8 (times)
> - **t_ack** = seconds lost per alarm (driver acknowledges alarm/distraction) = 5 (minutes) (Includes turning off the alarm after parking, etc.)
> - **t_review** = seconds for an engineer to review one flagged segment = 2 (minutes) (Includes writing simple report)
> - **H** = driving hours per day = 8 hours
>
> Then  
> - **Driver time lost/day** = **A × H × t_ack** = **0.8 × 8 × 5** = 32 (minutes)  
> - **Engineering review time/day** = **A × H × t_review** = **0.8 × 8 × 2** = 12.8 (minutes)

## **Expected Outcome**

**Objective:** build a **real-time AI driver monitoring system** using ECG to detect drowsiness and prevent accidents.

### **Key deliverables**

1. **Robust Signal Processing**  
   Automatically detect and filter artifacts (e.g., speaking/head movement) to recover a usable ECG.

2. **State Classification via HRV**  
   Use HR/HRV patterns to classify driver state:  
   - **Class 0 (Safe): Alert** → higher HR, lower HRV  
   - **Class 1 (Danger): Relaxed/Drowsy** → lower HR, higher HRV

3. **Automated Safety Mechanism**  
   If the system detects “Relaxed/Drowsy” while the vehicle is moving, trigger an **auditory alarm** immediately.

### **Efficiency + Before/After comparison (work efficiency, 前後比較)**

| Dimension | Before (manual / fragile rules) | After (agentic, automated pipeline) |
|---|---|---|
| Artifact handling | Often fails under motion artifacts; needs manual tuning | Artifact detection + filtering runs continuously |
| HRV extraction | R-peak detection unstable in noise; repeated reruns | Standardized R-peak + RR cleaning + feature extraction |
| Decision logic | Fixed thresholds, hard to personalize | Individual baseline + adaptive thresholds (per driver) |
| Response time | Delayed / offline | Real-time alerting (low latency) |
| Work efficiency | Frequent false alarms + frequent manual reviews | Reduced false alarms + fewer reviews; scalable to more drivers |

> **Target operational improvements (define measurable metrics):**  
> - Reduce false alarms/hour from **A_before** → **A_after** = **0.1 (times)**
> - Reduce review workload/day from **R_before** → **R_after** = **2 (minutes)**
> - Improve sensitivity/recall for true drowsy events

---

## Solution proposal (Chat Based, e.g., ChatGPT through a web browser)

### **Workflow**

1. During testing, an operator exports a short ECG segment (or precomputed HR/HRV values).
2. The operator pastes the segment/values into a chat interface and asks:  
   “Is this driver alert or drowsy? Which artifacts might exist?”
3. The operator manually adjusts filtering/thresholds based on the suggestion.
4. Repeat for each new segment or driver.

### **Observed limitation**

- **Not real-time:** copy/paste and human-in-the-loop cannot keep up with streaming ECG.
- **No direct sensor access:** the chat model does not see raw telemetry context (speed, noise labels) unless manually provided.
- **Inconsistent & hard to validate:** explanations may vary; hard to guarantee stable behavior for safety-critical monitoring.
- **Doesn’t scale:** becomes slower as the number of drivers/sessions increases.

---

## Solution proposal (Agentic)

We use a **multi-agent AI architecture** to handle noisy ECG streams and make safe decisions.

### **Workflow**

1. **ECG Stream → Agent 1: Artifact Handler (Preprocessing)**  
   - **Role:** Noise Filter  
   - **Input:** raw ECG contaminated by speaking/movement artifacts  
   - **Task:** detect artifacts and apply filters (e.g., band-pass, wavelet)  
   - **Output:** clean, normalized ECG

2. **Clean ECG → Agent 2: HRV Analyst (Feature Extraction)**  
   - **Role:** Physiological Calculator  
   - **Task:** detect R-peaks → compute RR intervals → compute key metrics (e.g., **SDNN** and **Mean Heart Rate**)  
   - **Output:** feature vector (e.g., **{HR: 55 bpm, HRV: High}**)

3. **Features → Agent 3: Safety Guardian (Decision & Action)**  
   - **Role:** Decision Maker  
   - **Task:** compare current metrics to **individual baseline**  
   - **Logic:** **IF (HRV > threshold) AND (HR < threshold) → Drowsy**  
   - **Output:** **TRIGGER ALARM** or **CONTINUE MONITORING**

4. **Logging & Evaluation (recommended addition)**  
   - Store alarms, detected artifacts, HR/HRV trends, and outcomes for later calibration.  
   - Periodically update thresholds and validate per-driver performance.
