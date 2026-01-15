# Case Brief: ECG-Based Physiological Sensing and Agentic AI Code Development Guardian

**Author:** 2026-Chen-KunYu, 2026-Liu-YungHsin, 2026-Chen-Guo-Zhu
**License:** CC-BY-4.0

## 1. Problem Statement

Software engineers often fall into "Tunnel Vision" during development, ignoring physical signs of fatigue or high stress, leading to long-term burnout and decreased code quality.

While wearable devices can collect ECG signals, most users lack the ability to interpret waveforms (e.g., P wave, QRS complex) and rhythm changes. Users often feel anxious when seeing abnormal heart rate values but cannot determine if these are merely non-pathological changes caused by stress or fatigue. Furthermore, when engineers are fatigued, they often lack the motivation to organize code, resulting in work being interrupted without being committed, leaving the project in an unstable state.

## 2. Context/Background

**Industry Context:** 
* Software development is a high-cognitive-load task; prolonged "ineffective diligence" often leads to a decline in code quality.


**Limitations of Current Solutions:**
* Standard smartwatches only provide heart rate numbers, lacking interpretation combined with "work context".


* Tools like Pomodoro timers are too rigid and cannot adjust dynamically based on the user's current physiological stress.





## 3. Analysis

### Root Causes

1. **Physiological Awareness Gap:** Users cannot distinguish between "benign work excitement" and "malignant stress-induced Sinus Tachycardia".


2. **Resistance to Wrapping Up:** When fatigued, the brain resists the tedious work of writing commit messages or comments.


3. **Lack of Immediate Guidance:** There is a lack of immediate behavioral guidance when abnormal rhythms (such as arrhythmia or excessive speed) are detected.



### Requirements

* **Precise Sensing:** Must automatically annotate ECG waveforms (P-QRS-T) and establish personalized baselines.


* **Active Intervention:** The Agent requires Generative AI capabilities to reduce the user's cognitive burden.



## 4. Proposed Approach (Agentic Solution)

This system integrates **ECG waveform analysis** and **Agentic decision-making** to convert complex physiological signals into actionable workflow feedback.

### Step 1: Perception

* **Physiological Layer:** Monitors ECG rhythm to identify **Sinus Tachycardia** (stress indicator) , **Irregular Rhythm** , and **Fast Rhythm Patterns**.


* **Digital Layer:** Monitors IDE activity and Git status (Modified but not Staged).



### Step 2: Reasoning

* **Logic:** "User heart rate persistently above baseline (excessive stress) + Long time without code submission = High risk of burnout, immediate intervention required."
* **Decision:** Trigger a gentle interruption and provide four solutions at different levels.



### Step 3: Action (State Display & Four Options)

When a risk is detected, the Agent will first **visually display the current physiological state** in a pop-up window (e.g., a dashboard showing "Current Stress Index: High" or "Fatigue Accumulation: 90%", accompanied by an HRV downward trend chart) to help the user build self-awareness, followed by providing the following options:

1. **【Rest Immediately】**:
* The user chooses to stop immediately. The Agent performs no additional actions, respecting the user's autonomy.




2. **【Snooze & Loop】**:
* Suitable for "Want to push a bit longer" situations. The Agent enters a **5-minute supervision loop**.
* Condition: Checks `git commit` every 5 minutes. If a submission is detected, the alarm is automatically lifted; otherwise, the reminder continues.




3. **【Breathing & Guidance (Biofeedback)】**:
* Suitable for "Too stressed but don't want to leave the seat" situations.
* The Agent activates a **1-5 minute breathing training module** to guide the user in adjusting the parasympathetic nervous system.
* Provides text-based stress reduction suggestions.




4. **【Generate & Archive (GenAI Assist)】**:
* Suitable for "Too tired to wrap up" situations.
* **Generate Commit Message:** Reads `git diff`, automatically writes a summary, and commits.


* **Generate Context Anchor:** If the code cannot be committed yet (Broken Code), the Agent automatically generates a comment at the cursor (e.g., `// TODO: Stopped here last time, refactoring API...`) to mark the current thought process for easier resumption.



## 5. Expected Outcomes

* **Ensure Project Health:** Reduce "rotten tail" code caused by fatigue through AI-assisted archiving.


* **Improve Self-Awareness:** Visualized state feedback helps engineers realize the connection between physical fatigue and work efficiency.


* **Immediate Risk Mitigation:** Provide immediate behavioral guidance through breathing training when critical waveform patterns are detected.


