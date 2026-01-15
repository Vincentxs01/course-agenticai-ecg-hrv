# Case Brief: Automated Sleep Stage Detection in Smart Wearables

**Author:** Chu,Yen-Chieh
**License:** CC-BY-4.0

## Problem Statement

A smart wearable company needs to analyze long-term sleep data (Actigraphy and Heart Rate) from multiple users for sleep quality assessment. The primary challenge is that current systems use a **single, fixed sleep scoring algorithm**, which ignores individual differences in circadian rhythms and resting heart rates. This results in frequent misclassification of sleep stages and requires manual data correction.

## Context/Background

- Average manual data correction times exceed **2 hours** for sleep specialists.
- Analysts spend **5-10 minutes** per session on manual calibration.
- Misclassification rates range from **10-30%** depending on user experience.
- Current electronic health record (EHR) systems provide limited decision support.

## Analysis

### Root Causes
1. High user-to-analyst ratios during peak hours.
2. Subjective assessment varies by analyst experience.
3. Information overload from multiple data sources (vitals, history, symptoms).

### Constraints
- Must integrate with existing EHR systems.
- Cannot replace clinical judgment (regulatory requirements).
- Must handle edge cases safely (fail-safe design).

### Requirements
- Real-time processing of patient data.
- Explainable recommendations (not black-box).
- Continuous learning from outcomes.

## Proposed Approach (Chat-based)

An analyst could:

1. **Collect** sleep data from multiple sources (vitals monitors, intake forms) and copy-paste them into the chat with a standardized prompt.
2. **Analyze** using the inputted data and prompt.
3. **Recommend** priority level with explanation.
4. **Learn** by updating the standardized prompt based on analyst feedback and patient outcomes.

## Proposed Approach (Agentic)

An AI agent system could:

1. **Collect** patient data from multiple sources (vitals monitors, intake forms).
2. **Analyze** using trained models on historical triage outcomes.
3. **Recommend** priority level with explanation.
4. **Learn** from analyst feedback and patient outcomes.

The agent would augment (not replace) analyst decision-making, flagging high-risk cases for immediate attention.

## Expected Outcomes

- **30% reduction** in triage time.
- Improved consistency in priority assignment.
- Early warning for deteriorating patients.
- Reduced cognitive load on triage analysts.