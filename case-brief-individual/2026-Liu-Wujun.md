# Case Brief: Performance Comparison of Real-time QRS Detection Algorithms

## Problem Definition

With the rapid growth of mobile health (mHealth) and wearable ECG monitoring devices, accurate real-time detection of ECG signals has become a critical requirement for home-based and continuous health monitoring. Among ECG components, the QRS complex is the most prominent and energetic feature, and its detection accuracy directly affects the reliability of Heart Rate (BPM) and Heart Rate Variability (HRV).

However, in practical scenarios, ECG signals are frequently contaminated by baseline wander, muscle artifacts, and power-line interference. These noise sources significantly degrade the performance of simple threshold-based QRS detection methods, making them unreliable in real-world, noisy environments—especially on resource-constrained mobile devices.

## Objectives

This project aims to:

1. Implement and compare two real-time ECG QRS detection strategies:
   - Basic Dynamic Thresholding
   - Pan–Tompkins Algorithm

2. Evaluate algorithm performance on mobile platforms, focusing on:
   - Noise suppression capability
   - Real-time computational feasibility
   - R-peak detection accuracy

3. Develop a mobile ECG analysis suite capable of real-time waveform visualization and automated cardiac feature extraction.

## Methodology

### Digital Signal Processing

A 5–15 Hz bandpass filter is implemented to enhance QRS complex features while suppressing baseline drift and high-frequency noise.

### Pan–Tompkins Algorithm Implementation

The Pan–Tompkins pipeline consists of:

- Differentiation to extract slope information
- Squaring to amplify dominant peaks and suppress noise
- Moving Window Integration (MWI) to generate a smoothed envelope for adaptive thresholding

### Mobile System Integration

A circular buffer architecture is designed to support low-latency signal streaming, continuous waveform rendering, and automatic R-peak annotation in real time under smartphone computational constraints.

## Expected Outcomes

### Functional Outcomes

- A real-time mobile ECG signal processing suite with continuous signal acquisition and visualization
- Automated and stable R-peak detection

### Algorithmic Outcomes

- Improved robustness and stability of the Pan–Tompkins algorithm compared with basic dynamic thresholding in noisy environments

### Analytical Outcomes

- Integrated time-domain HRV analysis, including:
  - SDNN (Standard Deviation of NN intervals)
  - RMSSD (Root Mean Square of Successive Differences)

## Value Proposition

This project demonstrates that clinically established ECG signal processing algorithms, such as Pan–Tompkins, can be successfully deployed in real-time mobile health applications, improving reliability while remaining computationally efficient.
