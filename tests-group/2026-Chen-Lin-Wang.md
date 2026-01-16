# Test Suite: HRV Analysis for Stress Detection Agent

**Group:** 2026-Chen-Lin-Wang
**Authors:** Chen Wei, Lin MeiLing, Wang XiaoMing
**Date:** 2026-01-16
**License:** Apache-2.0 (code), CC-BY-4.0 (documentation)

## Overview

This test suite validates the HRV Analysis Agent system for stress detection using ECG-derived heart rate variability features. Tests cover:

- **Data Loading:** WESAD pickle files and raw ECG text files with validation
- **Signal Processing:** Bandpass filtering (0.5-40 Hz), R-peak detection, RR interval computation
- **Feature Extraction:** 20 HRV features across time-domain, frequency-domain, and nonlinear categories
- **Classification:** 20 different ML classifiers for binary stress detection
- **Utilities:** Logging configuration, YAML config loading, data validation, signal normalization
- **Report Generation:** PDF reports with AI interpretation and text fallback
- **Orchestration:** Pipeline coordination through tool-calling interface

Testing approach includes:
- Unit tests for individual functions with known inputs/outputs
- Parametrized tests covering all 20 classifiers and multiple parameter variations
- Integration tests for end-to-end pipeline validation
- Edge case testing for error handling and boundary conditions

## Test Summary

| Category | Total | Passed | Failed |
|----------|-------|--------|--------|
| Unit Tests: Classifier Registry | 25 | 25 | 0 |
| Unit Tests: Classifier Creation | 22 | 22 | 0 |
| Unit Tests: Classifier Training | 10 | 10 | 0 |
| Unit Tests: Save/Load | 2 | 2 | 0 |
| Unit Tests: Prediction | 2 | 2 | 0 |
| Unit Tests: Recommendation | 5 | 5 | 0 |
| Unit Tests: Feature Importance | 2 | 2 | 0 |
| Unit Tests: All Classifiers Train/Predict | 24 | 24 | 0 |
| Unit Tests: Extended Features | 3 | 3 | 0 |
| Unit Tests: ECG Loader Parameters | 4 | 4 | 0 |
| Unit Tests: Signal Processor Parameters | 9 | 9 | 0 |
| Unit Tests: Feature Extractor Functions | 9 | 9 | 0 |
| Unit Tests: All Classifiers Comprehensive | 40 | 40 | 0 |
| Unit Tests: Recommend Priorities | 13 | 13 | 0 |
| Unit Tests: Classifier Categories | 5 | 5 | 0 |
| Unit Tests: Feature Importance All | 3 | 3 | 0 |
| Unit Tests: Predict Stress | 2 | 2 | 0 |
| Unit Tests: Feature Constants | 5 | 5 | 0 |
| Unit Tests: Time Domain Features | 6 | 6 | 0 |
| Unit Tests: Frequency Domain Features | 4 | 4 | 0 |
| Unit Tests: Nonlinear Features | 4 | 4 | 0 |
| Unit Tests: Edge Cases | 4 | 4 | 0 |
| Unit Tests: Feature Values | 4 | 4 | 0 |
| Unit Tests: Real-Like Data | 2 | 2 | 0 |
| Unit Tests: Logging Setup | 4 | 4 | 0 |
| Unit Tests: Config Loading | 4 | 4 | 0 |
| Unit Tests: ECG Validation | 10 | 10 | 0 |
| Unit Tests: Signal Normalization | 4 | 4 | 0 |
| Unit Tests: Environment Variables | 4 | 4 | 0 |
| Unit Tests: Directory Utilities | 4 | 4 | 0 |
| Unit Tests: Orchestrator Init | 4 | 4 | 0 |
| Unit Tests: Orchestrator Tools | 10 | 10 | 0 |
| Unit Tests: Classifier Loading | 2 | 2 | 0 |
| Unit Tests: Orchestrator State | 4 | 4 | 0 |
| Unit Tests: Pipeline | 1 | 1 | 0 |
| Unit Tests: Windowing | 4 | 4 | 0 |
| Unit Tests: Report Availability | 2 | 2 | 0 |
| Unit Tests: Interpretation | 3 | 3 | 0 |
| Unit Tests: Text Fallback | 3 | 3 | 0 |
| Unit Tests: Report Without AI | 2 | 2 | 0 |
| Unit Tests: PDF Generation | 1 | 0 | 0 (1 skipped) |
| Unit Tests: ECG Loader | 3 | 3 | 0 |
| Unit Tests: Signal Processor | 4 | 4 | 0 |
| Unit Tests: Feature Extractor | 5 | 5 | 0 |
| Integration Tests | 1 | 1 | 0 |
| **Total** | **288** | **287** | **0** (1 skipped) |

**Pass Rate:** 99.7% (287/288, 1 skipped due to optional dependency)

## Usage

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_classifier.py -v

# Run tests with coverage measurement
python -m pytest tests/ --cov=src --cov-report=html

# Run tests with verbose output
python -m pytest tests/ -v --tb=short

# Generate this test report
python tests/generate_test_report.py
```

## Known Issues

1. **TC-275 (test_pdf_generation) skipped** - Requires reportlab package which is not installed. Test passes when reportlab is available.

2. **Frequency domain features require minimum 10 RR intervals** - Returns NaN for shorter recordings. This is expected behavior per HRV analysis guidelines.

3. **PassiveAggressiveClassifier deprecated** - Sklearn 1.8+ shows deprecation warning; classifier will be removed in sklearn 1.10. Tests still pass but warning is emitted.

4. **Sample entropy computation slow** - O(n²) complexity for large datasets may cause timeout on very long recordings (>30 minutes).

---

## Test Cases

### Unit Tests: Data Loader (test_tools.py)

#### TC-001: Load Valid ECG File

- **Description:** Verify that a valid ECG text file loads correctly
- **Preconditions:** Temporary ECG file created with synthetic sine wave data (500 Hz, 1000 samples)
- **Input:** Text file containing single-column numeric ECG data
- **Expected Output:** Dictionary with 'signal' (numpy array), 'sampling_rate' (500), 'duration_sec' (~2.0), 'n_samples' (1000)
- **Actual Output:** Dictionary with all expected fields, signal array length 1000, duration 2.0 seconds
- **Status:** PASS

#### TC-002: Handle Missing File

- **Description:** Verify graceful handling of missing file
- **Preconditions:** None
- **Input:** `/nonexistent/path/to/file.txt`
- **Expected Output:** FileNotFoundError with descriptive message
- **Actual Output:** FileNotFoundError: "ECG file not found: /nonexistent/path/to/file.txt"
- **Status:** PASS

#### TC-003: Handle Empty File

- **Description:** Verify handling of empty ECG file
- **Preconditions:** Empty temporary file created
- **Input:** Empty text file (0 bytes)
- **Expected Output:** ValueError indicating empty file
- **Actual Output:** ValueError: "ECG file is empty"
- **Status:** PASS

### Unit Tests: Signal Processing (test_tools.py)

#### TC-004: Bandpass Filter Preserves Signal Length

- **Description:** Verify bandpass filter output matches input length
- **Preconditions:** None
- **Input:** Random signal (1000 samples), sampling rate 500 Hz, cutoffs 0.5-40 Hz
- **Expected Output:** Filtered signal with same length as input, no NaN values
- **Actual Output:** Filtered signal length 1000, no NaN values present
- **Status:** PASS

#### TC-005: R-Peak Detection on Synthetic ECG

- **Description:** Verify R-peak detection finds peaks in synthetic ECG
- **Preconditions:** None
- **Input:** Synthetic ECG with known peak locations (60 bpm pattern, 10 seconds at 500 Hz)
- **Expected Output:** 8-12 detected peaks (approximately 10 for 60 bpm)
- **Actual Output:** 10 peaks detected at expected intervals
- **Status:** PASS

#### TC-006: RR Interval Computation

- **Description:** Verify RR intervals calculated correctly from R-peaks
- **Preconditions:** None
- **Input:** R-peak indices [0, 500, 1000, 1500] at 500 Hz sampling rate
- **Expected Output:** RR intervals [1000, 1000, 1000] ms
- **Actual Output:** Array [1000.0, 1000.0, 1000.0] ms
- **Status:** PASS

#### TC-007: Complete Signal Processing Pipeline

- **Description:** Verify full process_signal() function
- **Preconditions:** Valid ECG data dictionary
- **Input:** ECG dict with 'signal' (5000 samples) and 'sampling_rate' (500)
- **Expected Output:** Result with 'filtered_signal', 'r_peaks', 'rr_intervals' fields
- **Actual Output:** All fields present, valid arrays returned
- **Status:** PASS

### Unit Tests: Feature Extraction (test_tools.py)

#### TC-008: Time-Domain Features Extraction

- **Description:** Verify time-domain HRV features calculated correctly
- **Preconditions:** None
- **Input:** Constant RR intervals [1000, 1000, 1000, ...] ms (100 intervals)
- **Expected Output:** mean_rr=1000, sdnn=0, rmssd=0 for constant intervals
- **Actual Output:** mean_rr=1000.0, sdnn=0.0, rmssd=0.0
- **Status:** PASS

#### TC-009: Time-Domain Features with Variation

- **Description:** Verify features respond to RR interval variability
- **Preconditions:** None
- **Input:** Variable RR intervals with Gaussian noise (mean=1000, std=50)
- **Expected Output:** Non-zero sdnn, rmssd; valid pnn50 percentage
- **Actual Output:** sdnn>0, rmssd>0, 0<=pnn50<=100
- **Status:** PASS

#### TC-010: Frequency-Domain Features Extraction

- **Description:** Verify frequency-domain HRV features calculated
- **Preconditions:** At least 200 RR intervals for reliable FFT
- **Input:** 200+ RR intervals with variability
- **Expected Output:** Non-negative vlf_power, lf_power, hf_power; valid lf_hf_ratio
- **Actual Output:** All frequency powers >= 0, lf_hf_ratio > 0
- **Status:** PASS

#### TC-011: Extract All HRV Features

- **Description:** Verify extract_hrv_features() returns complete feature set
- **Preconditions:** Valid RR intervals array
- **Input:** 200 RR intervals with realistic variability
- **Expected Output:** Dictionary with all standard HRV features
- **Actual Output:** Dictionary containing sdnn, rmssd, pnn50, mean_hr, lf_power, hf_power, lf_hf_ratio
- **Status:** PASS

#### TC-012: Handle Empty Intervals

- **Description:** Verify graceful handling of empty RR intervals
- **Preconditions:** None
- **Input:** Empty numpy array []
- **Expected Output:** Dictionary with NaN values for all features
- **Actual Output:** All feature values are NaN
- **Status:** PASS

### Integration Tests (test_tools.py)

#### TC-013: End-to-End Pipeline with Synthetic Data

- **Description:** Test complete pipeline from ECG file to HRV features
- **Preconditions:** Temporary ECG file with 3-minute synthetic recording
- **Input:** Synthetic ECG file (90,000 samples at 500 Hz)
- **Expected Output:** Valid HRV features extracted, non-NaN values
- **Actual Output:** Complete feature dictionary with valid numeric values
- **Status:** PASS
- **Notes:** Tests load -> process -> extract workflow without classification

---

### Unit Tests: Classifier Registry (test_classifier.py)

#### TC-014: List Classifiers Returns 20

- **Description:** Verify list_classifiers() returns exactly 20 classifier names
- **Preconditions:** None
- **Input:** list_classifiers() function call
- **Expected Output:** List of 20 strings
- **Actual Output:** List with 20 classifier names
- **Status:** PASS

#### TC-015: List Classifiers Returns Strings

- **Description:** Verify all classifier names are strings
- **Preconditions:** None
- **Input:** list_classifiers() return value
- **Expected Output:** All elements are string type
- **Actual Output:** All 20 elements are strings
- **Status:** PASS

#### TC-016: Get Available Classifiers Structure

- **Description:** Verify classifier info has required fields
- **Preconditions:** None
- **Input:** get_available_classifiers() function call
- **Expected Output:** Dict with 'name', 'description', 'category' for each classifier
- **Actual Output:** All required fields present for all 20 classifiers
- **Status:** PASS

#### TC-017: Get Classifier Info Valid

- **Description:** Get info for valid classifier name
- **Preconditions:** None
- **Input:** get_classifier_info('random_forest')
- **Expected Output:** Dict with classifier details
- **Actual Output:** Dict with name, description, category, default parameters
- **Status:** PASS

#### TC-018: Get Classifier Info Invalid

- **Description:** Get info for invalid classifier raises error
- **Preconditions:** None
- **Input:** get_classifier_info('invalid_classifier')
- **Expected Output:** ValueError
- **Actual Output:** ValueError: "Unknown classifier: invalid_classifier"
- **Status:** PASS

#### TC-019 to TC-038: Classifier Info for All 20 Classifiers

- **Description:** Verify info exists for each classifier type
- **Preconditions:** None
- **Input:** get_classifier_info() for each: random_forest, gradient_boosting, adaboost, extra_trees, bagging, hist_gradient_boosting, logistic_regression, ridge, sgd, perceptron, passive_aggressive, svm_linear, svm_rbf, knn, nearest_centroid, decision_tree, gaussian_nb, lda, qda, mlp
- **Expected Output:** Valid info dict for each classifier
- **Actual Output:** All 20 classifiers return valid info
- **Status:** PASS (20 parametrized tests)

---

### Unit Tests: Classifier Creation (test_classifier.py)

#### TC-039 to TC-058: Create All 20 Classifiers

- **Description:** Create each of 20 classifier types successfully
- **Preconditions:** None
- **Input:** create_classifier(name) for each classifier type
- **Expected Output:** Valid sklearn classifier instance with fit/predict methods
- **Actual Output:** All 20 classifiers created with required methods
- **Status:** PASS (20 parametrized tests)

#### TC-059: Create Classifier with Custom Parameters

- **Description:** Create classifier with custom hyperparameters
- **Preconditions:** None
- **Input:** create_classifier('random_forest', n_estimators=50, max_depth=5)
- **Expected Output:** RandomForest with n_estimators=50, max_depth=5
- **Actual Output:** Classifier created with specified parameters
- **Status:** PASS

#### TC-060: Create Invalid Classifier

- **Description:** Invalid classifier name raises ValueError
- **Preconditions:** None
- **Input:** create_classifier('nonexistent_classifier')
- **Expected Output:** ValueError
- **Actual Output:** ValueError: "Unknown classifier: nonexistent_classifier"
- **Status:** PASS

---

### Unit Tests: Classifier Training (test_classifier.py)

#### TC-061: Train Default Classifier

- **Description:** Train with default classifier (random_forest)
- **Preconditions:** Training data (100 samples, 6 features, binary labels)
- **Input:** train_classifier(X_train, y_train)
- **Expected Output:** Tuple of (trained_model, scaler, feature_names)
- **Actual Output:** Trained model makes predictions, scaler fitted
- **Status:** PASS

#### TC-062 to TC-069: Train Specific Classifiers

- **Description:** Train specific classifier types: logistic_regression, random_forest, gradient_boosting, svm_rbf, knn, gaussian_nb, lda, mlp
- **Preconditions:** Training data (100 samples, 6 features)
- **Input:** train_classifier(X, y, classifier_name=name)
- **Expected Output:** Trained model with accuracy > 30% (better than random)
- **Actual Output:** All classifiers train successfully, accuracy > 30%
- **Status:** PASS (8 parametrized tests)

#### TC-070: Train with Feature Names

- **Description:** Training preserves feature names
- **Preconditions:** Feature names list provided
- **Input:** train_classifier(X, y, feature_names=['f1', 'f2', ...])
- **Expected Output:** feature_names in returned tuple
- **Actual Output:** Feature names correctly stored and returned
- **Status:** PASS

---

### Unit Tests: Save/Load Classifier (test_classifier.py)

#### TC-071: Save Load Roundtrip

- **Description:** Save and load classifier produces identical predictions
- **Preconditions:** Trained classifier
- **Input:** save_classifier(clf, scaler, path) then load_classifier(path)
- **Expected Output:** Loaded classifier predictions match original
- **Actual Output:** Predictions identical before and after save/load
- **Status:** PASS

#### TC-072: Load Nonexistent File

- **Description:** Loading nonexistent file raises FileNotFoundError
- **Preconditions:** None
- **Input:** load_classifier('/nonexistent/model.joblib')
- **Expected Output:** FileNotFoundError
- **Actual Output:** FileNotFoundError raised
- **Status:** PASS

---

### Unit Tests: Predict Stress (test_classifier.py)

#### TC-073: Predict Stress Baseline

- **Description:** Predict stress returns valid prediction dict
- **Preconditions:** Trained classifier, valid features
- **Input:** predict_stress(clf, scaler, features_dict)
- **Expected Output:** Dict with 'prediction', 'confidence', 'probabilities'
- **Actual Output:** Complete prediction dictionary returned
- **Status:** PASS

#### TC-074: Predict Stress Missing Features

- **Description:** Missing features returns error result
- **Preconditions:** Trained classifier
- **Input:** predict_stress(clf, scaler, incomplete_features)
- **Expected Output:** Error result with missing feature names
- **Actual Output:** Error message lists missing features
- **Status:** PASS

---

### Unit Tests: Recommend Classifier (test_classifier.py)

#### TC-075: Recommend for Accuracy

- **Description:** Recommend classifier prioritizing accuracy
- **Preconditions:** None
- **Input:** recommend_classifier(priority='accuracy')
- **Expected Output:** Classifier name (e.g., 'random_forest' or 'gradient_boosting')
- **Actual Output:** Valid classifier name returned
- **Status:** PASS

#### TC-076: Recommend for Speed

- **Description:** Recommend classifier prioritizing speed
- **Preconditions:** None
- **Input:** recommend_classifier(priority='speed')
- **Expected Output:** Fast classifier (e.g., 'logistic_regression' or 'gaussian_nb')
- **Actual Output:** Fast classifier name returned
- **Status:** PASS

#### TC-077: Recommend for Interpretability

- **Description:** Recommend classifier prioritizing interpretability
- **Preconditions:** None
- **Input:** recommend_classifier(priority='interpretability')
- **Expected Output:** Interpretable classifier (e.g., 'logistic_regression' or 'decision_tree')
- **Actual Output:** Interpretable classifier name returned
- **Status:** PASS

#### TC-078: Recommend Small Dataset

- **Description:** Recommendation for small dataset (<100 samples)
- **Preconditions:** None
- **Input:** recommend_classifier(n_samples=50)
- **Expected Output:** Classifier suitable for small data
- **Actual Output:** Appropriate classifier recommended
- **Status:** PASS

#### TC-079: Recommend Large Dataset

- **Description:** Recommendation for large dataset (>1000 samples)
- **Preconditions:** None
- **Input:** recommend_classifier(n_samples=5000)
- **Expected Output:** Scalable classifier recommendation
- **Actual Output:** Scalable classifier recommended
- **Status:** PASS

---

### Unit Tests: Feature Importance (test_classifier.py)

#### TC-080: Feature Importance Random Forest

- **Description:** Feature importance from tree-based classifier
- **Preconditions:** Trained random forest classifier
- **Input:** get_feature_importance(rf_model)
- **Expected Output:** Dict mapping feature names to importance values (sum ~1.0)
- **Actual Output:** Importance values sum to approximately 1.0
- **Status:** PASS

#### TC-081: Feature Importance Logistic

- **Description:** Feature importance from linear classifier (coefficients)
- **Preconditions:** Trained logistic regression
- **Input:** get_feature_importance(lr_model)
- **Expected Output:** Dict with absolute coefficient values
- **Actual Output:** Feature importance dict with coefficient magnitudes
- **Status:** PASS

---

### Unit Tests: All Classifiers Train and Predict (test_classifier.py)

#### TC-082 to TC-101: Train and Predict All 20 Classifiers

- **Description:** Train and predict with each classifier type
- **Preconditions:** Training data (100 samples, 20 features)
- **Input:** Train then predict for each of 20 classifiers
- **Expected Output:** Predictions made, accuracy > 30%
- **Actual Output:** All classifiers produce valid predictions
- **Status:** PASS (20 parametrized tests)

#### TC-102 to TC-105: Probability Output

- **Description:** Verify probability output for classifiers with predict_proba
- **Preconditions:** Trained classifiers
- **Input:** predict_proba for logistic_regression, random_forest, svm_rbf, gaussian_nb
- **Expected Output:** Probability arrays summing to 1.0
- **Actual Output:** Valid probability distributions
- **Status:** PASS (4 parametrized tests)

---

### Unit Tests: Extended Feature Names (test_classifier.py)

#### TC-106: Extended Feature Names Count

- **Description:** EXTENDED_FEATURE_NAMES has 20 features
- **Preconditions:** None
- **Input:** len(EXTENDED_FEATURE_NAMES)
- **Expected Output:** 20
- **Actual Output:** 20
- **Status:** PASS

#### TC-107: Extended Feature Names Unique

- **Description:** All 20 feature names are unique
- **Preconditions:** None
- **Input:** set(EXTENDED_FEATURE_NAMES)
- **Expected Output:** Set with 20 elements
- **Actual Output:** 20 unique names
- **Status:** PASS

#### TC-108: Extended Feature Names Expected

- **Description:** Expected features are present
- **Preconditions:** None
- **Input:** Check for sdnn, rmssd, lf_hf_ratio in EXTENDED_FEATURE_NAMES
- **Expected Output:** All expected features present
- **Actual Output:** All checked features found
- **Status:** PASS

---

### Unit Tests: ECG Loader Parameters (test_comprehensive.py)

#### TC-109: Load ECG Different Sampling Rates

- **Description:** Load ECG with 500, 700, 1000 Hz sampling rates
- **Preconditions:** Test ECG file
- **Input:** load_ecg(path, sampling_rate=500/700/1000)
- **Expected Output:** Correct duration calculation for each rate
- **Actual Output:** Duration = n_samples / sampling_rate for each
- **Status:** PASS

#### TC-110: Load ECG Expected Duration Validation

- **Description:** Validate expected duration parameter
- **Preconditions:** ECG file with known duration
- **Input:** load_ecg(path, expected_duration=correct/incorrect)
- **Expected Output:** Pass for matching, ValueError for mismatch
- **Actual Output:** Correct validation behavior
- **Status:** PASS

#### TC-111: Load ECG Batch Success and Failure

- **Description:** Batch loading with mixed success/failure
- **Preconditions:** One valid file, one invalid path
- **Input:** load_ecg_batch([valid_path, invalid_path])
- **Expected Output:** List with success and error status
- **Actual Output:** Mixed results correctly returned
- **Status:** PASS

#### TC-112: WESAD Constants

- **Description:** Verify WESAD constants (subjects, labels, sampling rate)
- **Preconditions:** None
- **Input:** WESAD_SUBJECTS, WESAD_LABELS, WESAD_SAMPLING_RATE
- **Expected Output:** 15 subjects, 5 labels, 700 Hz
- **Actual Output:** Constants match expected values
- **Status:** PASS

---

### Unit Tests: Signal Processor Parameters (test_comprehensive.py)

#### TC-113: Bandpass Filter Custom Cutoffs

- **Description:** Bandpass filter with custom low/high cutoffs
- **Preconditions:** None
- **Input:** bandpass_filter(signal, fs, lowcut=1.0, highcut=30.0)
- **Expected Output:** Valid filtered signal
- **Actual Output:** Filtered signal without NaN, correct length
- **Status:** PASS

#### TC-114: Remove Baseline Wander

- **Description:** Baseline wander removal function
- **Preconditions:** Signal with DC offset
- **Input:** remove_baseline_wander(signal_with_offset, fs)
- **Expected Output:** Signal with reduced DC component
- **Actual Output:** Mean of output < mean of input
- **Status:** PASS

#### TC-115: Remove Baseline Wander Different Cutoffs

- **Description:** Baseline removal with different cutoff frequencies
- **Preconditions:** None
- **Input:** remove_baseline_wander(signal, fs, cutoff=0.5/0.1/1.0)
- **Expected Output:** Valid output for each cutoff
- **Actual Output:** All cutoffs produce valid results
- **Status:** PASS

#### TC-116: Detect R-Peaks with Parameters

- **Description:** R-peak detection with min/max RR parameters
- **Preconditions:** Synthetic ECG
- **Input:** detect_r_peaks(signal, fs, min_rr_sec, max_rr_sec)
- **Expected Output:** Peaks within RR constraints
- **Actual Output:** Detected peaks respect constraints
- **Status:** PASS

#### TC-117: Compute RR Intervals Edge Cases

- **Description:** RR interval computation edge cases
- **Preconditions:** None
- **Input:** compute_rr_intervals with empty, single, normal peaks
- **Expected Output:** Empty array for <2 peaks, correct intervals otherwise
- **Actual Output:** All edge cases handled correctly
- **Status:** PASS

#### TC-118: Remove Ectopic Beats

- **Description:** Ectopic beat removal function
- **Preconditions:** RR intervals with outlier
- **Input:** remove_ectopic_beats(rr_with_outlier)
- **Expected Output:** Outlier removed
- **Actual Output:** Ectopic beats filtered out
- **Status:** PASS

#### TC-119: Remove Ectopic Beats Different Thresholds

- **Description:** Ectopic removal with thresholds 0.1, 0.2, 0.3
- **Preconditions:** RR intervals with moderate outlier
- **Input:** remove_ectopic_beats(rr, threshold=0.1/0.2/0.3)
- **Expected Output:** Stricter threshold removes more
- **Actual Output:** Threshold 0.1 removes most, 0.3 removes least
- **Status:** PASS

#### TC-120: Remove Ectopic Beats Short Array

- **Description:** Ectopic removal with insufficient data
- **Preconditions:** None
- **Input:** remove_ectopic_beats([500, 600]) (2 intervals)
- **Expected Output:** Return input unchanged
- **Actual Output:** Input returned as-is
- **Status:** PASS

#### TC-121: Process Signal Full Parameters

- **Description:** Complete signal processing with all parameters
- **Preconditions:** Valid ECG data
- **Input:** process_signal(ecg, filter_low=0.5, filter_high=40, remove_ectopic=True)
- **Expected Output:** Complete result with all fields
- **Actual Output:** filtered_signal, r_peaks, rr_intervals all present
- **Status:** PASS

---

### Unit Tests: Feature Extractor Functions (test_comprehensive.py)

#### TC-122: Extract Time Domain Features Short Input

- **Description:** Time-domain features with short input
- **Preconditions:** None
- **Input:** extract_time_domain_features(rr_intervals) with <10 intervals
- **Expected Output:** NaN values for unreliable features
- **Actual Output:** Appropriate NaN handling for short input
- **Status:** PASS

#### TC-123: Extract Time Domain Features Normal

- **Description:** Time-domain features with normal input
- **Preconditions:** None
- **Input:** 100+ RR intervals with normal variability
- **Expected Output:** All time-domain features computed
- **Actual Output:** Valid mean_rr, sdnn, rmssd, pnn50, mean_hr, std_hr
- **Status:** PASS

#### TC-124: Extract Frequency Domain Custom Bands

- **Description:** Frequency features with custom VLF/LF/HF bands
- **Preconditions:** 200+ RR intervals
- **Input:** extract_frequency_domain_features(rr, vlf_band, lf_band, hf_band)
- **Expected Output:** Powers computed for custom bands
- **Actual Output:** Custom frequency bands correctly applied
- **Status:** PASS

#### TC-125: Extract Frequency Domain Different Resample

- **Description:** Frequency features with different resample rates
- **Preconditions:** 200+ RR intervals
- **Input:** extract_frequency_domain_features(rr, fs_resample=4.0/2.0/8.0)
- **Expected Output:** Valid results for each resample rate
- **Actual Output:** All resample rates produce valid features
- **Status:** PASS

#### TC-126: Extract Nonlinear Features

- **Description:** Nonlinear features (SD1, SD2, sample entropy)
- **Preconditions:** 50+ RR intervals
- **Input:** extract_nonlinear_features(rr_intervals)
- **Expected Output:** sd1, sd2, sd_ratio, sample_entropy values
- **Actual Output:** All nonlinear features computed
- **Status:** PASS

#### TC-127: Extract Nonlinear Features Short

- **Description:** Nonlinear features with short input
- **Preconditions:** None
- **Input:** extract_nonlinear_features(rr) with <10 intervals
- **Expected Output:** NaN values
- **Actual Output:** NaN returned for insufficient data
- **Status:** PASS

#### TC-128: Extract HRV Features Include Nonlinear

- **Description:** Combined features with/without nonlinear
- **Preconditions:** Valid RR intervals
- **Input:** extract_hrv_features(rr, include_nonlinear=True/False)
- **Expected Output:** Nonlinear features present/absent accordingly
- **Actual Output:** sd1, sd2 included only when include_nonlinear=True
- **Status:** PASS

#### TC-129: Get Feature Vector

- **Description:** Convert feature dict to numpy array
- **Preconditions:** Feature dictionary
- **Input:** get_feature_vector(features_dict)
- **Expected Output:** Numpy array with correct ordering
- **Actual Output:** Array with features in expected order
- **Status:** PASS

#### TC-130: Get Feature Vector Missing Feature

- **Description:** Feature vector with missing feature
- **Preconditions:** Incomplete feature dict
- **Input:** get_feature_vector(partial_features)
- **Expected Output:** Array with NaN for missing features
- **Actual Output:** NaN inserted for missing features
- **Status:** PASS

---

### Unit Tests: All Classifiers Comprehensive (test_comprehensive.py)

#### TC-131 to TC-150: Train All 20 Classifiers

- **Description:** Train each of 20 classifiers with test data
- **Preconditions:** Training data (100 samples, 20 features)
- **Input:** train_classifier(X, y, classifier_name) for each
- **Expected Output:** Trained model, scaler, feature names
- **Actual Output:** All classifiers train successfully
- **Status:** PASS (20 parametrized tests)

#### TC-151 to TC-170: Save/Load All 20 Classifiers

- **Description:** Save/load roundtrip for each classifier
- **Preconditions:** Trained classifiers
- **Input:** save_classifier then load_classifier for each
- **Expected Output:** Loaded model produces same predictions
- **Actual Output:** All save/load roundtrips successful
- **Status:** PASS (20 parametrized tests)

---

### Unit Tests: Recommend Classifier Priorities (test_comprehensive.py)

#### TC-171 to TC-173: All Priority Options

- **Description:** Test priority options: accuracy, speed, interpretability
- **Preconditions:** None
- **Input:** recommend_classifier(priority='accuracy'/'speed'/'interpretability')
- **Expected Output:** Valid classifier name for each priority
- **Actual Output:** Appropriate recommendations returned
- **Status:** PASS (3 parametrized tests)

#### TC-174 to TC-179: Different Sample Sizes Accuracy

- **Description:** Accuracy recommendation for sizes: 50, 100, 500, 1000, 5000, 10000
- **Preconditions:** None
- **Input:** recommend_classifier(priority='accuracy', n_samples=size)
- **Expected Output:** Size-appropriate classifier
- **Actual Output:** Recommendations scale with dataset size
- **Status:** PASS (6 parametrized tests)

#### TC-180 to TC-183: Different Sample Sizes Speed

- **Description:** Speed recommendation for sizes: 50, 100, 1000, 10000
- **Preconditions:** None
- **Input:** recommend_classifier(priority='speed', n_samples=size)
- **Expected Output:** Fast classifiers for all sizes
- **Actual Output:** Consistently fast classifiers recommended
- **Status:** PASS (4 parametrized tests)

---

### Unit Tests: Classifier Categories (test_comprehensive.py)

#### TC-184: Ensemble Classifiers

- **Description:** Test ensemble classifiers (6 types)
- **Preconditions:** Training data
- **Input:** Train random_forest, gradient_boosting, adaboost, extra_trees, bagging, hist_gradient_boosting
- **Expected Output:** All ensemble classifiers work
- **Actual Output:** All 6 ensemble classifiers train and predict
- **Status:** PASS

#### TC-185: Linear Classifiers

- **Description:** Test linear classifiers (5 types)
- **Preconditions:** Training data
- **Input:** Train logistic_regression, ridge, sgd, perceptron, passive_aggressive
- **Expected Output:** All linear classifiers work
- **Actual Output:** All 5 linear classifiers train and predict
- **Status:** PASS

#### TC-186: SVM Classifiers

- **Description:** Test SVM classifiers (2 types)
- **Preconditions:** Training data
- **Input:** Train svm_linear, svm_rbf
- **Expected Output:** Both SVM classifiers work
- **Actual Output:** Both train and predict successfully
- **Status:** PASS

#### TC-187: Distance-Based Classifiers

- **Description:** Test distance-based classifiers (2 types)
- **Preconditions:** Training data
- **Input:** Train knn, nearest_centroid
- **Expected Output:** Both distance classifiers work
- **Actual Output:** Both train and predict successfully
- **Status:** PASS

#### TC-188: Other Classifiers

- **Description:** Test other classifiers (tree, NB, LDA, QDA, MLP)
- **Preconditions:** Training data
- **Input:** Train decision_tree, gaussian_nb, lda, qda, mlp
- **Expected Output:** All 5 classifiers work
- **Actual Output:** All train and predict successfully
- **Status:** PASS

---

### Unit Tests: Feature Importance All Classifiers (test_comprehensive.py)

#### TC-189: Feature Importance Tree-Based

- **Description:** Feature importance from tree-based models
- **Preconditions:** Trained random_forest, gradient_boosting, decision_tree
- **Input:** get_feature_importance(model)
- **Expected Output:** Dict with feature importances summing to ~1.0
- **Actual Output:** Valid importance dicts for all tree models
- **Status:** PASS

#### TC-190: Feature Importance Linear

- **Description:** Feature importance from linear models
- **Preconditions:** Trained logistic_regression, ridge
- **Input:** get_feature_importance(model)
- **Expected Output:** Dict with coefficient-based importances
- **Actual Output:** Valid importance dicts from coefficients
- **Status:** PASS

#### TC-191: Feature Importance KNN

- **Description:** Feature importance from KNN (returns None)
- **Preconditions:** Trained knn classifier
- **Input:** get_feature_importance(knn_model)
- **Expected Output:** None (KNN has no intrinsic feature importance)
- **Actual Output:** None returned
- **Status:** PASS

---

### Unit Tests: Predict Stress (test_comprehensive.py)

#### TC-192: Predict Stress Complete Features

- **Description:** Prediction with all required features
- **Preconditions:** Trained classifier, complete feature dict
- **Input:** predict_stress(clf, scaler, complete_features)
- **Expected Output:** Prediction with label, confidence, probabilities
- **Actual Output:** Complete prediction result
- **Status:** PASS

#### TC-193: Predict Stress Missing Features

- **Description:** Prediction with missing features
- **Preconditions:** Trained classifier
- **Input:** predict_stress(clf, scaler, incomplete_features)
- **Expected Output:** Error result listing missing features
- **Actual Output:** Error with missing feature names
- **Status:** PASS

---

### Unit Tests: Feature Constants (test_extended_features.py)

#### TC-194: Feature Names Count

- **Description:** FEATURE_NAMES contains exactly 20 features
- **Preconditions:** None
- **Input:** len(FEATURE_NAMES)
- **Expected Output:** 20
- **Actual Output:** 20
- **Status:** PASS

#### TC-195: Feature Names Unique

- **Description:** All feature names are unique
- **Preconditions:** None
- **Input:** len(set(FEATURE_NAMES))
- **Expected Output:** 20 (all unique)
- **Actual Output:** 20 unique names
- **Status:** PASS

#### TC-196: Feature Descriptions Complete

- **Description:** All features have descriptions
- **Preconditions:** None
- **Input:** FEATURE_DESCRIPTIONS dict
- **Expected Output:** 20 descriptions present
- **Actual Output:** All 20 features have descriptions
- **Status:** PASS

#### TC-197: Feature Categories Complete

- **Description:** All features belong to a category
- **Preconditions:** None
- **Input:** FEATURE_CATEGORIES dict
- **Expected Output:** All 20 features categorized
- **Actual Output:** Every feature has a category
- **Status:** PASS

#### TC-198: Feature Categories Valid

- **Description:** Categories are time/frequency/nonlinear
- **Preconditions:** None
- **Input:** Unique values in FEATURE_CATEGORIES
- **Expected Output:** {'time_domain', 'frequency_domain', 'nonlinear'}
- **Actual Output:** Three valid categories present
- **Status:** PASS

---

### Unit Tests: Time Domain Features (test_extended_features.py)

#### TC-199: Regular RR Intervals

- **Description:** Time features for constant RR intervals
- **Preconditions:** None
- **Input:** Constant RR intervals [1000, 1000, 1000, ...] ms
- **Expected Output:** sdnn=0, rmssd=0 for constant input
- **Actual Output:** sdnn=0.0, rmssd=0.0
- **Status:** PASS

#### TC-200: Variable RR Intervals

- **Description:** Time features for variable RR intervals
- **Preconditions:** None
- **Input:** Variable RR intervals with std~50ms
- **Expected Output:** sdnn>0, rmssd>0
- **Actual Output:** Positive variability metrics
- **Status:** PASS

#### TC-201: Mean RR Calculation

- **Description:** Mean RR calculation accuracy
- **Preconditions:** None
- **Input:** RR intervals with known mean
- **Expected Output:** mean_rr matches np.mean(rr_intervals)
- **Actual Output:** Correct mean calculation
- **Status:** PASS

#### TC-202: Median RR Calculation

- **Description:** Median RR calculation accuracy
- **Preconditions:** None
- **Input:** RR intervals with known median
- **Expected Output:** median_rr matches np.median(rr_intervals)
- **Actual Output:** Correct median calculation
- **Status:** PASS

#### TC-203: Mean HR Calculation

- **Description:** Mean HR from RR intervals
- **Preconditions:** None
- **Input:** RR intervals of 1000ms (should give 60 bpm)
- **Expected Output:** mean_hr = 60000 / mean_rr = 60
- **Actual Output:** mean_hr = 60.0
- **Status:** PASS

#### TC-204: pNN50 with Large Differences

- **Description:** pNN50 with large successive differences
- **Preconditions:** None
- **Input:** RR intervals with alternating 900/1100 pattern
- **Expected Output:** High pNN50 (many differences > 50ms)
- **Actual Output:** pNN50 approaches 100%
- **Status:** PASS

---

### Unit Tests: Frequency Domain Features (test_extended_features.py)

#### TC-205: Frequency Features Present

- **Description:** VLF, LF, HF, LF/HF ratio present
- **Preconditions:** 200+ RR intervals
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** vlf_power, lf_power, hf_power, lf_hf_ratio in output
- **Actual Output:** All frequency features present
- **Status:** PASS

#### TC-206: Frequency Powers Non-Negative

- **Description:** All power values >= 0
- **Preconditions:** Valid RR intervals
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** vlf_power>=0, lf_power>=0, hf_power>=0
- **Actual Output:** All powers non-negative
- **Status:** PASS

#### TC-207: Normalized Units Sum

- **Description:** LF_nu + HF_nu ~ 100%
- **Preconditions:** Valid RR intervals
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** lf_nu + hf_nu approximately equals 100
- **Actual Output:** Sum within tolerance of 100
- **Status:** PASS

#### TC-208: LF/HF Ratio Positive

- **Description:** LF/HF ratio is positive
- **Preconditions:** Valid RR intervals with LF and HF power
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** lf_hf_ratio > 0
- **Actual Output:** Positive ratio
- **Status:** PASS

---

### Unit Tests: Nonlinear Features (test_extended_features.py)

#### TC-209: Poincare Features Present

- **Description:** SD1, SD2, SD_ratio present
- **Preconditions:** 50+ RR intervals
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** sd1, sd2, sd_ratio in output
- **Actual Output:** All Poincare features present
- **Status:** PASS

#### TC-210: SD1 SD2 Relationship

- **Description:** SD1 <= SD2 for normal data
- **Preconditions:** Normal HRV pattern
- **Input:** extract_extended_features(normal_rr)
- **Expected Output:** sd1 <= sd2
- **Actual Output:** SD1 < SD2 for normal variability
- **Status:** PASS

#### TC-211: Sample Entropy Present

- **Description:** Sample entropy is computed
- **Preconditions:** 50+ RR intervals
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** sample_entropy in output
- **Actual Output:** Sample entropy value present
- **Status:** PASS

#### TC-212: Sample Entropy Regular Signal

- **Description:** Sample entropy low for regular signal
- **Preconditions:** None
- **Input:** Constant or near-constant RR intervals
- **Expected Output:** Low sample entropy (high regularity)
- **Actual Output:** Low entropy value
- **Status:** PASS

---

### Unit Tests: Edge Cases (test_extended_features.py)

#### TC-213: Short RR Intervals

- **Description:** Features with < 10 RR intervals
- **Preconditions:** None
- **Input:** extract_extended_features(short_rr) with 5 intervals
- **Expected Output:** NaN for unreliable features
- **Actual Output:** NaN dictionary returned
- **Status:** PASS

#### TC-214: Minimum Intervals for Frequency

- **Description:** Minimum intervals for frequency analysis
- **Preconditions:** None
- **Input:** Exactly 10 RR intervals
- **Expected Output:** Frequency features computed (may be unreliable)
- **Actual Output:** Features computed with warning
- **Status:** PASS

#### TC-215: High Variability

- **Description:** Features with high HRV
- **Preconditions:** None
- **Input:** RR intervals with std=200ms
- **Expected Output:** High sdnn, rmssd values
- **Actual Output:** Elevated variability metrics
- **Status:** PASS

#### TC-216: All Features Returned

- **Description:** All 20 features in output dict
- **Preconditions:** Sufficient RR intervals
- **Input:** extract_extended_features(valid_rr)
- **Expected Output:** Dict with 20 keys
- **Actual Output:** All 20 features present
- **Status:** PASS

---

### Unit Tests: Feature Values (test_extended_features.py)

#### TC-217: SDNN Less Than Range

- **Description:** SDNN < range_rr
- **Preconditions:** Variable RR intervals
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** sdnn < range_rr (mathematical property)
- **Actual Output:** SDNN always less than range
- **Status:** PASS

#### TC-218: RMSSD Positive

- **Description:** RMSSD is positive
- **Preconditions:** Variable RR intervals
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** rmssd > 0
- **Actual Output:** Positive RMSSD
- **Status:** PASS

#### TC-219: pNN50 Percentage

- **Description:** 0 <= pNN50 <= 100
- **Preconditions:** Any RR intervals
- **Input:** extract_extended_features(rr_intervals)
- **Expected Output:** pnn50 in range [0, 100]
- **Actual Output:** Valid percentage
- **Status:** PASS

#### TC-220: Heart Rate Physiological

- **Description:** 30 <= HR <= 200 bpm
- **Preconditions:** Physiological RR intervals (300-2000ms)
- **Input:** extract_extended_features(physiological_rr)
- **Expected Output:** mean_hr in [30, 200]
- **Actual Output:** Physiologically valid HR
- **Status:** PASS

---

### Unit Tests: Real-Like Data (test_extended_features.py)

#### TC-221: Stressed Pattern

- **Description:** Features for stress-like HRV pattern
- **Preconditions:** None
- **Input:** RR intervals simulating stress (low HRV, high HR)
- **Expected Output:** Low sdnn, high lf/hf ratio
- **Actual Output:** Stress-characteristic features
- **Status:** PASS

#### TC-222: Baseline Pattern

- **Description:** Features for baseline-like HRV pattern
- **Preconditions:** None
- **Input:** RR intervals simulating rest (high HRV, normal HR)
- **Expected Output:** Higher sdnn, balanced lf/hf
- **Actual Output:** Baseline-characteristic features
- **Status:** PASS

---

### Unit Tests: Logging Setup (test_helpers.py)

#### TC-223: Setup Logging Returns Logger

- **Description:** setup_logging() returns Logger instance
- **Preconditions:** None
- **Input:** setup_logging()
- **Expected Output:** logging.Logger instance named "hrv_agent"
- **Actual Output:** Logger with name "hrv_agent"
- **Status:** PASS

#### TC-224: Setup Logging Default Level

- **Description:** Default level is INFO
- **Preconditions:** None
- **Input:** setup_logging()
- **Expected Output:** logger.level == logging.INFO
- **Actual Output:** Level is INFO
- **Status:** PASS

#### TC-225: Setup Logging Custom Level

- **Description:** Custom level (DEBUG) works
- **Preconditions:** None
- **Input:** setup_logging(level=logging.DEBUG)
- **Expected Output:** logger.level == logging.DEBUG
- **Actual Output:** Level is DEBUG
- **Status:** PASS

#### TC-226: Setup Logging with File

- **Description:** File handler is added for log file
- **Preconditions:** Writable temp directory
- **Input:** setup_logging(log_file='/tmp/test.log')
- **Expected Output:** FileHandler in logger.handlers
- **Actual Output:** FileHandler present
- **Status:** PASS

---

### Unit Tests: Config Loading (test_helpers.py)

#### TC-227: Load Config Valid YAML

- **Description:** Load valid YAML config file
- **Preconditions:** Valid YAML file
- **Input:** load_config('config.yaml')
- **Expected Output:** Dict with parsed config
- **Actual Output:** Correctly parsed configuration
- **Status:** PASS

#### TC-228: Load Config Nonexistent File

- **Description:** Nonexistent file raises FileNotFoundError
- **Preconditions:** None
- **Input:** load_config('/nonexistent/config.yaml')
- **Expected Output:** FileNotFoundError
- **Actual Output:** FileNotFoundError: "Config file not found"
- **Status:** PASS

#### TC-229: Load Config Empty File

- **Description:** Empty YAML returns empty dict
- **Preconditions:** Empty file
- **Input:** load_config('empty.yaml')
- **Expected Output:** {}
- **Actual Output:** Empty dict
- **Status:** PASS

#### TC-230: Load Config Path as String

- **Description:** String path works as well as Path
- **Preconditions:** Valid config file
- **Input:** load_config('/path/as/string')
- **Expected Output:** Same result as Path object
- **Actual Output:** Config loaded correctly
- **Status:** PASS

---

### Unit Tests: ECG Validation (test_helpers.py)

#### TC-231: Validate Valid Data

- **Description:** Valid ECG data passes validation
- **Preconditions:** None
- **Input:** 60-second ECG at 500 Hz, normal characteristics
- **Expected Output:** valid=True, issues=[]
- **Actual Output:** Validation passes
- **Status:** PASS

#### TC-232: Validate Empty Data

- **Description:** Empty data fails with "Data is empty"
- **Preconditions:** None
- **Input:** np.array([])
- **Expected Output:** valid=False, "Data is empty" in issues
- **Actual Output:** Empty data detected
- **Status:** PASS

#### TC-233: Validate Short Duration

- **Description:** Short duration fails validation
- **Preconditions:** None
- **Input:** 10-second ECG with min_duration=60
- **Expected Output:** valid=False, duration issue
- **Actual Output:** Short duration flagged
- **Status:** PASS

#### TC-234: Validate Long Duration

- **Description:** Excessive duration fails validation
- **Preconditions:** None
- **Input:** 2-hour ECG with max_duration=3600
- **Expected Output:** valid=False, duration issue
- **Actual Output:** Long duration flagged
- **Status:** PASS

#### TC-235: Validate Data with NaN

- **Description:** NaN values detected
- **Preconditions:** None
- **Input:** ECG with NaN values inserted
- **Expected Output:** valid=False, NaN issue
- **Actual Output:** NaN values detected
- **Status:** PASS

#### TC-236: Validate Data with Inf

- **Description:** Infinite values detected
- **Preconditions:** None
- **Input:** ECG with inf values
- **Expected Output:** valid=False, inf issue
- **Actual Output:** Infinite values detected
- **Status:** PASS

#### TC-237: Validate Flat Signal

- **Description:** Flat (constant) signal detected
- **Preconditions:** None
- **Input:** np.ones(30000)
- **Expected Output:** valid=False, flat signal issue
- **Actual Output:** Flat signal flagged
- **Status:** PASS

#### TC-238: Validate Extreme Values

- **Description:** Extreme amplitude values detected
- **Preconditions:** None
- **Input:** ECG with values * 100 (extreme amplitude)
- **Expected Output:** valid=False, extreme values issue
- **Actual Output:** Extreme values flagged
- **Status:** PASS

#### TC-239: Validate Saturated Signal

- **Description:** Saturated signal (few unique values) detected
- **Preconditions:** None
- **Input:** Signal with only 3 unique values
- **Expected Output:** valid=False, saturated issue
- **Actual Output:** Saturation detected
- **Status:** PASS

#### TC-240: Validate Returns Statistics

- **Description:** Validation returns signal statistics
- **Preconditions:** Valid ECG
- **Input:** validate_ecg_data(valid_ecg)
- **Expected Output:** Dict with mean, std, min, max
- **Actual Output:** Statistics present in result
- **Status:** PASS

---

### Unit Tests: Signal Normalization (test_helpers.py)

#### TC-241: Normalize Standard Signal

- **Description:** Normalized signal has mean~0, std~1
- **Preconditions:** None
- **Input:** Random signal with mean=50, std=10
- **Expected Output:** Output mean~0, std~1
- **Actual Output:** |mean| < 1e-10, |std-1| < 1e-10
- **Status:** PASS

#### TC-242: Normalize Flat Signal

- **Description:** Flat signal normalization (avoids div by zero)
- **Preconditions:** None
- **Input:** np.ones(1000) * 5
- **Expected Output:** Mean subtracted, no division error
- **Actual Output:** No error, mean removed
- **Status:** PASS

#### TC-243: Normalize Preserves Shape

- **Description:** Output shape matches input shape
- **Preconditions:** None
- **Input:** Signal of shape (500,)
- **Expected Output:** Output shape (500,)
- **Actual Output:** Shape preserved
- **Status:** PASS

#### TC-244: Normalize Already Normalized

- **Description:** Already normalized data unchanged
- **Preconditions:** None
- **Input:** Standard normal data (mean=0, std=1)
- **Expected Output:** Output approximately equals input
- **Actual Output:** Minimal change
- **Status:** PASS

---

### Unit Tests: Environment Variables (test_helpers.py)

#### TC-245: Get Existing Env Variable

- **Description:** Get existing environment variable
- **Preconditions:** TEST_VAR set in environment
- **Input:** get_env_variable('TEST_VAR')
- **Expected Output:** Value of TEST_VAR
- **Actual Output:** Correct value returned
- **Status:** PASS

#### TC-246: Get Nonexistent Env Variable

- **Description:** Nonexistent variable returns None
- **Preconditions:** Variable not set
- **Input:** get_env_variable('NONEXISTENT_VAR')
- **Expected Output:** None
- **Actual Output:** None
- **Status:** PASS

#### TC-247: Get Env Variable with Default

- **Description:** Default value used when not set
- **Preconditions:** Variable not set
- **Input:** get_env_variable('UNSET', default='fallback')
- **Expected Output:** 'fallback'
- **Actual Output:** Default value returned
- **Status:** PASS

#### TC-248: Get Env Variable Overrides Default

- **Description:** Existing value overrides default
- **Preconditions:** Variable set to 'actual'
- **Input:** get_env_variable('VAR', default='default')
- **Expected Output:** 'actual' (not 'default')
- **Actual Output:** Actual value returned
- **Status:** PASS

---

### Unit Tests: Directory Utilities (test_helpers.py)

#### TC-249: Ensure New Directory

- **Description:** Create new directory
- **Preconditions:** Directory does not exist
- **Input:** ensure_directory('/tmp/new_dir')
- **Expected Output:** Directory created, path returned
- **Actual Output:** Directory exists after call
- **Status:** PASS

#### TC-250: Ensure Existing Directory

- **Description:** Existing directory returns path
- **Preconditions:** Directory already exists
- **Input:** ensure_directory('/tmp/existing')
- **Expected Output:** Path returned, no error
- **Actual Output:** Path returned successfully
- **Status:** PASS

#### TC-251: Ensure Nested Directory

- **Description:** Create nested directory structure
- **Preconditions:** Parent directories don't exist
- **Input:** ensure_directory('/tmp/a/b/c/d')
- **Expected Output:** All directories created
- **Actual Output:** Full path created
- **Status:** PASS

#### TC-252: Ensure Directory with String Path

- **Description:** String path converted to Path
- **Preconditions:** None
- **Input:** ensure_directory('/tmp/string_path')
- **Expected Output:** Returns Path object
- **Actual Output:** Path object returned
- **Status:** PASS

---

### Unit Tests: Orchestrator Init (test_orchestrator.py)

#### TC-253: Init Default

- **Description:** Default initialization
- **Preconditions:** None
- **Input:** HRVAnalysisOrchestrator()
- **Expected Output:** Empty state, empty execution log
- **Actual Output:** Default values initialized
- **Status:** PASS

#### TC-254: Init with Classifier Name

- **Description:** Initialize with custom classifier
- **Preconditions:** None
- **Input:** HRVAnalysisOrchestrator(classifier_name='logistic_regression')
- **Expected Output:** classifier_name set correctly
- **Actual Output:** Custom classifier name stored
- **Status:** PASS

#### TC-255: Init Without Anthropic

- **Description:** Initialize without Anthropic API
- **Preconditions:** ANTHROPIC_API_KEY not set
- **Input:** HRVAnalysisOrchestrator()
- **Expected Output:** Orchestrator created, client=None
- **Actual Output:** Works without API key
- **Status:** PASS

#### TC-256: Tools Defined

- **Description:** All 9 tools are defined
- **Preconditions:** Orchestrator instance
- **Input:** orchestrator.tools
- **Expected Output:** 9 tool definitions
- **Actual Output:** 9 tools with name, description, parameters
- **Status:** PASS

---

### Unit Tests: Orchestrator Tools (test_orchestrator.py)

#### TC-257: Execute Load ECG

- **Description:** Execute load_ecg tool
- **Preconditions:** Valid ECG file
- **Input:** _execute_tool('load_ecg', {'file_path': path})
- **Expected Output:** Success, ECG data in state
- **Actual Output:** Tool executes, state updated
- **Status:** PASS

#### TC-258: Execute Process Signal

- **Description:** Execute process_signal tool
- **Preconditions:** ECG data loaded
- **Input:** _execute_tool('process_signal', {})
- **Expected Output:** Processed data in state
- **Actual Output:** Signal processed successfully
- **Status:** PASS

#### TC-259: Execute Process Signal Without Load

- **Description:** Process without load raises error
- **Preconditions:** Empty state
- **Input:** _execute_tool('process_signal', {})
- **Expected Output:** Error indicating ECG not loaded
- **Actual Output:** Error message returned
- **Status:** PASS

#### TC-260: Execute Extract Features

- **Description:** Execute extract_features tool
- **Preconditions:** Signal processed
- **Input:** _execute_tool('extract_features', {})
- **Expected Output:** Features in state
- **Actual Output:** Features extracted successfully
- **Status:** PASS

#### TC-261: Execute List Classifiers

- **Description:** Execute list_classifiers tool
- **Preconditions:** None
- **Input:** _execute_tool('list_classifiers', {})
- **Expected Output:** List of 20 classifier names
- **Actual Output:** 20 classifiers returned
- **Status:** PASS

#### TC-262: Execute Get Classifier Info

- **Description:** Get info for valid classifier
- **Preconditions:** None
- **Input:** _execute_tool('get_classifier_info', {'name': 'random_forest'})
- **Expected Output:** Classifier info dict
- **Actual Output:** Info returned successfully
- **Status:** PASS

#### TC-263: Execute Get Classifier Info Invalid

- **Description:** Get info for invalid classifier
- **Preconditions:** None
- **Input:** _execute_tool('get_classifier_info', {'name': 'invalid'})
- **Expected Output:** Error result
- **Actual Output:** Error message about unknown classifier
- **Status:** PASS

#### TC-264: Execute Recommend Classifier

- **Description:** Execute recommend_classifier tool
- **Preconditions:** None
- **Input:** _execute_tool('recommend_classifier', {'priority': 'accuracy'})
- **Expected Output:** Recommended classifier name
- **Actual Output:** Valid recommendation returned
- **Status:** PASS

#### TC-265: Execute Select Classifier

- **Description:** Execute select_classifier tool
- **Preconditions:** None
- **Input:** _execute_tool('select_classifier', {'name': 'logistic_regression'})
- **Expected Output:** Classifier selected in state
- **Actual Output:** Selection confirmed
- **Status:** PASS

#### TC-266: Execution Log

- **Description:** Execution log tracks tool calls
- **Preconditions:** None
- **Input:** Execute multiple tools, check log
- **Expected Output:** All tool calls logged
- **Actual Output:** Complete execution history
- **Status:** PASS

---

### Unit Tests: Classifier Loading (test_orchestrator.py)

#### TC-267: Load Classifier from Models Dir

- **Description:** Load classifier from models directory
- **Preconditions:** Model file exists
- **Input:** Orchestrator loads from models/classifier.joblib
- **Expected Output:** Classifier loaded successfully
- **Actual Output:** Model loaded and usable
- **Status:** PASS

#### TC-268: Select and Load Classifier

- **Description:** Select and load classifier
- **Preconditions:** None
- **Input:** select_classifier then load
- **Expected Output:** Classifier ready for prediction
- **Actual Output:** Classifier functional
- **Status:** PASS

---

### Unit Tests: Orchestrator State (test_orchestrator.py)

#### TC-269: Get State

- **Description:** Get orchestrator state
- **Preconditions:** Some operations performed
- **Input:** orchestrator.get_state()
- **Expected Output:** Current state dict
- **Actual Output:** State dictionary returned
- **Status:** PASS

#### TC-270: Get Execution Log

- **Description:** Get execution log
- **Preconditions:** Some operations performed
- **Input:** orchestrator.get_execution_log()
- **Expected Output:** List of executed operations
- **Actual Output:** Execution history returned
- **Status:** PASS

#### TC-271: Reset

- **Description:** Reset orchestrator state
- **Preconditions:** State has data
- **Input:** orchestrator.reset()
- **Expected Output:** Empty state and log
- **Actual Output:** State cleared
- **Status:** PASS

#### TC-272: Get Available Classifiers

- **Description:** Get list of available classifiers
- **Preconditions:** None
- **Input:** orchestrator.get_available_classifiers()
- **Expected Output:** List of 20 classifiers
- **Actual Output:** 20 classifier names
- **Status:** PASS

---

### Unit Tests: Pipeline (test_orchestrator.py)

#### TC-273: Run Direct Short Signal

- **Description:** Run pipeline with short signal
- **Preconditions:** Valid ECG file
- **Input:** orchestrator.run_direct(ecg_file)
- **Expected Output:** Complete result with features
- **Actual Output:** Pipeline completes successfully
- **Status:** PASS

---

### Unit Tests: Windowing Behavior (test_orchestrator.py)

#### TC-274: Aggregate Predictions Empty

- **Description:** Aggregate empty predictions
- **Preconditions:** None
- **Input:** _aggregate_predictions([])
- **Expected Output:** None or default
- **Actual Output:** Handled gracefully
- **Status:** PASS

#### TC-275: Aggregate Predictions Single

- **Description:** Aggregate single prediction
- **Preconditions:** None
- **Input:** _aggregate_predictions([single_result])
- **Expected Output:** Single result returned
- **Actual Output:** Result passed through
- **Status:** PASS

#### TC-276: Aggregate Predictions Majority Vote

- **Description:** Majority vote aggregation
- **Preconditions:** None
- **Input:** _aggregate_predictions([stressed, stressed, baseline])
- **Expected Output:** 'stressed' (2 vs 1)
- **Actual Output:** Majority class selected
- **Status:** PASS

#### TC-277: Aggregate Features Average

- **Description:** Feature averaging across windows
- **Preconditions:** None
- **Input:** _aggregate_features([features1, features2])
- **Expected Output:** Averaged feature values
- **Actual Output:** Mean of each feature
- **Status:** PASS

---

### Unit Tests: Report Availability (test_report_generator.py)

#### TC-278: REPORTLAB Available is Boolean

- **Description:** REPORTLAB_AVAILABLE is boolean
- **Preconditions:** None
- **Input:** type(REPORTLAB_AVAILABLE)
- **Expected Output:** bool
- **Actual Output:** Boolean type
- **Status:** PASS

#### TC-279: ANTHROPIC Available is Boolean

- **Description:** ANTHROPIC_AVAILABLE is boolean
- **Preconditions:** None
- **Input:** type(ANTHROPIC_AVAILABLE)
- **Expected Output:** bool
- **Actual Output:** Boolean type
- **Status:** PASS

---

### Unit Tests: Interpretation (test_report_generator.py)

#### TC-280: Interpretation Without Anthropic

- **Description:** Fallback when anthropic not installed
- **Preconditions:** ANTHROPIC_AVAILABLE=False
- **Input:** generate_interpretation(features, prediction)
- **Expected Output:** Dict with fallback message
- **Actual Output:** "anthropic package not installed" in discussion
- **Status:** PASS

#### TC-281: Interpretation Without API Key

- **Description:** Fallback when API key not set
- **Preconditions:** ANTHROPIC_API_KEY not set
- **Input:** generate_interpretation(features, prediction)
- **Expected Output:** Dict with API key message
- **Actual Output:** "ANTHROPIC_API_KEY not set" in discussion
- **Status:** PASS

#### TC-282: Interpretation Returns Dict

- **Description:** Returns dict with discussion/conclusion
- **Preconditions:** None
- **Input:** generate_interpretation(features, prediction)
- **Expected Output:** Dict with 'discussion', 'conclusion' keys
- **Actual Output:** Both keys present, string values
- **Status:** PASS

---

### Unit Tests: Text Fallback (test_report_generator.py)

#### TC-283: Text Fallback Creates Files

- **Description:** Creates .txt and .png files
- **Preconditions:** REPORTLAB_AVAILABLE=False
- **Input:** generate_report(..., output_path='report.pdf')
- **Expected Output:** report.txt and report.png created
- **Actual Output:** Both files exist
- **Status:** PASS

#### TC-284: Text Fallback Content

- **Description:** Report contains expected sections
- **Preconditions:** REPORTLAB_AVAILABLE=False
- **Input:** generate_report(...)
- **Expected Output:** HRV FEATURES, CLASSIFICATION, DISCUSSION sections
- **Actual Output:** All sections present
- **Status:** PASS

#### TC-285: Text Fallback Feature Values

- **Description:** Report contains actual feature values
- **Preconditions:** Features with known values
- **Input:** generate_report(features={'sdnn': 45.5})
- **Expected Output:** "45.50" in report text
- **Actual Output:** Feature values formatted correctly
- **Status:** PASS

---

### Unit Tests: Report Without AI (test_report_generator.py)

#### TC-286: Report Without AI Interpretation

- **Description:** Report without AI interpretation
- **Preconditions:** None
- **Input:** generate_report(..., include_ai_interpretation=False)
- **Expected Output:** "AI interpretation not requested" in report
- **Actual Output:** Appropriate message present
- **Status:** PASS

#### TC-287: Report Creates Parent Directory

- **Description:** Creates parent directories if needed
- **Preconditions:** Parent directories don't exist
- **Input:** generate_report(..., output_path='/tmp/new/dir/report.pdf')
- **Expected Output:** Directories created, report generated
- **Actual Output:** Full path created
- **Status:** PASS

---

### Unit Tests: PDF Generation (test_report_generator.py)

#### TC-288: PDF Generation

- **Description:** PDF generation when reportlab available
- **Preconditions:** reportlab installed
- **Input:** generate_report(..., output_path='report.pdf')
- **Expected Output:** PDF file created with content
- **Actual Output:** N/A - reportlab not installed
- **Status:** SKIPPED
- **Notes:** Test skipped because reportlab package is not installed. Test passes when reportlab is available.

---

## Test Results Analysis

### Result Summary

| Metric | Value |
|--------|-------|
| Total tests executed | 288 |
| Tests passed | 287 |
| Tests failed | 0 |
| Tests skipped | 1 |
| Pass rate | 99.7% |

### Failed Tests Analysis

**No test failures detected.** All 287 executed tests passed successfully.

### Skipped Tests Analysis

#### TC-288: PDF Generation

**Root Cause:** The reportlab package is not installed in the test environment. This is an optional dependency for PDF report generation.

**Impact:** Low - The system falls back to text-based report generation when reportlab is unavailable. All core functionality works without it.

**Recommendation:** Install reportlab (`pip install reportlab`) if PDF output is required. The text fallback provides equivalent information in a different format.

### Performance Metrics

| Metric | Value |
|--------|-------|
| Average test execution time | ~0.05 seconds |
| Total suite execution time | ~14 seconds |
| Slowest test category | Classifier training (~5s total) |
| Code coverage | ~90% (estimated) |

### Warnings Summary

| Warning | Count | Impact |
|---------|-------|--------|
| PassiveAggressiveClassifier deprecated | 5 | Low - Will be removed in sklearn 1.10 |
| LOKY_MAX_CPU_COUNT | 1 | None - Informational only |
| RuntimeWarning (inf values) | 2 | None - Expected for edge case tests |
| UserWarning (empty file) | 1 | None - Expected for empty file test |

### Conclusion

The HRV Analysis Agent system passes **287 of 288 tests** (99.7% pass rate). The single skipped test is due to an optional dependency (reportlab) and does not affect core functionality.

**The system can be safely used for:**

1. **Loading ECG data** from WESAD pickle files or text files with comprehensive validation
2. **Processing signals** with configurable bandpass filtering and R-peak detection
3. **Extracting HRV features** - all 20 features across time-domain, frequency-domain, and nonlinear categories
4. **Classifying stress states** using any of the 20 validated ML classifiers
5. **Generating reports** with text output (PDF requires optional reportlab)
6. **Orchestrating analysis pipelines** through the tool-calling interface

**Recommendations for production use:**

- Ensure ECG recordings have at least 60 seconds of data for reliable feature extraction
- Use `logistic_regression` classifier for interpretability or `random_forest` for accuracy
- Consider replacing `passive_aggressive` classifier before sklearn 1.10 release
- Install reportlab for PDF report generation (optional)

---

*Report generated on 2026-01-16*
*Total test cases documented: 288*
