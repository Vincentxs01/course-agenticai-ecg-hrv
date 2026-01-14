# Test Cases - Group Submission

This folder contains group test cases and their outcomes for validating AI agent systems.

## Requirements

### File Format
- **Format:** Folder containing test files and documentation
- **Folder Naming:** `YYYY-FamilyName1-FamilyName2-FamilyName3/` (alphabetical order, ASCII only)
- **Required:** `README.md` explaining tests and results
- **License:** Include license declaration (Apache-2.0 for code, CC-BY-4.0 for docs)

### Folder Structure

```
2026-Chen-Lin-Wang/
├── README.md              # Test documentation (required)
├── test_cases.md          # Test case descriptions
├── test_results.md        # Test outcomes and analysis
└── test_*.py              # Test scripts (if applicable)
```

Use data from `data-group/2026-Chen-Lin-Wang/`

---

## Content Requirements

Your test submission must include:

| File | Description |
|------|-------------|
| `README.md` | Overview of testing approach, how to run tests, summary of results |
| `test_cases.md` | Detailed description of each test case |
| `test_results.md` | Outcomes, pass/fail status, analysis |

### Test Case Documentation

Each test case should specify:

| Field | Description |
|-------|-------------|
| **Test ID** | Unique identifier (e.g., TC-001) |
| **Description** | What the test verifies |
| **Preconditions** | Required setup or state |
| **Input** | Test input data or parameters |
| **Expected Output** | What should happen |
| **Actual Output** | What actually happened |
| **Status** | PASS / FAIL / PARTIAL |
| **Notes** | Additional observations |

---

## Grading Criteria

Deliverable scored as passed (1) if handed in with acceptable quality before the end of the course.

---

## Example Test Documentation

### README.md

```markdown
# Test Suite: HRV Analysis Agent

**Group:** 2026-Chen-Lin-Wang
**License:** Apache-2.0 (code), CC-BY-4.0 (documentation)

## Overview

This test suite validates the HRV Analysis Agent system. Tests cover:
- Data loading and validation
- Signal processing accuracy
- Feature extraction correctness
- Classification performance
- Report generation

## Test Summary

| Category | Total | Passed | Failed |
|----------|-------|--------|--------|
| Unit Tests | 12 | 11 | 1 |
| Integration Tests | 5 | 5 | 0 |
| End-to-End Tests | 3 | 2 | 1 |
| **Total** | **20** | **18** | **2** |

**Pass Rate:** 90%

## How to Run Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/test_signal_processing.py
```

## Known Issues

1. TC-003 fails on corrupted data files (expected behavior, but error message unclear)
2. TC-018 timeout on very large files (>10 min recordings)
```

### test_cases.md

```markdown
# Test Cases

## Unit Tests: Data Loader

### TC-001: Load Valid ECG File

- **Description:** Verify that a valid ECG text file loads correctly
- **Preconditions:** Sample ECG file exists at test path
- **Input:** `sample_ecg.txt` (500 Hz, 6-minute recording)
- **Expected Output:** NumPy array with 180,000 samples
- **Actual Output:** NumPy array with 180,000 samples
- **Status:** PASS

### TC-002: Handle Missing File

- **Description:** Verify graceful handling of missing file
- **Preconditions:** None
- **Input:** `nonexistent_file.txt`
- **Expected Output:** FileNotFoundError with descriptive message
- **Actual Output:** FileNotFoundError: "File not found: nonexistent_file.txt"
- **Status:** PASS

### TC-003: Handle Corrupted File

- **Description:** Verify handling of corrupted/malformed data
- **Preconditions:** Corrupted file with non-numeric content
- **Input:** `corrupted_ecg.txt`
- **Expected Output:** ValueError with line number of error
- **Actual Output:** ValueError: "Could not convert string to float"
- **Status:** FAIL
- **Notes:** Error message does not indicate which line caused the problem

---

## Unit Tests: Signal Processing

### TC-004: Bandpass Filter Frequency Response

- **Description:** Verify filter passes 0.5-40 Hz, attenuates outside
- **Preconditions:** None
- **Input:** Synthetic signal with known frequency components
- **Expected Output:** >90% power in passband, <10% outside
- **Actual Output:** 94% in passband, 4% outside
- **Status:** PASS

### TC-005: R-Peak Detection Accuracy

- **Description:** Verify R-peak detection on annotated dataset
- **Preconditions:** MIT-BIH sample with known R-peak locations
- **Input:** Record 100 from MIT-BIH database
- **Expected Output:** Sensitivity >95%, Precision >95%
- **Actual Output:** Sensitivity 97.2%, Precision 96.8%
- **Status:** PASS

---

## Integration Tests

### TC-010: Full Pipeline Execution

- **Description:** Verify complete pipeline from file to report
- **Preconditions:** Valid ECG file, trained model
- **Input:** `sample_ecg_6min.txt`
- **Expected Output:** PDF report with HRV metrics and classification
- **Actual Output:** PDF generated with all expected sections
- **Status:** PASS
- **Execution Time:** 28 seconds

---

## End-to-End Tests

### TC-018: Large File Processing

- **Description:** Verify system handles 30-minute recording
- **Preconditions:** Large ECG file
- **Input:** `large_ecg_30min.txt` (900,000 samples)
- **Expected Output:** Complete processing within 5 minutes
- **Actual Output:** Timeout after 10 minutes
- **Status:** FAIL
- **Notes:** Memory usage exceeded available RAM; needs optimization
```

### test_results.md

```markdown
# Test Results Analysis

## Summary

Overall pass rate: 90% (18/20 tests)

## Failed Tests Analysis

### TC-003: Handle Corrupted File

**Root Cause:** The data loader catches the ValueError from NumPy but does not
preserve line number information when re-raising.

**Impact:** Low - user cannot identify exact location of corruption, but the
file is still rejected appropriately.

**Recommendation:** Enhance error handling to include line number in exception
message.

### TC-018: Large File Processing

**Root Cause:** The feature extraction step loads entire signal into memory
for FFT computation. 30-minute recordings at 500 Hz require ~7.2 MB for raw
data, but intermediate calculations expand this to ~500 MB.

**Impact:** Medium - system cannot process recordings longer than ~15 minutes
on machines with 8 GB RAM.

**Recommendation:** Implement windowed processing for long recordings.

## Performance Metrics

| Metric | Value |
|--------|-------|
| Average test execution time | 1.2 seconds |
| Total suite execution time | 24 seconds |
| Code coverage | 78% |

## Conclusion

The system passes core functionality tests. Two failures identified are
edge cases that do not affect normal operation but should be addressed
in future iterations.
```

---

## Best Practices for Testing

### Test Design
- Test both normal operation and edge cases
- Include positive tests (should work) and negative tests (should fail gracefully)
- Use realistic test data when possible
- Document assumptions and constraints

### Test Data
- Use anonymized or synthetic data only
- Document data sources and generation methods
- Include sample data files in your submission
- Never include real patient data

### Documentation
- Write tests as if someone else will run them
- Include setup instructions
- Explain any non-obvious test logic
- Analyze failures, don't just report them

---

## Submission Checklist

Before submitting, verify:

- [ ] Folder named correctly: `YYYY-Name1-Name2-Name3/` (alphabetical)
- [ ] Only ASCII characters in folder and file names
- [ ] `README.md` present with overview and summary
- [ ] `test_cases.md` with detailed test descriptions
- [ ] `test_results.md` with outcomes and analysis
- [ ] License declaration included
- [ ] No personally identifiable information (PII) in test data
- [ ] Test data is anonymized or synthetic
- [ ] All group members listed
