# Data Submissions - Group Submission

This folder contains group data submissions used for training, testing, and demonstrating AI agent systems.

## Requirements

### Folder Format
- **Format:** Folder containing data files and documentation
- **Folder Naming:** `YYYY-FamilyName1-FamilyName2-FamilyName3/` (alphabetical order, ASCII only)
- **Required:** `README.md` explaining the data
- **License:** Include license declaration (CC-BY-4.0 for data, or specify data source license)

### Folder Structure

```
2026-Chen-Lin-Wang/
├── README.md              # Data documentation (required)
├── raw/                   # Raw data files (if applicable)
│   └── *.txt
├── processed/             # Processed data files (if applicable)
│   └── *.csv
└── metadata.json          # Data metadata (optional)
```

---

## Content Requirements

Your data submission must include:

| File | Description |
|------|-------------|
| `README.md` | Data source, format, preprocessing steps, usage instructions |
| Data files | Actual data used in your project |
| `metadata.json` | Optional structured metadata |

### README.md Documentation

Your data README must specify:

| Field | Description |
|-------|-------------|
| **Source** | Where the data came from (public dataset, self-collected, synthetic) |
| **License** | License under which the data can be shared |
| **Format** | File format, encoding, structure |
| **Preprocessing** | Any transformations applied |
| **Privacy** | Confirmation that no PII is included |
| **Usage** | How to load and use the data |

---

## Grading Criteria

Deliverable scored as passed (1) if handed in with acceptable quality before the end of the course.

---

## Data Privacy Requirements

**CRITICAL: No Personally Identifiable Information (PII)**

- All data must be anonymized or synthetic
- No names, IDs, or identifying information linked to physiological data
- If using self-collected data, ensure written consent was obtained
- If using public datasets, cite the source and comply with its license

### Acceptable Data Sources

| Source Type | Requirements |
|-------------|--------------|
| **Public datasets** | Cite source, comply with license, document preprocessing |
| **Synthetic data** | Document generation method |
| **Self-collected** | Anonymized, consent obtained, IRB approval if applicable |

### NOT Acceptable

- Real patient data without proper anonymization
- Data scraped from websites without permission
- Proprietary data without license to share

---

## Example Data Submission

### README.md

```markdown
# Data: HRV Analysis Dataset

**Group:** 2026-Chen-Lin-Wang
**License:** CC-BY-4.0

## Source

This dataset contains preprocessed HRV features extracted from the SWELL-KW dataset.

**Original Source:** Koldijk, S., et al. (2014). The SWELL Knowledge Work Dataset.
Available at: http://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html

**Original License:** CC-BY-4.0

## Data Description

| File | Description | Format |
|------|-------------|--------|
| `hrv_features.csv` | Extracted HRV features | CSV, UTF-8 |
| `labels.csv` | Stress condition labels | CSV, UTF-8 |
| `subject_info.csv` | Anonymized subject metadata | CSV, UTF-8 |

### hrv_features.csv

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | string | Anonymized subject identifier (S01-S25) |
| `window_id` | int | Analysis window number |
| `sdnn` | float | Standard deviation of NN intervals (ms) |
| `rmssd` | float | Root mean square of successive differences (ms) |
| `pnn50` | float | Percentage of intervals > 50ms (%) |
| `lf_power` | float | Low frequency power (ms²) |
| `hf_power` | float | High frequency power (ms²) |
| `lf_hf_ratio` | float | LF/HF ratio |

### labels.csv

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | string | Anonymized subject identifier |
| `window_id` | int | Analysis window number |
| `condition` | string | Stress condition (neutral/time_pressure/interruptions) |
| `stress_label` | int | Binary label (0=neutral, 1=stressed) |

## Preprocessing

1. ECG signals bandpass filtered (0.5-40 Hz)
2. R-peaks detected using Pan-Tompkins algorithm
3. HRV features computed over 3-minute windows with 50% overlap
4. Outlier windows removed (RR interval < 300ms or > 2000ms)

## Usage

```python
import pandas as pd

# Load features and labels
features = pd.read_csv('hrv_features.csv')
labels = pd.read_csv('labels.csv')

# Merge on subject_id and window_id
data = features.merge(labels, on=['subject_id', 'window_id'])

# Split by subject for training/testing
train_subjects = [f'S{i:02d}' for i in range(1, 18)]
test_subjects = [f'S{i:02d}' for i in range(22, 26)]

train_data = data[data['subject_id'].isin(train_subjects)]
test_data = data[data['subject_id'].isin(test_subjects)]
```

## Privacy Statement

This dataset contains no personally identifiable information. Subject IDs are
anonymized (S01-S25) and cannot be linked to individuals. The original SWELL-KW
dataset was collected with informed consent and IRB approval.

## Statistics

| Metric | Value |
|--------|-------|
| Total subjects | 25 |
| Total windows | ~5,000 |
| Features | 6 HRV metrics |
| Conditions | 3 (neutral, time pressure, interruptions) |
```

---

## Data Format Guidelines

### Tabular Data (CSV)

- Use UTF-8 encoding
- Include header row with column names
- Use consistent delimiters (comma recommended)
- Document missing value representation (e.g., `NA`, empty string)

### Signal Data (TXT)

- One value per line for single-channel data
- Document sampling rate in README
- Include units in documentation

### Metadata (JSON)

```json
{
  "dataset_name": "HRV Analysis Dataset",
  "version": "1.0",
  "created": "2026-01-15",
  "authors": ["Chen Wei", "Lin MeiLing", "Wang XiaoMing"],
  "license": "CC-BY-4.0",
  "source": "SWELL-KW",
  "n_subjects": 25,
  "n_samples": 5000,
  "features": ["sdnn", "rmssd", "pnn50", "lf_power", "hf_power", "lf_hf_ratio"],
  "sampling_info": {
    "window_size_sec": 180,
    "overlap_percent": 50
  }
}
```

---

## Recommended Public Datasets

For HRV/ECG analysis projects, consider these publicly available datasets:

| Dataset | Description | License | URL |
|---------|-------------|---------|-----|
| **SWELL-KW** | Stress during knowledge work | CC-BY-4.0 | [Link](http://cs.ru.nl/~skoldijk/SWELL-KW/) |
| **WESAD** | Wearable stress detection | CC-BY-4.0 | [Link](https://archive.ics.uci.edu/ml/datasets/WESAD) |
| **MIT-BIH** | Arrhythmia database | ODC-BY | [Link](https://physionet.org/content/mitdb/) |
| **PTB-XL** | Large ECG dataset | CC-BY-4.0 | [Link](https://physionet.org/content/ptb-xl/) |

---

## Submission Checklist

Before submitting, verify:

- [ ] Folder named correctly: `YYYY-Name1-Name2-Name3/` (alphabetical)
- [ ] Only ASCII characters in folder and file names
- [ ] `README.md` present with complete documentation
- [ ] License declaration included
- [ ] Data source cited properly
- [ ] No personally identifiable information (PII)
- [ ] Data files are readable and not corrupted
- [ ] Usage example provided
- [ ] All group members listed
