# Data: Synthetic ECG Dataset for Driver Drowsiness Detection

**Group:** 2026-Wei-Wu-Zheng
**Authors:** Wei JungYing, Wu KunChe, Zheng YiKai
**License:** CC-BY-4.0

## Dataset Overview

### Source

This dataset contains **synthetic ECG signals** generated using a custom Python script (`utils/data_generator.py`) for testing and demonstrating the Driver Drowsiness Detection System. The signals simulate realistic cardiac patterns with configurable heart rate, heart rate variability (HRV), and noise characteristics.

**Generation Method:** Algorithmic simulation using NumPy
**Generator Location:** `project-code-group/2026-Wei-Wu-Zheng/utils/data_generator.py`

### Why Synthetic Data?

1. **Privacy:** Real ECG data from drivers would contain sensitive physiological information
2. **Controllability:** Synthetic data allows precise control over drowsy vs. alert states
3. **Reproducibility:** Anyone can regenerate the exact same test data
4. **Safety:** No human subjects required for initial system development and testing

### Statistics

| Property | Value |
|----------|-------|
| Data Type | Synthetic ECG signals |
| Sampling Rate | 250 Hz |
| Total Files | 3 |
| Total Samples | 37,500 (150 seconds total) |
| Signal Components | QRS complex, T wave, baseline wander, random noise |

### File Descriptions

| File | Duration | Samples | Heart Rate | State | Description |
|------|----------|---------|------------|-------|-------------|
| `ecg_normal.csv` | 30 sec | 7,500 | ~75 bpm | Alert | Normal driving state with low HRV |
| `ecg_drowsy.csv` | 30 sec | 7,500 | ~58 bpm | Drowsy | Fatigue state with high HRV |
| `ecg_long_drowsy.csv` | 60 sec | 15,000 | ~65 bpm | Drowsy | Extended drowsy recording |

## Format

### File Format

- **Format:** CSV (Comma-Separated Values)
- **Encoding:** UTF-8
- **Header:** Yes (first row contains column name)

### Column Structure

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `ECG` | float | mV (normalized) | ECG amplitude value |

### Example Content

```csv
ECG
0.05
0.08
0.12
-0.02
1.00
-0.15
...
```

## Preprocessing

### Raw Data Generation

The synthetic signals are generated with the following characteristics already included:

1. **QRS Complex:** Sharp R-peak (amplitude = 1.0) with Q wave (-0.1) and S wave (-0.15)
2. **T Wave:** Gaussian-shaped positive deflection (amplitude = 0.3)
3. **Baseline Wander:** Sinusoidal oscillation at 0.25 Hz simulating respiration
4. **Random Noise:** Gaussian noise (configurable level 0.05-0.08)
5. **Motion Artifacts:** Random high-amplitude segments (30% probability)

### No Additional Preprocessing Required

Since this is synthetic data generated for testing purposes, no preprocessing has been applied to the raw generated signals. The preprocessing (bandpass filtering, notch filtering, normalization) is performed by **Agent 1 (Signal Filter Agent)** in the detection system during runtime.

## Privacy

**Confirmation:** This dataset contains **NO personally identifiable information (PII)**.

All signals are computationally generated using mathematical algorithms and random number generation. The data does not correspond to any real individuals or actual physiological recordings.

## Usage

### Option 1: Use Pre-generated Data

The data files are located in `project-code-group/2026-Wei-Wu-Zheng/data/`:

```python
import pandas as pd

# Load ECG data
ecg_data = pd.read_csv('project-code-group/2026-Wei-Wu-Zheng/data/ecg_normal.csv')
signal = ecg_data['ECG'].values

# Parameters
sampling_rate = 250  # Hz
duration = len(signal) / sampling_rate  # seconds

print(f"Loaded {len(signal)} samples ({duration:.1f} seconds)")
```

### Option 2: Regenerate Data

To regenerate or create custom test data:

```bash
cd project-code-group/2026-Wei-Wu-Zheng
python utils/data_generator.py
```

### Option 3: Generate Custom Data

```python
import sys
sys.path.append('project-code-group/2026-Wei-Wu-Zheng')
from utils.data_generator import ECGDataGenerator

generator = ECGDataGenerator(sampling_rate=250)

# Generate custom ECG
ecg = generator.generate_ecg(
    duration=45,        # seconds
    heart_rate=65,      # bpm
    noise_level=0.1,    # noise amplitude
    drowsy=True         # drowsy state flag
)

generator.save_to_csv(ecg, "ecg_custom.csv")
```

### Using with the Detection System

```bash
cd project-code-group/2026-Wei-Wu-Zheng
streamlit run app.py
# Upload CSV files through the web interface
```

### Expected Results

| File | Expected Risk Level | Expected Behavior |
|------|---------------------|-------------------|
| `ecg_normal.csv` | Low | System detects alert state |
| `ecg_drowsy.csv` | High | System detects drowsy state, triggers alert |
| `ecg_long_drowsy.csv` | Very High | Extended drowsy state with accumulated risk |

## Signal Characteristics

### Normal (Alert) State Parameters

| Parameter | Value |
|-----------|-------|
| Heart Rate | 70-80 bpm |
| HRV Variation | 5% (low) |
| Noise Level | 0.05 |

### Drowsy (Fatigue) State Parameters

| Parameter | Value |
|-----------|-------|
| Heart Rate | 55-65 bpm (decreased) |
| HRV Variation | 15% (high) |
| Noise Level | 0.075-0.08 (slightly elevated) |

## Limitations

1. **Simplified cardiac model:** Real ECG signals have more complex morphology (P wave not modeled)
2. **No individual variation:** All signals use the same base waveform template
3. **Limited artifact types:** Only random noise and baseline wander simulated
4. **No pathological patterns:** Does not include arrhythmias or other cardiac abnormalities
5. **Fixed sampling rate:** Generated at 250 Hz only

## References

1. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE Transactions on Biomedical Engineering, 32(3), 230-236.
2. Task Force of ESC and NASPE. (1996). Heart rate variability: Standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065.
