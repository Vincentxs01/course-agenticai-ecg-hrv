# Project Code - Group Submission

This folder contains group project source code for AI agent systems.

## Requirements

### Folder Format
- **Format:** Folder containing source code and documentation
- **Folder Naming:** `YYYY-FamilyName1-FamilyName2-FamilyName3/` (alphabetical order, ASCII only)
- **Required:** `README.md` with setup and usage instructions
- **License:** Apache-2.0 for code (required)

### Folder Structure

```
2026-Chen-Lin-Wang/
├── README.md              # Project documentation (required)
├── requirements.txt       # Python dependencies (if applicable)
├── setup.py              # Package setup (optional)
├── src/                  # Source code
│   ├── __init__.py
│   ├── orchestrator.py   # Agent orchestrator
│   ├── tools/            # Tool implementations
│   │   ├── __init__.py
│   │   ├── ecg_loader.py
│   │   ├── signal_processor.py
│   │   ├── feature_extractor.py
│   │   ├── classifier.py
│   │   └── report_generator.py
│   └── utils/            # Utility functions
│       └── __init__.py
├── models/               # Trained models (if small enough)
│   └── classifier.joblib
├── config/               # Configuration files
│   └── config.yaml
└── scripts/              # Utility scripts
    └── run_analysis.py
```

---

## Content Requirements

Your code submission must include:

| File | Description |
|------|-------------|
| `README.md` | Setup instructions, usage examples, architecture overview |
| `requirements.txt` | All Python dependencies with versions |
| Source code | Well-organized, documented code |
| Configuration | Any config files needed to run the system |

### README.md Documentation

Your project README must include:

| Section | Description |
|---------|-------------|
| **Project Title** | Clear, descriptive name |
| **Description** | What the system does |
| **Installation** | Step-by-step setup instructions |
| **Usage** | How to run the system with examples |
| **Architecture** | Brief overview of system design |
| **API Keys** | What credentials are needed (without including actual keys) |
| **License** | Apache-2.0 declaration |
| **Authors** | Group member names |

---

## Grading Criteria

Deliverable scored as passed (1) if handed in with acceptable quality before the end of the course.

---

## Code Quality Guidelines

### Style and Formatting

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Use type hints where appropriate

### Documentation

- Include docstrings for all public functions and classes
- Add inline comments for complex logic
- Document any non-obvious design decisions

### Security

- **NEVER** commit API keys, passwords, or secrets
- Use environment variables for sensitive configuration
- Include a `.env.example` file showing required variables (without values)

---

## Example Project Structure

### README.md

```markdown
# HRV Analysis Agent

An AI agent system for automated heart rate variability analysis and fatigue detection.

**Group:** 2026-Chen-Lin-Wang
**License:** Apache-2.0

## Description

This system uses Claude Opus 4.5 as an orchestrator to coordinate ECG signal
processing, HRV feature extraction, machine learning classification, and
report generation. Given a raw ECG recording, it produces an interpretable
PDF report with fatigue assessment.

## Requirements

- Python 3.11+
- Anthropic API key

## Installation

```bash
# Clone or download this folder
cd 2026-Chen-Lin-Wang

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Configuration

Create a `.env` file with the following:

```
ANTHROPIC_API_KEY=your_api_key_here
```

**Note:** Never commit your actual API key to the repository.

## Usage

### Command Line

```bash
# Analyze a single ECG file
python scripts/run_analysis.py --input data/ecg_sample.txt --output reports/

# With custom configuration
python scripts/run_analysis.py --input data/ecg_sample.txt --config config/config.yaml
```

### Python API

```python
from src.orchestrator import HRVAnalysisOrchestrator

# Initialize orchestrator
agent = HRVAnalysisOrchestrator()

# Run analysis
result = agent.run(
    ecg_file="data/ecg_sample.txt",
    output_path="reports/analysis_report.pdf"
)

print(f"Report generated: {result}")
```

## Architecture

```
┌─────────────────┐
│   Orchestrator  │ ← Claude Opus 4.5
│  (coordinates)  │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┬─────────┐
    ▼         ▼          ▼          ▼         ▼
┌───────┐ ┌───────┐ ┌─────────┐ ┌────────┐ ┌───────┐
│Loader │→│Process│→│ Extract │→│Classify│→│Report │
└───────┘ └───────┘ └─────────┘ └────────┘ └───────┘
```

## Project Structure

```
├── src/
│   ├── orchestrator.py      # Main agent orchestrator
│   ├── tools/
│   │   ├── ecg_loader.py    # ECG file loading
│   │   ├── signal_processor.py  # Filtering, R-peak detection
│   │   ├── feature_extractor.py # HRV metric calculation
│   │   ├── classifier.py    # Fatigue classification
│   │   └── report_generator.py  # PDF report generation
│   └── utils/
│       └── helpers.py       # Utility functions
├── models/
│   └── classifier.joblib    # Trained Random Forest model
├── config/
│   └── config.yaml          # Configuration settings
└── scripts/
    └── run_analysis.py      # CLI entry point
```

## Dependencies

Key dependencies (see `requirements.txt` for full list):

- `anthropic>=0.18.0` - Claude API client
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.11.0` - Signal processing
- `scikit-learn>=1.3.0` - Machine learning
- `matplotlib>=3.7.0` - Visualization
- `reportlab>=4.0.0` - PDF generation

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src
```

## Known Issues

- Large files (>30 minutes) may cause memory issues
- Requires stable internet connection for Claude API

## Authors

- Chen Wei
- Lin MeiLing
- Wang XiaoMing

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
```

### requirements.txt

```
anthropic>=0.18.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
reportlab>=4.0.0
joblib>=1.3.0
python-dotenv>=1.0.0
pyyaml>=6.0
```

### .env.example

```
# Anthropic API Key (required)
ANTHROPIC_API_KEY=

# Optional: Model to use (default: claude-opus-4-5-20251101)
ANTHROPIC_MODEL=claude-opus-4-5-20251101

# Optional: Output directory
OUTPUT_DIR=./reports
```

### config/config.yaml

```yaml
# HRV Analysis Configuration

# Signal processing parameters
signal:
  sampling_rate: 500  # Hz
  bandpass_low: 0.5   # Hz
  bandpass_high: 40.0 # Hz
  filter_order: 4

# Feature extraction
features:
  window_size: 180    # seconds
  overlap: 0.5        # 50% overlap

# Classification
classifier:
  model_path: models/classifier.joblib
  threshold: 0.5

# Report generation
report:
  template: default
  include_plots: true
  language: en
```

---

## Code Examples

### Orchestrator Pattern

```python
import anthropic
from typing import Any

class HRVAnalysisOrchestrator:
    """Orchestrates HRV analysis workflow using Claude as the coordinator."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-opus-4-5-20251101"
        self.tools = self._define_tools()
        self.state = {}

    def _define_tools(self) -> list[dict]:
        """Define available tools for the agent."""
        return [
            {
                "name": "load_ecg",
                "description": "Load ECG data from a text file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "sampling_rate": {"type": "integer", "default": 500}
                    },
                    "required": ["file_path"]
                }
            },
            # ... additional tools
        ]

    def run(self, ecg_file: str, output_path: str) -> str:
        """Execute the complete analysis pipeline."""
        # Implementation here
        pass
```

### Tool Implementation Pattern

```python
import numpy as np
from scipy.signal import butter, filtfilt

def process_signal(ecg_data: dict) -> dict:
    """
    Process raw ECG signal: filter and detect R-peaks.

    Args:
        ecg_data: Dictionary with 'signal' and 'sampling_rate' keys

    Returns:
        Dictionary with processed signal and detected R-peak indices
    """
    signal = ecg_data["signal"]
    fs = ecg_data["sampling_rate"]

    # Bandpass filter
    filtered = bandpass_filter(signal, fs, 0.5, 40.0)

    # R-peak detection
    r_peaks = detect_r_peaks(filtered, fs)

    return {
        "filtered_signal": filtered,
        "r_peaks": r_peaks,
        "rr_intervals": np.diff(r_peaks) / fs * 1000  # in ms
    }
```

---

## API Key Security

### DO NOT:
- Commit API keys to the repository
- Hardcode keys in source files
- Share keys in documentation

### DO:
- Use environment variables
- Include `.env` in `.gitignore`
- Provide `.env.example` as template

### .gitignore should include:

```
# Environment variables
.env
.env.local

# API keys and secrets
*.key
*secret*

# Model files (if large)
*.joblib
*.pkl
*.h5

# Python
__pycache__/
*.pyc
.pytest_cache/
venv/
```

---

## Submission Checklist

Before submitting, verify:

- [ ] Folder named correctly: `YYYY-Name1-Name2-Name3/` (alphabetical)
- [ ] Only ASCII characters in folder and file names
- [ ] `README.md` present with setup and usage instructions
- [ ] `requirements.txt` with all dependencies
- [ ] License declaration (Apache-2.0) included
- [ ] No API keys or secrets in code
- [ ] `.env.example` provided (without actual keys)
- [ ] Code runs without errors (tested)
- [ ] No personally identifiable information (PII)
- [ ] All group members listed as authors
