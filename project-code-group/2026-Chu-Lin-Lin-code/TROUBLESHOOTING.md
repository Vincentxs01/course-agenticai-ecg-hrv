# Troubleshooting Guide

Common issues and solutions for the HRV Stress Classifier.

---

## Error: "externally-managed-environment"

**Symptom:** On macOS with Homebrew Python:
```
error: externally-managed-environment
× This environment is externally managed
```

**Cause:** PEP 668 prevents pip from installing packages system-wide on Homebrew Python.

**Solution:** Always use a virtual environment:
```bash
# Create venv in a writable location (e.g., /tmp for testing)
python3 -m venv /tmp/hrv_venv
source /tmp/hrv_venv/bin/activate
pip install -r requirements.txt
```

Or use `uv` which handles this automatically:
```bash
uv venv venv && source venv/bin/activate && uv pip install -r requirements.txt
```

---

## Error: "Fontconfig error: No writable cache directories"

**Symptom:**
```
Fontconfig error: No writable cache directories
    /opt/homebrew/var/cache/fontconfig
    /Users/username/.cache/fontconfig
```

**Cause:** Matplotlib cannot write to font cache directories due to permissions.

**Solution:** This is a warning and usually doesn't affect functionality. To suppress:
```bash
# Set writable config directory before running scripts
export MPLCONFIGDIR=/tmp/matplotlib
python scripts/visualize_ecg_conditions.py ...
```

---

## Error: "mkdir -p failed for path ~/.matplotlib"

**Symptom:**
```
mkdir -p failed for path /Users/username/.matplotlib: Operation not permitted
```

**Solution:** Set the matplotlib config directory to a writable location:
```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

Add to your shell profile (`~/.zshrc` or `~/.bashrc`) to make permanent:
```bash
echo 'export MPLCONFIGDIR=/tmp/matplotlib' >> ~/.zshrc
```

---

## Error: "ModuleNotFoundError: No module named 'xxx'"

**Symptom:** Missing modules like `matplotlib`, `joblib`, `sklearn`.

**Solution:** Ensure venv is activated and dependencies installed:
```bash
# Check if venv is active (should show venv path)
which python

# If not in venv, activate it
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Or install specific missing package
pip install matplotlib numpy scipy scikit-learn joblib
```

---

## Error: "uv" cache permission issues

**Symptom:**
```
error: failed to open file `~/.cache/uv/...`: Operation not permitted
```

**Solution:** Set a writable cache directory:
```bash
export UV_CACHE_DIR=/tmp/uv_cache
uv pip install -r requirements.txt
```

---

## Error: VisibleDeprecationWarning with NumPy 2.4+

**Symptom:**
```
VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean
```

**Cause:** WESAD pickle files were created with older NumPy. This is a warning, not an error.

**Solution:** The warning can be safely ignored. To suppress:
```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

---

## Performance: Slow package installation

**Symptom:** `pip install` takes a long time.

**Solution:** Use `uv` instead of pip (10-100x faster):
```bash
# Install uv
brew install uv  # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Use uv for installation
uv pip install -r requirements.txt
```
