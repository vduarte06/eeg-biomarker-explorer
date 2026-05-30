# mdr-processing-pipe

EEG processing pipeline built with [MNE-Python](https://mne.tools/) for MDR (EMDR) psychology research.

## Features

- **Preprocessing pipeline**: notch filter, bad channel detection, re-referencing, ICA, interpolation, bad segment rejection, and epoching
- **Band power analysis**: theta, alpha, beta, gamma
- **Frontal Alpha Asymmetry (FAA)**: F3/F4 asymmetry computation
- **Session logging**: event annotation utilities
- **Simulation**: synthetic EEG session generation for testing

## Installation

### From PyPI (once published)

```bash
pip install mdr-processing-pipe
```

### From GitHub

```bash
pip install git+https://github.com/viniciusduarte06/mdr-processing-pipe.git
```

### For development

```bash
git clone https://github.com/viniciusduarte06/mdr-processing-pipe.git
cd mdr-processing-pipe
pip install -e .
# or with Poetry:
poetry install
```

## Quick start

```python
import mne
import yaml
from mdr_processing_pipe.preprocessing.pipeline import preprocess
from mdr_processing_pipe.analysis.bandpower import compute_bandpower
from mdr_processing_pipe.analysis.faa import compute_faa

# Load config and raw EEG
with open("data/config.yaml") as f:
    cfg = yaml.safe_load(f)

raw = mne.io.read_raw_edf("your_file.edf", preload=True)

# Run full preprocessing pipeline
raw_clean, epochs = preprocess(raw, cfg["preprocessing"])

# Compute band power and FAA
bp = compute_bandpower(raw_clean)
faa = compute_faa(raw_clean)
```

## Requirements

- Python >= 3.11
- MNE-Python >= 1.12
- pandas, pyyaml, scikit-learn

## License

MIT
