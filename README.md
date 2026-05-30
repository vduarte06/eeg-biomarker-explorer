# eeg-biomarker-explorer

A declarative EEG analysis framework built on [MNE-Python](https://mne.tools/). Define reproducible preprocessing and feature-extraction workflows using YAML — no code changes needed between studies.

## Quick start

```bash
# create a new pipeline from template
eeg-pipe new pipelines/my_study.yaml

# edit it, then run
eeg-pipe run pipelines/my_study.yaml
```

## How it works

Each study is a single YAML file:

```yaml
version: 1

input:
  dataset:
    path: ./data/raw/subject01.edf

  events:
    path:   ./data/raw/events.json   # session log with timestamps → annotations
    task_a: LABEL_IN_RECORDING
    rest:   BASELINE_LABEL

  regions:
    frontal:   [Fp1, Fp2, F3, F4, Fz]
    occipital: [O1, O2]

pipeline:

  preprocessing:
    - filter:      { l_freq: 1.0, h_freq: 35.0 }
    - line_noise:  { freq: 60.0 }
    - bad_channels:
    - rereference: { method: average }
    - ica:         { n_components: 15 }
    - interpolate:
    - bad_segment_annotation:
    - plot:        { title: After preprocessing, sensitivity: 50 }

  feature_extraction:
    - spectral_power:
        events: [task_a, rest]
        bands:  { theta: [4, 8], alpha: [8, 12], beta: [12, 30] }
        groupby: region
    - faa:
        events: [task_a, rest]
        channels: { left: F3, right: F4 }

output:
  format: [csv, parquet]
  path: ./data/processed
```

## Installation

```bash
git clone https://github.com/viniciusduarte06/eeg-biomarker-explorer.git
cd eeg-biomarker-explorer
poetry install
```

## Supported input formats

`.edf` · `.fif` · `.bdf` · `.set` (EEGLAB) · `.vhdr` (BrainVision)

## Preprocessing steps

| Key | What it does | Notable options |
|---|---|---|
| `filter` | Bandpass filter | `l_freq`, `h_freq` |
| `line_noise` | Notch filter | `freq` — 50 Hz (EU) or 60 Hz (BR/US) |
| `bad_channels` | Flags noisy/flat channels | `z_threshold` |
| `rereference` | Re-referencing | `method: average \| REST \| channel` |
| `ica` | ICA eye-movement removal | `n_components`, `method`, `eog_channels`, `eog_threshold` |
| `interpolate` | Spherical spline interpolation of flagged channels | — |
| `bad_segment_annotation` | Marks amplitude-spike segments as BAD | `peak_amplitude` (µV) |
| `plot` | Signal viewer — blocks until window closed | `sensitivity` (µV/div), `duration`, `n_channels` |

`plot` can appear anywhere in either section, as many times as needed.

## Feature extraction

| Key | Description |
|---|---|
| `spectral_power` | Relative band power per region or channel (Welch / periodogram / multitaper) |
| `peak_power` | Peak PSD value and its frequency per band |
| `faa` | Frontal alpha asymmetry — four equations (Fox 1995, Allen 2004, O'Reilly 2017, Harrewijn 2019) |

## Output

Results are saved to `output.path` in all requested formats:

```
data/processed/
  spectral_power.csv
  spectral_power.parquet
  faa.csv
  faa.parquet
```

## Development

```bash
poetry install
poetry run pre-commit install   # black + lint hooks on every commit
```

## Requirements

Python ≥ 3.11 · MNE ≥ 1.12 · pandas · PyYAML · scikit-learn · pyarrow · pydantic
