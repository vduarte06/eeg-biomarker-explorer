# eeg-biomarker-explorer

A declarative EEG analysis framework built on [MNE-Python](https://mne.tools/). Define reproducible preprocessing and feature-extraction workflows using YAML — no code changes needed between studies.


## Requirements

- **Python** ≥ 3.11
- **[Poetry](https://python-poetry.org/)** — install

Runtime libraries (installed with `poetry install`): MNE ≥ 1.12, pandas, PyYAML, scikit-learn, pyarrow, pydantic.

## Installation

```bash
git clone https://github.com/viniciusduarte06/eeg-biomarker-explorer.git
cd eeg-biomarker-explorer
poetry install
```

## Development

```bash
poetry shell
pre-commit install   # black + lint hooks on every commit
```

## Quick start

```bash
# create a new pipeline from template
eeg-pipe new pipelines/my_study.yaml

# edit it, then run
eeg-pipe run pipelines/my_study.yaml
```

## Utilities

### Annotate epochs interactively

After ICA or preprocessing, visually mark time windows of interest and save them as a JSON events file that the pipeline can consume in `feature_extraction`.

**Standalone command** — works on any EDF without a pipeline:

```bash
eeg-pipe annotate data/raw/subject01.edf

# load existing events.json as reference annotations while you annotate
eeg-pipe annotate data/raw/subject01.edf --events data/raw/events.json

# custom output path and plot sensitivity
eeg-pipe annotate data/raw/subject01.edf \
    --events data/raw/events.json \
    --output data/raw/my_epochs.json \
    --sensitivity 50
```

**Workflow inside the viewer:**

1. Press `a` to enter annotation mode.
2. Click and drag on the signal to mark a region.
3. Type a label name in the Annotations dialog (e.g. `BASELINE`, `TASK`).
4. Repeat for every segment you want to label.
5. Close the window — the annotations are saved to the output JSON file.

The output uses the same `START_/END_` format as `events.json`, with `00:00:00.000` representing the recording start:

```json
[
  ["00:00:00.000", "START_SESSION"],
  ["00:01:07.500", "START_BASELINE"],
  ["00:02:10.200", "END_BASELINE"],
  ["00:05:33.000", "START_TASK"],
  ["00:06:45.800", "END_TASK"]
]
```

**As a pipeline step** — add `annotate` anywhere in `preprocessing` (typically after ICA):

```yaml
pipeline:
  preprocessing:
    - ica:
        n_components: 15
    - annotate:
        output: ./data/raw/user_epochs.json   # where to save
        sensitivity: 50                        # µV/div
```

After the pipeline runs the `annotate` step, the marked annotations are immediately available to subsequent `feature_extraction` steps via `user_events` (see below).

**Using annotated epochs in feature extraction:**

Add a `user_events` block under `input:` pointing to your saved file, then reference those labels in `feature_extraction`:

```yaml
input:
  events:
    path:   ./data/raw/events.json
    emdr_t: EMDR_T

  user_events:
    path:     ./data/raw/user_epochs.json
    baseline: BASELINE
    task:     TASK

pipeline:
  feature_extraction:
    - spectral_power:
        events: [emdr_t, baseline, task]
        bands:  { theta: [4, 8], alpha: [8, 12] }
        groupby: region
    - faa:
        events: [baseline, task]
        channels: { left: EEG F3, right: EEG F4 }
```

### Select and export epoch segments as EDF

Extract annotated epoch windows from an EDF and save a concatenated EDF file:

```bash
eeg-pipe epochs data/raw/subject01.edf \
    --events data/raw/events.json \
    --output data/processed/subject01_epochs.edf
```

### Crop an EDF file

Cut an EDF to a wall-clock time window (UTC) and save a new file:

```bash
eeg-pipe crop data/raw/subject01.edf \
    --start 09:15 \
    --end   09:45 \
    --output data/raw/subject01_cropped.edf
```

Times must match the UTC clock time embedded in the EDF header (`meas_date`).

## How it works

Each study is a single YAML file:

```yaml
version: 1

input:
  dataset:
    path: ./data/raw/subject01.edf
    exclude: [DC1+, DC2+, trx1+, trx2+, Flw1+, Flw2+] #channels that can should excluded

  events:
    path:   ./data/raw/events.json   # session log with timestamps → annotations
    task_a: LABEL_IN_RECORDING
    rest:   BASELINE_LABEL

  # optional: user-annotated epochs from eeg-pipe annotate
  user_events:
    path:     ./data/raw/user_epochs.json
    baseline: BASELINE

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
    - annotate:    { output: ./data/raw/user_epochs.json, sensitivity: 50 }
    - plot:        { title: After preprocessing, sensitivity: 50 }

  feature_extraction:
    - spectral_power:
        events: [task_a, rest, baseline]
        bands:  { theta: [4, 8], alpha: [8, 12], beta: [12, 30] }
        groupby: region
    - faa:
        events: [task_a, rest, baseline]
        channels: { left: F3, right: F4 }

output:
  format: [csv, parquet]
  path: ./data/processed
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
| `annotate` | Interactive epoch labeler — mark time windows, save to JSON | `output`, `sensitivity` |
| `plot` | Signal viewer — blocks until window closed | `sensitivity` (µV/div), `duration`, `n_channels` |

`plot` and `annotate` can appear anywhere in the preprocessing section, as many times as needed.

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
