import mne

sample_data_folder = mne.datasets.sample.data_path()
raw_path = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"

raw = mne.io.read_raw_fif(raw_path, preload=True)

# Keep only EEG channels
raw.pick("eeg")

# Basic bandpass filter
raw.filter(1.0, 40.0)

raw.plot(
    duration=10,
    n_channels=20,
    title="Sample EEG — MNE auditory/visual dataset",
    scalings="auto",
    block=True,
)
