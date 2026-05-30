from pathlib import Path

import numpy as np
import mne
from scipy.spatial.distance import cdist


def load_sample() -> mne.io.Raw:
    """Load the MNE sample dataset (placeholder until real EEG files are available).

    Channel names are renamed to standard 10-20 labels by matching each
    electrode's digitised position to the nearest standard location, so
    region-based analysis works out of the box.
    """
    folder = mne.datasets.sample.data_path()
    path = folder / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    raw.pick("eeg")
    _rename_to_1020(raw)
    return raw


def _rename_to_1020(raw: mne.io.Raw) -> None:
    """Rename channels to the nearest standard 10-20 label in-place."""
    montage = mne.channels.make_standard_montage("standard_1020")
    std_pos = montage.get_positions()["ch_pos"]
    std_names = list(std_pos.keys())
    std_coords = np.array(list(std_pos.values()))

    ch_coords = np.array(
        [raw.info["chs"][i]["loc"][:3] for i in range(len(raw.ch_names))]
    )
    dists = cdist(ch_coords, std_coords)

    used: set[str] = set()
    rename_map: dict[str, str] = {}
    for ch_name, row in zip(raw.ch_names, dists):
        for candidate in row.argsort():
            name = std_names[candidate]
            if name not in used:
                rename_map[ch_name] = name
                used.add(name)
                break

    raw.rename_channels(rename_map)


def load_raw(path: str | Path) -> mne.io.Raw:
    """Load a real EEG file. Supports .fif, .edf, .bdf, .set, .vhdr."""
    path = Path(path)
    readers = {
        ".fif": mne.io.read_raw_fif,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".vhdr": mne.io.read_raw_brainvision,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: {list(readers)}"
        )
    raw = reader(path, preload=True)
    raw.pick("eeg")
    return raw
