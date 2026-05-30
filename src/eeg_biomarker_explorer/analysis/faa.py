# Frontal Alpha Asymmetry (FAA) analysis.
#
# Reference for equation comparison and validation:
# ? et al. (2023). "Comparison of FAA equations in large samples."
# Clinical Neurophysiology, doi: 10.1016/j.clinph.2023.xx.xxx
# https://www.sciencedirect.com/science/article/abs/pii/S1388245723006934
#
# Equations implemented:
#   eq1 — ln(F4) - ln(F3)                              Fox et al., 1995 ++used
#   eq2 — (F4 - F3) / (F3 + F4)                        Allen et al., 2004
#   eq3 — (ln(F4) - ln(F3)) / (ln(F3) + ln(F4))        O'Reilly et al., 2017
#   eq4 — ln(rel(F4)) - ln(rel(F3))                     Harrewijn et al., 2019
#
# All equations use alpha power (8–13 Hz). Positive values indicate greater
# right frontal alpha (left hemisphere dominance); negative values indicate
# greater left frontal alpha (right hemisphere dominance).

import numpy as np
import pandas as pd
import mne

from eeg_biomarker_explorer.analysis.bandpower import bandpower, BANDS
from eeg_biomarker_explorer.utils.session_log import SessionLog, extract_segments

ALPHA_BAND = BANDS["alpha"]  # (8.0, 13.0)


def compute_faa(
    raw: mne.io.Raw,
    log: SessionLog | None = None,
    method: str = "welch",
    ch_left: str = "F3",
    ch_right: str = "F4",
    segments: list[tuple[float, float, str]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute Frontal Alpha Asymmetry for every session segment.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG recording. Must contain ch_left and ch_right channels.
    log : SessionLog, optional
        Parsed session log. Used when segments is None.
    method : 'periodogram' | 'welch' | 'multitaper'
        PSD estimation method.
    ch_left : str
        Left frontal channel name (default 'F3').
    ch_right : str
        Right frontal channel name (default 'F4').
    segments : list of (onset, offset, kind), optional
        Pre-computed segments. If provided, log is ignored.
    **kwargs
        Passed to the PSD estimator.

    Returns
    -------
    pd.DataFrame with columns:
        segment_kind, segment_idx,
        alpha_F3_abs, alpha_F4_abs,   — absolute alpha power
        alpha_F3_rel, alpha_F4_rel,   — relative alpha power
        faa_eq1, faa_eq2, faa_eq3, faa_eq4
    """
    missing = [ch for ch in (ch_left, ch_right) if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Channels {missing} not found in recording. " f"Available: {raw.ch_names}"
        )

    fs = raw.info["sfreq"]
    rec_dur = raw.times[-1]
    if segments is None:
        segments = extract_segments(log)

    idx_l = raw.ch_names.index(ch_left)
    idx_r = raw.ch_names.index(ch_right)

    rows = []
    kind_counters: dict[str, int] = {}

    for onset, offset, kind in segments:
        if onset >= rec_dur:
            continue
        offset = min(offset, rec_dur)

        kind_counters[kind] = kind_counters.get(kind, -1) + 1
        seg_idx = kind_counters[kind]

        data, _ = raw[:, raw.time_as_index(onset)[0] : raw.time_as_index(offset)[0]]

        # Absolute alpha power for F3 and F4
        abs_bp = bandpower(
            data[[idx_l, idx_r]], fs, method, ALPHA_BAND, relative=False, **kwargs
        )
        f3_abs, f4_abs = float(abs_bp[0]), float(abs_bp[1])

        # Relative alpha power for F3 and F4
        rel_bp = bandpower(
            data[[idx_l, idx_r]], fs, method, ALPHA_BAND, relative=True, **kwargs
        )
        f3_rel, f4_rel = float(rel_bp[0]), float(rel_bp[1])

        # eq1: ln(F4) - ln(F3)                             Fox et al., 1995
        eq1 = np.log(f4_abs) - np.log(f3_abs)

        # eq2: (F4 - F3) / (F3 + F4)                       Allen et al., 2004
        eq2 = (f4_abs - f3_abs) / (f3_abs + f4_abs)

        # eq3: (ln(F4) - ln(F3)) / (ln(F3) + ln(F4))       O'Reilly et al., 2017
        eq3 = (np.log(f4_abs) - np.log(f3_abs)) / (np.log(f3_abs) + np.log(f4_abs))

        # eq4: ln(rel(F4)) - ln(rel(F3))                    Harrewijn et al., 2019
        eq4 = np.log(f4_rel) - np.log(f3_rel)

        rows.append(
            {
                "segment_kind": kind,
                "segment_idx": seg_idx,
                "alpha_F3_abs": f3_abs,
                "alpha_F4_abs": f4_abs,
                "alpha_F3_rel": f3_rel,
                "alpha_F4_rel": f4_rel,
                "faa_eq1": eq1,
                "faa_eq2": eq2,
                "faa_eq3": eq3,
                "faa_eq4": eq4,
            }
        )

    return pd.DataFrame(rows)
