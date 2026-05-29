from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.signal import periodogram, welch
from mne.time_frequency import psd_array_multitaper

from mdr_processing_pipe.utils.session_log import SessionLog, extract_segments


BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpower(
    data: NDArray[np.float64],
    fs: float,
    method: str,
    band: tuple[float, float],
    relative: bool = True,
    **kwargs,
) -> NDArray[np.float64]:
    """Compute the bandpower of the individual channels.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz.
    method : 'periodogram' | 'welch' | 'multitaper'
    band : tuple of shape (2,)
        Frequency band of interest in Hz, edges included.
    relative : bool
        If True, returns relative bandpower (band / total power).
    **kwargs
        Passed to the underlying PSD estimator.

    Returns
    -------
    bandpower : array of shape (n_channels,)
    """
    assert data.ndim == 2, (
        "The provided data must be a 2D array of shape (n_channels, n_samples)."
    )
    freqs, psd = _compute_psd(data, fs, method, **kwargs)

    assert len(band) == 2, "The 'band' argument must be a 2-length tuple."
    assert band[0] <= band[1], "The 'band' argument must be defined as (low, high) in Hz."

    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp = simpson(psd[:, idx_band], dx=freq_res)
    if relative:
        bp = bp / simpson(psd, dx=freq_res)
    return bp


def _compute_psd(
    data: NDArray[np.float64],
    fs: float,
    method: str,
    **kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (freqs, psd) for data using the requested method."""
    if method == "periodogram":
        return periodogram(data, fs, **kwargs)
    elif method == "welch":
        return welch(data, fs, **kwargs)
    elif method == "multitaper":
        psd, freqs = psd_array_multitaper(data, fs, verbose="ERROR", **kwargs)
        return freqs, psd
    raise RuntimeError(f"Unsupported method '{method}'.")


def peak_bandpower(
    data: NDArray[np.float64],
    fs: float,
    method: str,
    band: tuple[float, float],
    **kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the peak PSD value and its frequency within a band for each channel.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
    fs : float
    method : 'periodogram' | 'welch' | 'multitaper'
    band : (low_hz, high_hz)

    Returns
    -------
    peak_power : array of shape (n_channels,) — max PSD value in band
    peak_freq  : array of shape (n_channels,) — frequency of that maximum
    """
    assert data.ndim == 2
    freqs, psd = _compute_psd(data, fs, method, **kwargs)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    psd_band = psd[:, idx_band]
    freqs_band = freqs[idx_band]
    peak_idx = psd_band.argmax(axis=1)
    return psd_band[np.arange(len(data)), peak_idx], freqs_band[peak_idx]


def run_peak_analysis(
    raw: mne.io.Raw,
    log: SessionLog,
    method: str = "welch",
    bands: dict[str, tuple[float, float]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute per-channel peak PSD power and peak frequency for every session segment.

    For each band the function returns the maximum PSD value within the band
    and the frequency at which that maximum occurs. This complements
    run_band_analysis (integral-based) with spectral peak information.

    Parameters
    ----------
    raw : mne.io.Raw
    log : SessionLog
    method : 'periodogram' | 'welch' | 'multitaper'
    bands : dict mapping band name → (low_hz, high_hz). Defaults to BANDS.

    Returns
    -------
    pd.DataFrame with columns:
        segment_kind, segment_idx, channel,
        {band}_peak_power, {band}_peak_freq  — one pair per band
    """
    if bands is None:
        bands = BANDS

    fs = raw.info["sfreq"]
    ch_names = raw.ch_names
    rec_dur = raw.times[-1]
    segments = extract_segments(log)

    rows = []
    kind_counters: dict[str, int] = {}

    for onset, offset, kind in segments:
        if onset >= rec_dur:
            continue
        offset = min(offset, rec_dur)

        kind_counters[kind] = kind_counters.get(kind, -1) + 1
        seg_idx = kind_counters[kind]

        data, _ = raw[:, raw.time_as_index(onset)[0]:raw.time_as_index(offset)[0]]

        peaks: dict[str, tuple[NDArray, NDArray]] = {
            band_name: peak_bandpower(data, fs, method, band_range, **kwargs)
            for band_name, band_range in bands.items()
        }

        for ci, ch in enumerate(ch_names):
            row = {"segment_kind": kind, "segment_idx": seg_idx, "channel": ch}
            for band_name, (peak_pwr, peak_frq) in peaks.items():
                row[f"{band_name}_peak_power"] = peak_pwr[ci]
                row[f"{band_name}_peak_freq"]  = peak_frq[ci]
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_peak_by_region(
    raw: mne.io.Raw,
    log: SessionLog,
    regions: dict[str, list[str]],
    method: str = "welch",
    bands: dict[str, tuple[float, float]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute peak band power per cerebral region for every session segment.

    Channels within each region are averaged first, then peak power and peak
    frequency are estimated on the averaged signal (consistent with
    aggregate_by_region). Regions with no matching channels are silently skipped.

    Returns
    -------
    pd.DataFrame with columns:
        segment_kind, segment_idx, region,
        {band}_peak_power, {band}_peak_freq  — one pair per band
    """
    if bands is None:
        bands = BANDS

    fs = raw.info["sfreq"]
    ch_index = {ch: i for i, ch in enumerate(raw.ch_names)}
    rec_dur = raw.times[-1]
    segments = extract_segments(log)

    region_indices = {
        name: [ch_index[ch] for ch in chs if ch in ch_index]
        for name, chs in regions.items()
    }
    active_regions = {n: idx for n, idx in region_indices.items() if idx}

    if not active_regions:
        cols = (
            ["segment_kind", "segment_idx", "region"]
            + [f"{b}_{s}" for b in bands for s in ("peak_power", "peak_freq")]
        )
        return pd.DataFrame(columns=cols)

    rows = []
    kind_counters: dict[str, int] = {}

    for onset, offset, kind in segments:
        if onset >= rec_dur:
            continue
        offset = min(offset, rec_dur)

        kind_counters[kind] = kind_counters.get(kind, -1) + 1
        seg_idx = kind_counters[kind]

        data, _ = raw[:, raw.time_as_index(onset)[0]:raw.time_as_index(offset)[0]]

        for region_name, indices in active_regions.items():
            region_sig = data[indices].mean(axis=0, keepdims=True)  # (1, n_samples)
            row = {"segment_kind": kind, "segment_idx": seg_idx, "region": region_name}
            for band_name, band_range in bands.items():
                pwr, frq = peak_bandpower(region_sig, fs, method, band_range, **kwargs)
                row[f"{band_name}_peak_power"] = float(pwr[0])
                row[f"{band_name}_peak_freq"]  = float(frq[0])
            rows.append(row)

    return pd.DataFrame(rows)


def run_band_analysis(
    raw: mne.io.Raw,
    log: SessionLog,
    method: str = "welch",
    bands: dict[str, tuple[float, float]] | None = None,
    relative: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Compute per-electrode relative bandpower for every session segment.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG recording.
    log : SessionLog
        Parsed session log (list of (time_str, label) tuples).
    method : 'periodogram' | 'welch' | 'multitaper'
    bands : dict mapping band name → (low_hz, high_hz). Defaults to BANDS.
    relative : bool
        Passed to bandpower().
    **kwargs
        Passed to the PSD estimator.

    Returns
    -------
    pd.DataFrame with columns:
        segment_kind, segment_idx, channel, <band_name>, ...
    """
    if bands is None:
        bands = BANDS

    fs = raw.info["sfreq"]
    ch_names = raw.ch_names
    rec_dur = raw.times[-1]
    segments = extract_segments(log)

    rows = []
    kind_counters: dict[str, int] = {}

    for onset, offset, kind in segments:
        if onset >= rec_dur:
            continue
        offset = min(offset, rec_dur)

        kind_counters[kind] = kind_counters.get(kind, -1) + 1
        seg_idx = kind_counters[kind]

        tmin, tmax = onset, offset
        data, _ = raw[:, raw.time_as_index(tmin)[0]:raw.time_as_index(tmax)[0]]

        bp_per_band = {
            band_name: bandpower(data, fs, method, band_range, relative, **kwargs)
            for band_name, band_range in bands.items()
        }

        for ci, ch in enumerate(ch_names):
            row = {"segment_kind": kind, "segment_idx": seg_idx, "channel": ch}
            for band_name, bp_arr in bp_per_band.items():
                row[band_name] = bp_arr[ci]
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_region(
    raw: mne.io.Raw,
    log: SessionLog,
    regions: dict[str, list[str]],
    method: str = "welch",
    bands: dict[str, tuple[float, float]] | None = None,
    relative: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Compute bandpower per cerebral region for every session segment.

    For each region the time-series of its channels are averaged first, then
    a single PSD is estimated on the averaged signal (Option A aggregation).
    Channels in a region that are absent from the recording are silently skipped;
    a region with no matching channels is omitted from the output.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG recording.
    log : SessionLog
        Parsed session log.
    regions : dict mapping region name → list of channel names.
    method : 'periodogram' | 'welch' | 'multitaper'
    bands : dict mapping band name → (low_hz, high_hz). Defaults to BANDS.
    relative : bool
        Passed to bandpower().
    **kwargs
        Passed to the PSD estimator.

    Returns
    -------
    pd.DataFrame with columns:
        segment_kind, segment_idx, region, <band_name>, ...
    """
    if bands is None:
        bands = BANDS

    fs = raw.info["sfreq"]
    ch_index = {ch: i for i, ch in enumerate(raw.ch_names)}
    rec_dur = raw.times[-1]
    segments = extract_segments(log)

    # Pre-resolve channel indices per region (skip missing channels)
    region_indices: dict[str, list[int]] = {
        name: [ch_index[ch] for ch in chs if ch in ch_index]
        for name, chs in regions.items()
    }

    active_regions = {n: idx for n, idx in region_indices.items() if idx}
    if not active_regions:
        matched = set(raw.ch_names) & {ch for chs in regions.values() for ch in chs}
        print(
            f"[aggregate_by_region] Warning: none of the region channels were found in the "
            f"recording ({len(raw.ch_names)} channels, e.g. '{raw.ch_names[0]}'). "
            f"Matched: {matched or 'none'}. Returning empty DataFrame."
        )
        cols = ["segment_kind", "segment_idx", "region"] + list(bands.keys())
        return pd.DataFrame(columns=cols)

    rows = []
    kind_counters: dict[str, int] = {}

    for onset, offset, kind in segments:
        if onset >= rec_dur:
            continue
        offset = min(offset, rec_dur)

        kind_counters[kind] = kind_counters.get(kind, -1) + 1
        seg_idx = kind_counters[kind]

        data, _ = raw[:, raw.time_as_index(onset)[0]:raw.time_as_index(offset)[0]]

        for region_name, indices in active_regions.items():
            region_sig = data[indices].mean(axis=0, keepdims=True)  # (1, n_samples)
            row = {"segment_kind": kind, "segment_idx": seg_idx, "region": region_name}
            for band_name, band_range in bands.items():
                row[band_name] = float(
                    bandpower(region_sig, fs, method, band_range, relative, **kwargs)[0]
                )
            rows.append(row)

    return pd.DataFrame(rows)


def save_analysis(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_band_analysis(df: pd.DataFrame, group_col: str = "channel") -> plt.Figure:
    """Heatmap of mean relative bandpower per group × band, one subplot per phase.

    Parameters
    ----------
    df : DataFrame returned by run_band_analysis (group_col='channel') or
         aggregate_by_region (group_col='region').
    group_col : column to use as the Y-axis grouping.
    """
    band_cols = [c for c in df.columns if c not in ("segment_kind", "segment_idx", group_col)]
    kinds = df["segment_kind"].unique()

    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 8), sharey=True)
    if len(kinds) == 1:
        axes = [axes]

    for ax, kind in zip(axes, kinds):
        subset = df[df["segment_kind"] == kind]
        pivot = subset.groupby(group_col)[band_cols].mean()

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_title(kind, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(band_cols)))
        ax.set_xticklabels(band_cols, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        plt.colorbar(im, ax=ax, label="Relative power")

    fig.suptitle(f"Relative bandpower per {group_col} — EMDR session", fontsize=13)
    fig.tight_layout()
    return fig
