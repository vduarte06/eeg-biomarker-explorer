"""EEG preprocessing pipeline.

Steps (following the standard order):
  1. Line noise removal      — notch filter at power line frequency + harmonics
  2. Bad channel rejection   — z-score on channel variance; flat channel detection
  3. Re-referencing          — average reference (or any MNE-supported ref)
  4. ICA                     — artifact removal with automatic EOG detection
  5. Bad channel interpolation — spherical spline interpolation of flagged channels
  6. Bad time segments       — amplitude-threshold annotation and rejection
  7. Data epoching           — epochs from session-log annotations
"""

import numpy as np
import mne
from mne.preprocessing import ICA

# ── Step 1 ────────────────────────────────────────────────────────────────────


def remove_line_noise(raw: mne.io.Raw, line_freq: float = 60.0) -> mne.io.Raw:
    """Notch filter at line frequency and its first harmonic."""
    freqs = [line_freq, line_freq * 2]
    raw.notch_filter(freqs=freqs, verbose=False)
    return raw


# ── Step 2 ────────────────────────────────────────────────────────────────────


def detect_bad_channels(raw: mne.io.Raw, z_threshold: float = 3.0) -> mne.io.Raw:
    """Flag channels with abnormal or near-zero variance as bad.

    Channels whose std z-score exceeds z_threshold, or whose std is below 1 nV
    (flat), are added to raw.info['bads'].
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[i] for i in picks]
    stds = data.std(axis=1)

    flat_mask = stds < 1e-9
    z_scores = (stds - stds.mean()) / (stds.std() + 1e-30)
    noisy_mask = np.abs(z_scores) > z_threshold

    new_bads = [ch for ch, bad in zip(ch_names, flat_mask | noisy_mask) if bad]
    raw.info["bads"] = list(set(raw.info["bads"]) | set(new_bads))

    if new_bads:
        print(f"  [bad channels] flagged {len(new_bads)}: {new_bads}")
    else:
        print("  [bad channels] none detected")
    return raw


# ── Step 3 ────────────────────────────────────────────────────────────────────


def rereference(raw: mne.io.Raw, ref: str = "average") -> mne.io.Raw:
    """Apply EEG re-referencing (average, REST, or a named channel)."""
    if ref == "average":
        good_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        good_chs = [raw.ch_names[i] for i in good_picks]
        print(f"  [rereference] {len(good_chs)} channels in average: {good_chs}")
        if raw.info["bads"]:
            print(f"  [rereference] excluded (bads): {raw.info['bads']}")
    raw.set_eeg_reference(ref, projection=False, verbose=False)
    return raw


# ── Step 4 ────────────────────────────────────────────────────────────────────


def run_ica(
    raw: mne.io.Raw,
    n_components: int = 15,
    method: str = "fastica",
    eog_channels: list[str] | None = None,
    eog_threshold: float | str = 3.0,
    l_freq: float = 1.0,
    h_freq: float | None = None,
) -> mne.io.Raw:
    """Fit ICA, auto-detect EOG components, and apply to raw.

    Follows the MNE recommended workflow:
    - Fit on a filtered copy
    - Apply to the original raw (preserves low-frequency content)
    - Test each EOG surrogate channel independently and union the results,
      so a blink component caught by Fp1 OR Fp2 is excluded.

    Parameters
    ----------
    eog_channels : list of channel names used as eye-movement surrogates.
        Fp1/Fp2 are typical proxies when no dedicated EOG channel exists.
    eog_threshold : z-score threshold for find_bads_eog. Default 3.0.
        Lower (e.g. 2.5) is more aggressive. Use 'auto' for adaptive tuning.
    l_freq, h_freq : bandpass cutoffs (Hz) for the ICA fit copy.
    """
    raw_hp = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    ica = ICA(
        n_components=n_components,
        method=method,
        max_iter="auto",
        random_state=42,
        verbose=False,
    )
    ica.fit(raw_hp, verbose=False)
    print(f"  [ICA] fitted {ica.n_components_} components")

    # If eog_channels is explicitly set in the YAML, use those.
    # Otherwise auto-detect dedicated EOG channels from the recording.
    if eog_channels:
        eog_chs = [ch for ch in eog_channels if ch in raw.ch_names]
        missing = [ch for ch in eog_channels if ch not in raw.ch_names]
        if missing:
            print(f"  [ICA] EOG channel(s) not in recording, skipping: {missing}")
    else:
        eog_chs = [
            ch
            for ch in raw.ch_names
            if mne.channel_type(raw.info, raw.ch_names.index(ch)) == "eog"
        ]
        if eog_chs:
            print(f"  [ICA] auto-detected EOG channel(s): {eog_chs}")

    # Test each channel independently and union results.
    exclude: set[int] = set()
    for ch in eog_chs:
        idx, scores = ica.find_bads_eog(
            raw, ch_name=ch, threshold=eog_threshold, verbose=False
        )
        if idx:
            print(
                f"  [ICA] {ch} → component(s) {idx}  (max |r|={max(abs(scores)):.2f})"
            )
        exclude.update(idx)

    if not eog_chs:
        print("  [ICA] no EOG channels found — skipping eye movement removal")
    elif not exclude:
        print(
            f"  [ICA] no components exceeded threshold={eog_threshold}. "
            "Consider lowering eog_threshold (e.g. 2.5)."
        )
    else:
        print(f"  [ICA] excluding {len(exclude)} EOG component(s): {sorted(exclude)}")

    ica.exclude = sorted(exclude)
    ica.apply(raw, verbose=False)
    return raw


# ── Step 5 ────────────────────────────────────────────────────────────────────


def interpolate_bad_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """Spherical spline interpolation of channels in raw.info['bads']."""
    if raw.info["bads"]:
        print(f"  [interpolation] interpolating {len(raw.info['bads'])} channel(s)")
        raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
        print("  [interpolation] no bad channels to interpolate")
    return raw


# ── Step 6 ────────────────────────────────────────────────────────────────────


def remove_bad_segments(
    raw: mne.io.Raw,
    peak_amplitude: float = 150e-6,
) -> mne.io.Raw:
    """Annotate time segments exceeding peak_amplitude as 'BAD_amplitude'."""
    annots, _ = mne.preprocessing.annotate_amplitude(
        raw, peak=peak_amplitude, verbose=False
    )
    n = len(annots)
    raw.set_annotations(raw.annotations + annots)
    print(
        f"  [bad segments] annotated {n} segment(s) above {peak_amplitude*1e6:.0f} µV"
    )
    return raw


# ── Step 7 ────────────────────────────────────────────────────────────────────


def epoch_by_annotations(
    raw: mne.io.Raw,
    tmin: float = -0.2,
    tmax: float = 0.8,
) -> tuple[mne.Epochs | None, np.ndarray, dict]:
    """Create fixed-length epochs time-locked to session annotation onsets.

    Each START_* annotation (EMDR_T, EMDR_O, INTROCEPTION) becomes an epoch.
    Bad segments are automatically rejected via the 'BAD_' annotation prefix.

    Returns
    -------
    epochs   : mne.Epochs or None
    events   : np.ndarray, shape (n_events, 3)
    phase_id : dict mapping phase label → event code
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    phase_id = {
        k: v
        for k, v in event_id.items()
        if any(k.startswith(p) for p in ("EMDR_T", "EMDR_O", "INTROCEPTION"))
    }

    if not phase_id:
        print("  [epoching] no phase annotations found — returning None")
        return None, events, phase_id

    epochs = mne.Epochs(
        raw,
        events,
        event_id=phase_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )
    print(f"  [epoching] {len(epochs)} epochs across {list(phase_id.keys())}")
    return epochs, events, phase_id


def crop_raw_to_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    phase_id: dict,
    tmin: float = -0.2,
    tmax: float = 0.8,
) -> mne.io.Raw:
    """Crop raw to the union of epoch windows and concatenate into a single Raw.

    Parameters
    ----------
    raw      : cleaned mne.io.Raw (not modified)
    events   : events array returned by epoch_by_annotations
    phase_id : phase-label → event-code mapping returned by epoch_by_annotations
    tmin, tmax : epoch window relative to each event onset (seconds)

    Returns
    -------
    mne.io.Raw — concatenation of all cropped segments
    """
    phase_codes = set(phase_id.values())
    sfreq = raw.info["sfreq"]
    segments = []

    for sample, _, code in events:
        if code not in phase_codes:
            continue
        onset = sample / sfreq
        t_start = max(raw.times[0] + raw.first_samp / sfreq, onset + tmin)
        t_end = min(raw.times[-1] + raw.first_samp / sfreq, onset + tmax)
        segments.append(raw.copy().crop(tmin=t_start, tmax=t_end))

    if not segments:
        print("  [crop] no matching events found — returning original raw")
        return raw

    cropped = mne.concatenate_raws(segments)
    print(
        f"  [crop] concatenated {len(segments)} segments → {cropped.times[-1]:.1f}s total"
    )
    return cropped


# ── Full pipeline ─────────────────────────────────────────────────────────────


def preprocess(
    raw: mne.io.Raw,
    cfg: dict,
    do_epoch: bool = False,
) -> tuple[mne.io.Raw, mne.Epochs | None]:
    """Run the preprocessing pipeline from a config dict.

    Parameters
    ----------
    raw : mne.io.Raw  (preloaded, EEG channels already picked)
    cfg : dict  — the 'pipeline' section of the pipeline YAML.
        Supports both old key 'reference' and new key 'rereference.method'.
    do_epoch : bool
        If False (default), skip step 7. The runner applies annotations and
        epochs externally after this function returns.

    Returns
    -------
    raw    : cleaned mne.io.Raw
    epochs : mne.Epochs or None
    """
    # Resolve re-reference key from either schema
    ref = cfg.get("rereference", {}).get("method") or cfg.get("reference", "average")

    print("\n── Preprocessing ─────────────────────────────────────────────")

    print("1. Line noise removal")
    raw = remove_line_noise(raw, line_freq=cfg["line_noise_hz"])

    print("2. Bad channel detection")
    raw = detect_bad_channels(raw, z_threshold=cfg["bad_channels"]["z_threshold"])

    print("3. Re-referencing")
    raw = rereference(raw, ref=ref)

    print("4. ICA")
    ica_cfg = cfg["ica"]
    raw = run_ica(
        raw,
        n_components=ica_cfg["n_components"],
        method=ica_cfg["method"],
        eog_channels=ica_cfg.get("eog_channels"),
    )

    print("5. Bad channel interpolation")
    raw = interpolate_bad_channels(raw)

    print("6. Bad time segment removal")
    raw = remove_bad_segments(raw, peak_amplitude=cfg["bad_segments"]["peak_amplitude"])

    epochs = None
    if do_epoch:
        print("7. Epoching")
        ep_cfg = cfg.get("epoching", {"tmin": -0.2, "tmax": 0.8})
        epochs, _, _ = epoch_by_annotations(
            raw, tmin=ep_cfg["tmin"], tmax=ep_cfg["tmax"]
        )
    else:
        print("7. Epoching — skipped (handled by runner after annotation)")

    print("── Preprocessing complete ─────────────────────────────────────\n")
    return raw, epochs
