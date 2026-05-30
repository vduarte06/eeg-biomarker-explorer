import numpy as np
import mne

from eeg_biomarker_explorer.utils.session_log import SessionLog, extract_segments, session_duration

CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7",  "C3",  "Cz", "C4", "T8",
    "P7",  "P3",  "Pz", "P4", "P8",
    "O1",  "O2",
]
_POSTERIOR = {"O1", "O2", "P3", "P4", "Pz", "P7", "P8"}
_FRONTAL   = {"Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"}


def simulate_session(log: SessionLog, sfreq: int = 250, seed: int = 0) -> mne.io.RawArray:
    """Simulate a full-session EEG RawArray from a session log.

    Parameters
    ----------
    log:   list of (time_str, label) tuples — must contain START_SESSION and FIM_SESSAO
    sfreq: sampling frequency in Hz
    seed:  random seed for reproducibility

    Returns
    -------
    mne.io.RawArray with annotations marking each phase segment
    """
    dur = session_duration(log)
    segments = extract_segments(log)
    n = int(dur * sfreq)
    rng = np.random.default_rng(seed)

    def _pink(length: int) -> np.ndarray:
        freqs = np.fft.rfftfreq(length, 1 / sfreq)
        amp = np.zeros(len(freqs))
        amp[1:] = 1.0 / np.sqrt(freqs[1:])
        phase = rng.uniform(0, 2 * np.pi, len(freqs))
        sig = np.fft.irfft(amp * np.exp(1j * phase), length)
        return sig * (8e-6 / sig.std())

    def _osc(freq: float, length: int, amp: float) -> np.ndarray:
        t = np.arange(length) / sfreq
        return amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

    data = np.zeros((len(CH_NAMES), n))
    for ci, ch in enumerate(CH_NAMES):
        alpha_amp = 15e-6 if ch in _POSTERIOR else 5e-6
        data[ci] = (
            _pink(n)
            + _osc(10, n, alpha_amp)   # alpha
            + _osc(20, n, 2e-6)        # beta
            + _osc(6,  n, 1.5e-6)      # theta bg
        )

    for onset, offset, kind in segments:
        s = int(onset  * sfreq)
        e = min(int(offset * sfreq), n)
        length = e - s

        if kind == "EMDR_T":
            for ci in range(len(CH_NAMES)):
                data[ci, s:e] += (
                    _osc(6,  length, 10e-6)
                    - _osc(10, length, 6e-6)
                    + rng.normal(0, 2e-6, length)
                )
        elif kind == "EMDR_O":
            for ci, ch in enumerate(CH_NAMES):
                if ch in _FRONTAL:
                    data[ci, s:e] += _osc(0.8, length, 60e-6)
                data[ci, s:e] -= _osc(10, length, 8e-6)
        elif kind == "INTROCEPTION":
            for ci, ch in enumerate(CH_NAMES):
                if ch in _FRONTAL:
                    data[ci, s:e] += _osc(6, length, 8e-6)
                if ch in _POSTERIOR:
                    data[ci, s:e] += _osc(10, length, 10e-6)

    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_eeg_reference("average", projection=True)

    onsets      = [s for s, _, _ in segments]
    durations   = [e - s for s, e, _ in segments]
    descriptions = [k for _, _, k in segments]
    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))

    return raw
