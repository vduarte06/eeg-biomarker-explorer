import mne


def bandpass(raw: mne.io.Raw, l_freq: float = 1.0, h_freq: float = 35.0) -> mne.io.Raw:
    raw.filter(l_freq, h_freq, verbose=False)
    return raw
