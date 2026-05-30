"""Interactive epoch annotator.

Opens an MNE plot where the user marks time windows with labels. Annotations are
saved as a events.json-compatible file (START_/END_ pairs, origin at 00:00:00.000).
"""

import json
from datetime import timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne


def _seconds_to_clock(seconds: float) -> str:
    """Convert seconds-from-recording-start to HH:MM:SS.mmm string."""
    total_ms = int(round(seconds * 1000))
    millis = total_ms % 1000
    total_s = total_ms // 1000
    hours = total_s // 3600
    minutes = (total_s % 3600) // 60
    secs = total_s % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _snapshot(raw: mne.io.Raw) -> set[tuple]:
    """Capture current annotations as (onset_rounded, duration_rounded, description) tuples."""
    return {
        (
            round(float(ann["onset"]), 4),
            round(float(ann["duration"]), 4),
            ann["description"],
        )
        for ann in raw.annotations
    }


def annotate_interactive(
    raw: mne.io.Raw,
    sensitivity: float = 50.0,
    title: str = "Annotate epochs — press 'a' to enter annotation mode",
) -> list[tuple[float, float, str]]:
    """Open an interactive MNE plot and return new annotations created by the user.

    Returns
    -------
    list of (onset_sec, duration_sec, label) tuples for each new non-BAD annotation.
    Annotations are already applied to raw.annotations in place by MNE.
    """
    before = _snapshot(raw)

    if matplotlib.get_backend().lower() in ("agg", ""):
        matplotlib.use("MacOSX")

    scalings = {"eeg": float(sensitivity) * 1e-6}
    time_format = "clock" if raw.info.get("meas_date") else "float"
    print(f"\n  [annotate] Close the window when done.")
    print(f"             Press 'a' in the plot to enter annotation mode,")
    print(f"             then click and drag to mark a segment.\n")

    raw.plot(
        duration=30,
        n_channels=20,
        scalings=scalings,
        title=title,
        time_format=time_format,
        block=False,
    )
    plt.show(block=True)

    after = _snapshot(raw)
    new_tuples = sorted(
        (onset, dur, desc)
        for onset, dur, desc in after - before
        if not desc.startswith("BAD")
    )

    print(f"  [annotate] {len(new_tuples)} new annotation(s) captured.")
    return new_tuples


def annotations_to_events_json(
    annotations: list[tuple[float, float, str]],
) -> list[list[str]]:
    """Convert (onset_sec, duration_sec, label) tuples to events.json START_/END_ format.

    Recording start is represented as 00:00:00.000.
    """
    events: list[list[str]] = [["00:00:00.000", "START_SESSION"]]
    for onset, duration, label in sorted(annotations):
        upper = label.upper()
        events.append([_seconds_to_clock(onset), f"START_{upper}"])
        events.append([_seconds_to_clock(onset + duration), f"END_{upper}"])
    return events


def save_user_events(events_json: list[list[str]], path: str | Path) -> None:
    """Write events list to a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events_json, indent=2))
    print(f"  [annotate] {len(events_json)} entries saved → {path}")
