import json
import yaml
from datetime import datetime
from pathlib import Path

SessionLog = list[tuple[str, str]]


def load_log(path: str | Path) -> SessionLog:
    """Load a session log from a JSON file."""
    return [tuple(entry) for entry in json.loads(Path(path).read_text())]


def load_config(path: str | Path) -> dict:
    """Load study configuration from a JSON or YAML file."""
    path = Path(path)
    content = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(content)
    return json.loads(content)


def parse_time(s: str) -> datetime:
    fmt = "%H:%M:%S.%f" if "." in s else "%H:%M:%S"
    return datetime.strptime(s, fmt)


def extract_segments(log: SessionLog) -> list[tuple[float, float, str]]:
    """Return (onset_sec, offset_sec, kind) for every START/END pair in log.

    onset/offset are seconds relative to the first START_SESSION entry.
    kind is the label suffix (e.g. 'EMDR_T', 'EMDR_O', 'INTROCEPTION').
    """
    t0 = next(parse_time(t) for t, label in log if label == "START_SESSION")

    def rel(t_str: str) -> float:
        return (parse_time(t_str) - t0).total_seconds()

    segments: list[tuple[float, float, str]] = []
    i = 0
    while i < len(log):
        ts, label = log[i]
        if label.startswith("START_") and label != "START_SESSION":
            kind = label[6:]
            end_label = "END_" + kind
            for j in range(i + 1, len(log)):
                if log[j][1] == end_label:
                    segments.append((rel(ts), rel(log[j][0]), kind))
                    i = j
                    break
        i += 1
    return segments


def session_duration(log: SessionLog) -> float:
    """Total session length in seconds (START_SESSION → FIM_SESSAO)."""
    t0 = next(parse_time(t) for t, l in log if l == "START_SESSION")
    t_end = next(parse_time(t) for t, l in log if l == "FIM_SESSAO")
    return (t_end - t0).total_seconds()


def get_segments_from_raw(
    raw,
    event_map: dict[str, str],
) -> list[tuple[float, float, str]]:
    """Extract segments from raw annotations using a pipeline event map.

    Parameters
    ----------
    raw : mne.io.Raw
        Recording with annotations already applied.
    event_map : dict
        Maps analysis event names to annotation labels, e.g.
        {'emdr': 'EMDR_T', 'baseline': 'INTROCEPTION'}

    Returns
    -------
    List of (onset_sec, offset_sec, annotation_label) sorted by onset.
    """
    label_set = set(event_map.values())
    segments = []
    for ann in raw.annotations:
        if ann["description"] in label_set:
            onset = float(ann["onset"])
            offset = onset + float(ann["duration"])
            segments.append((onset, offset, ann["description"]))
    return sorted(segments)
