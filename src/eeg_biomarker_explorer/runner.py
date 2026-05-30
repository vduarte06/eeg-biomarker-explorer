"""Declarative EEG pipeline runner.

Reads a YAML pipeline definition and executes it end-to-end:
  input → preprocessing → feature_extraction → output

Usage
-----
    poetry run mdr-pipe run pipelines/emdr_session.yaml
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_biomarker_explorer.preprocessing.loader import load_sample, load_raw
from eeg_biomarker_explorer.preprocessing.filters import bandpass
from eeg_biomarker_explorer.preprocessing.pipeline import (
    remove_line_noise,
    detect_bad_channels,
    rereference,
    run_ica,
    interpolate_bad_channels,
    remove_bad_segments,
    epoch_by_annotations,
    crop_raw_to_epochs,
)
from eeg_biomarker_explorer.analysis.bandpower import (
    run_band_analysis,
    run_peak_analysis,
    aggregate_by_region,
    aggregate_peak_by_region,
    append_segment_means,
    save_analysis,
)
from eeg_biomarker_explorer.analysis.faa import compute_faa
from eeg_biomarker_explorer.utils.session_log import (
    load_log,
    load_config,
    extract_segments,
    get_segments_from_raw,
    parse_time,
)
from eeg_biomarker_explorer.schema import PipelineSchema

_PREPROCESSING_STEPS = {
    "filter",
    "line_noise",
    "bad_channels",
    "rereference",
    "ica",
    "interpolate",
    "bad_segment_annotation",
    "annotate",
    "plot",
    "save",
}
_FEATURE_STEPS = {"spectral_power", "peak_power", "faa", "plot"}

_DEFAULTS = {
    "filter": {"l_freq": 1.0, "h_freq": 35.0},
    "line_noise": {"freq": 60.0},
    "bad_channels": {"z_threshold": 3.0},
    "rereference": {"method": "average"},
    "ica": {
        "n_components": 15,
        "method": "fastica",
        "eog_channels": None,
        "eog_threshold": 3.0,
        "l_freq": 1.0,
        "h_freq": None,
    },
    "interpolate": {},
    "bad_segment_annotation": {"peak_amplitude": 150e-6},
    "annotate": {"output": "./data/raw/user_epochs.json", "sensitivity": 50.0},
    "save": {"suffix": "_after_ica", "format": "edf"},
    "plot": {
        "duration": 20,
        "n_channels": 20,
        "scalings": "auto",
        "title": None,
        "sensitivity": None,
        "time_format": "clock",
    },
    "spectral_power": {
        "events": [],
        "method": "welch",
        "bands": {},
        "groupby": "region",
        "relative": True,
    },
    "peak_power": {"events": [], "method": "welch", "bands": {}, "groupby": "region"},
    "faa": {"events": [], "method": "welch", "channels": {"left": "F3", "right": "F4"}},
}


def _clock_to_edf_onset(
    raw: mne.io.Raw, log: list, onset_rel_to_session: float
) -> float:
    """Convert a session-relative onset (seconds from START_SESSION) to an EDF-relative onset.

    events.json stores absolute wall-clock times. extract_segments converts those to
    seconds relative to START_SESSION. This function adds the offset between EDF
    meas_date and START_SESSION so the onset is correctly anchored in the recording.
    """
    meas_date = raw.info.get("meas_date")
    if meas_date is None:
        return onset_rel_to_session
    t0_str = next(t for t, label in log if label == "START_SESSION")
    t0 = parse_time(t0_str)
    meas = meas_date.replace(tzinfo=None)
    t0_full = t0.replace(year=meas.year, month=meas.month, day=meas.day)
    session_offset = (t0_full - meas).total_seconds()
    return onset_rel_to_session + session_offset


class PipelineRunner:
    def __init__(self, pipeline_path: str | Path):
        self.root = Path(pipeline_path).parent.parent
        raw_cfg = load_config(Path(pipeline_path))
        self.schema = PipelineSchema.model_validate(raw_cfg)
        self.cfg = raw_cfg  # kept for step-level dispatch
        self.results: dict[str, pd.DataFrame] = {}

    def run(self) -> dict[str, pd.DataFrame]:
        inp = self.cfg.get("input", {})

        self._input_path = inp["dataset"].get("path")
        raw = self._load_data(inp["dataset"])

        raw_events = inp.get("events", {})
        log_path = raw_events.get("path")
        event_map = {k: v for k, v in raw_events.items() if k != "path"}

        if log_path and inp["dataset"].get("path") == "sample" and event_map:
            self._ensure_sample_events(log_path, event_map)

        self._apply_session_log(raw, log_path)

        raw_user_events = inp.get("user_events", {})
        user_log_path = raw_user_events.get("path") if raw_user_events else None
        user_event_map = {
            k: v for k, v in (raw_user_events or {}).items() if k != "path"
        }
        if user_log_path:
            self._apply_user_events(raw, user_log_path)
            event_map.update(user_event_map)

        regions = {
            name: (cfg["channels"] if isinstance(cfg, dict) else cfg)
            for name, cfg in inp.get("regions", {}).items()
        }

        pipe = self.cfg.get("pipeline", {})
        self._run_preprocessing(raw, pipe.get("preprocessing", []))
        self._run_feature_extraction(
            raw, pipe.get("feature_extraction", []), event_map, regions
        )
        self._save_outputs()
        return self.results

    # ── input ─────────────────────────────────────────────────────────────────

    def _load_data(self, dataset: dict) -> mne.io.Raw:
        path = dataset["path"]
        exclude = dataset.get("exclude", [])
        print(f"\n── Input ──────────────────────────────────────────────────────")
        if path == "sample":
            print("  dataset  : MNE sample")
            return load_sample()
        full = self.root / path
        print(f"  dataset  : {full}")
        if exclude:
            print(f"  exclude  : {exclude}")
        return load_raw(full, exclude=exclude)

    def _ensure_sample_events(self, log_path: str, event_map: dict) -> None:
        path = self.root / log_path
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = list(dict.fromkeys(event_map.values()))
        events: list[list[str]] = [["00:00:00.000", "START_SESSION"]]
        t = 10
        for i, label in enumerate(labels):
            if i > 0:
                t += 20
            for rep in range(2):
                if rep > 0:
                    t += 15
                h, m, s = t // 3600, (t % 3600) // 60, t % 60
                events.append([f"{h:02d}:{m:02d}:{s:02d}.000", f"START_{label}"])
                t += 30
                h, m, s = t // 3600, (t % 3600) // 60, t % 60
                events.append([f"{h:02d}:{m:02d}:{s:02d}.000", f"END_{label}"])
        t += 10
        h, m, s = t // 3600, (t % 3600) // 60, t % 60
        events.append([f"{h:02d}:{m:02d}:{s:02d}.000", "FIM_SESSAO"])
        path.write_text(json.dumps(events, indent=2))
        print(f"  events   : generated sample events → {path}")

    def _apply_session_log(self, raw: mne.io.Raw, log_path: str | None) -> None:
        if not log_path:
            return
        log = load_log(self.root / log_path)
        segments = extract_segments(log)
        rec_dur = raw.times[-1]
        adjusted = [
            (_clock_to_edf_onset(raw, log, s), _clock_to_edf_onset(raw, log, e), k)
            for s, e, k in segments
        ]
        visible = [(s, e, k) for s, e, k in adjusted if 0 <= s < rec_dur]
        raw.set_annotations(
            mne.Annotations(
                onset=[s for s, e, k in visible],
                duration=[e - s for s, e, k in visible],
                description=[k for s, e, k in visible],
            )
        )
        print(f"  log      : {len(visible)} annotation(s) applied")

    def _apply_user_events(self, raw: mne.io.Raw, log_path: str) -> None:
        log = load_log(self.root / log_path)
        segments = extract_segments(log)
        rec_dur = raw.times[-1]
        visible = [(s, e, k) for s, e, k in segments if s < rec_dur]
        new_annots = mne.Annotations(
            onset=[s for s, e, k in visible],
            duration=[e - s for s, e, k in visible],
            description=[k for s, e, k in visible],
        )
        raw.set_annotations(raw.annotations + new_annots)
        print(f"  user_events: {len(visible)} annotation(s) merged from {log_path}")

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _run_preprocessing(self, raw: mne.io.Raw, steps: list) -> None:
        if not steps:
            return
        print(
            f"\n── Preprocessing ({len(steps)} step(s)) ────────────────────────────────"
        )
        for i, step in enumerate(steps, 1):
            kind, cfg = next(iter(step.items()))
            cfg = {**_DEFAULTS.get(kind, {}), **(cfg or {})}
            raw = self._preprocess_step(raw, i, kind, cfg)

    def _preprocess_step(
        self, raw: mne.io.Raw, idx: int, kind: str, cfg: dict
    ) -> mne.io.Raw:
        label = f"  {idx}. {kind}"

        if kind == "filter":
            print(f"{label}  ({cfg['l_freq']}–{cfg['h_freq']} Hz)")
            return bandpass(raw, l_freq=cfg["l_freq"], h_freq=cfg["h_freq"])

        elif kind == "line_noise":
            print(f"{label}  ({cfg['freq']} Hz notch)")
            return remove_line_noise(raw, line_freq=cfg["freq"])

        elif kind == "bad_channels":
            print(f"{label}  (z > {cfg['z_threshold']})")
            return detect_bad_channels(raw, z_threshold=cfg["z_threshold"])

        elif kind == "rereference":
            print(f"{label}  ({cfg['method']})")
            return rereference(raw, ref=cfg["method"])

        elif kind == "ica":
            print(
                f"{label}  ({cfg['n_components']} components, {cfg['method']}, "
                f"fit filter {cfg['l_freq']}–{cfg['h_freq']} Hz)"
            )
            return run_ica(
                raw,
                n_components=cfg["n_components"],
                method=cfg["method"],
                eog_channels=cfg.get("eog_channels"),
                eog_threshold=cfg.get("eog_threshold", 3.0),
                l_freq=cfg["l_freq"],
                h_freq=cfg["h_freq"],
            )

        elif kind == "interpolate":
            print(f"{label}")
            return interpolate_bad_channels(raw)

        elif kind == "bad_segment_annotation":
            print(f"{label}  (peak > {cfg['peak_amplitude']*1e6:.0f} µV)")
            return remove_bad_segments(raw, peak_amplitude=cfg["peak_amplitude"])

        elif kind == "annotate":
            output = cfg.get("output", "./data/raw/user_epochs.json")
            sensitivity = cfg.get("sensitivity", 50.0)
            print(f"{label}  → '{output}'")
            return self._annotate_step(raw, output, sensitivity)

        elif kind == "plot":
            title = cfg.get("title") or f"Preprocessing step {idx}"
            print(f"{label}  → '{title}' (close window to continue)")
            self._show_plot(raw, cfg, title)
            return raw

        elif kind == "save":
            suffix = cfg.get("suffix", "_after_ica")
            fmt = cfg.get("format", "edf")
            print(f"{label}  (suffix='{suffix}', format={fmt})")
            self._save_raw_step(raw, suffix, fmt)
            return raw

        print(f"{label}  [unknown — skipped]")
        return raw

    def _annotate_step(
        self, raw: mne.io.Raw, output: str, sensitivity: float
    ) -> mne.io.Raw:
        from eeg_biomarker_explorer.preprocessing.annotator import (
            annotate_interactive,
            annotations_to_events_json,
            save_user_events,
        )

        new_anns = annotate_interactive(raw, sensitivity=sensitivity)
        if new_anns:
            events_json = annotations_to_events_json(new_anns)
            save_user_events(events_json, self.root / output)
        return raw

    def _save_raw_step(self, raw: mne.io.Raw, suffix: str, fmt: str) -> None:
        if not self._input_path or self._input_path == "sample":
            print("     [save] skipped — no input file path")
            return
        inp = self.root / self._input_path
        out = inp.with_name(inp.stem + suffix + inp.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "edf":
            data = raw.get_data()
            max_v = 9.999999
            encodable = [
                ch
                for i, ch in enumerate(raw.ch_names)
                if abs(data[i].min()) <= max_v and abs(data[i].max()) <= max_v
            ]
            dropped = [ch for ch in raw.ch_names if ch not in encodable]
            if dropped:
                print(
                    f"     [save] dropped channels (range too large for EDF): {dropped}"
                )
                raw_out = raw.copy().pick(encodable)
            else:
                raw_out = raw
            raw_out.export(
                str(out),
                fmt="edf",
                physical_range="channelwise",
                overwrite=True,
                verbose=False,
            )
        else:
            raw.save(str(out), overwrite=True, verbose=False)
        print(f"     [save] → {out}")

    # ── feature extraction ────────────────────────────────────────────────────

    def _run_feature_extraction(
        self, raw: mne.io.Raw, steps: list, event_map: dict, regions: dict
    ) -> None:
        if not steps:
            return
        all_segments = get_segments_from_raw(raw, event_map)
        print(
            f"\n── Feature extraction ({len(steps)} step(s)) ──────────────────────────────"
        )
        for i, step in enumerate(steps, 1):
            kind, cfg = next(iter(step.items()))
            cfg = {**_DEFAULTS.get(kind, {}), **(cfg or {})}

            if kind == "plot":
                title = cfg.get("title") or f"Feature extraction step {i}"
                print(f"  {i}. plot  → '{title}' (close window to continue)")
                self._show_plot(raw, cfg, title)
                continue

            event_names = cfg.get("events") or list(event_map.keys())
            requested = {event_map[n] for n in event_names if n in event_map}
            segs = [s for s in all_segments if s[2] in requested]
            method = cfg.get("method", "welch")
            bands = {k: tuple(v) for k, v in cfg.get("bands", {}).items()} or None
            groupby = cfg.get("groupby", "region")

            print(f"  {i}. {kind}  ({groupby}, {len(segs)} segment(s))")

            if kind == "spectral_power":
                relative = cfg.get("relative", True)
                df = (
                    aggregate_by_region(
                        raw,
                        regions=regions,
                        method=method,
                        bands=bands,
                        relative=relative,
                        segments=segs,
                    )
                    if groupby == "region"
                    else run_band_analysis(
                        raw,
                        method=method,
                        bands=bands,
                        relative=relative,
                        segments=segs,
                    )
                )

            elif kind == "peak_power":
                df = (
                    aggregate_peak_by_region(
                        raw, regions=regions, method=method, bands=bands, segments=segs
                    )
                    if groupby == "region"
                    else run_peak_analysis(
                        raw, method=method, bands=bands, segments=segs
                    )
                )

            elif kind == "faa":
                ch = cfg.get("channels", {})
                df = compute_faa(
                    raw,
                    method=method,
                    segments=segs,
                    ch_left=ch.get("left", "F3"),
                    ch_right=ch.get("right", "F4"),
                )

            else:
                print(f"     [unknown — skipped]")
                continue

            self.results[kind] = df

    # ── plot ──────────────────────────────────────────────────────────────────

    def _show_plot(self, raw: mne.io.Raw, cfg: dict, title: str) -> None:
        if matplotlib.get_backend().lower() in ("agg", ""):
            matplotlib.use("MacOSX")
        sensitivity = cfg.get("sensitivity")
        scalings = (
            {"eeg": float(sensitivity) * 1e-6}
            if sensitivity is not None
            else cfg.get("scalings", "auto")
        )
        time_format = cfg.get(
            "time_format", "clock" if raw.info.get("meas_date") else "float"
        )
        raw.plot(
            duration=cfg.get("duration", 20),
            n_channels=cfg.get("n_channels", 20),
            scalings=scalings,
            title=title,
            time_format=time_format,
            block=False,
        )
        plt.show(block=True)

    # ── output ────────────────────────────────────────────────────────────────

    def _save_outputs(self) -> None:
        out_cfg = self.cfg.get("output", {})
        formats = out_cfg.get("format", ["csv"])
        out_dir = self.root / out_cfg.get("path", "data/processed")

        print("\n── Output ─────────────────────────────────────────────────────")
        for name, df in self.results.items():
            df = append_segment_means(df)
            for fmt in formats:
                path = out_dir / f"{name}.{fmt}"
                save_analysis(df, path)
                print(f"  {path}")


# ── Pipeline template ─────────────────────────────────────────────────────────

_TEMPLATE = """\
version: 1

# -----------------------------------------------------------
# Input
# -----------------------------------------------------------
input:

  dataset:
    path: sample                       # "sample" | path to .edf / .fif / .bdf / .set / .vhdr

  events:
    path:        ./data/raw/events.json  # auto-generated when path=sample and file is absent
    condition_a: LABEL_A
    condition_b: LABEL_B

  regions:
    frontal:   [Fp1, Fp2, F3, F4, Fz, F7, F8]
    central:   [C3, Cz, C4]
    temporal:  [T7, T8]
    parietal:  [P3, P4, Pz, P7, P8]
    occipital: [O1, O2]

# -----------------------------------------------------------
# Pipeline — steps run in order; insert plot anywhere
# -----------------------------------------------------------
pipeline:

  preprocessing:
    - filter:
        l_freq: 1.0
        h_freq: 35.0

    - line_noise:
        freq: 60.0         # 50 Hz (EU) | 60 Hz (BR/US)

    - bad_channels:
        z_threshold: 3.0

    - rereference:
        method: average

    - ica:
        n_components: 15
        method: fastica
        # eog_channels: [Fp1, Fp2]  # override auto-detection if needed
        eog_threshold: 3.0

    - interpolate:

    - bad_segment_annotation:
        peak_amplitude: 150.0e-6

    - plot:
        title: After preprocessing
        sensitivity: 50    # µV/div

  feature_extraction:
    - spectral_power:
        events: [condition_a, condition_b]
        method: welch
        bands:
          theta: [4, 8]
          alpha: [8, 13]
          beta:  [13, 30]
        groupby: region    # channel | region
        relative: true

    - faa:
        events: [condition_a, condition_b]
        method: welch
        channels:
          left: F3
          right: F4

output:
  format: [csv, parquet]
  path: ./data/processed
"""


def _parse_clock(hhmm: str, ref: datetime.datetime) -> datetime.datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    candidate = ref.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate < ref:
        candidate += datetime.timedelta(days=1)
    return candidate


def crop_edf(input_path: str, start: str, end: str, output: str) -> None:
    """Crop an EDF file between two HH:MM wall-clock times and save a new EDF."""
    inp = Path(input_path)
    out = Path(output)

    raw = mne.io.read_raw_edf(str(inp), preload=False, verbose=False)
    meas_date = raw.info["meas_date"]  # timezone-aware UTC datetime

    t_start = _parse_clock(start, meas_date)
    t_end = _parse_clock(end, meas_date)

    tmin = (t_start - meas_date).total_seconds()
    tmax = (t_end - meas_date).total_seconds()

    rec_dur = raw.times[-1]
    if tmin < 0 or tmax > rec_dur:
        print(
            f"Error: window [{start}–{end}] maps to [{tmin:.0f}s–{tmax:.0f}s] but "
            f"recording is {rec_dur:.0f}s long (starts {meas_date.strftime('%H:%M')} UTC)."
        )
        sys.exit(1)

    print(f"  input    : {inp}")
    print(f"  start    : {start} UTC  →  {tmin:.0f}s from recording start")
    print(f"  end      : {end} UTC  →  {tmax:.0f}s from recording start")
    print(
        f"  duration : {tmax - tmin:.0f}s  ({datetime.timedelta(seconds=int(tmax - tmin))})"
    )
    print(f"  output   : {out}")

    raw.load_data(verbose=False)
    raw.crop(tmin=tmin, tmax=tmax)

    # EDF stores physical range as µV with an 8-char ASCII field (7 digits + sign).
    # Channels whose |value| > ~10 V overflow the field; drop them with a warning.
    data = raw.get_data()
    max_v = 9.999999  # 9999999 µV fits in 7 chars, leaving room for '-'
    encodable = [
        ch
        for i, ch in enumerate(raw.ch_names)
        if abs(data[i].min()) <= max_v and abs(data[i].max()) <= max_v
    ]
    dropped = [ch for ch in raw.ch_names if ch not in encodable]
    if dropped:
        print(f"  dropped  : {dropped}  (physical range too large for EDF format)")
        raw.pick(encodable)

    out.parent.mkdir(parents=True, exist_ok=True)
    raw.export(
        str(out), fmt="edf", physical_range="channelwise", overwrite=True, verbose=False
    )
    print("  done.")


def annotate_command(
    edf_path: str,
    events_path: str | None,
    output: str | None,
    sensitivity: float,
) -> None:
    """Open an interactive plot on an EDF file and save user-drawn annotations as JSON.

    The output file uses the same START_/END_ format as events.json, with
    00:00:00.000 representing the recording start.
    """
    from eeg_biomarker_explorer.preprocessing.annotator import (
        annotate_interactive,
        annotations_to_events_json,
        save_user_events,
    )

    inp = Path(edf_path)
    out = Path(output) if output else inp.with_name(inp.stem + "_user_epochs.json")

    print(f"\n── Annotate ────────────────────────────────────────────────────")
    print(f"  input    : {inp}")

    raw = load_raw(inp)

    if events_path:
        log = load_log(Path(events_path))
        segments = extract_segments(log)
        rec_dur = raw.times[-1]
        adjusted = [
            (_clock_to_edf_onset(raw, log, s), _clock_to_edf_onset(raw, log, e), k)
            for s, e, k in segments
        ]
        visible = [(s, e, k) for s, e, k in adjusted if 0 <= s < rec_dur]
        raw.set_annotations(
            mne.Annotations(
                onset=[s for s, e, k in visible],
                duration=[e - s for s, e, k in visible],
                description=[k for s, e, k in visible],
            )
        )
        print(f"  events   : {len(visible)} reference annotation(s) from {events_path}")

    new_anns = annotate_interactive(raw, sensitivity=sensitivity)

    if not new_anns:
        print("  no new annotations created — nothing saved.")
        return

    events_json = annotations_to_events_json(new_anns)
    save_user_events(events_json, out)
    print(f"  output   : {out}")
    print("  done.")


def new_pipeline(path: str) -> None:
    dest = Path(path)
    if dest.exists():
        print(f"Error: {dest} already exists. Choose a different name.")
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_TEMPLATE)
    print(f"Created {dest}")
    print("Edit the file, then run:  eeg-pipe run", dest)


def epochs_command(
    edf_path: str,
    events_path: str | None,
    output: str | None,
    tmin: float,
    tmax: float,
) -> None:
    """Extract epoch windows from an EDF and save a cropped EDF."""
    inp = Path(edf_path)
    out = Path(output) if output else inp.with_stem(inp.stem + "_epochs")

    print(f"\n── Epochs ─────────────────────────────────────────────────────")
    print(f"  input    : {inp}")

    raw = load_raw(inp)

    if events_path:
        log = load_log(Path(events_path))
        segments = extract_segments(log)
        rec_dur = raw.times[-1]
        adjusted = [
            (_clock_to_edf_onset(raw, log, s), _clock_to_edf_onset(raw, log, e), k)
            for s, e, k in segments
        ]
        visible = [(s, e, k) for s, e, k in adjusted if 0 <= s < rec_dur]
        raw.set_annotations(
            mne.Annotations(
                onset=[s for s, e, k in visible],
                duration=[e - s for s, e, k in visible],
                description=[k for s, e, k in visible],
            )
        )
        print(f"  events   : {len(visible)} annotation(s) applied from {events_path}")

    epochs, events, phase_id = epoch_by_annotations(raw, tmin=tmin, tmax=tmax)

    if epochs is None:
        print("  no epochs found — nothing to save.")
        sys.exit(1)

    cropped = crop_raw_to_epochs(raw, events, phase_id, tmin=tmin, tmax=tmax)

    # EDF physical range constraint: drop channels whose values exceed what
    # the 8-char ASCII field can encode (±9 999 999 µV).
    data = cropped.get_data()
    max_v = 9.999999
    encodable = [
        ch
        for i, ch in enumerate(cropped.ch_names)
        if abs(data[i].min()) <= max_v and abs(data[i].max()) <= max_v
    ]
    dropped = [ch for ch in cropped.ch_names if ch not in encodable]
    if dropped:
        print(f"  dropped  : {dropped}  (physical range too large for EDF format)")
        cropped.pick(encodable)

    out.parent.mkdir(parents=True, exist_ok=True)
    cropped.export(
        str(out), fmt="edf", physical_range="channelwise", overwrite=True, verbose=False
    )
    print(f"  output   : {out}")
    print("  done.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="eeg-pipe", description="Declarative EEG pipeline runner"
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("run", help="Execute a pipeline YAML").add_argument(
        "pipeline", help="Path to pipeline YAML"
    )
    sub.add_parser("new", help="Create a new pipeline YAML from template").add_argument(
        "path", help="Destination path (e.g. pipelines/my_study.yaml)"
    )

    ann_p = sub.add_parser(
        "annotate", help="Interactively annotate epochs on an EDF and save to JSON"
    )
    ann_p.add_argument("edf", help="Path to EDF file")
    ann_p.add_argument(
        "--events",
        "-e",
        metavar="FILE",
        help="Path to events.json to display as reference annotations (optional)",
    )
    ann_p.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output JSON path (default: <edf-stem>_user_epochs.json)",
    )
    ann_p.add_argument(
        "--sensitivity",
        "-s",
        type=float,
        default=50.0,
        metavar="UV",
        help="Plot sensitivity in µV/div (default: 50)",
    )

    crop_p = sub.add_parser("crop", help="Crop an EDF file to a clock-time window")
    crop_p.add_argument("input", help="Path to source EDF file")
    crop_p.add_argument(
        "--start", required=True, metavar="HH:MM", help="Start time (UTC)"
    )
    crop_p.add_argument("--end", required=True, metavar="HH:MM", help="End time (UTC)")
    crop_p.add_argument(
        "--output", "-o", required=True, metavar="FILE", help="Output EDF path"
    )

    ep_p = sub.add_parser("epochs", help="Crop an EDF to its annotated epoch windows")
    ep_p.add_argument("edf", help="Path to source EDF file")
    ep_p.add_argument(
        "--events",
        "-e",
        metavar="FILE",
        help="Path to events.json session log (optional if EDF already has annotations)",
    )
    ep_p.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output EDF path (default: <input>_epochs.edf)",
    )
    ep_p.add_argument(
        "--tmin",
        type=float,
        default=-0.2,
        metavar="SEC",
        help="Epoch start relative to onset in seconds (default: -0.2)",
    )
    ep_p.add_argument(
        "--tmax",
        type=float,
        default=0.8,
        metavar="SEC",
        help="Epoch end relative to onset in seconds (default: 0.8)",
    )

    args = parser.parse_args()
    if args.cmd == "run":
        PipelineRunner(args.pipeline).run()
    elif args.cmd == "annotate":
        annotate_command(
            args.edf,
            events_path=args.events,
            output=args.output,
            sensitivity=args.sensitivity,
        )
    elif args.cmd == "new":
        new_pipeline(args.path)
    elif args.cmd == "crop":
        crop_edf(args.input, args.start, args.end, args.output)
    elif args.cmd == "epochs":
        epochs_command(
            args.edf,
            events_path=args.events,
            output=args.output,
            tmin=args.tmin,
            tmax=args.tmax,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
