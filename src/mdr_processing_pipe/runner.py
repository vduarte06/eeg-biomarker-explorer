"""Declarative EEG pipeline runner.

Reads a YAML pipeline definition and executes it end-to-end:
  input → preprocessing → feature_extraction → output

Usage
-----
    poetry run mdr-pipe run pipelines/emdr_session.yaml
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import pandas as pd

from mdr_processing_pipe.preprocessing.loader import load_sample, load_raw
from mdr_processing_pipe.preprocessing.filters import bandpass
from mdr_processing_pipe.preprocessing.pipeline import (
    remove_line_noise,
    detect_bad_channels,
    rereference,
    run_ica,
    interpolate_bad_channels,
    remove_bad_segments,
)
from mdr_processing_pipe.analysis.bandpower import (
    run_band_analysis,
    run_peak_analysis,
    aggregate_by_region,
    aggregate_peak_by_region,
    save_analysis,
)
from mdr_processing_pipe.analysis.faa import compute_faa
from mdr_processing_pipe.utils.session_log import (
    load_log,
    load_config,
    extract_segments,
    get_segments_from_raw,
)
from mdr_processing_pipe.schema import PipelineSchema

_PREPROCESSING_STEPS  = {"filter", "line_noise", "bad_channels", "rereference",
                          "ica", "interpolate", "bad_segments", "plot"}
_FEATURE_STEPS        = {"spectral_power", "peak_power", "faa", "plot"}

_DEFAULTS = {
    "filter":         {"l_freq": 1.0, "h_freq": 35.0},
    "line_noise":     {"freq": 60.0},
    "bad_channels":   {"z_threshold": 3.0},
    "rereference":    {"method": "average"},
    "ica":            {"n_components": 15, "method": "fastica", "eog_channels": None},
    "interpolate":    {},
    "bad_segments":   {"peak_amplitude": 150e-6},
    "plot":           {"duration": 20, "n_channels": 20, "scalings": "auto",
                       "title": None, "sensitivity": None},
    "spectral_power": {"events": [], "method": "welch", "bands": {},
                       "groupby": "region", "relative": True},
    "peak_power":     {"events": [], "method": "welch", "bands": {}, "groupby": "region"},
    "faa":            {"events": [], "method": "welch",
                       "channels": {"left": "F3", "right": "F4"}},
}


class PipelineRunner:
    def __init__(self, pipeline_path: str | Path):
        self.root = Path(pipeline_path).parent.parent
        raw_cfg   = load_config(Path(pipeline_path))
        self.schema  = PipelineSchema.model_validate(raw_cfg)
        self.cfg     = raw_cfg                          # kept for step-level dispatch
        self.results: dict[str, pd.DataFrame] = {}

    def run(self) -> dict[str, pd.DataFrame]:
        inp = self.cfg.get("input", {})

        raw = self._load_data(inp["dataset"])
        self._apply_session_log(raw, inp["dataset"])

        event_map = inp.get("events", {})
        regions   = {
            name: (cfg["channels"] if isinstance(cfg, dict) else cfg)
            for name, cfg in inp.get("regions", {}).items()
        }

        pipe = self.cfg.get("pipeline", {})
        self._run_preprocessing(raw, pipe.get("preprocessing", []))
        self._run_feature_extraction(raw, pipe.get("feature_extraction", []),
                                     event_map, regions)
        self._save_outputs()
        return self.results

    # ── input ─────────────────────────────────────────────────────────────────

    def _load_data(self, dataset: dict) -> mne.io.Raw:
        path = dataset["path"]
        print(f"\n── Input ──────────────────────────────────────────────────────")
        if path == "sample":
            print("  dataset  : MNE sample")
            return load_sample()
        full = self.root / path
        print(f"  dataset  : {full}")
        return load_raw(full)

    def _apply_session_log(self, raw: mne.io.Raw, dataset: dict) -> None:
        log_path = dataset.get("session_log")
        if not log_path:
            return
        log      = load_log(self.root / log_path)
        segments = extract_segments(log)
        rec_dur  = raw.times[-1]
        visible  = [(s, e, k) for s, e, k in segments if s < rec_dur]
        raw.set_annotations(mne.Annotations(
            onset       = [s for s, e, k in visible],
            duration    = [e - s for s, e, k in visible],
            description = [k for s, e, k in visible],
        ))
        print(f"  log      : {len(visible)} annotation(s) applied")

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _run_preprocessing(self, raw: mne.io.Raw, steps: list) -> None:
        if not steps:
            return
        print(f"\n── Preprocessing ({len(steps)} step(s)) ────────────────────────────────")
        for i, step in enumerate(steps, 1):
            kind, cfg = next(iter(step.items()))
            cfg = {**_DEFAULTS.get(kind, {}), **(cfg or {})}
            raw = self._preprocess_step(raw, i, kind, cfg)

    def _preprocess_step(self, raw: mne.io.Raw, idx: int, kind: str, cfg: dict) -> mne.io.Raw:
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
            print(f"{label}  ({cfg['n_components']} components, {cfg['method']})")
            return run_ica(raw, n_components=cfg["n_components"],
                           method=cfg["method"], eog_channels=cfg.get("eog_channels"))

        elif kind == "interpolate":
            print(f"{label}")
            return interpolate_bad_channels(raw)

        elif kind == "bad_segments":
            print(f"{label}  (peak > {cfg['peak_amplitude']*1e6:.0f} µV)")
            return remove_bad_segments(raw, peak_amplitude=cfg["peak_amplitude"])

        elif kind == "plot":
            title = cfg.get("title") or f"Preprocessing step {idx}"
            print(f"{label}  → '{title}' (close window to continue)")
            self._show_plot(raw, cfg, title)
            return raw

        print(f"{label}  [unknown — skipped]")
        return raw

    # ── feature extraction ────────────────────────────────────────────────────

    def _run_feature_extraction(
        self, raw: mne.io.Raw, steps: list, event_map: dict, regions: dict
    ) -> None:
        if not steps:
            return
        all_segments = get_segments_from_raw(raw, event_map)
        print(f"\n── Feature extraction ({len(steps)} step(s)) ──────────────────────────────")
        for i, step in enumerate(steps, 1):
            kind, cfg = next(iter(step.items()))
            cfg = {**_DEFAULTS.get(kind, {}), **(cfg or {})}

            if kind == "plot":
                title = cfg.get("title") or f"Feature extraction step {i}"
                print(f"  {i}. plot  → '{title}' (close window to continue)")
                self._show_plot(raw, cfg, title)
                continue

            event_names = cfg.get("events") or list(event_map.keys())
            requested   = {event_map[n]["annotation"] for n in event_names if n in event_map}
            segs        = [s for s in all_segments if s[2] in requested]
            method      = cfg.get("method", "welch")
            bands       = {k: tuple(v) for k, v in cfg.get("bands", {}).items()} or None
            groupby     = cfg.get("groupby", "region")

            print(f"  {i}. {kind}  ({groupby}, {len(segs)} segment(s))")

            if kind == "spectral_power":
                relative = cfg.get("relative", True)
                df = (
                    aggregate_by_region(raw, regions=regions, method=method,
                                        bands=bands, relative=relative, segments=segs)
                    if groupby == "region" else
                    run_band_analysis(raw, method=method, bands=bands,
                                      relative=relative, segments=segs)
                )

            elif kind == "peak_power":
                df = (
                    aggregate_peak_by_region(raw, regions=regions, method=method,
                                             bands=bands, segments=segs)
                    if groupby == "region" else
                    run_peak_analysis(raw, method=method, bands=bands, segments=segs)
                )

            elif kind == "faa":
                ch = cfg.get("channels", {})
                df = compute_faa(raw, method=method, segments=segs,
                                 ch_left=ch.get("left", "F3"), ch_right=ch.get("right", "F4"))

            else:
                print(f"     [unknown — skipped]")
                continue

            self.results[kind] = df

    # ── plot ──────────────────────────────────────────────────────────────────

    def _show_plot(self, raw: mne.io.Raw, cfg: dict, title: str) -> None:
        if matplotlib.get_backend().lower() in ("agg", ""):
            matplotlib.use("MacOSX")
        sensitivity = cfg.get("sensitivity")
        scalings = ({"eeg": float(sensitivity) * 1e-6}
                    if sensitivity is not None else cfg.get("scalings", "auto"))
        raw.plot(duration=cfg.get("duration", 20), n_channels=cfg.get("n_channels", 20),
                 scalings=scalings, title=title, block=False)
        plt.show(block=True)

    # ── output ────────────────────────────────────────────────────────────────

    def _save_outputs(self) -> None:
        out_cfg = self.cfg.get("output", {})
        formats = out_cfg.get("format", ["csv"])
        out_dir = self.root / out_cfg.get("path", "data/processed")

        print("\n── Output ─────────────────────────────────────────────────────")
        for name, df in self.results.items():
            for fmt in formats:
                path = out_dir / f"{name}.{fmt}"
                save_analysis(df, path)
                print(f"  {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def cli() -> None:
    parser = argparse.ArgumentParser(prog="mdr-pipe",
                                     description="Declarative EEG pipeline runner")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("run", help="Execute a pipeline YAML").add_argument(
        "pipeline", help="Path to pipeline YAML")

    args = parser.parse_args()
    if args.cmd == "run":
        PipelineRunner(args.pipeline).run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
