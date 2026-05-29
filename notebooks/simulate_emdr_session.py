from pathlib import Path

import mne
import pandas as pd

from mdr_processing_pipe.preprocessing.loader import load_sample  # swap → load_raw("file.edf")
from mdr_processing_pipe.preprocessing.filters import bandpass
from mdr_processing_pipe.preprocessing.pipeline import preprocess
from mdr_processing_pipe.utils.session_log import load_log, load_config, extract_segments
from mdr_processing_pipe.analysis.bandpower import (
    run_band_analysis,
    run_peak_analysis,
    aggregate_by_region,
    aggregate_peak_by_region,
    save_analysis,
    plot_band_analysis,
)
from mdr_processing_pipe.analysis.faa import compute_faa

ROOT = Path(__file__).parents[1]

# ── config & inputs ───────────────────────────────────────────────────────────
config   = load_config(ROOT / "data" / "config.yaml")
ANALYSIS = config["analysis"]          # channel | region | both
FILTERS  = config["filters"]
REGIONS  = config["regions"]
PREPROC  = config["preprocessing"]

PLOT_RAW   = False      # set True to open the interactive EEG browser
PLOT_BANDS = False      # set True to generate and save the heatmap

log = load_log(ROOT / "data" / "raw" / "events.json")

# ── EEG source ────────────────────────────────────────────────────────────────
raw = load_sample()                                         # ← swap for real file

# ── preprocessing ─────────────────────────────────────────────────────────────
raw = bandpass(raw, l_freq=FILTERS["l_freq"], h_freq=FILTERS["h_freq"])
raw, epochs = preprocess(raw, PREPROC)

# ── annotate phases ───────────────────────────────────────────────────────────
rec_duration = raw.times[-1]
segments = extract_segments(log)
visible = [(s, e, k) for s, e, k in segments if s < rec_duration]

raw.set_annotations(mne.Annotations(
    onset       = [s for s, e, k in visible],
    duration    = [e - s for s, e, k in visible],
    description = [k for s, e, k in visible],
))

print(f"Recording : {rec_duration:.1f} s  |  Annotated segments: {len(visible)}")
for s, e, k in visible:
    print(f"  {k:<14} {s:6.1f}–{e:6.1f} s  ({e-s:.1f} s)")

# ── band power analysis ───────────────────────────────────────────────────────
print(f"\nRunning band analysis: {ANALYSIS!r}")

if ANALYSIS in ("channel", "both"):
    print("  → per channel...")
    df_ch = run_band_analysis(raw, log, method="welch")
    save_analysis(df_ch, ROOT / "data" / "processed" / "band_analysis_channels.csv")
    print(df_ch.groupby("segment_kind")[["delta","theta","alpha","beta","gamma"]].mean().round(3))
    if PLOT_BANDS:
        fig = plot_band_analysis(df_ch, group_col="channel")
        fig.savefig(ROOT / "data" / "processed" / "band_analysis_channels.png", dpi=150)
        fig.show()

if ANALYSIS in ("region", "both"):
    print("  → per region...")
    df_reg = aggregate_by_region(raw, log, regions=REGIONS, method="welch")
    save_analysis(df_reg, ROOT / "data" / "processed" / "band_analysis_regions.csv")
    if not df_reg.empty:
        print(df_reg.groupby("segment_kind")[["delta","theta","alpha","beta","gamma"]].mean().round(3))
    if PLOT_BANDS and not df_reg.empty:
        fig = plot_band_analysis(df_reg, group_col="region")
        fig.savefig(ROOT / "data" / "processed" / "band_analysis_regions.png", dpi=150)
        fig.show()

# ── peak band power ───────────────────────────────────────────────────────────
print(f"\nRunning peak analysis: {ANALYSIS!r}")

if ANALYSIS in ("channel", "both"):
    print("  → per channel...")
    df_peak = run_peak_analysis(raw, log, method="welch")
    save_analysis(df_peak, ROOT / "data" / "processed" / "peak_analysis_channels.csv")

if ANALYSIS in ("region", "both"):
    print("  → per region...")
    df_peak = aggregate_peak_by_region(raw, log, regions=REGIONS, method="welch")
    save_analysis(df_peak, ROOT / "data" / "processed" / "peak_analysis_regions.csv")

if not df_peak.empty:
    peak_power_cols = [c for c in df_peak.columns if c.endswith("_peak_power")]
    peak_freq_cols  = [c for c in df_peak.columns if c.endswith("_peak_freq")]
    with pd.option_context("display.float_format", "{:.3e}".format):
        print(df_peak.groupby("segment_kind")[peak_power_cols].mean())
    print(df_peak.groupby("segment_kind")[peak_freq_cols].mean().round(2))

# ── frontal alpha asymmetry ───────────────────────────────────────────────────
print("\nComputing FAA...")
df_faa = compute_faa(raw, log, method="welch")
save_analysis(df_faa, ROOT / "data" / "processed" / "faa_analysis.csv")
faa_cols = ["faa_eq1", "faa_eq2", "faa_eq3", "faa_eq4"]
print(df_faa.groupby("segment_kind")[faa_cols].mean().round(4))

# ── raw plot ──────────────────────────────────────────────────────────────────
if PLOT_RAW:
    raw.plot(
        duration=20,
        n_channels=60,
        title="EEG — EMDR session annotations",
        scalings="auto",
        block=True,
    )
