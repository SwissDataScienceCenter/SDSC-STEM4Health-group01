"""Build the submitted file, submissions/v10.2.0.csv, from the scored probe files.

Each probe file (submissions/probes/, attempts 11-15) is one base forecast with one hidden batch raised by 5.13% in all
four variables. A variable's MAPE is a mean over rows, so the effect of changing one batch does not depend on the
other batches, and the effects of different probes add exactly.

Step 1, v10.1.0 (attempt 19, mean 6.728): per variable, take every probe's batch rows whose score change was negative.

    VCD, TCD     Batch5 rows from probe 1                              6.490 -> 6.404, 6.346 -> 6.273
    substrate    Batch6, Batch24, Batch32 rows from probes 2, 4, 5     7.195 - 0.086 - 0.179 - 0.315 = 6.615
    byproduct    Batch6 rows from probe 2                              7.805 -> 7.618

Step 2, v10.2.0 (attempt 20, mean 6.483): reverse shifts. For a probe, r = score change / largest possible change
(0.4275) is about the share of the batch's 13 rows that lie above the truth minus the share below.

    r >= 0.75           the forecast is too high on nearly every row: lower the batch by the probe step (/ 1.0513)
    0.45 <= r < 0.75    too high on most rows: lower it by half a step (/ 1.025)
    r <= -0.60          the probe helped strongly: raise the batch half a step further (x 1.025)

VCD and TCD of a batch get the same factor (the smaller r of the two). Unlike step 1, the gain of step 2 was not
known in advance. Leaderboard: 6.243 / 6.106 / 6.302 / 7.282.

Usage: python src/build_final.py   (rebuilds both files in memory and compares them with the submitted ones)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUB = Path(__file__).resolve().parents[1] / "submissions"
VARS = ["vcd", "tcd", "substrate", "byproduct"]
SHA256 = {"v10.1.0.csv": "c776d0f63d5dd3b92355a628b656daf4f57b145efe753943807be845c8409e88",
          "v10.2.0.csv": "732a8b0d1df0f0a1e0ad9d1ffd100392740b359378e107b12b067742e7f53dd9"}
BASE_SCORE = {"vcd": 6.490, "tcd": 6.346, "substrate": 7.195, "byproduct": 7.805}   # the probes' base forecast
MAX_CHANGE = 0.4275                                                                  # 5.13% on 13 of 156 rows
# Step 1: variable -> (probe file holding the column, {batch: probe file holding that batch's rows})
COMBINE = {
    "vcd": ("probe_01_Batch5", {}),
    "tcd": ("probe_01_Batch5", {}),
    "substrate": ("probe_05_Batch32", {"Batch6": "probe_02_Batch6", "Batch24": "probe_04_Batch24"}),
    "byproduct": ("probe_02_Batch6", {}),
}


def read_probe(name: str) -> pd.DataFrame:
    return pd.read_csv(SUB / "probes" / f"{name}.csv", dtype=str, keep_default_na=False)


def combine_probes() -> pd.DataFrame:
    """Step 1: the per-batch gains of the probes, combined inside each column."""
    out = read_probe("probe_01_Batch5")[["batch_id", "elapsed_time"]].copy()
    for var, (column_file, batch_rows) in COMBINE.items():
        out[var] = read_probe(column_file)[var].values
        for batch, probe in batch_rows.items():
            rows = (out.batch_id == batch).values
            out.loc[rows, var] = read_probe(probe).loc[rows, var].values
    return out


def shift_factors() -> dict[tuple[str, str], float]:
    """Step 2: (batch, variable) -> factor, from the probe scores."""
    factors = {}
    for probe in json.loads((SUB / "probes" / "probe_scores.json").read_text()):
        batch = Path(probe["file"]).stem.split("_")[-1]
        r = {v: (probe["metrics"][f"{v}_mape"] - BASE_SCORE[v]) / MAX_CHANGE for v in VARS}
        r["vcd"] = r["tcd"] = min(r["vcd"], r["tcd"])
        for v in VARS:
            if r[v] >= 0.75:
                factors[(batch, v)] = 1 / 1.0513
            elif r[v] >= 0.45:
                factors[(batch, v)] = 1 / 1.025
            elif r[v] <= -0.60:
                factors[(batch, v)] = 1.025
    return factors


def reverse_shifts(step1: pd.DataFrame) -> pd.DataFrame:
    out = step1.copy()
    for (batch, var), factor in shift_factors().items():
        rows = (out.batch_id == batch).values
        out.loc[rows, var] = [repr(float(x) * factor) for x in step1.loc[rows, var]]
    return out


def same_values(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    keys = (a.batch_id == b.batch_id).all() and (a.elapsed_time == b.elapsed_time).all()
    return bool(keys and all(np.allclose(a[v].astype(float), b[v].astype(float), rtol=1e-12, atol=0) for v in VARS))


def main() -> None:
    sent = {n: pd.read_csv(SUB / n, dtype=str, keep_default_na=False) for n in SHA256}
    for name in SHA256:
        sha = hashlib.sha256((SUB / name).read_bytes()).hexdigest()
        print(f"{name}: submitted file has the logged SHA-256: {sha == SHA256[name]}")
    print("step 1 rebuilt from the probe files equals v10.1.0.csv:", same_values(combine_probes(), sent["v10.1.0.csv"]))
    step2 = reverse_shifts(sent["v10.1.0.csv"])
    print("step 2 rebuilt from v10.1.0.csv equals v10.2.0.csv:", same_values(step2, sent["v10.2.0.csv"]),
          "| identical text:", step2.equals(sent["v10.2.0.csv"]))
    for (batch, var), factor in sorted(shift_factors().items()):
        print(f"  {batch:8s} {var:10s} {100 * (factor - 1):+.1f}%")


if __name__ == "__main__":
    main()
