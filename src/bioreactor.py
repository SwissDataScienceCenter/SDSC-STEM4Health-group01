"""Shared building blocks: data, team folds, a vectorised ODE simulator, score.

Everything the model scripts (fit_ode.py, sindy.py) need, in one place, so that
every model is scored identically.

Data
    data/bioreaction_remaining.csv, long format, ~2858 rows x 12 columns.
    49 training batches (14 daily samples) and 12 test batches (day 0 only).
    Each row is either a sample (vcd, tcd, substrate, byproduct, volume) or
    one feed bolus (feed_substrate, feed_b, feed_c, feed_d = volume added).

Units are inferred, not given: cells 10^6/mL, substrate/byproduct g/L, volume L.

Simulator
    State array of shape (n_states, n_batches); all batches integrated at once
    with RK4 on a fixed grid of step DT. At each grid step: volume reset to the
    recorded value (if a sample is there), then feed boluses (dilute
    everything, add glucose c_k * v_k / V), then one RK4 step.
"""

from __future__ import annotations  # allows "X | None" hints on Python 3.9

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

DATA_PATH = Path("data/bioreaction_remaining.csv")
STATE = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
FEED_GLUCOSE_KEY = {"feed_substrate": "c_sub", "feed_b": "c_b", "feed_c": None, "feed_d": "c_d"}
DT = 0.05          # d (72 min).
T_END = 13.1       # d, covers the last sample (~13.0)

# Team split (TEAM_HOLDOUT).
# It is identical to the folds on the team page (49 batches, seed 42, 10/10/10/10/9).
TEAM_HOLDOUT = {
    1: ["Batch8", "Batch25", "Batch28", "Batch30", "Batch36", "Batch39", "Batch42", "Batch46", "Batch58", "Batch61"],
    2: ["Batch7", "Batch10", "Batch13", "Batch21", "Batch22", "Batch33", "Batch45", "Batch47", "Batch53", "Batch55"],
    3: ["Batch4", "Batch19", "Batch20", "Batch26", "Batch29", "Batch34", "Batch38", "Batch49", "Batch50", "Batch52"],
    4: ["Batch1", "Batch9", "Batch14", "Batch15", "Batch23", "Batch27", "Batch31", "Batch37", "Batch43", "Batch51"],
    5: ["Batch2", "Batch3", "Batch11", "Batch16", "Batch17", "Batch18", "Batch41", "Batch44", "Batch59"],
}


# --- data --------------------------------------------------------------------
def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the long-format table, sorted by batch and time."""
    return pd.read_csv(path).sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)


def batch_number(batch_id: str) -> int:
    """'Batch12' -> 12, for natural sorting."""
    return int(batch_id.removeprefix("Batch"))


def train_ids(raw: pd.DataFrame) -> list[str]:
    """The 49 batches whose later samples are given (to_predict is never True)."""
    test = set(raw.loc[raw["to_predict"], "batch_id"])
    return sorted(set(raw["batch_id"]) - test, key=batch_number)


def team_folds(raw: pd.DataFrame) -> list[tuple[list[str], list[str]]]:
    """[(train_ids, validation_ids)] x 5, the team split."""
    ids = train_ids(raw)
    held = sorted(b for v in TEAM_HOLDOUT.values() for b in v)
    assert held == sorted(ids), "team folds must cover each training batch exactly once"
    return [([b for b in ids if b not in va], va) for va in TEAM_HOLDOUT.values()]


def targets(raw: pd.DataFrame, batch_ids: list[str]) -> pd.DataFrame:
    """Scored samples: every measured sample after day 0. ~13 rows per batch."""
    t = raw[raw["batch_id"].isin(batch_ids) & raw["vcd"].notna() & (raw["elapsed_time"] > 0.5)]
    return t[["batch_id", "elapsed_time"] + STATE].reset_index(drop=True)


# --- simulator ---------------------------------------------------------------
def build_inputs(raw: pd.DataFrame, batch_ids: list[str]) -> dict:
    """Everything a test batch provides, put on the time grid.

    Returns dict with x0 (4 x B: Xv, Xd, S, L at day 0), feed (4 feeds x B x T,
    bolus volume added at that grid step), vol (B x T, recorded volume or NaN),
    and index (batch_id -> column).
    """
    grid_n = int(round(T_END / DT)) + 1
    idx = {b: i for i, b in enumerate(batch_ids)}
    sub = raw[raw["batch_id"].isin(batch_ids)]
    d0 = sub[sub["vcd"].notna() & (sub["elapsed_time"] < 0.05)].set_index("batch_id").loc[batch_ids]
    x0 = np.vstack([d0.vcd, d0.tcd - d0.vcd, d0.substrate, d0.byproduct]).astype(float)

    # A feed at time t is applied at the first grid step >= t, so a sample taken
    # just before a feed (e.g. 4.96 vs 4.99) sees the state before the bolus.
    feed = np.zeros((len(FEEDS), len(batch_ids), grid_n))
    for k, col in enumerate(FEEDS):
        ev = sub.dropna(subset=[col])
        t_idx = np.minimum(np.ceil(ev["elapsed_time"].values / DT).astype(int), grid_n - 1)
        np.add.at(feed[k], (ev["batch_id"].map(idx).values, t_idx), ev[col].values)

    vol = np.full((len(batch_ids), grid_n), np.nan)
    vs = sub.dropna(subset=["volume"])
    v_idx = np.minimum(np.floor(vs["elapsed_time"].values / DT).astype(int), grid_n - 1)
    vol[vs["batch_id"].map(idx).values, v_idx] = vs["volume"].values
    return {"x0": x0, "feed": feed, "vol": vol, "index": idx, "v0": d0.volume.values.astype(float)}


def simulate(rates: Callable, p: dict, inp: dict, extra0: np.ndarray | None = None) -> np.ndarray:
    """Integrate all batches; returns array (n_states, B, T).

    rates(x, p) -> dx/dt, with x of shape (n_states, B). States beyond the
    first four (e.g. a hidden nutrient) start from extra0 and are diluted by
    feeds like the others.
    """
    x = inp["x0"].copy() if extra0 is None else np.vstack([inp["x0"], extra0])
    feed, vol_rec = inp["feed"], inp["vol"]
    V = inp["v0"].copy()
    n_t = feed.shape[2]
    out = np.empty((x.shape[0], x.shape[1], n_t))
    c_in = np.array([p[k] if k else 0.0 for k in FEED_GLUCOSE_KEY.values()])  # per feed column
    for t in range(n_t):
        rec = vol_rec[:, t]
        V = np.where(np.isnan(rec), V, rec)               # reset to recorded volume
        fv = feed[:, :, t]
        if fv.any():
            v_new = V + fv.sum(0)
            x = x * V / v_new                              # dilution
            x[2] += (c_in[:, None] * fv).sum(0) / v_new    # glucose carried by the feeds
            V = v_new
        out[:, :, t] = x
        k1 = rates(x, p)
        k2 = rates(x + DT / 2 * k1, p)
        k3 = rates(x + DT / 2 * k2, p)
        k4 = rates(x + DT * k3, p)
        x = np.maximum(x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
    return out


def sim_at(sim: np.ndarray, inp: dict, query: pd.DataFrame) -> pd.DataFrame:
    """Simulated vcd/tcd/substrate/byproduct at the query rows (batch_id, elapsed_time)."""
    b = query["batch_id"].map(inp["index"]).values
    t = np.minimum(np.floor(query["elapsed_time"].values / DT).astype(int), sim.shape[2] - 1)
    xv, xd, s, lac = (sim[k, b, t] for k in range(4))
    return pd.DataFrame({"vcd": xv, "tcd": xv + xd, "substrate": s, "byproduct": lac})


# --- score -------------------------------------------------------------------
def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Pooled MAPE (%) per variable and their mean: the competition metric."""
    s = {v: float(np.mean(np.abs(pred[v].values - truth[v].values) / np.abs(truth[v].values)) * 100)
         for v in STATE}
    s["mean"] = float(np.mean(list(s.values())))
    return s
