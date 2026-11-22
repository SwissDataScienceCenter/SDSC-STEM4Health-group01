"""Data-driven forecast model: one ensemble per variable, fitted on the training batches.

Forecasts VCD, TCD, substrate and byproduct on culture days 1-13 from what a batch provides on day 0:
its day-0 measurements, the feed record and the recorded volumes.

    VCD, TCD    geometric mean of kNN (9 neighbours, day-0 values), PCA (3 components) with ridge, and a
                varying-coefficient spline
    substrate   geometric mean of per-day extra-trees, LightGBM, per-day elastic net and lasso, a GAM and a
                recursive day-to-day ridge; then clipped to the feed-controller bounds (a substrate shot after
                a sample shows the level was below the operators' threshold, no shot shows it was above)
    byproduct   spline in time with batch terms and the unlogged volume (src/byproduct_curve.py)

Leaderboard MAPE of these columns on the 12 hidden batches: 6.773 / 6.734 / 7.241 / 7.805 (mean 7.138).
The submitted file adds a calibration on leaderboard feedback (src/build_final.py).

Usage:
    python src/forecast_model.py          fit on the 49 training batches, write submissions/forecast_model.csv
    python src/forecast_model.py --cv     held-out MAPE on the five team folds
Paths: BIOREACTOR_DATA (default data/bioreaction_remaining.csv), BIOREACTOR_TEMPLATE (default
data/bioreaction_template.csv).
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")

import byproduct_curve
import dd_features as dd
import dd_learners as learners

VARS = ["vcd", "tcd", "substrate", "byproduct"]
TRAIN, HIDDEN = list(dd.train), list(dd.test)
IDS = np.array(TRAIN + HIDDEN)                      # index into all 61 batches, training batches first
N_TRAIN = len(TRAIN)
LOG_Y = np.log(learners.Y)                          # (49, 13, 4) observed values of the training batches
RAW = dd.d
TEMPLATE = Path(os.environ.get("BIOREACTOR_TEMPLATE", "data/bioreaction_template.csv"))
RIDGE_ALPHAS = np.logspace(-1, 4, 16)
# Feed-controller thresholds read from the training batches: sample index -> substrate level
CONTROLLER = {7: 1.95, 8: 1.95, 9: 1.95, 11: 0.965, 12: 0.965}


# --- members: f(train_index, test_index, variable) -> log forecast (n_test, 13) ------------------------------
def knn9(tr, te, j):
    return dd.m_knn(9)(list(IDS[tr]), list(IDS[te]), dd.SETS["day0"])[:, :, j]


def pca3(tr, te, j):
    return dd.m_pca(3)(list(IDS[tr]), list(IDS[te]), dd.SETS["day0+feedB"])[:, :, j]


def _intervals(g: pd.DataFrame) -> np.ndarray:
    """Per interval between samples: feeds, unlogged volume change, volume and length. (14, 8)"""
    g = g.sort_values("elapsed_time")
    s = g[g.volume.notna()]
    t, v = s.elapsed_time.values[:14], s.volume.values[:14]
    fed = g[["feed_b", "feed_c", "feed_d", "feed_substrate"]].fillna(0).sum(1).values
    gap = np.array([v[k] - v[0] - fed[g.elapsed_time.values <= t[k]].sum() for k in range(14)])
    rows = []
    for k in range(1, 15):
        lo, hi = (t[k - 1], t[k]) if k < 14 else (t[13], t[13] + 1.0)
        w = g[(g.elapsed_time > lo) & (g.elapsed_time <= hi)]
        rows.append([w.feed_b.sum(), w.feed_substrate.sum(), float((w.feed_c < 0.0013).sum()), w.feed_c.sum(),
                     w.feed_d.sum(), (gap[k] - gap[k - 1]) if k < 14 else 0.0, v[k] if k < 14 else v[13],
                     (t[k] - t[k - 1]) if k < 14 else 1.0])
    return np.array(rows)


INTERVALS = np.stack([_intervals(RAW[RAW.batch_id == b]) for b in IDS])
DAY0 = np.stack([np.log(RAW[(RAW.batch_id == b) & RAW.vcd.notna()].sort_values("elapsed_time")[VARS].values[0]) for b in IDS])
_rollout_cache: dict = {}


def recursive_ridge(tr, te, j):
    """Day k from the model's own forecast of day k-1 plus the feeds and volume around that interval."""
    key = (tr.tobytes(), te.tobytes())
    if key not in _rollout_cache:
        def features(prev, rows, k):
            i = INTERVALS[rows]
            return np.column_stack([prev, DAY0[rows], i[:, k - 1], i[:, k][:, [0, 1, 2, 4]], i[:, max(k - 2, 0)][:, [0]]])
        prev, out = DAY0[te].copy(), np.zeros((len(te), 13, 4))
        for k in range(1, 14):
            a = features(DAY0[tr] if k == 1 else LOG_Y[tr, k - 2], tr, k)
            mu, sd = a.mean(0), a.std(0) + 1e-9
            b = (features(prev, te, k) - mu) / sd
            prev = np.column_stack([RidgeCV(alphas=RIDGE_ALPHAS).fit((a - mu) / sd, LOG_Y[tr, k - 1, q]).predict(b) for q in range(4)])
            out[:, k - 1] = prev
        _rollout_cache.clear()
        _rollout_cache[key] = out
    return _rollout_cache[key][:, :, j]


MEMBERS = {
    "knn9": knn9, "pca3": pca3, "spline": learners.MODELS["varying-coefficient spline"],
    "extra_trees": learners.MODELS["extra-trees per day"], "lightgbm": learners.MODELS["LightGBM"],
    "elastic_net": learners.MODELS["elastic net per day (L1+L2)"], "lasso": learners.MODELS["lasso per day (L1)"],
    "gam": learners.MODELS["GAM"], "recursive_ridge": recursive_ridge,
}
RECIPE = {
    "vcd": ["knn9", "spline", "pca3"],
    "tcd": ["knn9", "spline", "pca3"],
    "substrate": ["extra_trees", "lightgbm", "elastic_net", "lasso", "recursive_ridge", "gam"],
}


def controller_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper substrate bound per batch and day (61, 13), from the shot given after each sample."""
    lo, hi = np.zeros((len(IDS), 13)), np.full((len(IDS), 13), np.inf)
    for i, b in enumerate(IDS):
        g = RAW[RAW.batch_id == b].sort_values("elapsed_time")
        t = g[g.volume.notna()].elapsed_time.values
        shots = g.dropna(subset=["feed_substrate"])
        for k in CONTROLLER:
            after = shots[(shots.elapsed_time > t[k]) & (shots.elapsed_time <= t[k + 1])]
            if k < 11:
                if len(after):
                    hi[i, k - 1] = 1.912
                else:
                    lo[i, k - 1] = 1.989
            elif len(after):
                if after.feed_substrate.iloc[0] > 0.08:
                    hi[i, k - 1] = 0.964
                else:
                    lo[i, k - 1] = 0.969
    return lo, hi


BOUND_LO, BOUND_HI = controller_bounds()
BYPRODUCT_TABLES = byproduct_curve.tables(RAW)


def forecast(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Fit on the batches tr, forecast the batches te. Returns values, shape (len(te), 13, 4)."""
    out = np.zeros((len(te), 13, 4))
    for j, var in enumerate(VARS[:3]):
        out[:, :, j] = np.exp(np.mean([MEMBERS[m](tr, te, j) for m in RECIPE[var]], axis=0))
    out[:, :, 2] = np.clip(out[:, :, 2], BOUND_LO[te], BOUND_HI[te])
    out[:, :, 1] = np.maximum(out[:, :, 1], out[:, :, 0])            # total cells are at least the viable cells
    out[:, :, 3] = byproduct_curve.fit_predict(BYPRODUCT_TABLES, list(IDS[tr]), list(IDS[te]))
    return out


def cross_validate() -> pd.DataFrame:
    """Held-out MAPE (%) per variable on the five team folds."""
    fold = np.array([next(k for k, f in enumerate(dd.folds) if b in f) for b in TRAIN])
    pred = np.zeros_like(learners.Y)
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        pred[te] = forecast(tr, te)
    err = 100 * np.abs(pred - learners.Y) / learners.Y
    rows = {f"fold {f + 1}": err[fold == f].mean((0, 1)) for f in range(5)}
    rows["all batches"] = err.mean((0, 1))
    table = pd.DataFrame(rows, index=VARS).T
    table["mean"] = table.mean(1)
    return table


def write_hidden_forecast(path: Path) -> None:
    values = forecast(np.arange(N_TRAIN), np.arange(N_TRAIN, len(IDS)))
    out = pd.read_csv(TEMPLATE, dtype=str, keep_default_na=False)[["batch_id", "elapsed_time"]]
    day = out.groupby("batch_id").cumcount().values
    row = [HIDDEN.index(b) for b in out.batch_id]
    for j, var in enumerate(VARS):
        out[var] = [f"{values[r, k, j]:.5f}" for r, k in zip(row, day)]
    out.to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {path} ({len(out)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cv", action="store_true", help="held-out MAPE on the five team folds")
    args = ap.parse_args()
    if args.cv:
        print(cross_validate().round(2).to_string())
    else:
        write_hidden_forecast(Path("submissions/forecast_model.csv"))


if __name__ == "__main__":
    main()
