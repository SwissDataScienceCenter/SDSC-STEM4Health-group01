"""Byproduct forecast: spline in time plus batch terms, ridge with robust reweighting.

    log y = s(t) + z_b'beta + z_b'gamma (t - 7) / 6

z_b = day-0 values, initial volume, feed totals, and the unlogged volume on days 2, 3, 4, 5 and 13 (recorded volume
minus initial volume minus logged feeds). The unlogged volume is the base added for pH control, the only byproduct
signal in the inputs. The penalty is chosen by an inner 4-fold split by batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

import bioreactor as br

N_KNOTS = 12                           # cubic spline in elapsed time
RIDGE_ALPHAS = np.logspace(-3, 2, 11)  # penalty grid
SPLINE_SCALE = 100.0                   # spline columns x100: the penalty then shrinks only the batch terms
ROBUST_PASSES, ROBUST_C = 10, 0.10     # points with |log residual| > 10% are down-weighted
INNER_FOLDS = 4
RESID_DAYS = [2, 3, 4, 5, 13]          # sample index of the unlogged-volume features


def sample_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Sample rows (those with a recorded volume), k = 0..13 within each batch. (61*14) rows."""
    s = raw[raw.volume.notna()].sort_values(["batch_id", "elapsed_time"]).copy()
    s["k"] = s.groupby("batch_id").cumcount()
    return s.reset_index(drop=True)


def batch_z(raw: pd.DataFrame, samp: pd.DataFrame) -> pd.DataFrame:
    """Batch features z_b: day-0 values, V0, log viability0, total of each feed, Feed C count. (61, 11)"""
    rows = {}
    for b, g in raw.groupby("batch_id"):
        d0 = samp[(samp.batch_id == b) & (samp.k == 0)].iloc[0]
        rows[b] = {"vcd0": d0.vcd, "tcd0": d0.tcd, "S0": d0.substrate, "B0": d0.byproduct,
                   "V0": d0.volume, "logviab0": np.log(d0.vcd / d0.tcd),
                   **{f"tot_{f}": g[f].sum() for f in br.FEEDS}, "n_feed_c": int(g.feed_c.notna().sum())}
    return pd.DataFrame.from_dict(rows, orient="index")


def volume_residuals(raw: pd.DataFrame, samp: pd.DataFrame) -> pd.DataFrame:
    """residual_k = V_k - V_0 - sum of logged feeds up to sample k, for k in RESID_DAYS. (61, 5)"""
    out = {}
    for b, g in samp.groupby("batch_id"):
        ev = raw[raw.batch_id == b]
        fed = ev[br.FEEDS].fillna(0).sum(axis=1)
        out[b] = {f"vres_{k}": g.volume.iloc[k] - g.volume.iloc[0] - fed[ev.elapsed_time <= g.elapsed_time.iloc[k]].sum()
                  for k in RESID_DAYS}
    return pd.DataFrame.from_dict(out, orient="index")


def choose_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Ridge penalty with the lowest squared log error over an inner 4-fold batch split."""
    parts = np.array_split(np.random.default_rng(0).permutation(np.unique(groups)), INNER_FOLDS)
    errs = []
    for a in RIDGE_ALPHAS:
        e = 0.0
        for p in parts:
            te = np.isin(groups, p)
            e += ((Ridge(alpha=a).fit(X[~te], y[~te]).predict(X[te]) - y[te]) ** 2).sum()
        errs.append(e)
    return float(RIDGE_ALPHAS[int(np.argmin(errs))])


class CurveModel:
    """log y = s(t) + z'beta + z'gamma (t-7)/6, ridge with robust reweighting."""

    def design(self, t: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Spline basis, standardised z, and z times rescaled time (-1 on day 1, +1 on day 13)."""
        zs = (Z - self.mu) / self.sd
        tau = ((t - 7) / 6)[:, None]
        return np.hstack([SPLINE_SCALE * self.spline.transform(t[:, None]), zs, zs * tau])

    def fit(self, t: np.ndarray, Z: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "CurveModel":
        self.spline = SplineTransformer(n_knots=N_KNOTS, degree=3).fit(t[:, None])
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-12
        X, ly = self.design(t, Z), np.log(y)
        self.alpha = choose_alpha(X, ly, groups)
        w = np.ones(len(ly))
        for _ in range(ROBUST_PASSES):
            self.model = Ridge(alpha=self.alpha).fit(X, ly, sample_weight=w)
            w = np.minimum(1.0, ROBUST_C / np.maximum(np.abs(ly - self.model.predict(X)), 1e-9))
        return self

    def predict(self, t: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.exp(self.model.predict(self.design(t, Z)))



def tables(raw: pd.DataFrame) -> dict:
    """Input-only tables for every batch (training and hidden)."""
    samp = sample_table(raw)
    return {"samp": samp, "z": batch_z(raw, samp), "vres": volume_residuals(raw, samp)}


def design_rows(T: dict, ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Scored rows (k = 1..13) of ids, ordered by ids then k, and their batch features."""
    s = T["samp"][T["samp"].batch_id.isin(ids) & (T["samp"].k >= 1)].copy()
    s["order"] = s.batch_id.map({b: i for i, b in enumerate(ids)})
    s = s.sort_values(["order", "k"]).reset_index(drop=True)
    Z = np.hstack([T["z"].loc[s.batch_id].values, T["vres"].loc[s.batch_id].values])
    return s, Z


def fit_predict(T: dict, train_ids: list[str], test_ids: list[str]) -> np.ndarray:
    """Byproduct forecast for test_ids, shape (len(test_ids), 13)."""
    a, Za = design_rows(T, train_ids)
    b, Zb = design_rows(T, test_ids)
    model = CurveModel().fit(a.elapsed_time.values, Za, a.byproduct.values, a.batch_id.values)
    return model.predict(b.elapsed_time.values, Zb).reshape(len(test_ids), 13)
