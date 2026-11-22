"""Grey-box model, reproduced from GREYBOX_MODEL.md: ode-S v5 with per-batch rates set from a data-driven forecast.

Part 1, dd-three: the data-driven forecast (one model per kind of variable, all on log y)
    VCD, TCD   log y = s(t) + z_b'beta + z_b'gamma (t-7)/6        ridge, robust reweighting
    substrate  one extra-trees per sampling slot, 28 feed/volume/day-0 features
    byproduct  as VCD, with z_b extended by the volume residuals on days 2, 3, 4, 5, 13
Part 2, grey box: 11 ODE rates get a per-batch multiplier exp(theta_b), the MAP estimate that
    makes the ODE trajectory match the dd-three forecast (see greybox_step).

Shapes
    samples      14 per batch (day 0 + 13), every batch has a recorded volume at each
    rows         per fold: (n_batches * 13) scored rows, ordered by batch, then sample k = 1..13
    arrays       (B, 13, 4): batches x samples x [vcd, tcd, substrate, byproduct]

Inputs   data/bioreaction_remaining.csv
Options  --evap 1   add the unlogged-volume (evaporation) term of ode_s5.py to the shared ODE
                   (dC/dt -= f_evap F/V C); default 0 = the doc's model
         --sub pooled_ode   substrate model: one extra-trees over all days with the shared-rate ODE
                   substrate (log) as an input (substrate_cv.py: 8.47 -> 7.96, 5 x 4-fold); default per_day
         --bfeat vol   byproduct batch features: volume residuals + B0 + V0 + Feed C volume on days
                   2-4, 5-8, 9-13 only (byproduct_cv.py: 8.68 -> 8.54); default all = z + volume residuals
         --bmech   byproduct structure: base (doc), floor (consumption stops at B_eq:
                   k_bc max(B - B_eq, 0) K_i/(K_i+S); lactate <-> pyruvate is reversible),
                   death (dying cells make byproduct: + y_bd k_d Xv). The new parameter is
                   shared and also gets a per-batch multiplier.
Outputs  results/greybox_scores_evap<0|1>.csv  (model x MAPE per variable + mean, pooled over the team folds)
         results/greybox_folds_evap<0|1>.csv   (the same per fold)
         results/greybox_oof_evap<0|1>.csv     (637 held-out rows: observed + every model's prediction)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

import bioreactor as br
import ode_s5 as s5

RESULTS_DIR = Path("results")
VARS = br.STATE                       # vcd, tcd, substrate, byproduct
N_SAMPLES = 13                        # scored samples per batch (days 1-13)

# --- dd-three settings (GREYBOX_MODEL.md, "Data-driven forecast") ------------
N_KNOTS = 12                          # cubic spline in elapsed time, 12 knots
RIDGE_ALPHAS = np.logspace(-3, 2, 11) # penalty grid; chosen by an inner 4-fold batch split
SPLINE_SCALE = 100.0                  # spline columns x100: the ridge penalty then shrinks only the batch
                                      # terms, not the mean time course (penalised spline: VCD 7.06 vs 6.62)
ROBUST_PASSES, ROBUST_C = 10, 0.10    # points with |log residual| > 10% down-weighted, 10 passes
INNER_FOLDS = 4
ET_SETTINGS = dict(n_estimators=300, max_features=0.7, random_state=0, n_jobs=-1)
RESID_DAYS = [2, 3, 4, 5, 13]         # sample index k of the volume residuals for the byproduct

# --- grey-box settings (GREYBOX_MODEL.md, "Grey-box step") -------------------
# Shared ode-S v5 parameters: the doc's fitted values, used as start; refitted on each fold.
SHARED_START = dict(r_N=1.47, Y_XN=89.4, N0=0.556, tau=0.079, k_dx=0.0069, k_l=4.00, m_S=0.102,
                    q_o=0.588, k_g=0.254, Y_bo=0.689, y_bg=0.171, k_bc=0.0644, K_i=4.04,
                    c_sub=223.0, c_B=48.1, lam=0.512, gam=0.382)
SHARED_BOUND_LN = 2.0                 # each shared parameter within e^±2 of its start
GB_PARAMS = ["r_N", "k_dx", "k_l", "m_S", "q_o", "k_g", "Y_bo", "y_bg", "k_bc", "c_sub", "c_B"]
GN_ITERS, FD_STEP = 10, 0.05          # Gauss-Newton iterations; forward-difference step in theta
STEP_MAX, THETA_MAX = 0.3, 1.2        # safeguards: step per iteration, |theta|
ORACLE_PRIOR_SD = 1.0                 # weak prior for the per-batch fits to own data (Sigma)
ORACLE_NOISE = 0.10                   # ~10% log noise of a measurement in those fits
SIGMA_SHRINK = 0.5                    # Sigma = half the covariance of the oracle fits (doc; not tuned)
EPS = 0.01                            # as in ode_s5: log((pred + eps) / (obs + eps))
F_EVAP = 0.0                          # set by --evap; 1 = unlogged volume loss concentrates every state
BMECH = "base"                        # set by --bmech
SUB_MODE = "per_day"                  # set by --sub
BFEAT = "all"                         # set by --bfeat
ODE_CTX: dict = {}                    # shared ODE of the current fold (p_sh, xbar, raw), for SUB_MODE pooled_ode
ET_POOLED = dict(n_estimators=500, max_features=0.5, min_samples_leaf=2, random_state=0, n_jobs=-1)
# Start values of the extra byproduct parameter. B_eq = 0.4 g/L: below the day-12/13 median
# plateau (0.53-0.56) so the fit can move up or down; y_bd = 0.5 g per 10^6 dead cells/mL (order
# of magnitude of y_bg + Y_bo); both refitted on each fold within e^±2.
BMECH_START = {"base": {}, "floor": {"B_eq": 0.4}, "death": {"y_bd": 0.5}}


def make_rates(bmech: str):
    """ode-S v5 rates with the chosen byproduct structure (same signature as ode_s5.rates)."""
    def rates(x: np.ndarray, p: dict, t: float) -> np.ndarray:
        xv, xd, s, b, n, g = x
        s = np.maximum(s, 0.0)
        sat = s / (s + 0.05)
        mu = p["r_N"] * n * sat * (1 - np.exp(-t / p["tau"]))
        kd = p["k_dx"] * xv
        q_ov = p["q_o"] * g
        b_eff = np.maximum(b - p["B_eq"], 0.0) if bmech == "floor" else b
        q_b = p["y_bg"] * mu + p["Y_bo"] * q_ov - p["k_bc"] * b_eff * p["K_i"] / (p["K_i"] + s)
        if bmech == "death":
            q_b = q_b + p["y_bd"] * kd
        return np.array([(mu - kd) * xv, kd * xv - p["k_l"] * xd, -(p["m_S"] * sat + q_ov) * xv,
                         q_b * xv, -mu / p["Y_XN"] * xv, -p["k_g"] * xv * g])
    return rates


# --- inputs: tables built once for all 61 batches ----------------------------
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


def substrate_features(raw: pd.DataFrame, samp: pd.DataFrame) -> pd.DataFrame:
    """28 features per scored sample (batch_id, k = 1..13) for the substrate extra-trees.

    Local feeds: substrate feed and Feed B in the intervals (k-3,k-2], (k-2,k-1], (k-1,k], (k,k+1];
    Feed C count and Feed D volume in (k-1,k] and (k,k+1]. A feed given just after a sample is
    evidence of the level at that sample (the operators dosed on what they measured).
    Late shots: substrate feed after the day-11 and day-12 samples, total and count of shots.
    Early Feed B: the first two doses. Volume/timing: V_k/V_0, interval length, t. Day 0 and totals.
    """
    rows = []
    for b, g in samp.groupby("batch_id"):
        ev = raw[raw.batch_id == b]
        t, ts = ev.elapsed_time.values, g.elapsed_time.values
        n = len(ts)

        def vol(col: str, j: int) -> float:
            """Volume of feed col in interval j = (ts[j-1], ts[j]]; 0 outside the run."""
            if j < 1 or j >= n:
                return 0.0
            return float(ev.loc[(t > ts[j - 1]) & (t <= ts[j]), col].sum())

        def count(col: str, j: int) -> float:
            if j < 1 or j >= n:
                return 0.0
            return float(ev.loc[(t > ts[j - 1]) & (t <= ts[j]), col].notna().sum())

        shots = ev.feed_substrate.dropna()
        fb = ev.feed_b.dropna().values
        d0 = g.iloc[0]
        late11, late12 = vol("feed_substrate", 12), vol("feed_substrate", 13)   # after the day-11 / day-12 samples
        for k in range(1, n):
            r = {"batch_id": b, "k": k}
            for col, short in [("feed_substrate", "sub"), ("feed_b", "fb")]:
                for lag, j in [("m2", k - 2), ("m1", k - 1), ("0", k), ("p1", k + 1)]:
                    r[f"{short}_{lag}"] = vol(col, j)
            r |= {"fc_n_0": count("feed_c", k), "fc_n_p1": count("feed_c", k + 1),
                  "fd_0": vol("feed_d", k), "fd_p1": vol("feed_d", k + 1),
                  "shot_after11": late11, "shot_after12": late12,
                  "shot_total": float(shots.sum()), "shot_n": float(len(shots)),
                  "fb_1": fb[0] if len(fb) > 0 else 0.0, "fb_2": fb[1] if len(fb) > 1 else 0.0,
                  "V_rel": g.volume.iloc[k] / d0.volume, "dt": ts[k] - ts[k - 1], "t": ts[k],
                  "vcd0": d0.vcd, "S0": d0.substrate, "B0": d0.byproduct, "V0": d0.volume,
                  "tot_fb": float(ev.feed_b.sum()), "tot_fd": float(ev.feed_d.sum()),
                  "n_fc": float(ev.feed_c.notna().sum())}
            rows.append(r)
    return pd.DataFrame(rows)


def feed_c_windows(raw: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Feed C volume on days 2-4, 5-8, 9-13 per batch (61, 3)."""
    fc = raw.dropna(subset=["feed_c"]).assign(day=lambda d: np.ceil(d.elapsed_time).astype(int))
    v = fc.pivot_table(index="batch_id", columns="day", values="feed_c", aggfunc="sum").fillna(0)
    w = pd.DataFrame({f"fcv_d{lo}_{hi}": v[[c for c in v.columns if lo <= c <= hi]].sum(axis=1)
                      for lo, hi in [(2, 4), (5, 8), (9, 13)]})
    return w.reindex(index).fillna(0)


def build_tables(raw: pd.DataFrame) -> dict:
    """All input-only tables, for every batch (train and hidden)."""
    samp = sample_table(raw)
    z = batch_z(raw, samp)
    return {"samp": samp, "z": z, "vres": volume_residuals(raw, samp),
            "sfeat": substrate_features(raw, samp), "fcwin": feed_c_windows(raw, z.index)}


def rows_for(T: dict, ids: list[str]) -> pd.DataFrame:
    """Scored rows (k >= 1) of ids, ordered by ids then k, with the substrate features joined."""
    s = T["samp"][T["samp"].batch_id.isin(ids) & (T["samp"].k >= 1)].copy()
    s["order"] = s.batch_id.map({b: i for i, b in enumerate(ids)})
    s = s.sort_values(["order", "k"]).drop(columns="order").reset_index(drop=True)
    return s.merge(T["sfeat"], on=["batch_id", "k"], how="left", suffixes=("", "_f"))


# --- dd-three models ---------------------------------------------------------
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


def dd_inputs(T: dict, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time, z_b (cells) and z_b + volume residuals (byproduct) for each row."""
    Z = T["z"].loc[rows.batch_id].values
    if BFEAT == "vol":
        Zb = pd.concat([T["vres"], T["z"][["B0", "V0"]], T["fcwin"]], axis=1).loc[rows.batch_id].values
    else:
        Zb = np.hstack([Z, T["vres"].loc[rows.batch_id].values])
    return rows.elapsed_time.values, Z, Zb


def substrate_X(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[[c for c in rows.columns if c in SUB_COLS]]


def ode_log_substrate(T: dict, ids: list[str]) -> np.ndarray:
    """log substrate of the fold's shared-rate ODE at the scored rows of ids."""
    p_sh, xbar, raw = ODE_CTX["p_sh"], ODE_CTX["xbar"], ODE_CTX["raw"]
    rows = rows_for(T, ids)
    a = ode_out(p_sh, np.zeros((len(ids), len(GB_PARAMS))), br.build_inputs(raw, ids), xbar,
                rows[["batch_id", "elapsed_time"]])
    return np.log(np.maximum(a[:, :, 2].ravel(), 1e-6))


def pooled_substrate_X(T: dict, rows: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Local-feed features + k + z_b + volume residuals + ODE log substrate, all days together."""
    batch = pd.concat([T["z"], T["vres"]], axis=1).loc[rows.batch_id].reset_index(drop=True)
    X = substrate_X(rows).reset_index(drop=True)
    X = pd.concat([X, batch.drop(columns=[c for c in batch.columns if c in X.columns])], axis=1)
    return X.assign(k=rows.k.values, log_S_ode=ode_log_substrate(T, ids))


def dd_fit(T: dict, ids: list[str]) -> dict:
    """Fit the three dd-three models on the training batches ids."""
    rows = rows_for(T, ids)
    t, Z, Zb = dd_inputs(T, rows)
    g = rows.batch_id.values
    m = {"vcd": CurveModel().fit(t, Z, rows.vcd.values, g),
         "tcd": CurveModel().fit(t, Z, rows.tcd.values, g),
         "byproduct": CurveModel().fit(t, Zb, rows.byproduct.values, g)}
    if SUB_MODE == "pooled_ode":
        m["substrate"] = ExtraTreesRegressor(**ET_POOLED).fit(pooled_substrate_X(T, rows, ids), np.log(rows.substrate))
        return m
    X = substrate_X(rows)
    m["substrate"] = {k: ExtraTreesRegressor(**ET_SETTINGS).fit(X[rows.k == k], np.log(rows.substrate[rows.k == k]))
                      for k in range(1, N_SAMPLES + 1)}
    return m


def dd_predict(m: dict, T: dict, ids: list[str]) -> pd.DataFrame:
    """dd-three forecast for ids at their scored rows (columns VARS), TCD raised to VCD."""
    rows = rows_for(T, ids)
    t, Z, Zb = dd_inputs(T, rows)
    out = pd.DataFrame({"vcd": m["vcd"].predict(t, Z), "tcd": m["tcd"].predict(t, Z),
                        "byproduct": m["byproduct"].predict(t, Zb)})
    if SUB_MODE == "pooled_ode":
        out["substrate"] = np.exp(m["substrate"].predict(pooled_substrate_X(T, rows, ids)))
    else:
        X, sub = substrate_X(rows), np.empty(len(rows))
        for k, et in m["substrate"].items():
            sub[(rows.k == k).values] = np.exp(et.predict(X[rows.k == k]))
        out["substrate"] = sub
    out["tcd"] = np.maximum(out.tcd, out.vcd)
    return out[VARS]


def dd_inner_oof(T: dict, ids: list[str]) -> pd.DataFrame:
    """Out-of-fold dd-three forecast for ids from an inner 4-fold split (rows in rows_for order)."""
    parts = np.array_split(np.random.default_rng(1).permutation(ids), INNER_FOLDS)
    pred = {}
    for va in parts:
        va = list(va)
        p = dd_predict(dd_fit(T, [b for b in ids if b not in set(va)]), T, va)
        for i, b in enumerate(va):
            pred[b] = p.iloc[i * N_SAMPLES:(i + 1) * N_SAMPLES].values
    return pd.DataFrame(np.vstack([pred[b] for b in ids]), columns=VARS)


SUB_COLS: set[str] = set()   # filled in main() from the substrate feature table


# --- grey box ----------------------------------------------------------------
def fit_shared(raw: pd.DataFrame, ids: list[str]) -> tuple[dict, float]:
    """Refit the 17 shared ode-S v5 parameters on ids (evaporation term if F_EVAP = 1)."""
    inp, obs = br.build_inputs(raw, ids), br.targets(raw, ids)
    xbar = float(np.exp(np.mean(np.log(inp["x0"][0]))))
    names, lp0 = list(SHARED_START), np.log(list(SHARED_START.values()))
    y = obs[VARS].values

    def residuals(lp: np.ndarray) -> np.ndarray:
        p = dict(zip(names, np.exp(lp))) | {"f_evap": F_EVAP}
        pred = s5.predict(p, inp, obs, xbar)[VARS].values
        return np.log((np.nan_to_num(pred, nan=1e6) + EPS) / (y + EPS)).ravel()

    res = least_squares(residuals, lp0, bounds=(lp0 - SHARED_BOUND_LN, lp0 + SHARED_BOUND_LN),
                        loss="soft_l1", f_scale=0.1, x_scale="jac", max_nfev=300)
    return dict(zip(names, np.exp(res.x))) | {"f_evap": F_EVAP}, xbar


def ode_out(p_sh: dict, theta: np.ndarray, inp: dict, xbar: float, query: pd.DataFrame) -> np.ndarray:
    """ODE output (B, 13, 4) with per-batch parameters p_i exp(theta_bi); theta (B, 11)."""
    p = dict(p_sh)
    for i, name in enumerate(GB_PARAMS):
        p[name] = p_sh[name] * np.exp(theta[:, i])        # arrays (B,) broadcast in the rates
    pred = s5.predict(p, inp, query, xbar)[VARS].values
    return np.nan_to_num(pred, nan=1e6).reshape(theta.shape[0], N_SAMPLES, len(VARS))


def gauss_newton(p_sh: dict, inp: dict, xbar: float, query: pd.DataFrame, target_log: np.ndarray,
                 r_inv: np.ndarray, s_inv: np.ndarray) -> np.ndarray:
    """MAP theta per batch: min sum (log target - log f)^2 R^-1 + theta' Sigma^-1 theta.

    All batches are simulated together; the forward-difference Jacobian perturbs one
    parameter for every batch at once (batches are independent). Returns theta (B, 11).
    """
    n_b, n_p = target_log.shape[0], len(GB_PARAMS)
    w = np.sqrt(np.broadcast_to(r_inv, target_log.shape)).reshape(n_b, -1)
    theta = np.zeros((n_b, n_p))
    for _ in range(GN_ITERS):
        f0 = np.log(np.maximum(ode_out(p_sh, theta, inp, xbar, query), 1e-6))
        r0 = (target_log - f0).reshape(n_b, -1) * w
        J = np.empty((n_b, r0.shape[1], n_p))
        for i in range(n_p):
            th = theta.copy()
            th[:, i] += FD_STEP
            fi = np.log(np.maximum(ode_out(p_sh, th, inp, xbar, query), 1e-6))
            J[:, :, i] = ((fi - f0) / FD_STEP).reshape(n_b, -1) * w
        for b in range(n_b):
            A = J[b].T @ J[b] + s_inv
            g = J[b].T @ r0[b] - s_inv @ theta[b]
            step = np.clip(np.linalg.solve(A, g), -STEP_MAX, STEP_MAX)
            theta[b] = np.clip(theta[b] + step, -THETA_MAX, THETA_MAX)
    return theta


def as_array(df: pd.DataFrame) -> np.ndarray:
    """Rows in rows_for order -> (B, 13, 4)."""
    return df[VARS].values.reshape(-1, N_SAMPLES, len(VARS))


def as_frame(a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(a.reshape(-1, len(VARS)), columns=VARS)


def greybox_fold(raw: pd.DataFrame, T: dict, tr: list[str], va: list[str]) -> tuple[dict, dict]:
    """One outer fold. Returns held-out predictions per model (B, 13, 4) and diagnostics."""
    p_sh, xbar = fit_shared(raw, tr)
    ODE_CTX.update(p_sh=p_sh, xbar=xbar, raw=raw)   # note: inner dd fits reuse this fold-level ODE
    rows_tr, rows_va = rows_for(T, tr), rows_for(T, va)
    q_tr, q_va = rows_tr[["batch_id", "elapsed_time"]], rows_va[["batch_id", "elapsed_time"]]
    inp_tr, inp_va = br.build_inputs(raw, tr), br.build_inputs(raw, va)
    log_obs_tr = np.log(as_array(rows_tr))

    # Sigma: spread of the 11 multipliers fitted to each training batch's own data
    theta_or = gauss_newton(p_sh, inp_tr, xbar, q_tr, log_obs_tr, np.full((N_SAMPLES, 4), ORACLE_NOISE ** -2),
                            np.eye(len(GB_PARAMS)) / ORACLE_PRIOR_SD ** 2)
    sigma = SIGMA_SHRINK * np.cov(theta_or.T) + 1e-4 * np.eye(len(GB_PARAMS))
    s_inv = np.linalg.inv(sigma)

    # R: squared log error of dd-three per sample and variable, out of fold on the training batches
    dd_oof_tr = as_array(dd_inner_oof(T, tr))
    r_inv = 1.0 / np.mean((np.log(dd_oof_tr) - log_obs_tr) ** 2, axis=0)

    # held-out batches
    dd_va = as_array(dd_predict(dd_fit(T, tr), T, va))
    zero = np.zeros((len(va), len(GB_PARAMS)))
    theta_va = gauss_newton(p_sh, inp_va, xbar, q_va, np.log(dd_va), r_inv, s_inv)
    gb_va = ode_out(p_sh, theta_va, inp_va, xbar, q_va)
    theta_orv = gauss_newton(p_sh, inp_va, xbar, q_va, np.log(as_array(rows_va)),
                             np.full((N_SAMPLES, 4), ORACLE_NOISE ** -2), s_inv)
    preds = {"ODE, shared rates": ode_out(p_sh, zero, inp_va, xbar, q_va),
             "grey box": gb_va, "dd-three": dd_va,
             "geo mean grey box + dd-three": np.sqrt(gb_va * dd_va),
             "oracle (own measurements)": ode_out(p_sh, theta_orv, inp_va, xbar, q_va)}
    spread = dict(zip(GB_PARAMS, np.sqrt(np.diag(np.cov(theta_or.T)))))
    return preds, {"spread": spread, "theta": theta_va}


# --- main --------------------------------------------------------------------
def score_rows(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    return br.score(pred.reset_index(drop=True), truth[VARS].reset_index(drop=True))


def forecast_hidden(raw: pd.DataFrame, T: dict, out_dir: Path) -> pd.DataFrame:
    """Refit everything on the 49 training batches and forecast the 12 hidden batches.

    Same steps as one fold with tr = all training batches and va = hidden batches. Writes
    grey box, dd-three and their geometric mean (ens-grey) in the submission format: the
    template's batch_id and elapsed_time copied as text, TCD >= VCD, positive values.
    """
    tr = br.train_ids(raw)
    hidden = sorted(set(raw.batch_id) - set(tr), key=br.batch_number)
    p_sh, xbar = fit_shared(raw, tr)
    ODE_CTX.update(p_sh=p_sh, xbar=xbar, raw=raw)
    rows_tr, rows_h = rows_for(T, tr), rows_for(T, hidden)
    q_tr, q_h = rows_tr[["batch_id", "elapsed_time"]], rows_h[["batch_id", "elapsed_time"]]
    inp_tr, inp_h = br.build_inputs(raw, tr), br.build_inputs(raw, hidden)
    log_obs_tr = np.log(as_array(rows_tr))
    theta_or = gauss_newton(p_sh, inp_tr, xbar, q_tr, log_obs_tr, np.full((N_SAMPLES, 4), ORACLE_NOISE ** -2),
                            np.eye(len(GB_PARAMS)) / ORACLE_PRIOR_SD ** 2)
    s_inv = np.linalg.inv(SIGMA_SHRINK * np.cov(theta_or.T) + 1e-4 * np.eye(len(GB_PARAMS)))
    r_inv = 1.0 / np.mean((np.log(as_array(dd_inner_oof(T, tr))) - log_obs_tr) ** 2, axis=0)
    dd_h = as_array(dd_predict(dd_fit(T, tr), T, hidden))
    theta_h = gauss_newton(p_sh, inp_h, xbar, q_h, np.log(dd_h), r_inv, s_inv)
    gb_h = ode_out(p_sh, theta_h, inp_h, xbar, q_h)
    # template rows as text, in the template's order
    tmpl = pd.read_csv(br.DATA_PATH, dtype=str)
    tmpl = tmpl[tmpl.to_predict.str.lower() == "true"][["batch_id", "elapsed_time"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts = {}
    for name, a in [("greybox", gb_h), ("dd-three", dd_h), ("ens-grey", np.sqrt(gb_h * dd_h))]:
        f = as_frame(a).assign(batch_id=rows_h.batch_id.values, k=rows_h.k.values)
        f["tcd"] = np.maximum(f.tcd, f.vcd)
        key = tmpl.assign(k=tmpl.groupby("batch_id").cumcount() + 1)
        sub = key.merge(f, on=["batch_id", "k"], how="left")[["batch_id", "elapsed_time"] + VARS]
        assert len(sub) == 156 and sub[VARS].notna().all().all() and (sub[VARS] > 0).all().all()
        assert (sub.tcd >= sub.vcd).all()
        sub.to_csv(out_dir / f"{name}_forecast.csv", index=False, float_format="%.4f")
        forecasts[name] = sub
    return forecasts["ens-grey"]


def main() -> None:
    """load → tables → per team fold: shared ODE, Sigma, R, dd-three, grey box → pooled scores → save."""
    global F_EVAP, BMECH
    ap = argparse.ArgumentParser()
    ap.add_argument("--evap", type=int, choices=[0, 1], default=0)
    ap.add_argument("--bmech", choices=list(BMECH_START), default="base")
    ap.add_argument("--sub", choices=["per_day", "pooled_ode"], default="per_day")
    ap.add_argument("--bfeat", choices=["all", "vol"], default="all")
    ap.add_argument("--forecast", action="store_true", help="fit on all 49 batches, forecast the 12 hidden ones")
    args = ap.parse_args()
    global SUB_MODE, BFEAT
    F_EVAP, BMECH, SUB_MODE, BFEAT = float(args.evap), args.bmech, args.sub, args.bfeat
    SHARED_START.update(BMECH_START[BMECH])
    GB_PARAMS.extend(BMECH_START[BMECH])
    s5.rates = make_rates(BMECH)          # ode_s5.simulate looks up `rates` at call time
    tag = f"evap{int(F_EVAP)}" + ("" if BMECH == "base" else f"_{BMECH}") + ("" if SUB_MODE == "per_day" else f"_{SUB_MODE}") + ("" if BFEAT == "all" else f"_b{BFEAT}")
    raw = br.load_raw()
    T = build_tables(raw)
    SUB_COLS.update(c for c in T["sfeat"].columns if c not in ("batch_id", "k"))
    if args.forecast:
        ens = forecast_hidden(raw, T, RESULTS_DIR / ("forecast" if SUB_MODE == "per_day" else f"forecast_{SUB_MODE}"))
        print(ens.head(13).to_string(index=False))
        print("wrote greybox, dd-three, ens-grey forecasts (156 rows each)")
        return
    oof_rows = []
    pooled_pred, truths, per_fold, spreads = {}, [], [], []
    for f, (tr, va) in enumerate(br.team_folds(raw), start=1):
        t0 = time.time()
        preds, diag = greybox_fold(raw, T, tr, va)
        rows_va = rows_for(T, va)
        truth = rows_va[VARS]
        truths.append(truth)
        oof = rows_va[["batch_id", "k", "elapsed_time"] + VARS].assign(fold=f)
        for name, a in preds.items():
            oof = oof.join(as_frame(a).add_prefix(f"{name}|"))
        oof_rows.append(oof)
        spreads.append(diag["spread"])
        for name, a in preds.items():
            pooled_pred.setdefault(name, []).append(as_frame(a))
            per_fold.append({"model": name, "fold": f, **score_rows(as_frame(a), truth)})
        print(f"fold {f}: " + ", ".join(f"{r['model']} {r['mean']:.2f}" for r in per_fold[-len(preds):])
              + f"  ({time.time() - t0:.0f}s)")
    truth = pd.concat(truths)
    folds = pd.DataFrame(per_fold)
    rows = []
    print(f"\n{'pooled MAPE (%)':32s}" + "".join(f"{v:>10s}" for v in VARS + ['mean']) + "   fold SD")
    for name, parts in pooled_pred.items():
        sc = score_rows(pd.concat(parts), truth)
        sd = folds[folds.model == name]["mean"].std()
        rows.append({"model": name, **sc, "fold_sd": sd})
        print(f"{name:32s}" + "".join(f"{v:10.2f}" for v in sc.values()) + f"   {sd:.2f}")
    late = pd.concat(pooled_pred["grey box"]).reset_index(drop=True)
    tt = truth.reset_index(drop=True)
    d = pd.concat(oof_rows, ignore_index=True)
    for name in pooled_pred:
        ape = (d[f"{name}|byproduct"] - d.byproduct).abs() / d.byproduct * 100
        print(f"  byproduct days 11-13, {name}: {ape[d.k >= 11].mean():.2f}")
    sp = pd.DataFrame(spreads).median()
    print("\nbetween-batch spread of the oracle multipliers (SD, median over folds):")
    print("  " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sp.sort_values(ascending=False).items()))
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"greybox_scores_{tag}.csv", index=False)
    folds.to_csv(RESULTS_DIR / f"greybox_folds_{tag}.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(RESULTS_DIR / f"greybox_oof_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
