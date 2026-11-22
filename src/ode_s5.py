"""ode-S v5, the mechanistic member of the team ensemble, reproduced from the team page.

Six states, (6, B) per time step:
    Xv, Xd  viable / dead cells (10^6/mL)
    S, B    substrate / byproduct (g/L)
    N       hidden growth nutrient, delivered by Feed B (units: Feed-B fraction)
    G       hidden overflow driver, starts at 1 and is used up by the cells

    mu   = r_N N S/(S+0.05) (1 - exp(-t/tau))      growth follows the Feed-B nutrient
    k_d  = k_dx Xv                                  death rises with density
    q_ov = q_o G                                    early overflow, fades as G is used up
    dXv = (mu - k_d) Xv            dXd = k_d Xv - k_l Xd
    dS  = -(m_S S/(S+0.05) + q_ov) Xv
    dB  = (y_bg mu + Y_bo q_ov - k_bc B K_i/(K_i+S)) Xv
    dN  = -mu/Y_XN Xv              dG  = -k_g Xv G
    bolus: all states diluted; S += (c_sub v_sub + c_B v_B)/V+;  N += v_B/V+

    Volume (added here, not on the page). Between two recorded volumes the
    logged feeds do not explain the change: ~0.05 L/d is missing from day 5 on
    (results/volume_balance_intervals.csv). That unlogged flow F (L/d, from the
    recorded volumes and feeds, which a test batch also provides) enters as
        dV  = F
        dC  += -f_evap (F/V) C          for every state C
    F < 0 with f_evap = 1: water evaporates, everything is concentrated.
    f_evap = 0: the loss is sampling (broth leaves at the same concentration),
    so concentrations do not change; this is the page's model.
    day-0 correction: Xv(0) = xbar (x0/xbar)^lam;  predicted VCD, TCD x (x0/xbar)^gam

The page gives 15 of the 17 fitted values. k_l and y_bg are not published, so
they are fitted here (2 parameters, least squares on all 49 batches, in-sample).

Inputs   data/bioreaction_remaining.csv
Outputs  results/ode_s5_scores.csv
         results/ode_s5_forecast.csv   (156 rows: 12 test batches x 13 days, submission columns)
         figures/ode_s5/color/ode_s5_states.{png,svg}, ode_s5_forecast.{png,svg}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import bioreactor as br

RESULTS_DIR = Path("results")
FIG_DIR = Path("figures/ode_s5")
STYLES = {"color": "default"}

# Values from the team page, section "The mechanistic member: ode-S v5".
PAGE = dict(q_o=0.59, k_g=0.25, Y_bo=0.69, m_S=0.10, k_bc=0.064, K_i=4.0,
            r_N=1.47, Y_XN=89.0, N0=0.56, k_dx=0.0069, tau=0.08,
            c_sub=223.0, c_B=48.0, lam=0.51, gam=0.38)
# Fraction of the unlogged volume flow that changes concentrations (evaporation,
# unlogged base); the rest is sampling. Scenarios compared in main(): 0 = page model.
EVAP_SCENARIOS = (0.0, 1.0)
# Not on the page: start values for the 2-parameter fit.
# k_l = 1/d: dead cells lyse in ~1 day; y_bg = 0.5 g byproduct per 10^6 cells/mL of growth (order of magnitude).
MISSING_START = dict(k_l=1.0, y_bg=0.5)
EPS, F_SCALE = 0.01, 0.1   # same loss as fit_ode.py: soft-L1 on log ratio

FIGURE = dict(batches=["Batch1", "Batch25", "Batch52"], figsize=(13, 7.5),
              title="ode-S v5 on three training batches: measured and hidden states",
              xlabel="Time (d)", xlim=(0, 13.2))
PANELS = [("Viable cells (10⁶/mL)", 0), ("Total cells (10⁶/mL)", None), ("Substrate (g/L)", 2),
          ("Byproduct (g/L)", 3), ("Nutrient N (Feed-B units)", 4), ("Overflow driver G (–)", 5)]


def rates(x: np.ndarray, p: dict, t: float) -> np.ndarray:
    """dx/dt for the six states; t is needed for the growth lag."""
    xv, xd, s, b, n, g = x
    s = np.maximum(s, 0.0)
    sat = s / (s + 0.05)
    mu = p["r_N"] * n * sat * (1 - np.exp(-t / p["tau"]))
    kd = p["k_dx"] * xv
    q_ov = p["q_o"] * g
    return np.array([
        (mu - kd) * xv,
        kd * xv - p["k_l"] * xd,
        -(p["m_S"] * sat + q_ov) * xv,
        (p["y_bg"] * mu + p["Y_bo"] * q_ov - p["k_bc"] * b * p["K_i"] / (p["K_i"] + s)) * xv,
        -mu / p["Y_XN"] * xv,
        -p["k_g"] * xv * g,
    ])


def initial_state(inp: dict, p: dict, xbar: float) -> np.ndarray:
    """Day-0 state (6, B) with the noisy-count correction on Xv."""
    x0 = inp["x0"]
    n_b = x0.shape[1]
    xv0 = xbar * (x0[0] / xbar) ** p["lam"]
    return np.vstack([xv0, x0[1], x0[2], x0[3], np.full(n_b, p["N0"]), np.ones(n_b)])


def unlogged_flow(inp: dict) -> np.ndarray:
    """Volume flow not explained by the feed log, (B, T) in L/d.

    Between consecutive recorded volumes at grid steps i0 < i1:
        F = (V_i1 - V_i0 - feeds applied at steps i0..i1-1) / ((i1 - i0) DT)
    held constant over the interval; 0 after the last record.
    """
    feed, vol = inp["feed"].sum(0), inp["vol"]
    flow = np.zeros_like(vol)
    for b in range(vol.shape[0]):
        rec = np.flatnonzero(~np.isnan(vol[b]))
        for i0, i1 in zip(rec[:-1], rec[1:]):
            resid = vol[b, i1] - vol[b, i0] - feed[b, i0:i1].sum()
            flow[b, i0:i1] = resid / ((i1 - i0) * br.DT)
    return flow


def simulate(p: dict, inp: dict, xbar: float) -> np.ndarray:
    """RK4 on the shared grid; same volume reset and bolus rule as bioreactor.simulate,
    plus N += v_B/V+ and the unlogged volume flow. Returns (6, B, T)."""
    if "flow" not in inp:
        inp["flow"] = unlogged_flow(inp)
    x = initial_state(inp, p, xbar)
    feed, vol_rec, V = inp["feed"], inp["vol"], inp["v0"].copy()
    n_t, dt = feed.shape[2], br.DT
    out = np.empty((6, x.shape[1], n_t))
    with np.errstate(all="ignore"):
        for i in range(n_t):
            t = i * dt
            V = np.where(np.isnan(vol_rec[:, i]), V, vol_rec[:, i])
            fv = feed[:, :, i]
            if fv.any():
                v_new = V + fv.sum(0)
                x = x * V / v_new
                x[2] += (p["c_sub"] * fv[0] + p["c_B"] * fv[1]) / v_new   # Feed C, D: no substrate
                x[4] += fv[1] / v_new
                V = v_new
            out[:, :, i] = x
            F = inp["flow"][:, i]
            conc = -p["f_evap"] * F / V          # 1/d; > 0 when water leaves

            def f(y: np.ndarray, tt: float) -> np.ndarray:
                return rates(y, p, tt) + conc * y

            k1 = f(x, t)
            k2 = f(x + dt / 2 * k1, t + dt / 2)
            k3 = f(x + dt / 2 * k2, t + dt / 2)
            k4 = f(x + dt * k3, t + dt)
            x = np.maximum(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
            V = V + F * dt
    return out


def predict(p: dict, inp: dict, query: pd.DataFrame, xbar: float) -> pd.DataFrame:
    """Predictions at the sample times, with the gamma correction on VCD and TCD."""
    pred = br.sim_at(simulate(p, inp, xbar), inp, query)
    corr = (inp["x0"][0] / xbar) ** p["gam"]
    c = corr[query["batch_id"].map(inp["index"]).values]
    pred["vcd"] *= c
    pred["tcd"] *= c
    return pred


def fit_missing(inp: dict, obs: pd.DataFrame, xbar: float, f_evap: float) -> dict:
    """Fit k_l and y_bg with the 15 page values fixed; returns the full parameter dict."""
    names = list(MISSING_START)
    y = obs[br.STATE].values

    def residuals(lp: np.ndarray) -> np.ndarray:
        p = PAGE | {"f_evap": f_evap} | dict(zip(names, np.exp(lp)))
        pred = predict(p, inp, obs, xbar)[br.STATE].values
        return np.log((np.nan_to_num(pred, nan=1e6) + EPS) / (y + EPS)).ravel()

    lp0 = np.log(list(MISSING_START.values()))
    res = least_squares(residuals, lp0, bounds=(lp0 - 4, lp0 + 4),
                        loss="soft_l1", f_scale=F_SCALE, x_scale="jac")
    return PAGE | {"f_evap": f_evap} | dict(zip(names, np.exp(res.x)))


def plot_states(raw: pd.DataFrame, p: dict, xbar: float) -> None:
    """Rows = batches, columns = 4 measured + 2 hidden states; dots = samples."""
    ids = FIGURE["batches"]
    inp = br.build_inputs(raw, ids)
    sim = simulate(p, inp, xbar)
    t = np.arange(sim.shape[2]) * br.DT
    obs = br.targets(raw, ids)
    for mode, style in STYLES.items():
        with plt.style.context(style):
            fig, axes = plt.subplots(len(ids), len(PANELS), figsize=FIGURE["figsize"], sharex=True)
            for r, b in enumerate(ids):
                j = inp["index"][b]
                corr = (inp["x0"][0, j] / xbar) ** p["gam"]
                curves = [sim[0, j] * corr, (sim[0, j] + sim[1, j]) * corr,
                          sim[2, j], sim[3, j], sim[4, j], sim[5, j]]
                o = obs[obs.batch_id == b]
                for c, ((label, _), y) in enumerate(zip(PANELS, curves)):
                    ax = axes[r, c]
                    ax.plot(t, y, ls="-", lw=1.2)
                    if c < 4:
                        ax.plot(o.elapsed_time, o[br.STATE[c]], ls="none", marker="o",
                                ms=2.5, color="0.15")
                    ax.set_xlim(FIGURE["xlim"])
                    ax.set_ylim(0, None)
                    ax.tick_params(labelsize=8)
                    if r == 0:
                        ax.set_title(label, fontsize=9)
                axes[r, 0].set_ylabel(b, fontsize=9)
            for ax in axes[-1]:
                ax.set_xlabel(FIGURE["xlabel"], fontsize=9)
            fig.suptitle(FIGURE["title"], x=0.01, ha="left")
            fig.tight_layout()
            out = FIG_DIR / mode
            out.mkdir(parents=True, exist_ok=True)
            for ext in ("png", "svg"):
                fig.savefig(out / f"ode_s5_states.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)


def forecast(raw: pd.DataFrame, p: dict, xbar: float) -> pd.DataFrame:
    """Open-loop forecast of the 12 test batches at their to_predict rows (156 x 6)."""
    query = raw.loc[raw["to_predict"], ["batch_id", "elapsed_time"]].reset_index(drop=True)
    ids = sorted(query["batch_id"].unique(), key=br.batch_number)
    pred = predict(p, br.build_inputs(raw, ids), query, xbar)
    return pd.concat([query, pred], axis=1)


def plot_forecast(raw: pd.DataFrame, fc: pd.DataFrame) -> None:
    """One panel per variable: 12 forecast lines over the 5-95% band of the 49 training batches."""
    train = br.targets(raw, br.train_ids(raw)).assign(day=lambda d: d.elapsed_time.round())
    band = train.groupby("day")[br.STATE].quantile([0.05, 0.95]).unstack()
    labels = ["Viable cells (10⁶/mL)", "Total cells (10⁶/mL)", "Substrate (g/L)", "Byproduct (g/L)"]
    for mode, style in STYLES.items():
        with plt.style.context(style):
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
            for ax, var, lab in zip(axes, br.STATE, labels):
                ax.fill_between(band.index, band[(var, 0.05)], band[(var, 0.95)],
                                color="0.85", lw=0, label="Training batches, 5–95%")
                for k, (b, d) in enumerate(fc.groupby("batch_id")):
                    ax.plot(d.elapsed_time, d[var], ls="-", lw=1, color="0.2" if mode == "bw" else "#2a78d6",
                            alpha=0.6, label="Test batches (12)" if k == 0 else None)
                ax.set_title(lab, fontsize=10)
                ax.set_xlabel("Time (d)")
                ax.set_ylim(0, None)
            axes[0].legend(fontsize=8, frameon=False, loc="upper left")
            fig.suptitle("ode-S v5 forecast of the 12 test batches (open loop from day 0)", x=0.01, ha="left")
            fig.tight_layout()
            for ext in ("png", "svg"):
                fig.savefig(FIG_DIR / mode / f"ode_s5_forecast.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)


def evaluate(raw: pd.DataFrame, f_evap: float) -> tuple[dict, dict, dict]:
    """Fit k_l, y_bg on all 49 batches and per team fold; returns params, in-sample and fold scores.

    Only k_l and y_bg are refitted per fold; the 15 page values were fitted by the
    team on their folds, so the fold score is not a clean out-of-fold score.
    """
    ids = br.train_ids(raw)
    inp, obs = br.build_inputs(raw, ids), br.targets(raw, ids)
    xbar = float(np.exp(np.mean(np.log(inp["x0"][0]))))   # geometric mean of day-0 VCD
    p = fit_missing(inp, obs, xbar, f_evap)
    s_in = br.score(predict(p, inp, obs, xbar), obs)
    oof = []
    for tr, va in br.team_folds(raw):
        i_tr = br.build_inputs(raw, tr)
        xb = float(np.exp(np.mean(np.log(i_tr["x0"][0]))))
        pf = fit_missing(i_tr, br.targets(raw, tr), xb, f_evap)
        q = br.targets(raw, va)
        oof.append(pd.concat([q[["batch_id", "elapsed_time"]],
                              predict(pf, br.build_inputs(raw, va), q, xb).add_suffix("_pred"),
                              q[br.STATE]], axis=1))
    oof = pd.concat(oof)
    s_oof = br.score(oof[[f"{v}_pred" for v in br.STATE]].set_axis(br.STATE, axis=1), oof[br.STATE])
    return p, s_in, s_oof


def main() -> None:
    """load → per volume scenario: fit, score → keep the better → plot → forecast test → save."""
    raw = br.load_raw()
    rows, fits = [], {}
    for f_evap in EVAP_SCENARIOS:
        p, s_in, s_oof = evaluate(raw, f_evap)
        fits[f_evap] = (p, s_oof["mean"])
        print(f"f_evap = {f_evap}: k_l = {p['k_l']:.3g} 1/d, y_bg = {p['y_bg']:.3g}")
        print("  in-sample  (%): " + "  ".join(f"{k} {v:.2f}" for k, v in s_in.items()))
        print("  team folds (%): " + "  ".join(f"{k} {v:.2f}" for k, v in s_oof.items()))
        rows += [{"f_evap": f_evap, "protocol": "in-sample", **s_in, "k_l": p["k_l"], "y_bg": p["y_bg"]},
                 {"f_evap": f_evap, "protocol": "team folds (k_l, y_bg only)", **s_oof,
                  "k_l": p["k_l"], "y_bg": p["y_bg"]}]
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ode_s5_scores.csv", index=False)

    best = min(fits, key=lambda k: fits[k][1])
    p = fits[best][0]
    print(f"plots and forecast use f_evap = {best}")
    inp = br.build_inputs(raw, br.train_ids(raw))
    xbar = float(np.exp(np.mean(np.log(inp["x0"][0]))))
    plot_states(raw, p, xbar)
    fc = forecast(raw, p, xbar)
    fc.to_csv(RESULTS_DIR / "ode_s5_forecast.csv", index=False)
    plot_forecast(raw, fc)


if __name__ == "__main__":
    main()
