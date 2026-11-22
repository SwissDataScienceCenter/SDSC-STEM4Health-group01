"""Local sensitivity and practical identifiability of the mechanistic model (ode-S v5) at its documented shared values.

Sensitivity: elasticity d log y / d log p of each output (VCD, TCD, substrate, byproduct at the 13 sample days of the
49 training batches), by central differences of +/-5% in each rate.
Identifiability: with the 11 per-batch rates free and 6% log noise per sample, the standard error of each rate from one
batch's 52 samples (Fisher information of the sensitivities), the collinearity of the rates, and the same for the
shared values fitted on all batches.
Usage: BIOREACTOR_DATA=... python src/ode_sensitivity.py   (writes results/ode_sensitivity/)"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import bioreactor as br
import ode_s5 as s5

OUT = Path("results/ode_sensitivity"); NOISE = 0.06; STEP = 0.05
VARS = br.STATE; EPS = 0.01
# Shared values of ode-S v5 (fitted on the 49 training batches) and the eleven rates that vary by batch.
SHARED = dict(r_N=1.47, Y_XN=89.4, N0=0.556, tau=0.079, k_dx=0.0069, k_l=4.00, m_S=0.102, q_o=0.588, k_g=0.254,
              Y_bo=0.689, y_bg=0.171, k_bc=0.0644, K_i=4.04, c_sub=223.0, c_B=48.1, lam=0.512, gam=0.382)
RATES = ["r_N", "k_dx", "k_l", "m_S", "q_o", "k_g", "Y_bo", "y_bg", "k_bc", "c_sub", "c_B"]


def main() -> None:
    raw = br.load_raw(Path(os.environ.get("BIOREACTOR_DATA", br.DATA_PATH))); ids = br.train_ids(raw)
    inp, obs = br.build_inputs(raw, ids), br.targets(raw, ids); xbar = float(np.exp(np.mean(np.log(inp["x0"][0]))))
    p0 = dict(SHARED) | {"f_evap": 0.0}; names = list(RATES)
    f = lambda p: np.log(s5.predict(p, dict(inp), obs, xbar)[VARS].values + EPS)      # (rows, 4)
    S = np.stack([(f(p0 | {k: p0[k] * (1 + STEP)}) - f(p0 | {k: p0[k] * (1 - STEP)})) / (np.log(1 + STEP) - np.log(1 - STEP)) for k in names], -1)  # (rows, 4, P)
    OUT.mkdir(parents=True, exist_ok=True)
    el = pd.DataFrame(np.abs(S).mean(0), index=VARS, columns=names).T; el["signed_byproduct"] = S[:, 3].mean(0); el["signed_substrate"] = S[:, 2].mean(0)
    el.round(3).to_csv(OUT / "elasticity.csv"); print("mean absolute elasticity (% change in output per 1% change in rate):"); print(el.round(2).to_string())
    b = obs.batch_id.values; se = []
    for bid in ids:
        J = S[b == bid].reshape(-1, len(names)) / NOISE; cov = np.linalg.pinv(J.T @ J); se.append(np.sqrt(np.diag(cov)))
    se = np.array(se); J = S.reshape(-1, len(names)) / NOISE; cov = np.linalg.inv(J.T @ J); c = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    sv = np.linalg.svd(J / np.linalg.norm(J, axis=0), compute_uv=False)
    idf = pd.DataFrame({"se_one_batch_pct": 100 * np.median(se, 0), "se_all_batches_pct": 100 * np.sqrt(np.diag(cov)),
                        "most_correlated_with": [names[int(np.argsort(-np.abs(c[i] - np.eye(len(names))[i]))[0])] for i in range(len(names))],
                        "correlation": [float(c[i][int(np.argsort(-np.abs(c[i] - np.eye(len(names))[i]))[0])]) for i in range(len(names))]}, index=names)
    idf.round(2).to_csv(OUT / "identifiability.csv"); print("\nstandard error of each rate (%, 6% sample noise) and its strongest correlation:"); print(idf.round(2).to_string())
    print(f"\ncollinearity index of the 11 rates (1 / smallest singular value of the normalised sensitivities): {1 / sv.min():.1f}; rates with one-batch SE below 25%: {[n for n, s in zip(names, np.median(se, 0)) if s < 0.25]}")


if __name__ == "__main__":
    main()
