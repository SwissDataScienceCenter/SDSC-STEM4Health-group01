"""Batch-level features and the batch-level learners (ridge, PLS, PCA + ridge, kNN, boosting).
Inputs are only what a test batch provides: day-0 values, the full feed record, recorded volumes.
Each learner: f(train_ids, test_ids, columns) -> log forecasts (n_test, 13, 4)."""
import os, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
CSV = os.environ.get("BIOREACTOR_DATA", "data/bioreaction_remaining.csv")
MEAS = ["vcd", "tcd", "substrate", "byproduct"]
d = pd.read_csv(CSV); test = sorted(d.loc[d.to_predict, "batch_id"].unique(), key=lambda s: int(s[5:]))
train = sorted(set(d.batch_id) - set(test), key=lambda s: int(s[5:])); allb = train + test
folds = [list(f) for f in np.array_split(np.random.default_rng(42).permutation(train), 5)]

def features(g):
    g = g.sort_values("elapsed_time"); m0 = g[g.vcd.notna()].iloc[0]; f = {}
    for k in MEAS: f["log_" + k + "0"] = np.log(m0[k])
    f["v0"] = m0.volume
    fb = g.feed_b.dropna().values[:12]
    for i, v in enumerate(fb, 1): f[f"lfb{i:02d}"] = np.log(v)
    vol = g.volume.dropna().values
    for i in (3, 6, 9, 13): f[f"vol{i:02d}"] = vol[i]
    sub = g[g.feed_substrate.notna()]
    f["early_shot"] = float((sub.elapsed_time < 11).any()); f["early_shot_t"] = sub.elapsed_time.min() if f["early_shot"] else 11.5
    late = sub[sub.elapsed_time > 11].feed_substrate.values
    f["shot1"] = late[0] if len(late) > 0 else 0.0; f["shot2"] = late[1] if len(late) > 1 else 0.0
    c = g[g.feed_c.notna() & (g.feed_c < 0.0013)]
    f["c_small"] = len(c); f["c_small_2_4"] = int((c.elapsed_time < 5).sum()); f["c_small_5_8"] = int(((c.elapsed_time >= 5) & (c.elapsed_time < 9)).sum())
    f["fd_tot"] = g.feed_d.sum()
    return f
X = pd.DataFrame({b: features(d[d.batch_id == b]) for b in allb}).T.astype(float)
Yl = {b: np.log(d[(d.batch_id == b) & d.vcd.notna() & (d.elapsed_time > 0)].sort_values("elapsed_time")[MEAS].values) for b in train}   # (13, 4)
SETS = {"day0": [c for c in X if c.startswith("log_")] + ["v0"],
        "day0+feedB": [c for c in X if c.startswith("log_") or c.startswith("lfb")] + ["v0"],
        "all": list(X.columns)}

def std(tr, te, cols):
    mu, sd = X.loc[tr, cols].mean(), X.loc[tr, cols].std() + 1e-9
    return ((X.loc[tr, cols] - mu) / sd).values, ((X.loc[te, cols] - mu) / sd).values

def m_ridge(tr, te, cols):
    A, B_ = std(tr, te, cols); Y = np.stack([Yl[b] for b in tr]).reshape(len(tr), -1)
    return np.column_stack([RidgeCV(alphas=np.logspace(-1, 4, 16)).fit(A, Y[:, j]).predict(B_) for j in range(52)]).reshape(len(te), 13, 4)
def m_pls(k):
    def f(tr, te, cols):
        A, B_ = std(tr, te, cols); Y = np.stack([Yl[b] for b in tr]).reshape(len(tr), -1)
        return PLSRegression(k, scale=True).fit(A, Y).predict(B_).reshape(len(te), 13, 4)
    return f
def m_pca(k):
    def f(tr, te, cols):
        A, B_ = std(tr, te, cols); Y = np.stack([Yl[b] for b in tr]).reshape(len(tr), -1); ym = Y.mean(0); ys = Y.std(0) + 1e-9
        pc = PCA(k).fit((Y - ym) / ys); S = pc.transform((Y - ym) / ys)
        Sh = np.column_stack([RidgeCV(alphas=np.logspace(-1, 4, 16)).fit(A, S[:, j]).predict(B_) for j in range(k)])
        return (pc.inverse_transform(Sh) * ys + ym).reshape(len(te), 13, 4)
    return f
def m_knn(k, w=None):
    def f(tr, te, cols):
        A, B_ = std(tr, te, cols); Y = np.stack([Yl[b] for b in tr]); out = []
        for x in B_:
            nn = np.argsort(np.linalg.norm(A - x, axis=1))[:k]; out.append(np.median(Y[nn], 0))
        return np.array(out)
    return f
def m_gbm(tr, te, cols):
    A, B_ = X.loc[tr, cols].values, X.loc[te, cols].values; Y = np.stack([Yl[b] for b in tr]); ym = np.median(Y, 0); out = np.zeros((len(te), 13, 4))
    for j in range(4):   # one model per variable on stacked days: target = log y minus the day's median
        Xa = np.vstack([np.column_stack([A, np.full(len(tr), t)]) for t in range(13)]); ya = np.concatenate([Y[:, t, j] - ym[t, j] for t in range(13)])
        g = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=8, l2_regularization=1.0, random_state=0).fit(Xa, ya)
        for t in range(13): out[:, t, j] = ym[t, j] + g.predict(np.column_stack([B_, np.full(len(te), t)]))
    return out
MODELS = {"ridge per day": m_ridge, "PLS 2": m_pls(2), "PLS 3": m_pls(3), "PLS 5": m_pls(5), "PCA 3 + ridge": m_pca(3), "PCA 6 + ridge": m_pca(6),
          "kNN 5": m_knn(5), "kNN 9": m_knn(9), "boosting": m_gbm}
