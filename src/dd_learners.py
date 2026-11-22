"""Sample-level features and learners of the data-driven ensemble.
Long format = one row per batch and sampling day; features are built for all 61 batches (inputs only).
Each learner: f(train_index, test_index, variable) -> log forecasts (n_test, 13); indices into the 61 batches,
training batches first."""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import SplineTransformer
import dd_features as dd
d, train, folds = dd.d, dd.train, dd.folds; MEAS = dd.MEAS; FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
Y = np.stack([np.exp(dd.Yl[b]) for b in train]); lY = np.log(Y); F = np.array([next(k for k, f in enumerate(folds) if b in f) for b in train])
def build(g):
    g = g.sort_values("elapsed_time"); s = g[g.volume.notna()]; t, v = s.elapsed_time.values[:14], s.volume.values[:14]; m0 = g[g.vcd.notna()].iloc[0]
    fed = g[FEEDS].fillna(0).sum(1).values; ft = g.elapsed_time.values; gap = np.array([v[k] - v[0] - fed[ft <= t[k]].sum() for k in range(14)])
    edges = list(t) + [t[13] + 1.0]
    amt = {f: np.array([g.loc[(g.elapsed_time >= edges[k]) & (g.elapsed_time < edges[k + 1]), f].sum() for k in range(14)]) for f in FEEDS}
    csm = np.array([((g.elapsed_time >= edges[k]) & (g.elapsed_time < edges[k + 1]) & (g.feed_c < 0.0013)).sum() for k in range(14)])
    sub = g[g.feed_substrate.notna()]; early = float((sub.elapsed_time < 11).any())
    batch = {"l_vcd0": np.log(m0.vcd), "l_tcd0": np.log(m0.tcd), "l_s0": np.log(m0.substrate), "l_b0": np.log(m0.byproduct), "v0": m0.volume, "viab0": np.log(m0.vcd / m0.tcd),
             **{f + "_tot": g[f].sum() for f in FEEDS}, "c_count": g.feed_c.notna().sum(), "c_small": csm.sum(), "early_shot": early, "early_t": sub.elapsed_time.min() if early else 11.5,
             "shot11": amt["feed_substrate"][11], "shot12": amt["feed_substrate"][12], "fb1": amt["feed_b"][1], "fb2": amt["feed_b"][2],
             **{f"gap{k}": gap[k] for k in (2, 3, 4, 5, 13)}}
    rows = []
    for k in range(1, 14):
        r = dict(batch); r |= {"slot": k, "t": t[k], "dt": t[k] - t[k - 1], "vratio": v[k] / m0.volume, "gap_now": gap[k], "dgap": gap[k] - gap[k - 1],
                               "c_small_int": csm[k - 1], "fd_int": amt["feed_d"][k - 1], **{f"cum_{f}": amt[f][:k].sum() for f in FEEDS}}
        for f in ("feed_substrate", "feed_b"):
            for o in (-2, -1, 0, 1): r[f"{f}_o{o}"] = amt[f][k + o - 1 + 1] if 0 <= k + o < 14 else 0.0      # o=0: the interval starting at this sample; o=-1: the one ending at it
        rows.append(r)
    return pd.DataFrame(rows), batch
LONG = []; BATCH = {}
for b in train + list(dd.test): df, BATCH[b] = build(d[d.batch_id == b]); LONG.append(df)
COLS = list(LONG[0].columns); XL = np.stack([x.values for x in LONG]).astype(float)      # (49, 13, p)
XB = pd.DataFrame(BATCH).T.astype(float).values; BCOLS = list(pd.DataFrame(BATCH).T.columns)
GAPALL = np.stack([x.gap_now.values for x in LONG]); FBALL = np.stack([x["feed_b_o-1"].values for x in LONG])
XBW = np.column_stack([XB, FBALL, GAPALL])                                  # wide: batch features + 13 Feed B doses + 13 unlogged-volume levels
def stdz(A, B): mu, sd = A.mean(0), A.std(0) + 1e-9; return (A - mu) / sd, (B - mu) / sd
def centre(tr, j): return lY[tr, :, j].mean(0)                                           # typical log trajectory of the fitting batches
lY = np.concatenate([lY, np.full((len(dd.test), 13, 4), np.nan)])   # hidden batches have no targets
# ---------------- models: each returns log predictions (n_te, 13) for variable j
def m_perday(est):
    def f(tr, te, j):
        A, B = stdz(XBW[tr], XBW[te]); return np.column_stack([est().fit(A, lY[tr, k, j]).predict(B) for k in range(13)])
    return f
def m_long(make, centred=True):
    def f(tr, te, j):
        c = centre(tr, j) if centred else np.zeros(13); A = XL[tr].reshape(-1, XL.shape[2]); B = XL[te].reshape(-1, XL.shape[2])
        m = make().fit(A, (lY[tr, :, j] - c).ravel()); return m.predict(B).reshape(-1, 13) + c
    return f
def m_lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=8, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, random_state=0)
def m_lgbm_l1():
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=8, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=0)
def m_et_slot(tr, te, j):
    return np.column_stack([ExtraTreesRegressor(300, max_features=.7, random_state=17, n_jobs=4).fit(XL[tr, k], lY[tr, k, j]).predict(XL[te, k]) for k in range(13)])
def m_gam(tr, te, j):
    from pygam import LinearGAM, s, l
    c = centre(tr, j); A, B = stdz(XL[tr].reshape(-1, XL.shape[2]), XL[te].reshape(-1, XL.shape[2])); ti = COLS.index("t")
    smooth = [COLS.index(n) for n in ("t", "gap_now", "dgap", "feed_substrate_o0", "feed_b_o0", "feed_b_o-1", "vratio", "l_vcd0", "gap2", "gap13")]
    terms = None
    for i in range(A.shape[1]):
        tm = s(i, n_splines=10 if i == ti else 5) if i in smooth else l(i); terms = tm if terms is None else terms + tm
    y = (lY[tr, :, j] - c).ravel(); best = None
    for lam in (1, 10, 100):
        g = LinearGAM(terms, lam=lam).fit(A, y); sc = g.statistics_["GCV"]
        if best is None or sc < best[0]: best = (sc, g)
    return best[1].predict(B).reshape(-1, 13) + c
def m_vcspline(tr, te, j):
    """Varying-coefficient spline: log y = B(t) a + sum_i z_i B4(t) b_i, ridge; penalty by inner CV."""
    Zt, Ze = stdz(XB[tr], XB[te]); tt, te_t = XL[tr][:, :, COLS.index("t")], XL[te][:, :, COLS.index("t")]
    def des(Z, T, sp1=None, sp2=None):
        t = T.ravel()[:, None]; sp1 = sp1 or SplineTransformer(n_knots=12, degree=3).fit(t); sp2 = sp2 or SplineTransformer(n_knots=4, degree=2).fit(t)
        b2 = sp2.transform(t); z = np.repeat(Z, 13, 0); return np.column_stack([sp1.transform(t) * 100] + [z[:, [i]] * b2 for i in range(Z.shape[1])]), sp1, sp2
    y = lY[tr, :, j].ravel(); idx = np.arange(len(tr)); inner = np.array_split(np.random.default_rng(1).permutation(idx), 4); sc = []
    for a in (3, 10, 30, 100, 300):
        e = []
        for iv in inner:
            it = np.setdiff1d(idx, iv); A, s1, s2 = des(Zt[it], tt[it]); B, _, _ = des(Zt[iv], tt[iv], s1, s2)
            e.append(np.abs(np.exp(Ridge(alpha=a, fit_intercept=False).fit(A, lY[tr][it, :, j].ravel()).predict(B) - lY[tr][iv, :, j].ravel()) - 1).mean())
        sc.append(np.mean(e))
    a = (3, 10, 30, 100, 300)[int(np.argmin(sc))]; A, s1, s2 = des(Zt, tt); B, _, _ = des(Ze, te_t, s1, s2)
    return Ridge(alpha=a, fit_intercept=False).fit(A, y).predict(B).reshape(-1, 13)
AL = np.logspace(-1, 4, 16)
MODELS = {"ridge per day (L2)": m_perday(lambda: RidgeCV(alphas=AL)), "lasso per day (L1)": m_perday(lambda: LassoCV(cv=4, n_alphas=30, max_iter=20000)),
          "elastic net per day (L1+L2)": m_perday(lambda: ElasticNetCV(cv=4, l1_ratio=0.5, n_alphas=30, max_iter=20000)),
          "varying-coefficient spline": m_vcspline, "GAM": m_gam, "extra-trees per day": m_et_slot,
          "LightGBM": m_long(m_lgbm), "LightGBM, absolute-error loss": m_long(m_lgbm_l1)}
