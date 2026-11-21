"""Build the two submissions that combine ens-rb v4 with the team's median ensemble (README, step 17).

Inputs
  ../data/bioreaction_remaining.csv, ../data/bioreaction_template.csv   (not in git, see README §2)
  submission_ensemble_caps_volume_shortfall.csv                          ens-rb v4, written by model_playground.ipynb §5
  team_median_ensemble_test.json                                         team ensemble test predictions (in git)

Outputs (in this folder, not in git)
  submission_split_team_cells_ensrb_chem.csv   team ensemble for vcd/tcd, ens-rb v4 for substrate/byproduct (CV 7.71)
  submission_blend50_ensrb_team.csv            geometric mean of both for every variable, substrate caps re-applied (CV 7.82)

Run from testing/:
  python build_combined_submissions.py
To refresh the team predictions after the team results page changes, download its data.json and run:
  python build_combined_submissions.py --refresh-team /path/to/data.json
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TEAM_JSON = HERE / "team_median_ensemble_test.json"
ENSRB_CSV = HERE / "submission_ensemble_caps_volume_shortfall.csv"
STATE = ["vcd", "tcd", "substrate", "byproduct"]
CAP_DAYS = {7: "early", 8: "early", 9: "early", 11: "late", 12: "late"}   # same rule as SubstrateCaps in model_playground §3e


def refresh_team(page_json: Path) -> None:
    """Extract the median-ensemble test predictions from the team results page's data.json."""
    D = json.loads(page_json.read_text())
    preds = {b["id"]: b["ens"] for b in D["batches"] if b["role"] == "test"}
    assert len(preds) == 12 and all(len(v) == 13 for v in preds.values()), "expected 12 test batches × 13 predictions"
    TEAM_JSON.write_text(json.dumps({
        "model": "team median ensemble: median of ODE S5, 5 nearest batches and Gaussian process",
        "source": "data.json of the team results page 'Decode the Bioreactor' (batches[].ens for role == 'test')",
        "extracted": date.today().isoformat(),
        "columns": ["elapsed_time"] + STATE,
        "predictions": preds,
    }, indent=1))
    print(f"wrote {TEAM_JSON.name} from {page_json}")


def substrate_caps(raw: pd.DataFrame):
    """Controller rule for feed_substrate: thresholds learned on the 49 training batches, and for every constrained
    sample (incl. hidden test samples) whether the substrate was below the threshold. Mirrors model_playground §3e."""
    fs = raw.dropna(subset=["feed_substrate"])
    samp = raw[raw.vcd.notna() | raw.to_predict].assign(day=lambda d: d.elapsed_time.round().astype(int))
    rows = []
    for _, s in samp[samp.day.isin(list(CAP_DAYS))].iterrows():
        ev = fs[(fs.batch_id == s.batch_id) & (fs.elapsed_time > s.elapsed_time) & (fs.elapsed_time < s.elapsed_time + 0.8)]
        dose = ev.feed_substrate.iloc[0] if len(ev) else 0.0
        rows.append({"batch_id": s.batch_id, "day": s.day, "below": bool(dose > 0.08 if s.day >= 11 else dose > 0),
                     "substrate": s.substrate, "train": not s.to_predict})
    T = pd.DataFrame(rows)
    thresholds = {}
    for grp in ["early", "late"]:
        t = T[T.train & (T.day.map(CAP_DAYS) == grp)]
        thresholds[grp] = (t[t.below].substrate.max() + t[~t.below].substrate.min()) / 2
    return thresholds, T.set_index(["batch_id", "day"]).below


def finish(df: pd.DataFrame, tmpl: pd.DataFrame, name: str, caps=None) -> None:
    df = df.copy()
    n_cap = 0
    if caps:
        thresholds, below = caps
        for i, (b, d) in enumerate(zip(tmpl.batch_id, tmpl.elapsed_time.round().astype(int))):
            if d not in CAP_DAYS:
                continue
            thr, bl = thresholds[CAP_DAYS[d]], below.loc[(b, d)]
            if (bl and df.at[i, "substrate"] > thr) or (not bl and df.at[i, "substrate"] < thr):
                df.at[i, "substrate"] = thr
                n_cap += 1
    n_tcd = int((df.tcd < df.vcd).sum())
    df["tcd"] = np.maximum(df.tcd, df.vcd)
    sub = pd.concat([tmpl[["batch_id", "elapsed_time"]], df[STATE]], axis=1)
    assert len(sub) == 156 and sub[STATE].notna().all().all() and (sub[STATE] > 0).all().all() and (sub.tcd >= sub.vcd).all()
    sub.to_csv(HERE / name, index=False)
    print(f"{name}: 156 rows, all positive, TCD >= VCD ({n_tcd} raised), substrate caps applied to {n_cap} values")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-team", type=Path, metavar="DATA_JSON", help="re-extract the team predictions from the page's data.json first")
    args = ap.parse_args()
    if args.refresh_team:
        refresh_team(args.refresh_team)

    raw = pd.read_csv(DATA / "bioreaction_remaining.csv").sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)
    tmpl = pd.read_csv(DATA / "bioreaction_template.csv")
    if not ENSRB_CSV.exists():
        raise SystemExit(f"{ENSRB_CSV.name} not found: run model_playground.ipynb §5 first")
    mine = pd.read_csv(ENSRB_CSV)
    assert (mine.batch_id.values == tmpl.batch_id.values).all() and np.allclose(mine.elapsed_time, tmpl.elapsed_time), \
        "ens-rb v4 file doesn't follow the template order"

    team = json.loads(TEAM_JSON.read_text())["predictions"]
    rows = []
    for b, t in zip(tmpl.batch_id, tmpl.elapsed_time):
        r = min(team[b], key=lambda x: abs(x[0] - t))
        assert abs(r[0] - t) < 0.01, f"no team prediction near {b} t={t:.3f}"
        rows.append(r[1:])
    team_df = pd.DataFrame(rows, columns=STATE)

    finish(pd.DataFrame({"vcd": team_df.vcd, "tcd": team_df.tcd, "substrate": mine.substrate, "byproduct": mine.byproduct}),
           tmpl, "submission_split_team_cells_ensrb_chem.csv")
    finish(pd.DataFrame({v: np.sqrt(team_df[v] * mine[v]) for v in STATE}),
           tmpl, "submission_blend50_ensrb_team.csv", caps=substrate_caps(raw))


if __name__ == "__main__":
    main()
