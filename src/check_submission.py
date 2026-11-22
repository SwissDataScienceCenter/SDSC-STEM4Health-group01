"""Deterministic checks for a leaderboard submission file.

A file passes only if the server will receive exactly the rows it expects, in a form it can score:
the template's columns, rows, keys and order; finite, positive numbers; TCD >= VCD; values inside a
generous range of what the training batches show; and a JSON payload identical in shape to the one
`Leaderboard_template.ipynb` sends.

The template and the training data are read from `data/` (never versioned). Override the paths with
the environment variables BIOREACTOR_TEMPLATE and BIOREACTOR_DATA.

Usage:
    python src/check_submission.py submissions/v2.0.0.csv [more files ...]
Exit code 0 if every file passes, 1 otherwise. Prints the SHA-256 of each file, to log with the attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(os.environ.get("BIOREACTOR_TEMPLATE", ROOT / "data" / "bioreaction_template.csv"))
TRAIN = Path(os.environ.get("BIOREACTOR_DATA", ROOT / "data" / "bioreaction_remaining.csv"))
KEYS = ["batch_id", "elapsed_time"]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
N_ROWS = 156
# A forecast outside [LOW x training minimum, HIGH x training maximum] of a variable is treated as an error.
LOW, HIGH = 0.25, 2.0


class CheckError(Exception):
    """A submission file failed a check."""


def read_text(path: Path) -> pd.DataFrame:
    """Every column as text, exactly as written in the file."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_template() -> pd.DataFrame:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"template not found at {TEMPLATE}; set BIOREACTOR_TEMPLATE")
    return read_text(TEMPLATE)


def training_ranges() -> dict[str, tuple[float, float]] | None:
    """Observed minimum and maximum of each target over the training batches, or None if the data is absent."""
    if not TRAIN.exists():
        return None
    d = pd.read_csv(TRAIN)
    d = d[~d["to_predict"].astype(bool)]
    return {v: (float(d[v].min()), float(d[v].max())) for v in TARGETS}


def check_encoding(path: Path) -> None:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CheckError("file starts with a UTF-8 byte-order mark")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise CheckError(f"file is not valid UTF-8: {e}") from e
    if "\r" in text:
        raise CheckError("file contains carriage returns (Windows line endings)")
    lines = text.rstrip("\n").split("\n")
    if any(line != line.strip() for line in lines):
        raise CheckError("a line has leading or trailing whitespace")
    if any(line == "" for line in lines):
        raise CheckError("file contains an empty line")


def check_structure(sub: pd.DataFrame, tpl: pd.DataFrame) -> None:
    if list(sub.columns) != list(tpl.columns):
        raise CheckError(f"columns {list(sub.columns)} differ from the template's {list(tpl.columns)}")
    if len(sub) != len(tpl) or len(sub) != N_ROWS:
        raise CheckError(f"{len(sub)} rows; the template has {len(tpl)} (expected {N_ROWS})")
    if sub.duplicated(KEYS).any():
        raise CheckError(f"duplicated keys: {sub[sub.duplicated(KEYS, keep=False)][KEYS].values.tolist()[:4]}")
    diff = (sub[KEYS].values != tpl[KEYS].values).any(axis=1)
    if diff.any():
        i = int(np.argmax(diff))
        raise CheckError(f"{int(diff.sum())} rows have keys different from the template, first at row {i + 2}: "
                         f"{sub[KEYS].iloc[i].tolist()} against {tpl[KEYS].iloc[i].tolist()}")


def check_values(sub: pd.DataFrame, ranges: dict[str, tuple[float, float]] | None) -> pd.DataFrame:
    num = pd.DataFrame(index=sub.index)
    for v in TARGETS:
        col = pd.to_numeric(sub[v], errors="coerce")
        bad = col.isna() | ~np.isfinite(col)
        if bad.any():
            raise CheckError(f"{v}: {int(bad.sum())} values are missing or not finite numbers, e.g. {sub[v][bad].tolist()[:3]}")
        if (col <= 0).any():
            raise CheckError(f"{v}: {int((col <= 0).sum())} values are zero or negative (MAPE needs positive forecasts)")
        num[v] = col
    if (num["tcd"] < num["vcd"]).any():
        raise CheckError(f"TCD below VCD in {int((num['tcd'] < num['vcd']).sum())} rows")
    if ranges is not None:
        for v, (lo, hi) in ranges.items():
            out = (num[v] < LOW * lo) | (num[v] > HIGH * hi)
            if out.any():
                raise CheckError(f"{v}: {int(out.sum())} values outside [{LOW * lo:.3g}, {HIGH * hi:.3g}], "
                                 f"the plausible range from the training data, e.g. {num[v][out].tolist()[:3]}")
    return num


def check_payload(path: Path, tpl: pd.DataFrame) -> None:
    """Rebuild the request body the way Leaderboard_template.ipynb does and check it."""
    df = pd.read_csv(path, dtype={"batch_id": str})
    records = df.to_dict(orient="records")
    if len(records) != N_ROWS:
        raise CheckError(f"payload has {len(records)} records")
    if any(set(r) != set(tpl.columns) for r in records):
        raise CheckError("a payload record has keys different from the template columns")
    for r in records:
        for v in TARGETS + ["elapsed_time"]:
            if not isinstance(r[v], float) or not math.isfinite(r[v]):
                raise CheckError(f"payload value {v}={r[v]!r} is not a finite float")
    json.dumps({"predictions": records}, allow_nan=False)
    t = tpl["elapsed_time"].astype(float).values
    if not np.array_equal(df["elapsed_time"].astype(float).values, t):
        raise CheckError("elapsed_time parses to numbers different from the template's")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_file(path: Path) -> dict:
    """Run every check on one file; raise CheckError on the first failure."""
    path = Path(path)
    if not path.exists():
        raise CheckError(f"{path} does not exist")
    tpl = load_template()
    check_encoding(path)
    sub = read_text(path)
    check_structure(sub, tpl)
    num = check_values(sub, training_ranges())
    check_payload(path, tpl)
    return {"file": str(path), "sha256": sha256(path), "rows": len(sub),
            "medians": {v: round(float(num[v].median()), 4) for v in TARGETS}}


def main(paths: list[str]) -> int:
    ok = True
    for p in paths:
        try:
            r = check_file(Path(p))
            print(f"PASS  {p}  sha256 {r['sha256'][:16]}  medians {r['medians']}")
        except (CheckError, FileNotFoundError) as e:
            ok = False
            print(f"FAIL  {p}  {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
