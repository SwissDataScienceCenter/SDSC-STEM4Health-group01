"""Tests for the submission checker and for every forecast file in submissions/.

Needs the template in data/ (or BIOREACTOR_TEMPLATE). Run from the repository root:
    python -m pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import check_submission as cs  # noqa: E402

FORECASTS = sorted(p for p in (ROOT / "submissions").glob("*.csv") if not p.name.startswith("explanation"))
pytestmark = pytest.mark.skipif(not cs.TEMPLATE.exists(), reason=f"template not found at {cs.TEMPLATE}; set BIOREACTOR_TEMPLATE")


@pytest.fixture(scope="module")
def template() -> pd.DataFrame:
    return cs.load_template()


@pytest.fixture
def good(tmp_path, template) -> Path:
    """A valid file: the template's keys with plausible positive values, TCD above VCD."""
    df = template.copy()
    rng = np.random.default_rng(0)
    vcd = rng.uniform(2, 30, len(df))
    df["vcd"] = [f"{x:.5f}" for x in vcd]
    df["tcd"] = [f"{x * 1.05:.5f}" for x in vcd]
    df["substrate"] = [f"{x:.5f}" for x in rng.uniform(0.5, 4, len(df))]
    df["byproduct"] = [f"{x:.5f}" for x in rng.uniform(0.3, 2.5, len(df))]
    p = tmp_path / "good.csv"
    df.to_csv(p, index=False)
    return p


def write(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# ---------- every real forecast file passes

def test_there_are_forecast_files():
    assert FORECASTS, "no forecast CSVs in submissions/"


@pytest.mark.parametrize("path", FORECASTS, ids=[p.name for p in FORECASTS])
def test_forecast_file_passes(path):
    r = cs.check_file(path)
    assert r["rows"] == cs.N_ROWS


@pytest.mark.parametrize("path", FORECASTS, ids=[p.name for p in FORECASTS])
def test_forecast_keys_are_template_text(path, template):
    sub = cs.read_text(path)
    assert (sub[cs.KEYS].values == template[cs.KEYS].values).all()


def test_template_itself(template):
    assert list(template.columns) == cs.KEYS + cs.TARGETS
    assert len(template) == cs.N_ROWS and template["batch_id"].nunique() == 12
    assert (template.groupby("batch_id").size() == 13).all()


# ---------- the checker accepts a valid file and rejects each kind of error

def test_valid_file_passes(good):
    cs.check_file(good)


def test_check_is_deterministic(good):
    assert cs.check_file(good) == cs.check_file(good)


@pytest.mark.parametrize("break_it, message", [
    (lambda df: df.drop(columns="byproduct"), "columns"),
    (lambda df: df[["batch_id", "elapsed_time", "tcd", "vcd", "substrate", "byproduct"]], "columns"),
    (lambda df: df.iloc[:-1], "rows"),
    (lambda df: pd.concat([df, df.iloc[[0]]]), "rows"),
    (lambda df: df.iloc[::-1].reset_index(drop=True), "keys"),
    (lambda df: df.assign(elapsed_time=df["elapsed_time"].astype(float).map(lambda x: f"{x:.12f}")), "keys"),
    (lambda df: df.assign(batch_id=df["batch_id"].str.replace("Batch", "batch")), "keys"),
    (lambda df: df.assign(vcd=["" if i == 3 else x for i, x in enumerate(df["vcd"])]), "missing"),
    (lambda df: df.assign(substrate=["nan" if i == 3 else x for i, x in enumerate(df["substrate"])]), "missing"),
    (lambda df: df.assign(byproduct=["inf" if i == 3 else x for i, x in enumerate(df["byproduct"])]), "missing"),
    (lambda df: df.assign(substrate=["abc" if i == 3 else x for i, x in enumerate(df["substrate"])]), "missing"),
    (lambda df: df.assign(byproduct=["0" if i == 3 else x for i, x in enumerate(df["byproduct"])]), "zero or negative"),
    (lambda df: df.assign(substrate=["-1.0" if i == 3 else x for i, x in enumerate(df["substrate"])]), "zero or negative"),
    (lambda df: df.assign(tcd=df["vcd"].astype(float).mul(0.9).map(lambda x: f"{x:.5f}")), "TCD below VCD"),
])
def test_errors_are_caught(good, tmp_path, break_it, message):
    df = cs.read_text(good)
    bad = write(break_it(df), tmp_path / "bad.csv")
    with pytest.raises(cs.CheckError, match=message):
        cs.check_file(bad)


@pytest.mark.skipif(not cs.TRAIN.exists(), reason="training data not found; set BIOREACTOR_DATA")
def test_implausible_values_are_caught(good, tmp_path):
    df = cs.read_text(good)
    df.loc[5, "vcd"] = "999.0"
    df.loc[5, "tcd"] = "1000.0"
    with pytest.raises(cs.CheckError, match="plausible range"):
        cs.check_file(write(df, tmp_path / "bad.csv"))


def test_byte_order_mark_is_caught(good, tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbf" + good.read_bytes())
    with pytest.raises(cs.CheckError, match="byte-order mark"):
        cs.check_file(p)


def test_windows_line_endings_are_caught(good, tmp_path):
    p = tmp_path / "crlf.csv"
    p.write_bytes(good.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(cs.CheckError, match="carriage returns"):
        cs.check_file(p)


def test_missing_file_is_caught(tmp_path):
    with pytest.raises(cs.CheckError, match="does not exist"):
        cs.check_file(tmp_path / "nope.csv")


def test_command_line_exit_codes(good, tmp_path):
    df = cs.read_text(good).iloc[:-1]
    bad = write(df, tmp_path / "bad.csv")
    assert cs.main([str(good)]) == 0
    assert cs.main([str(good), str(bad)]) == 1
