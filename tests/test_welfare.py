"""Unit tests for the valuation layer (EV, WEVM, Atkinson). Pure-formula, no microdata."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.welfare import welfare as wf


def test_wevm_eps0_is_plain_mean():
    ev = np.array([100.0, 200.0, 300.0])
    y0 = np.array([500.0, 1500.0, 2500.0])
    assert wf.wevm(ev, y0, 0.0) == np.mean(ev)  # utilitarian benchmark


def test_wevm_rises_when_poor_gain_more():
    # poorest agent has the largest EV -> prioritarian weighting must raise the WEVM
    y0 = np.array([400.0, 1000.0, 4000.0])
    ev = np.array([900.0, 300.0, 100.0])
    vals = [wf.wevm(ev, y0, e) for e in (0.0, 1.0, 2.0)]
    assert vals[0] < vals[1] < vals[2]


def test_wevm_falls_when_rich_gain_more():
    # regressive case: richest gains most -> WEVM should fall as eps rises
    y0 = np.array([400.0, 1000.0, 4000.0])
    ev = np.array([100.0, 300.0, 900.0])
    vals = [wf.wevm(ev, y0, e) for e in (0.0, 1.0, 2.0)]
    assert vals[0] > vals[1] > vals[2]


def test_atkinson_zero_for_equal_distribution():
    assert abs(wf.atkinson(np.full(100, 1000.0), 1.0)) < 1e-9
    assert abs(wf.atkinson(np.full(100, 1000.0), 2.0)) < 1e-9


def test_atkinson_positive_for_unequal():
    y = np.array([100.0, 1000.0, 10000.0])
    assert wf.atkinson(y, 1.0) > 0
    assert wf.atkinson(y, 2.0) > wf.atkinson(y, 0.5)  # more aversion -> larger index


def test_equivalent_variation_recovers_transfer_pv():
    """Two matched panels differing only by a constant income top-up: EV ~ PV of top-up."""
    beta_y, delta, T = 0.1, 0.97, 5
    base_income, topup = 1000.0, 1000.0  # factual income doubles each tick
    rows_f, rows_cf = [], []
    for t in range(T):
        # utility = beta_y*ln(y) (+ identical non-income terms that cancel)
        rows_f.append(
            {
                "agent_id": 0,
                "tick": t,
                "utility": beta_y * np.log(base_income + topup),
                "income": base_income + topup,
                "y0": base_income,
                "region": "X",
            }
        )
        rows_cf.append(
            {
                "agent_id": 0,
                "tick": t,
                "utility": beta_y * np.log(base_income),
                "income": base_income,
            }
        )
    ev = wf.equivalent_variation(
        pd.DataFrame(rows_f), pd.DataFrame(rows_cf), beta_y=beta_y, delta=delta
    )
    pv = topup * sum(delta**t for t in range(T))
    assert abs(float(ev["EV"].iloc[0]) - pv) < 1.0  # closed form recovers PV of transfers


def test_subgroup_uses_priority_mass():
    ev_df = pd.DataFrame(
        {
            "agent_id": range(4),
            "region": ["A", "A", "B", "B"],
            "y0": [500.0, 700.0, 3000.0, 4000.0],
            "EV": [800.0, 600.0, 200.0, 100.0],
            "EV_linear": [0, 0, 0, 0],
        }
    )
    rep = wf.subgroup_decomposition(ev_df, by="region", eps=1.0)
    a = rep.set_index("region").loc["A"]
    # poor region A has > half the priority mass despite equal population share
    assert a["pop_share"] == 0.5 and a["priority_mass_share"] > 0.5
