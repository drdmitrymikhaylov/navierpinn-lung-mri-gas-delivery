"""Pin the two sections added 2026-10-19 -- "Which measurement would buy the
most" (section 1) and "All at once" (section 3) -- to
results/uncertainty_fixing.json and results/budget_extremes.json.  The table
rows on the page are built by the formatters below and pasted verbatim.
Run:  /usr/local/bin/python3 -m pytest tests/test_second_pass.py
"""

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAMES = {
    "rb_scale": "rubidium density factor",
    "sd_xe": "Rb–Xe spin destruction",
    "kappa_se": "binary spin exchange κ",
    "kappa_vdw": "van der Waals spin exchange κ",
    "sd_rb": "Rb–Rb spin destruction",
    "sd_n2": "Rb–N2 spin destruction",
    "sd_he": "Rb–He spin destruction",
    "broadening_ghz": "D1 pressure broadening",
    "gamma_xe": "in-cell xenon relaxation",
}


def _load(name):
    with open(os.path.join(HERE, "results", name)) as fh:
        return json.load(fh)


def _readme():
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
        return " ".join(fh.read().split())


def pct(frac):
    v = 100 * frac
    return f"{0.0 if abs(v) < 0.5 else v:.0f} %"      # no "-0 %"


def constant_rows(d):
    f, a, rho = d["one_fixed_at_nominal"], d["one_varying_alone"], d["spearman_stored"]
    order = sorted(f, key=lambda k: -f[k]["width_removed_frac"])
    return [f"| {NAMES[k]} | {rho[k]:+.2f} | {f[k]['p05']:.2f}–{f[k]['p95']:.2f} "
            f"| {pct(f[k]['width_removed_frac'])} | {pct(f[k]['var_removed_frac'])} "
            f"| {pct(a[k]['width_frac_of_full'])} |" for k in order]


def combo_rows(d):
    s = d["several_fixed"]
    lab = {"rb_scale+kappa_se": "density factor and binary κ",
           "rb_scale+kappa_se+kappa_vdw": "… and van der Waals κ",
           "rb_scale+kappa_se+kappa_vdw+sd_xe": "… and Rb–Xe spin destruction",
           "all_but_rb_scale": "every constant except the density factor"}
    rows = [f"| {lab[k]} | {s[k]['p05']:.2f}–{s[k]['p95']:.2f} | "
            f"{100 * s[k]['width_removed_frac']:.0f} % |" for k in lab]
    p, r = d["product_kappa_se_x_rb_scale_within_10pct"], d["rb_scale_within_10pct"]
    rows.append(f"| the product κ·f known to ±10 % (what a build-up series measures; "
                f"{p['n']} of {d['n_draws']} draws) | {p['p05']:.2f}–{p['p95']:.2f} | "
                f"{100 * p['width_removed_frac']:.0f} % |")
    rows.append(f"| the density factor alone known to ±10 % ({r['n']} draws) | "
                f"{r['p05']:.2f}–{r['p95']:.2f} | {100 * r['width_removed_frac']:.0f} % |")
    return rows


def floor_rows(b):
    fl = b["lung_only_floor_by_breath_hold_s"]
    sw = b["breath_hold_sweep"]
    idx = {str(s): i for i, s in enumerate(sw["breath_hold_s"])}
    return [f"| {s} s | {100 * fl[s]:.0f} % | {100 * sw['best_upstream'][idx[s]]:.0f} % "
            f"| {100 * sw['baseline'][idx[s]]:.0f} % |" for s in ("3", "5", "8", "10", "15", "20")]


def test_fixing_reproduces_and_ranks():
    d = _load("uncertainty_fixing.json")
    assert d["reproduces_exp1_samples_to"] < 1e-12 and d["n_draws"] == 400
    full = d["full"]
    assert round(full["p05"], 2) == 0.34 and round(full["p95"], 2) == 0.67
    f = d["one_fixed_at_nominal"]
    assert round(100 * f["sd_xe"]["width_removed_frac"]) == 29
    assert round(100 * f["rb_scale"]["width_removed_frac"]) == 24
    assert round(100 * f["sd_xe"]["var_removed_frac"]) == 40
    assert round(100 * f["rb_scale"]["var_removed_frac"]) == 40
    # the ranking by correlation puts rb_scale first; by what fixing it buys it is not
    rho = d["spearman_stored"]
    assert max(rho, key=lambda k: abs(rho[k])) == "rb_scale"
    assert max(f, key=lambda k: f[k]["width_removed_frac"]) == "sd_xe"
    # the five minor constants: nothing
    minor = ["sd_n2", "sd_he", "sd_rb", "broadening_ghz", "gamma_xe"]
    assert all(abs(f[k]["width_removed_frac"]) <= 0.015 for k in minor)
    assert round(100 * d["several_fixed"]["rb_scale+kappa_se+kappa_vdw+sd_xe"]["width_removed_frac"]) == 92
    assert round(100 * d["several_fixed"]["all_but_rb_scale"]["width_removed_frac"]) == 52
    assert round(100 * d["product_kappa_se_x_rb_scale_within_10pct"]["width_removed_frac"]) == 32
    assert round(100 * d["rb_scale_within_10pct"]["width_removed_frac"]) == 29
    assert round(100 * d["frac_draws_below_nominal"]) == 67
    r = _readme()
    for row in constant_rows(d) + combo_rows(d):
        assert row in r, row
    assert "removes **29 %** of the width" in r and "67 % of the draws" in r


def test_budget_extremes():
    b = _load("budget_extremes.json")
    e = _load("exp3_budget.json")
    assert abs(b["baseline_delivered"] - e["delivered"]) < 1e-12
    assert round(100 * b["all_five_together"]["delivered"]) == 64
    assert round(b["all_five_together"]["ratio"], 2) == 2.74
    assert round(b["product_of_five_ratios"], 2) == 3.52
    assert round(100 * b["all_but_breath_hold"]["delivered"]) == 45
    assert round(b["upstream_share_best_chain"], 2) == 0.96
    assert round(100 * b["two_bad_together"]["delivered"], 1) == 3.6
    assert round(b["two_bad_together"]["ratio"], 2) == 0.15
    assert round(b["two_bad_together"]["product_of_ratios"], 2) == 0.31
    fl = b["lung_only_floor_by_breath_hold_s"]
    assert round(100 * fl["8"]) == 67 and round(100 * fl["15"]) == 47
    assert round(b["lung_T1_s"], 1) == 19.8
    # the one-at-a-time ratios on the page are the stored ones
    for name, kw_ratio in b["one_at_a_time_ratio"].items():
        assert f"×{kw_ratio:.2f}" in _readme(), name
    r = _readme()
    for row in floor_rows(b):
        assert row in r, row
    assert "**64 %**" in r and "multiply to ×3.52" in r and "×0.15" in r
