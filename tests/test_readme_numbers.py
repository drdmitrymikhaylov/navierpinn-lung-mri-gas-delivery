"""Pin the numbers quoted in README.md to results/*.json.  Reads only the
results files; no PyTorch, no re-training.
Run:  /usr/local/bin/python3 -m pytest tests/test_readme_numbers.py
"""

import json
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUE_K, TRUE_G = 5.3e-16, 1 / 1800.0


def _load(name):
    with open(os.path.join(HERE, "results", name)) as fh:
        return json.load(fh)


def _readme():
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
        return fh.read()


def test_operating_point_and_monte_carlo():
    s = _load("exp1_seop.json")
    assert {p: s["temperature_scan"][p]["best_T_C"] for p in ("25", "50", "100", "200")} == \
        {"25": 115.0, "50": 120.0, "100": 130.0, "200": 145.0}
    mc = s["monte_carlo"]
    assert round(mc["P_xe_nominal"], 2) == 0.52 and round(mc["P_xe_median"], 2) == 0.47
    assert round(mc["P_xe_p05"], 2) == 0.34 and round(mc["P_xe_p95"], 2) == 0.67
    assert round(mc["spearman"]["rb_scale"], 2) == 0.54
    assert max(mc["spearman"], key=lambda k: abs(mc["spearman"][k])) == "rb_scale"
    o = s["optimum_T_under_uncertainty"]
    assert (o["median"], o["p05"], o["p95"]) == (125.0, 115.0, 135.0)


def test_identifiability_table():
    d = _load("exp2_identifiability.json")["designs"]
    assert round(d["A_single_T"]["k_ratio_median"], 2) == 1.02
    assert d["A_single_T"]["fisher"]["sd_log_G"] is None
    assert round(d["D_series_50W_laser"]["k_ratio_median"], 2) == 2.23
    assert round(d["D_series_50W_laser"]["reduced_chi2_median"]) == 123
    assert round(d["D_series_100W_laser"]["k_ratio_median"], 2) == 1.58
    assert round(d["D_series_100W_laser"]["reduced_chi2_median"]) == 76
    assert round(d["B_series"]["reduced_chi2_median"], 2) == 0.98
    deg = _load("exp2_identifiability.json")["rb_scale_degeneracy"]
    assert round(deg["0.6"], 2) == 0.60 and round(deg["1.6"], 2) == 1.60


def test_delivery_chain_and_per_minute_costs():
    b = _load("exp3_budget.json")
    c = _load("replicate_checks.json")
    assert round(b["delivered"], 2) == 0.23
    cum = [round(st["cumulative"], 2) for st in b["chain"]]
    assert cum == [0.97, 0.82, 0.76, 0.50, 0.23]
    v = b["variants"]
    assert round(v["breath-hold 8 s instead of 15"] / v["baseline"], 2) == 1.42
    assert round(v["transport 2 min instead of 10"] / v["baseline"], 2) == 1.40
    # was printed x0.005 until 2026-09-23; 0.00101/0.2346 = 0.0043
    assert round(v["bag O2 2% (a leaking valve)"] / v["baseline"], 3) == 0.004
    assert round(100 * v["bag O2 2% (a leaking valve)"], 1) == 0.1
    # per-minute costs quoted in the text (added 2026-09-23)
    pm = {r["stage"]: r for r in c["chain_per_minute"]}
    assert round(100 * pm["cryogenic accumulation"]["loss_per_minute"], 1) == 0.7
    assert round(100 * pm["bag: walls + residual O2"]["loss_per_minute"], 1) == 4.0
    assert round(100 * pm["transport to scanner"]["loss_per_minute"], 1) == 4.1
    assert round(100 * pm["inhalation and breath-hold"]["loss_per_second"], 1) == 4.9
    t_bag, t_tr = pm["bag: walls + residual O2"]["T1_s"], pm["transport to scanner"]["T1_s"]
    assert abs(t_bag / t_tr - 1) < 0.04
    assert round(t_bag / 60) == 25 and round(t_tr / 60) == 24
    # 7 s of breath-hold == 8 min of transport, both ~x1.4
    assert round(math.exp(7 / pm["inhalation and breath-hold"]["T1_s"]), 2) == 1.42
    assert round(math.exp(480 / t_tr), 2) == 1.40


def test_flip_angle_schedules():
    f = _load("exp3_budget.json")["flip"]
    c = _load("replicate_checks.json")["flip"]
    assert round(f["constant 15 deg"]["last_over_first"], 2) == 0.07
    assert round(1 - f["variable, ignoring T1"]["last_over_first"], 2) == 0.38
    assert abs(f["variable, T1-aware"]["last_over_first"] - 1) < 1e-9
    # corrected 2026-09-23: the variable schedule's mean is HIGHER than constant 15 deg
    r = c["variable_vs_constant15"]
    assert f["variable, T1-aware"]["mean_signal"] > f["constant 15 deg"]["mean_signal"]
    assert round(r["mean_ratio"], 2) == 1.06
    assert round(r["first_line_ratio"], 2) == 0.37
    assert round(r["energy_ratio"], 2) == 0.73
    assert round(f["variable, T1-aware"]["mean_signal"], 3) == 0.097
    assert round(f["constant 15 deg"]["mean_signal"], 3) == 0.091
    assert round(f["variable, T1-aware"]["total_signal_energy"], 2) == 0.60
    assert round(f["constant 15 deg"]["total_signal_energy"], 2) == 0.82
    assert "6 % *higher*" in _readme()


def test_sabre_numbers():
    s = _load("exp4_sabre.json")
    assert s["matching"]["0.2"]["B_match_uT"] == 0.205 and round(s["matching"]["1.0"]["B_match_uT"], 2) == 0.20
    assert round(s["B_naive_uT"], 2) == 0.17 and round(s["B_first_order_uT"], 2) == 0.30
    assert round(s["matching"]["0.2"]["P_max"], 2) == 0.19
    assert abs(s["symmetric_couplings_P_max"]) < 1e-10
    assert s["two_substrates"]["B_match_uT"] == 0.38
    assert round(s["two_substrates"]["P_max_per_N"] / s["matching"]["0.2"]["P_max"], 2) in (0.37, 0.38)
    assert 0.14 <= min(x["B_match_uT"] for x in s["coupling_scatter"])
    assert max(x["B_match_uT"] for x in s["coupling_scatter"]) <= 0.24 + 1e-9


def test_pinn_medians_table():
    p = _load("exp5_pinn.json")["inverse"]
    B, D = p["B_series"], p["D_series_50W_laser"]
    assert round(B["least_squares_const_Prb"]["k_ratio_median"], 2) == 1.00
    assert round(B["pinn_const_Prb"]["k_ratio_median"], 2) == 0.88
    assert round(B["pinn_Prb_of_T"]["k_ratio_median"], 2) == 0.97
    assert round(B["pinn_Prb_of_T"]["G_ratio_median"], 2) == 1.19
    assert round(D["least_squares_const_Prb"]["k_ratio_median"], 2) == 2.20
    assert round(D["pinn_Prb_of_T"]["k_ratio_median"], 2) == 0.88
    assert round(D["pinn_Prb_of_T"]["G_ratio_median"], 2) == 0.90
    fw = _load("exp5_pinn.json")["forward"]
    assert fw["max_abs_err_phi"] < 1e-4 and fw["max_abs_err_P"] < 1e-4


def test_pinn_replicates_bias_and_lost_diagnostic():
    """Added 2026-09-23: kappa below truth in 5/5 replicates on design D for
    the P_Rb(T) PINN; every PINN run has data chi2/dof < 1 on both designs."""
    p = _load("exp5_pinn.json")["inverse"]
    c = _load("replicate_checks.json")["pinn_replicates"]
    D = c["D_series_50W_laser"]["pinn_Prb_of_T"]
    B = c["B_series"]["pinn_Prb_of_T"]
    # the JSON in replicate_checks is derived from exp5_pinn.json
    for design in ("B_series", "D_series_50W_laser"):
        for method in ("least_squares_const_Prb", "pinn_const_Prb", "pinn_Prb_of_T"):
            reps = p[design][method]["reps"]
            assert c[design][method]["k_ratio"] == [r["k"] / TRUE_K for r in reps]
    assert D["k_below_truth"] == 5 and D["G_below_truth"] == 4
    assert [round(k, 2) for k in D["k_ratio"]] == [0.93, 0.87, 0.90, 0.88, 0.81]
    assert round(D["k_mean"], 2) == 0.88
    assert B["k_below_truth"] == 3
    assert sum(g > 1 for g in B["G_ratio"]) == 4
    # chi2/dof: least squares 0.98 vs 123; every PINN run below 1 on both designs
    chi = []
    for design in ("B_series", "D_series_50W_laser"):
        for method in ("pinn_const_Prb", "pinn_Prb_of_T"):
            chi += c[design][method]["chi2_dof"]
    assert len(chi) == 20 and max(chi) < 1
    assert round(min(chi), 2) == 0.10 and round(max(chi), 2) == 0.45
    ls = _load("replicate_checks.json")["least_squares_reduced_chi2"]
    assert round(ls["B_series"], 2) == 0.98 and round(ls["D_series_50W_laser"]) == 123
    # physics residual: lower on the wrong model for the P_Rb(T) variant
    assert max(D["phys_res"]) < min(B["phys_res"])
    assert round(min(B["phys_res"]), 4) == 0.0008 and round(max(B["phys_res"]), 4) == 0.0013
    assert round(min(D["phys_res"]), 4) == 0.0004 and round(max(D["phys_res"]), 4) == 0.0007
    Dc = c["D_series_50W_laser"]["pinn_const_Prb"]["phys_res"]
    assert round(min(Dc), 4) == 0.0016 and round(max(Dc), 4) == 0.0044
