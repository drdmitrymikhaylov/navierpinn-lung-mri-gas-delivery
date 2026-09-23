"""Second reading of results/exp5_pinn.json and results/exp3_budget.json.

Section 5 of the README reports medians and ranges over five replicates.
This script reads the replicates one by one: how many land on which side of
the truth, what the data-fit chi2/dof of each PINN run is (the number that,
for least squares, flagged the misspecified design D), and what each stage
of the delivery chain costs per minute.  Writes results/replicate_checks.json.
Needs no PyTorch and no re-training.
"""

import json
import math

TRUE_K = 5.3e-16
TRUE_G = 1 / 1800.0  # s^-1, the Gamma used to generate the synthetic series


def main():
    pinn = json.load(open("results/exp5_pinn.json"))["inverse"]
    budget = json.load(open("results/exp3_budget.json"))
    out = {"pinn_replicates": {}, "chain_per_minute": [], "flip": {}}

    for design in ("B_series", "D_series_50W_laser"):
        out["pinn_replicates"][design] = {}
        for method in ("least_squares_const_Prb", "pinn_const_Prb", "pinn_Prb_of_T"):
            reps = pinn[design][method]["reps"]
            kr = [r["k"] / TRUE_K for r in reps]
            gr = [r["G"] / TRUE_G for r in reps]
            row = {"k_ratio": kr, "G_ratio": gr,
                   "k_below_truth": sum(k < 1 for k in kr),
                   "G_below_truth": sum(g < 1 for g in gr),
                   "k_mean": sum(kr) / len(kr), "G_mean": sum(gr) / len(gr)}
            if "chi2_dof" in reps[0]:
                row["chi2_dof"] = [r["chi2_dof"] for r in reps]
                row["phys_res"] = [r["phys_res"] for r in reps]
                row["chi2_dof_max"] = max(row["chi2_dof"])
            out["pinn_replicates"][design][method] = row

    # least-squares reduced chi2 for the same designs, from exp2
    ident = json.load(open("results/exp2_identifiability.json"))["designs"]
    out["least_squares_reduced_chi2"] = {
        "B_series": ident["B_series"]["reduced_chi2_median"],
        "D_series_50W_laser": ident["D_series_50W_laser"]["reduced_chi2_median"]}

    # --- what a minute costs at each stage of the chain
    for st in budget["chain"]:
        T1 = st["T1_s"]
        per_min = (1 - math.exp(-60 / T1)) if T1 not in (None, float("inf")) else None
        per_s = (1 - math.exp(-1 / T1)) if T1 not in (None, float("inf")) else None
        out["chain_per_minute"].append({
            "stage": st["stage"], "T1_s": T1, "duration_s": st["duration_s"],
            "loss_per_minute": per_min, "loss_per_second": per_s,
            "stage_loss": 1 - st["survival"]})

    # --- flip-angle schedules: mean signal, first line, total energy
    for name, f in budget["flip"].items():
        out["flip"][name] = {"mean_signal": f["mean_signal"],
                             "first_signal": f["signal"][0],
                             "last_over_first": f["last_over_first"],
                             "total_signal_energy": f["total_signal_energy"]}
    va, c15 = out["flip"]["variable, T1-aware"], out["flip"]["constant 15 deg"]
    out["flip"]["variable_vs_constant15"] = {
        "mean_ratio": va["mean_signal"] / c15["mean_signal"],
        "first_line_ratio": va["first_signal"] / c15["first_signal"],
        "energy_ratio": va["total_signal_energy"] / c15["total_signal_energy"]}

    with open("results/replicate_checks.json", "w") as fh:
        json.dump(out, fh, indent=1)

    for design, methods in out["pinn_replicates"].items():
        print(design)
        for m, row in methods.items():
            print(f"  {m:26s} k: " + " ".join(f"{k:.2f}" for k in row["k_ratio"]) +
                  f"  ({row['k_below_truth']}/5 below)   G: " +
                  " ".join(f"{g:.2f}" for g in row["G_ratio"]) + f"  ({row['G_below_truth']}/5 below)")
            if "chi2_dof" in row:
                print(f"  {'':26s} chi2/dof " + " ".join(f"{c:.2f}" for c in row["chi2_dof"]))
    print("least-squares reduced chi2:", out["least_squares_reduced_chi2"])
    print("\nstage                          T1 s    %/min   %/s   stage loss")
    for r in out["chain_per_minute"]:
        pm = f"{100*r['loss_per_minute']:5.2f}" if r["loss_per_minute"] else "  n/a"
        ps = f"{100*r['loss_per_second']:5.2f}" if r["loss_per_second"] else "  n/a"
        print(f"{r['stage']:30s} {r['T1_s']:7.0f} {pm} {ps}   {100*r['stage_loss']:.1f}%")
    print("\nflip:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out["flip"].items()})


if __name__ == "__main__":
    main()
