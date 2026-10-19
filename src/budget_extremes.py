"""The delivery chain at its ends: every repair at once, and the floor.

Section 3 of the README scores one change at a time against the baseline
chain (23 % delivered).  This script asks what the changes are worth
together, whether the one-at-a-time ratios multiply, and what is left when
every stage but the lung is perfect.  Pure arithmetic on exp3_budget's
default_chain; no data, no PyTorch.  Writes results/budget_extremes.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_budget import default_chain, t1_oxygen     # noqa: E402

GOOD = {
    "breath-hold 8 s instead of 15": {"breath_hold_s": 8.0},
    "transport 2 min instead of 10": {"transport_min": 2.0},
    "coated glass instead of a bag": {"bag_wall_min": 240.0},
    "skip the cryogenic step": {"accumulate_min": 1e-3, "thaw_loss": 0.0},
    "bag oxygen 50 ppm instead of 500": {"bag_o2_ppm": 50.0},
    "thaw loss 5% instead of 15%": {"thaw_loss": 0.05},
}
BAD = {
    "transport 30 min": {"transport_min": 30.0},
    "5 uT/cm gradient during transport": {"transport_grad_uT_cm": 5.0},
}


def delivered(**kw):
    return float(default_chain(**kw)[-1]["cumulative"])


def merged(names, table):
    kw = {}
    for n in names:
        kw.update(table[n])
    return kw


def main():
    base = delivered()
    stored = json.load(open("results/exp3_budget.json"))
    assert abs(base - stored["delivered"]) < 1e-12
    ratios = {n: delivered(**kw) / base for n, kw in GOOD.items()}

    # all at once: the thaw row is moot once the cryogenic step is gone, so
    # "all" is the other five together
    five = [n for n in GOOD if "thaw" not in n]
    all_good = delivered(**merged(five, GOOD))
    rows_all = default_chain(**merged(five, GOOD))
    # all but the lung change
    all_but_lung = delivered(**merged([n for n in five if "breath" not in n], GOOD))
    # cryogenic step kept, with the 5 % thaw instead
    keep_cryo = delivered(**merged([n for n in GOOD if "cryogenic" not in n], GOOD))
    # the two worst non-catastrophic rows together
    all_bad = delivered(**merged(BAD, BAD))

    # the lung alone: everything upstream perfect
    t1_lung = float(t1_oxygen(0.13))
    floor = {str(s): float(np.exp(-s / t1_lung)) for s in (3, 5, 8, 10, 15, 20)}

    # breath-hold sweep, baseline chain and best chain
    bh = list(range(3, 21))
    sweep = {"breath_hold_s": bh,
             "baseline": [delivered(breath_hold_s=s) for s in bh],
             "best_upstream": [delivered(**dict(merged(
                 [n for n in GOOD if "breath" not in n], GOOD), breath_hold_s=s))
                 for s in bh]}

    # the chain's own numbers for a 5 s and a 10 s breath-hold in the best chain
    out = {
        "baseline_delivered": base,
        "one_at_a_time_ratio": ratios,
        "product_of_five_ratios": float(np.prod([ratios[n] for n in five])),
        "all_five_together": {"delivered": all_good, "ratio": all_good / base,
                              "stages": rows_all},
        "all_but_breath_hold": {"delivered": all_but_lung, "ratio": all_but_lung / base},
        "all_but_skip_cryo": {"delivered": keep_cryo, "ratio": keep_cryo / base},
        "two_bad_together": {"delivered": all_bad, "ratio": all_bad / base,
                             "product_of_ratios": float(np.prod(
                                 [delivered(**kw) / base for kw in BAD.values()]))},
        "lung_T1_s": t1_lung,
        "lung_only_floor_by_breath_hold_s": floor,
        "breath_hold_sweep": sweep,
        "upstream_share_best_chain": all_but_lung / floor["15"],
    }
    os.makedirs("results", exist_ok=True)
    with open("results/budget_extremes.json", "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"baseline {100*base:.1f} %")
    for n, r in ratios.items():
        print(f"  {n:36s} x{r:.2f}")
    print(f"product of the five ratios x{out['product_of_five_ratios']:.2f}; "
          f"all five together {100*all_good:.1f} % = x{all_good/base:.2f}")
    print(f"all but the breath-hold {100*all_but_lung:.1f} % (x{all_but_lung/base:.2f}); "
          f"all but skipping cryo {100*keep_cryo:.1f} %")
    print(f"two bad rows together {100*all_bad:.1f} % (x{all_bad/base:.2f}; product "
          f"x{out['two_bad_together']['product_of_ratios']:.2f})")
    print(f"lung T1 {t1_lung:.1f} s; lung-only floor:",
          {k: round(100*v, 1) for k, v in floor.items()})
    print("upstream survival in the best chain:",
          round(out["upstream_share_best_chain"], 3))
    for r in rows_all:
        print(f"   {r['stage']:28s} survives {r['survival']:.3f} cum {r['cumulative']:.3f}")


if __name__ == "__main__":
    main()
