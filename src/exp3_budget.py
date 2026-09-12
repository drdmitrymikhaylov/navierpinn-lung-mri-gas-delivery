"""From the polarizer to the image: where the polarization goes.

Hyperpolarization is not renewable.  Whatever leaves the cell decays from
that moment on, at a rate set by whatever the xenon touches -- oxygen, walls,
field gradients, other xenon -- and every radio-frequency pulse spends part of
what is left.  The delivered polarization is a product of survival factors,
one per stage, and this file computes that product for a realistic chain,
finds the stage that costs most, and asks what a given improvement buys.

Relaxation rates used (129Xe gas, literature-typical):

  oxygen        0.388 s^-1 per amagat of O2 at room temperature -- the
                dominant loss anywhere air can get in, and the reason the
                lung is a 20-second reservoir
  xenon-xenon   the intrinsic gas-phase limit, a few hours at one amagat and
                scaling as 1/density
  gradients     D |grad B|^2 / B0^2 in the motional-narrowing limit; D of
                xenon in xenon at 1 atm about 0.06 cm^2/s, an order of
                magnitude higher in helium-diluted mixtures
  walls         container-specific; tens of minutes in a polymer bag, hours
                in coated glass
  frozen xenon  the cryogenic accumulation step: xenon ice at 77 K in a
                holding field keeps for a couple of hours, and the thaw
                itself costs a fixed fraction

The imaging part is the flip-angle schedule: with N excitations of a
non-renewable magnetization, a constant flip angle gives a signal that
decays through the acquisition, while the variable schedule
tan(a_n) = E sin(a_{n+1}), a_N = 90 deg, spends it evenly.
"""

import json
import sys

sys.path.insert(0, 'src')

import numpy as np

O2_RATE = 0.388                    # s^-1 per amagat of O2
XE_XE_T1_1AMG_H = 4.6              # h, intrinsic T1 of xenon gas at 1 amagat
D_XE_PURE = 0.06                   # cm^2/s at 1 atm
D_XE_DILUTE = 0.5                  # cm^2/s, xenon in a helium-rich mixture


def t1_oxygen(o2_amagat):
    return np.inf if o2_amagat <= 0 else 1.0 / (O2_RATE * o2_amagat)


def t1_xe_xe(xe_amagat):
    return XE_XE_T1_1AMG_H * 3600.0 / max(xe_amagat, 1e-9)


def t1_gradient(grad_T_per_cm, B0_T, D_cm2_s):
    """Motional-narrowing gradient relaxation of a gas."""
    if grad_T_per_cm <= 0:
        return np.inf
    return 1.0 / (D_cm2_s * (grad_T_per_cm / B0_T) ** 2)


def combine(*t1s):
    r = sum(0.0 if not np.isfinite(t) else 1.0 / t for t in t1s)
    return np.inf if r == 0 else 1.0 / r


def accumulation_survival(t_acc, t1_solid):
    """Average survival of xenon accumulated at a constant rate for t_acc in
    a reservoir with relaxation time t1_solid: the first atom in waits the
    longest."""
    x = t_acc / t1_solid
    return (1.0 - np.exp(-x)) / x if x > 0 else 1.0


def chain(stages):
    """stages: list of (name, duration_s, T1_s, extra_factor).  Returns the
    survival per stage and the running product."""
    rows, running = [], 1.0
    for name, dur, t1, extra in stages:
        s = (np.exp(-dur / t1) if np.isfinite(t1) else 1.0) * extra
        running *= s
        rows.append({"stage": name, "duration_s": dur, "T1_s": t1,
                     "extra": extra, "survival": s, "cumulative": running})
    return rows


def default_chain(bag_o2_ppm=500.0, transport_min=10.0, breath_hold_s=15.0,
                  accumulate_min=10.0, thaw_loss=0.15, bag_wall_min=40.0,
                  transport_field_uT=50.0, transport_grad_uT_cm=1.0):
    """A representative clinical chain, dilute-xenon polarizer with cryogenic
    accumulation.  Every number is an argument so it can be swept."""
    bag_o2 = bag_o2_ppm * 1e-6
    st = [
        ("cryogenic accumulation", accumulate_min * 60.0, 2.5 * 3600.0, 1.0),
        ("thaw", 30.0, np.inf, 1.0 - thaw_loss),
        ("bag: walls + residual O2", 60.0 * 2,
         combine(bag_wall_min * 60.0, t1_oxygen(bag_o2), t1_xe_xe(1.0)), 1.0),
        ("transport to scanner", transport_min * 60.0,
         combine(bag_wall_min * 60.0, t1_oxygen(bag_o2), t1_xe_xe(1.0),
                 t1_gradient(transport_grad_uT_cm * 1e-6, transport_field_uT * 1e-6,
                             D_XE_PURE)), 1.0),
        ("inhalation and breath-hold", breath_hold_s,
         t1_oxygen(0.13), 1.0),        # alveolar O2 about 13%
    ]
    rows = chain(st)
    # the accumulation stage is not a simple exponential: use the average
    acc = accumulation_survival(accumulate_min * 60.0, 2.5 * 3600.0)
    rows[0]["survival"] = acc
    run = 1.0
    for r in rows:
        run *= r["survival"]
        r["cumulative"] = run
    return rows


# --- flip-angle schedules ----------------------------------------------------

def constant_flip(N, alpha_deg, TR_s=0.0, T1_s=np.inf):
    E = np.exp(-TR_s / T1_s) if np.isfinite(T1_s) else 1.0
    a = np.deg2rad(alpha_deg)
    M, S = 1.0, []
    for _ in range(N):
        S.append(M * np.sin(a))
        M *= np.cos(a) * E
    return np.array(S), np.full(N, alpha_deg)


def variable_flip(N, TR_s=0.0, T1_s=np.inf):
    """Backward recursion for equal signal from every excitation."""
    E = np.exp(-TR_s / T1_s) if np.isfinite(T1_s) else 1.0
    a = np.zeros(N)
    a[-1] = np.pi / 2
    for n in range(N - 2, -1, -1):
        a[n] = np.arctan(E * np.sin(a[n + 1]))
    M, S = 1.0, []
    for n in range(N):
        S.append(M * np.sin(a[n]))
        M *= np.cos(a[n]) * E
    return np.array(S), np.rad2deg(a)


def image_snr_proxy(S):
    """Mean signal and its unevenness across k-space lines; a decaying
    signal is an apodization that blurs and a low mean is lost SNR."""
    return {"mean_signal": float(S.mean()), "last_over_first": float(S[-1] / S[0]),
            "total_signal_energy": float(np.sum(S ** 2))}


def main():
    rows = default_chain()
    delivered = rows[-1]["cumulative"]
    print("default chain")
    for r in rows:
        t1 = "inf" if not np.isfinite(r["T1_s"]) else f"{r['T1_s']:.0f} s"
        print(f"  {r['stage']:28s} {r['duration_s']:6.0f} s   T1 {t1:>8s}   "
              f"survives {r['survival']:.3f}   cumulative {r['cumulative']:.3f}")
    print(f"  -> {delivered*100:.1f}% of the polarizer output reaches the first "
          f"excitation\n")

    # what each single improvement buys, and what each single failure costs
    variants = {
        "baseline": {},
        "bag O2 50 ppm instead of 500": {"bag_o2_ppm": 50.0},
        "bag O2 2% (a leaking valve)": {"bag_o2_ppm": 20000.0},
        "transport 2 min instead of 10": {"transport_min": 2.0},
        "transport 30 min": {"transport_min": 30.0},
        "breath-hold 8 s instead of 15": {"breath_hold_s": 8.0},
        "accumulate 30 min instead of 10": {"accumulate_min": 30.0},
        "no cryogenic step (direct dispense)": {"accumulate_min": 1e-3, "thaw_loss": 0.0},
        "thaw loss 5% instead of 15%": {"thaw_loss": 0.05},
        "coated glass instead of bag (T1 wall 4 h)": {"bag_wall_min": 240.0},
        "transport in Earth field, 5 uT/cm gradient": {"transport_grad_uT_cm": 5.0},
        "transport in a 1 mT carrier": {"transport_field_uT": 1000.0},
    }
    var_out = {}
    print("single changes, delivered fraction")
    for name, kw in variants.items():
        d = default_chain(**kw)[-1]["cumulative"]
        var_out[name] = d
        print(f"  {name:44s} {d*100:5.1f}%   x{d/delivered:.2f}")

    # oxygen: the one that matters, as a curve
    o2 = np.geomspace(10, 1e5, 60)
    o2_curve = [default_chain(bag_o2_ppm=p)[-1]["cumulative"] for p in o2]
    tr = np.linspace(0, 60, 31)
    tr_curve = [default_chain(transport_min=m)[-1]["cumulative"] for m in tr]

    # flip angles
    N, TR, T1_lung = 64, 0.01 * 1, 20.0
    TR = 0.15       # s between excitations for a breath-hold 2D acquisition
    fa = {}
    for label, (S, a) in {
        "constant 5 deg": constant_flip(N, 5.0, TR, T1_lung),
        "constant 10 deg": constant_flip(N, 10.0, TR, T1_lung),
        "constant 15 deg": constant_flip(N, 15.0, TR, T1_lung),
        "variable, T1-aware": variable_flip(N, TR, T1_lung),
        "variable, ignoring T1": variable_flip(N, 0.0, np.inf),
    }.items():
        # when T1 is ignored the schedule is applied in a lung that does relax:
        if label == "variable, ignoring T1":
            E = np.exp(-TR / T1_lung)
            M, S2 = 1.0, []
            for n in range(N):
                S2.append(M * np.sin(np.deg2rad(a[n])))
                M *= np.cos(np.deg2rad(a[n])) * E
            S = np.array(S2)
        fa[label] = {"signal": S.tolist(), "alpha_deg": a.tolist(),
                     **image_snr_proxy(S)}
    print(f"\nflip-angle schedules, {N} excitations, TR {TR} s, lung T1 {T1_lung} s")
    for k, v in fa.items():
        print(f"  {k:24s} mean signal {v['mean_signal']:.3f}   last/first "
              f"{v['last_over_first']:.2f}   energy {v['total_signal_energy']:.2f}")

    out = {"chain": rows, "delivered": delivered, "variants": var_out,
           "o2_ppm": o2.tolist(), "o2_delivered": o2_curve,
           "transport_min": tr.tolist(), "transport_delivered": tr_curve,
           "flip": fa, "flip_config": {"N": N, "TR_s": TR, "T1_lung_s": T1_lung},
           "constants": {"O2_rate_per_amagat": O2_RATE,
                         "xe_xe_T1_1amagat_h": XE_XE_T1_1AMG_H,
                         "D_xe_pure": D_XE_PURE, "D_xe_dilute": D_XE_DILUTE}}
    with open("results/exp3_budget.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: None if not np.isfinite(x) else x)


if __name__ == "__main__":
    main()
