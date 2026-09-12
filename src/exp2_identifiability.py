"""What a build-up curve can and cannot tell you about the rate constants.

The xenon polarization in a sealed cell grows as

    P(t) = P_inf (1 - exp(-g t)),    P_inf = P_Rb k n_Rb / (k n_Rb + G),
                                     g     = k n_Rb + G

with k the spin-exchange coefficient, G the xenon relaxation rate in the cell
and n_Rb the rubidium density.  The curve at one temperature has two
observable numbers, P_inf and g, and the model has three unknowns (k, G,
P_Rb) plus a fourth that is usually forgotten: the rubidium density is taken
from a vapour-pressure formula and a thermocouple, and the real density is
that times an unknown factor f that is rarely within 30% of one.

This file asks, by Fisher information and by fitting synthetic data, which
combinations of measurements pin down which parameters:

  A  one temperature                            the usual experiment
  B  four temperatures                          the usual "temperature series"
  C  B plus an independent measurement of P_Rb  e.g. from laser transmission
  D  B, but the rubidium polarization actually falls with temperature
     (the laser runs out) while the fit assumes it is constant

The point is not the arithmetic.  It is that k and f enter the model only
as a product, so no series of build-up curves at any set of temperatures can
separate them, and a "measured" spin-exchange rate carries the full
uncertainty of the rubidium density it was divided by.
"""

import json
import sys

sys.path.insert(0, 'src')

import numpy as np
from scipy.optimize import least_squares

from seop import polarizer, rb_density

TRUE = {"k": 5.3e-16, "G": 1 / 1800.0, "P_rb": 0.85, "f": 1.0}
T_SINGLE = [130.0]
T_SERIES = [90.0, 110.0, 130.0, 150.0]
TIMES = np.geomspace(5.0, 3000.0, 16)
NOISE_ABS, NOISE_REL = 0.005, 0.03
N_REP = 200
P_RB_PRIOR_SIGMA = 0.08              # design C: P_Rb known to about +-0.08


def buildup(theta, temps_C, times, p_rb_of_T=None):
    """theta = (log k, log G, P_rb); the Rb density uses the nominal formula
    (f = 1), which is what an experimenter would do."""
    lk, lG, p_rb = theta
    rows = []
    for T in temps_C:
        n = rb_density(T + 273.15)
        g_se = np.exp(lk) * n
        g = g_se + np.exp(lG)
        p = p_rb if p_rb_of_T is None else p_rb_of_T(T)
        rows.append(p * g_se / g * (1.0 - np.exp(-g * times)))
    return np.concatenate(rows)


def noise_sigma(y):
    return NOISE_ABS + NOISE_REL * y


def simulate(rng, temps_C, p_rb_of_T=None):
    th = (np.log(TRUE["k"] * TRUE["f"]), np.log(TRUE["G"]), TRUE["P_rb"])
    y = buildup(th, temps_C, TIMES, p_rb_of_T)
    return y + rng.normal(0, noise_sigma(y))


def fit(y, temps_C, prior_p_rb=None):
    sig = noise_sigma(np.clip(y, 0, None))

    def resid(th):
        r = (buildup(th, temps_C, TIMES) - y) / sig
        if prior_p_rb is not None:
            r = np.append(r, (th[2] - prior_p_rb) / P_RB_PRIOR_SIGMA)
        return r

    x0 = (np.log(3e-16), np.log(1e-3), 0.7)
    best = None
    for start in (x0, (np.log(1e-15), np.log(3e-4), 0.9),
                  (np.log(1e-16), np.log(3e-3), 0.5)):
        s = least_squares(resid, start, bounds=([-40, -12, 0.0], [-33, -3, 1.0]))
        if best is None or s.cost < best.cost:
            best = s
    return best


def fisher(temps_C, prior_p_rb=False):
    """Covariance of (log k, log G, P_rb) from the Jacobian at the truth."""
    th = np.array([np.log(TRUE["k"]), np.log(TRUE["G"]), TRUE["P_rb"]])
    y = buildup(th, temps_C, TIMES)
    sig = noise_sigma(y)
    J = np.zeros((len(y), 3))
    for j in range(3):
        h = 1e-5
        d = np.zeros(3); d[j] = h
        J[:, j] = (buildup(th + d, temps_C, TIMES)
                   - buildup(th - d, temps_C, TIMES)) / (2 * h)
    J = J / sig[:, None]
    F = J.T @ J
    if prior_p_rb:
        F[2, 2] += 1.0 / P_RB_PRIOR_SIGMA ** 2
    ev, vec = np.linalg.eigh(F)
    cond = float(ev[-1] / max(ev[0], 1e-300))
    if ev[0] / ev[-1] < 1e-10:
        # singular: the parameter that dominates the null direction is not
        # identifiable at all, and a pseudo-inverse would report it as
        # perfectly known.  Say so instead.
        cov = np.linalg.pinv(F)
        sd = np.sqrt(np.diag(cov))
        sd[np.argmax(np.abs(vec[:, 0]))] = np.inf
    else:
        cov = np.linalg.inv(F)
        sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(np.where(np.isfinite(sd), sd, 1.0),
                          np.where(np.isfinite(sd), sd, 1.0))
    sd = [None if not np.isfinite(v) else float(v) for v in sd]
    return {"sd_log_k": sd[0], "sd_log_G": sd[1],
            "sd_P_rb": sd[2], "condition": cond,
            "corr_k_G": float(corr[0, 1]), "corr_k_Prb": float(corr[0, 2]),
            "corr_G_Prb": float(corr[1, 2])}


def profile(temps_C, rng):
    """Profile chi-square along log k: how well is k pinned?"""
    y = simulate(rng, temps_C)
    sig = noise_sigma(np.clip(y, 0, None))
    grid = np.log(TRUE["k"]) + np.linspace(-1.5, 1.5, 31)
    prof = []
    for lk in grid:
        def resid(th2):
            return (buildup((lk, th2[0], th2[1]), temps_C, TIMES) - y) / sig
        s = least_squares(resid, (np.log(1e-3), 0.7),
                          bounds=([-12, 0.0], [-3, 1.0]))
        prof.append(2 * s.cost)
    prof = np.array(prof)
    return {"log_k_offset": (grid - np.log(TRUE["k"])).tolist(),
            "delta_chi2": (prof - prof.min()).tolist()}


def p_rb_starved(T, power=50.0):
    """Design D: the rubidium polarization the SEOP model gives for a laser
    of this power -- it falls as the cell gets hotter and the light runs
    out."""
    return polarizer(T + 273.15, power, 1.0, 0.15, 0.30, 2.55)["P_rb"]


def main():
    rng = np.random.default_rng(0)
    designs = {"A_single_T": (T_SINGLE, None, None),
               "B_series": (T_SERIES, None, None),
               "C_series_plus_Prb": (T_SERIES, TRUE["P_rb"], None),
               "D_series_50W_laser": (T_SERIES, None, 50.0),
               "D_series_100W_laser": (T_SERIES, None, 100.0)}
    out = {"truth": TRUE, "temps_series_C": T_SERIES, "times_s": TIMES.tolist(),
           "noise": {"abs": NOISE_ABS, "rel": NOISE_REL}, "designs": {}}
    out["P_rb_starved_truth"] = {}
    for name, (temps, prior, misspec) in designs.items():
        p_rb_fn = None
        if misspec:
            cache = {T: p_rb_starved(T, misspec) for T in temps}
            out["P_rb_starved_truth"][str(int(misspec))] = cache
            p_rb_fn = lambda T, c=cache: c[T]
        est, chi2 = [], []
        for _ in range(N_REP):
            y = simulate(rng, temps, p_rb_fn)
            s = fit(y, temps, prior)
            est.append(s.x)
            chi2.append(2 * s.cost / (len(y) - 3))     # reduced chi-square
        est = np.array(est)
        k_hat, G_hat, p_hat = np.exp(est[:, 0]), np.exp(est[:, 1]), est[:, 2]
        d = {"n_rep": N_REP,
             "reduced_chi2_median": float(np.median(chi2)),
             "k_ratio_median": float(np.median(k_hat) / TRUE["k"]),
             "k_ratio_p05": float(np.quantile(k_hat, 0.05) / TRUE["k"]),
             "k_ratio_p95": float(np.quantile(k_hat, 0.95) / TRUE["k"]),
             "G_ratio_median": float(np.median(G_hat) / TRUE["G"]),
             "G_ratio_p05": float(np.quantile(G_hat, 0.05) / TRUE["G"]),
             "G_ratio_p95": float(np.quantile(G_hat, 0.95) / TRUE["G"]),
             "P_rb_median": float(np.median(p_hat)),
             "P_rb_p05": float(np.quantile(p_hat, 0.05)),
             "P_rb_p95": float(np.quantile(p_hat, 0.95))}
        if not misspec:
            d["fisher"] = fisher(temps, prior is not None)
        out["designs"][name] = d

    out["profile"] = {"A_single_T": profile(T_SINGLE, rng),
                      "B_series": profile(T_SERIES, rng)}
    # the structural point: k and f are one parameter.  Show it by refitting
    # design B data generated with f = 0.6 and f = 1.6.
    out["rb_scale_degeneracy"] = {}
    for f in (0.6, 1.0, 1.6):
        est = []
        for _ in range(60):
            th = (np.log(TRUE["k"] * f), np.log(TRUE["G"]), TRUE["P_rb"])
            y = buildup(th, T_SERIES, TIMES)
            y = y + rng.normal(0, noise_sigma(y))
            est.append(np.exp(fit(y, T_SERIES).x[0]) / TRUE["k"])
        out["rb_scale_degeneracy"][str(f)] = float(np.median(est))

    with open("results/exp2_identifiability.json", "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"truth: k = {TRUE['k']:.2e} cm^3/s, G = 1/{1/TRUE['G']:.0f} s, "
          f"P_Rb = {TRUE['P_rb']}\n")
    print(f"{'design':26s} {'k / true':>18s} {'G / true':>18s} {'P_Rb':>18s}")
    for name, d in out["designs"].items():
        print(f"{name:26s} {d['k_ratio_median']:6.2f} [{d['k_ratio_p05']:5.2f},"
              f"{d['k_ratio_p95']:5.2f}]  {d['G_ratio_median']:6.2f} "
              f"[{d['G_ratio_p05']:5.2f},{d['G_ratio_p95']:5.2f}]  "
              f"{d['P_rb_median']:6.2f} [{d['P_rb_p05']:5.2f},{d['P_rb_p95']:5.2f}]"
              f"   chi2/dof {d['reduced_chi2_median']:.1f}")
    print("\nFisher information at the truth")
    for name, d in out["designs"].items():
        if "fisher" in d:
            fi = {k: (float("inf") if v is None else v) for k, v in d["fisher"].items()}
            print(f"  {name:26s} sd(log k) {fi['sd_log_k']:.2f}  sd(log G) "
                  f"{fi['sd_log_G']:.2f}  sd(P_Rb) {fi['sd_P_rb']:.2f}  "
                  f"corr(k,P_Rb) {fi['corr_k_Prb']:+.2f}  cond {fi['condition']:.1e}")
    print("\nfitted k / true k when the real Rb density is f x nominal:")
    for f, v in out["rb_scale_degeneracy"].items():
        print(f"  f = {f}: {v:.2f}")
    for pw, cache in out["P_rb_starved_truth"].items():
        print(f"\nP_Rb a {pw} W laser really delivers (design D):",
              {T: round(v, 2) for T, v in cache.items()})


if __name__ == "__main__":
    main()
