"""Two physics-informed neural networks for the polarizer, and what each buys.

The rate models in this repository are cheap to solve exactly, so a neural
network is not needed to solve them.  It is used here for the two things a
network with the physics inside it can do that a curve fit cannot:

  forward   solve the coupled light-attenuation / rubidium-polarization
            problem along the cell from the ODE alone, with no solver, as a
            check that the physics residual is enough to pin the solution
            (it is; the PINN matches the Runge-Kutta reference to about
            1e-3, and is slower -- that is stated, not hidden)

  inverse   fit the build-up curves of exp2 with the rate equation as a
            residual and the rate constants as trainable parameters.  The
            useful case is design D of exp2, where a least-squares fit that
            assumes a constant rubidium polarization returns kappa 2.2x too
            high: the PINN is given an *unknown function* P_Rb(T) instead
            of an unknown constant, and the question is whether the physics
            still identifies kappa and Gamma.  It does, because the rate
            information lives in the time constant gamma(T) = kappa n(T) +
            Gamma, which the residual enforces, and not in the plateau,
            which P_Rb(T) is free to absorb.

Runs on CPU in a few minutes.  torch only.
"""

import json
import sys
import time

sys.path.insert(0, 'src')

import numpy as np
import torch

from exp2_identifiability import (NOISE_ABS, NOISE_REL, T_SERIES, TIMES,
                                  TRUE, buildup, fit, p_rb_starved)
from seop import cell_profile, gamma_sd, photon_flux, rb_density, sigma_eff

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
DEV = "cpu"


def mlp(n_in, n_out, width=48, depth=3):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers.append(torch.nn.Linear(d, n_out))
    return torch.nn.Sequential(*layers)


# ---------------------------------------------------------------- forward --

def forward_cell_pinn(T_K=393.15, power_W=50.0, area=20.0, L=25.0,
                      xe=0.15, n2=0.30, he=2.55, laser_ghz=120.0,
                      steps=4000):
    """Solve d(phi)/dz = -n sigma phi (1 - P), P = sigma phi/(sigma phi + G)
    with a network for log phi(z), hard-constrained to phi(0) = phi0."""
    n_rb = rb_density(T_K)
    sig = sigma_eff(xe + n2 + he, laser_ghz)
    G = gamma_sd(n_rb, xe, n2, he)
    phi0 = photon_flux(power_W, area)
    # non-dimensionalise: u = phi/phi0, s = z/L, a = n sigma L (optical depth)
    a = n_rb * sig * L
    R0 = sig * phi0 / G                     # pumping rate / destruction, at z=0
    net = mlp(1, 1).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    s_col = torch.linspace(0, 1, 256, device=DEV).reshape(-1, 1).requires_grad_(True)

    def u_of(s):
        # log u = s * N(s)  ->  u(0) = 1 exactly
        return torch.exp(s * net(s))

    t0 = time.time()
    for it in range(steps):
        opt.zero_grad()
        u = u_of(s_col)
        du = torch.autograd.grad(u, s_col, torch.ones_like(u), create_graph=True)[0]
        P = R0 * u / (R0 * u + 1.0)
        res = du + a * u * (1.0 - P)
        loss = torch.mean((res / (a * u.detach() + 1e-9)) ** 2)   # relative residual
        loss.backward()
        opt.step()
    lb = torch.optim.LBFGS(net.parameters(), max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        u = u_of(s_col)
        du = torch.autograd.grad(u, s_col, torch.ones_like(u), create_graph=True)[0]
        P = R0 * u / (R0 * u + 1.0)
        res = du + a * u * (1.0 - P)
        l = torch.mean((res / (a * u.detach() + 1e-9)) ** 2)
        l.backward()
        return l
    lb.step(closure)
    t_pinn = time.time() - t0

    t0 = time.time()
    z, P_ref, phi_ref, Pm_ref = cell_profile(T_K, power_W, area, L, xe, n2, he,
                                             laser_ghz, nz=256)
    t_ref = time.time() - t0
    with torch.no_grad():
        s = torch.tensor(z / L).reshape(-1, 1)
        u = u_of(s).squeeze().numpy()
    P_pinn = R0 * u / (R0 * u + 1.0)
    return {"z_cm": z.tolist(), "phi_ref": phi_ref.tolist(), "phi_pinn": u.tolist(),
            "P_ref": P_ref.tolist(), "P_pinn": P_pinn.tolist(),
            "max_abs_err_phi": float(np.max(np.abs(u - phi_ref))),
            "max_abs_err_P": float(np.max(np.abs(P_pinn - P_ref))),
            "P_mean_ref": float(Pm_ref), "P_mean_pinn": float(np.trapezoid(P_pinn, z) / L),
            "time_pinn_s": t_pinn, "time_rk_s": t_ref, "optical_depth": float(a),
            "R0_over_G": float(R0)}


# ---------------------------------------------------------------- inverse --

class BuildupPINN(torch.nn.Module):
    """P(t, T) = t_s * N(t_s, T_s), so P(0) = 0 exactly.  Rate constants are
    trainable in log space; the rubidium polarization is either one trainable
    scalar (the exp2 assumption) or a small network of temperature."""

    def __init__(self, prb_is_function):
        super().__init__()
        self.net = mlp(2, 1)
        self.log_k = torch.nn.Parameter(torch.tensor(np.log(3e-16)))
        self.log_G = torch.nn.Parameter(torch.tensor(np.log(1e-3)))
        self.prb_is_function = prb_is_function
        if prb_is_function:
            self.prb_net = mlp(1, 1, width=16, depth=2)
        else:
            self.prb_logit = torch.nn.Parameter(torch.tensor(1.0))

    def P_rb(self, T_s):
        if self.prb_is_function:
            return torch.sigmoid(self.prb_net(T_s))
        return torch.sigmoid(self.prb_logit) * torch.ones_like(T_s)

    def forward(self, t_s, T_s):
        x = torch.cat([t_s, T_s], dim=1)
        return t_s * self.net(x)


T_SCALE = 1000.0          # seconds -> scaled time


def T_s_of(T_C):
    return (np.asarray(T_C, float) - 120.0) / 30.0


def inverse_pinn(y, temps_C, prb_is_function, steps=6000, seed=0):
    torch.manual_seed(seed)
    m = BuildupPINN(prb_is_function).to(DEV)
    n_rb = {T: rb_density(T + 273.15) for T in temps_C}
    # data
    t_d = np.tile(TIMES, len(temps_C)) / T_SCALE
    T_d = np.repeat(T_s_of(temps_C), len(TIMES))
    y_d = np.asarray(y)
    sig = NOISE_ABS + NOISE_REL * np.clip(y_d, 0, None)
    t_d, T_d, y_d, sig = (torch.tensor(v).reshape(-1, 1) for v in (t_d, T_d, y_d, sig))
    # collocation
    n_per = 150
    t_c = torch.rand(n_per * len(temps_C), 1) * 3.0
    T_c = torch.tensor(np.repeat(T_s_of(temps_C), n_per)).reshape(-1, 1)
    n_c = torch.tensor(np.repeat([n_rb[T] for T in temps_C], n_per)).reshape(-1, 1)
    t_c.requires_grad_(True)
    # fixed residual scale (a trainable one would reward a large kappa)
    g_ref2 = torch.mean((5e-16 * n_c * T_SCALE * 0.5) ** 2)

    def losses():
        P = m(t_d, T_d)
        l_data = torch.mean(((P - y_d) / sig) ** 2)
        Pc = m(t_c, T_c)
        dP = torch.autograd.grad(Pc, t_c, torch.ones_like(Pc), create_graph=True)[0]
        g_se = torch.exp(m.log_k) * n_c * T_SCALE
        G = torch.exp(m.log_G) * T_SCALE
        res = dP - (g_se * (m.P_rb(T_c) - Pc) - G * Pc)
        l_phys = torch.mean(res ** 2) / g_ref2
        return l_data, l_phys

    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for it in range(steps):
        opt.zero_grad()
        ld, lp = losses()
        (ld + 50.0 * lp).backward()
        opt.step()
    lb = torch.optim.LBFGS(m.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        ld, lp = losses()
        l = ld + 50.0 * lp
        l.backward()
        return l
    lb.step(closure)
    ld, lp = losses()
    with torch.no_grad():
        prb = {float(T): float(m.P_rb(torch.tensor([[v]]))) for T, v in
               zip(temps_C, T_s_of(temps_C))}
    return {"k": float(torch.exp(m.log_k)), "G": float(torch.exp(m.log_G)),
            "P_rb": prb, "chi2_dof": float(ld) * len(y_d) / (len(y_d) - 3),
            "phys_res": float(lp)}


def main():
    rng = np.random.default_rng(0)
    out = {"forward": forward_cell_pinn()}
    f = out["forward"]
    print(f"forward cell PINN: max|phi err| {f['max_abs_err_phi']:.1e}, "
          f"max|P err| {f['max_abs_err_P']:.1e}, mean P {f['P_mean_pinn']:.4f} vs "
          f"{f['P_mean_ref']:.4f}; {f['time_pinn_s']:.0f} s vs RK {f['time_rk_s']*1000:.0f} ms")

    # inverse: designs B (well specified) and D (laser-starved, 50 W)
    cases = {"B_series": None, "D_series_50W_laser": 50.0}
    N_REP = 5
    out["inverse"] = {}
    for name, power in cases.items():
        cache = {T: p_rb_starved(T, power) for T in T_SERIES} if power else None
        rows = {"least_squares_const_Prb": [], "pinn_const_Prb": [], "pinn_Prb_of_T": []}
        for rep in range(N_REP):
            th = (np.log(TRUE["k"]), np.log(TRUE["G"]), TRUE["P_rb"])
            y = buildup(th, T_SERIES, TIMES, (lambda T, c=cache: c[T]) if cache else None)
            y = y + rng.normal(0, NOISE_ABS + NOISE_REL * y)
            s = fit(y, T_SERIES)
            rows["least_squares_const_Prb"].append(
                {"k": float(np.exp(s.x[0])), "G": float(np.exp(s.x[1])), "P_rb": float(s.x[2])})
            rows["pinn_const_Prb"].append(inverse_pinn(y, T_SERIES, False, seed=rep))
            rows["pinn_Prb_of_T"].append(inverse_pinn(y, T_SERIES, True, seed=rep))
            print(f"  {name} rep {rep}: LS k/true {rows['least_squares_const_Prb'][-1]['k']/TRUE['k']:.2f}"
                  f"  PINN-const {rows['pinn_const_Prb'][-1]['k']/TRUE['k']:.2f}"
                  f"  PINN-P_rb(T) {rows['pinn_Prb_of_T'][-1]['k']/TRUE['k']:.2f}"
                  f"  G/true {rows['pinn_Prb_of_T'][-1]['G']/TRUE['G']:.2f}")
        summ = {}
        for meth, lst in rows.items():
            kk = np.array([r["k"] for r in lst]) / TRUE["k"]
            GG = np.array([r["G"] for r in lst]) / TRUE["G"]
            summ[meth] = {"k_ratio_median": float(np.median(kk)),
                          "k_ratio_min": float(kk.min()), "k_ratio_max": float(kk.max()),
                          "G_ratio_median": float(np.median(GG)),
                          "G_ratio_min": float(GG.min()), "G_ratio_max": float(GG.max()),
                          "reps": lst}
        if cache:
            summ["P_rb_true_of_T"] = cache
        out["inverse"][name] = summ
        print(f"{name}:")
        for meth, s_ in summ.items():
            if meth == "P_rb_true_of_T":
                continue
            print(f"  {meth:26s} k/true {s_['k_ratio_median']:.2f} "
                  f"[{s_['k_ratio_min']:.2f},{s_['k_ratio_max']:.2f}]   G/true "
                  f"{s_['G_ratio_median']:.2f} [{s_['G_ratio_min']:.2f},{s_['G_ratio_max']:.2f}]")

    with open("results/exp5_pinn.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
