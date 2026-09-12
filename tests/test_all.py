"""Checks on the physics, the estimators and the closed forms."""

import sys

sys.path.insert(0, 'src')

import numpy as np

from seop import (AMAGAT, CONST, cell_profile, gamma_sd, photon_flux,
                  rb_density, sigma_eff, xe_polarization)
from exp2_identifiability import TRUE, T_SERIES, TIMES, buildup, fit
from exp3_budget import (accumulation_survival, constant_flip, t1_oxygen,
                         variable_flip)
from exp4_sabre import System, three_spin, transferred


def test_rubidium_density_is_in_the_known_range():
    n100, n150 = rb_density(373.15), rb_density(423.15)
    assert 3e12 < n100 < 2e13, n100
    assert 5e13 < n150 < 2e14, n150
    T = np.linspace(330, 480, 50)
    assert np.all(np.diff(rb_density(T)) > 0)


def test_cross_section_and_flux_scale_correctly():
    assert sigma_eff(1.0, 120) > sigma_eff(3.0, 120) > sigma_eff(10.0, 120)
    # 1 W at 795 nm is 4.0e18 photons/s
    assert abs(photon_flux(1.0, 1.0) / 4.0e18 - 1) < 0.02


def test_photon_budget_is_conserved_along_the_cell():
    """Every photon absorbed must have destroyed one Rb spin at steady state:
    Phi(0) - Phi(L) = integral n_Rb Gamma_SD P_Rb dz."""
    T, pw, A, L = 393.15, 50.0, 20.0, 25.0
    xe, n2, he = 0.15, 0.30, 2.55
    z, P, phi, _ = cell_profile(T, pw, A, L, xe, n2, he, nz=400)
    n_rb = rb_density(T)
    G = gamma_sd(n_rb, xe, n2, he)
    absorbed = photon_flux(pw, A) * (1.0 - phi[-1])
    destroyed = np.trapezoid(n_rb * G * P, z)
    assert abs(absorbed / destroyed - 1) < 0.02, (absorbed, destroyed)


def test_thin_cell_limit():
    """With almost no rubidium the laser passes and P_Rb = R/(R+G)."""
    T = 313.15                                  # 40 C: n_Rb ~ 1e11
    z, P, phi, Pm = cell_profile(T, 50, 20, 25, 0.15, 0.3, 2.55)
    assert phi[-1] > 0.95
    n_rb = rb_density(T)
    R = sigma_eff(3.0, 120) * photon_flux(50, 20)
    assert abs(Pm - R / (R + gamma_sd(n_rb, 0.15, 0.3, 2.55))) < 1e-3


def test_xenon_buildup_limits():
    assert xe_polarization(0.8, 0.02, 0.002, 0.0) == 0.0
    assert abs(xe_polarization(0.8, 0.02, 0.002, 1e6) - 0.8 * 0.02 / 0.022) < 1e-9


def test_fit_recovers_truth_without_noise():
    th = (np.log(TRUE["k"]), np.log(TRUE["G"]), TRUE["P_rb"])
    y = buildup(th, T_SERIES, TIMES)
    s = fit(y, T_SERIES)
    assert abs(np.exp(s.x[0]) / TRUE["k"] - 1) < 0.01
    assert abs(np.exp(s.x[1]) / TRUE["G"] - 1) < 0.02
    assert abs(s.x[2] - TRUE["P_rb"]) < 0.005


def test_oxygen_gives_the_lung_its_twenty_seconds():
    t1 = t1_oxygen(0.13)
    assert 15 < t1 < 25, t1


def test_accumulation_survival_bounds():
    """The average over an accumulation lies between the survival of the
    last atom in (1) and of the first (exp(-t/T1))."""
    for x in (0.1, 1.0, 5.0):
        s = accumulation_survival(x, 1.0)
        assert np.exp(-x) < s < 1.0


def test_variable_flip_angle_is_flat_and_matches_the_closed_form():
    S, a = variable_flip(8, 0.0, np.inf)
    assert np.ptp(S) < 1e-12
    with np.errstate(divide="ignore"):
        expected = np.rad2deg(np.arctan(1 / np.sqrt(8 - np.arange(1, 9))))
    assert np.allclose(a, expected)
    S2, _ = variable_flip(64, 0.15, 20.0)
    assert np.ptp(S2) < 1e-12                      # still flat with T1
    Sc, _ = constant_flip(64, 10.0, 0.15, 20.0)
    assert Sc[-1] < Sc[0]


def test_sabre_needs_asymmetry_and_a_field():
    assert abs(transferred(0.3e-6, -8, -12, -12, 0.2)) < 1e-9
    assert abs(transferred(0.0, -8, -24, 0, 0.2)) < 1e-9
    p = transferred(0.2e-6, -8, -24, 0, 0.2)
    assert 0.1 < p < 0.25, p


def test_sabre_closed_form_equals_the_time_integral():
    sysm = three_spin(-8.0, -24.0, 0.0)
    tau = 0.05
    times = np.linspace(0, 12 * tau, 20000)
    p_t = sysm.n_polarization_vs_time(0.2e-6, times)
    grid = np.trapezoid(p_t * np.exp(-times / tau) / tau, times)
    assert abs(grid - sysm.transferred(0.2e-6, tau)) < 2e-3


def test_sabre_polarization_is_physical():
    sysm = System(2, {(0, 1): -8.0, (0, 2): -24.0, (1, 3): -24.0})
    for b in (0.1e-6, 0.4e-6, 1.0e-6):
        p = sysm.n_polarization_vs_time(b, np.linspace(0, 2, 200))
        assert np.all(np.abs(p) <= 1.0 + 1e-9)


def test_forward_pinn_matches_the_solver():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skipped: no torch)"); return
    from exp5_pinn import forward_cell_pinn
    f = forward_cell_pinn(steps=1500)
    assert f["max_abs_err_phi"] < 1e-3, f["max_abs_err_phi"]
    assert f["max_abs_err_P"] < 1e-3, f["max_abs_err_P"]


def test_inverse_pinn_hard_constraint_at_t0():
    try:
        import torch
    except ImportError:
        print("  (skipped: no torch)"); return
    from exp5_pinn import BuildupPINN
    m = BuildupPINN(True)
    t0 = torch.zeros(5, 1, dtype=torch.float64)
    T = torch.linspace(-1, 1, 5, dtype=torch.float64).reshape(-1, 1)
    assert torch.all(m(t0, T) == 0.0)
    assert torch.all((m.P_rb(T) > 0) & (m.P_rb(T) < 1))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
