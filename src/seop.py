"""Spin-exchange optical pumping of 129Xe, as a rate model.

Everything downstream of the laser is a competition between rates, and the
whole polarizer can be written in four lines:

  rubidium     n_Rb(T)          from the vapour pressure of liquid rubidium
  pumping      P_Rb = R / (R + G_SD)      photons in against spin destruction
  exchange     dP_Xe/dt = g_SE (P_Rb - P_Xe) - G_Xe P_Xe
  light        dPhi/dz = -n_Rb sigma Phi (1 - P_Rb(z))  the laser is eaten
                                                          by the atoms it has
                                                          not yet polarized

The rate constants are literature-typical values.  Each is uncertain by tens
of per cent at best, and several (the van der Waals molecular terms, the wall
relaxation) vary from cell to cell, so nothing here is used as a point value
without also being swept over its range (see exp1).

Units: cm, s, K, W.  Densities in cm^-3; 1 amagat = 2.687e19 cm^-3.
"""

import numpy as np
from scipy.integrate import solve_ivp

AMAGAT = 2.687e19        # cm^-3, number density of an ideal gas at 0 C, 1 atm
K_B = 1.380649e-23       # J/K
TORR = 133.322           # Pa
H_C = 1.98645e-25        # J m
LAMBDA_D1 = 794.98e-9    # m, Rb D1

# --- literature-typical rate constants, with the range used in exp1 -------
CONST = {
    # spin exchange Rb -> 129Xe, binary collisions, cm^3/s
    "kappa_se":        (3.7e-16, 2.5e-16, 5.5e-16),
    # spin exchange through Rb-Xe van der Waals molecules: rate per Rb atom
    # per xenon partner, quoted at a total density of one amagat and scaled
    # by 1/rho_eff, cm^3/s (mixture-weighted, see rho_eff)
    "kappa_vdw":       (2.0e-16, 1.0e-16, 4.0e-16),
    # Rb spin destruction, cm^3/s
    "sd_xe":           (5.2e-15, 3.5e-15, 7.0e-15),
    "sd_n2":           (9.0e-18, 4.0e-18, 1.5e-17),
    "sd_he":           (2.0e-18, 1.0e-18, 4.0e-18),
    "sd_rb":           (4.2e-13, 3.0e-13, 6.0e-13),
    # pressure broadening of the Rb D1 line, GHz FWHM per amagat
    "broadening_ghz":  (18.0, 16.0, 20.0),
    # 129Xe relaxation inside the cell (walls + gas), s^-1
    "gamma_xe":        (1 / 1800.0, 1 / 7200.0, 1 / 600.0),
    # ratio of true Rb density to the vapour-pressure formula (the cell
    # is rarely at the temperature of the thermocouple, and the Rb surface
    # is rarely clean)
    "rb_scale":        (1.0, 0.5, 1.6),
}

F_D1 = 0.342                 # oscillator strength of the D1 line
R_E = 2.818e-13              # classical electron radius, cm
C_CM = 2.998e10              # cm/s
SIGMA_INT = np.pi * R_E * C_CM * F_D1       # integrated cross-section, cm^2 Hz


def rb_density(T_K, scale=1.0):
    """Rubidium number density above liquid Rb, cm^-3 (vapour-pressure fit).

    log10 P[torr] = 2.881 + 4.312 - 4040 / T   for T above the melting point.
    Roughly 1e13 cm^-3 at 100 C and 1e14 at 150 C.
    """
    T_K = np.asarray(T_K, float)
    p_torr = 10.0 ** (2.881 + 4.312 - 4040.0 / T_K)
    n_m3 = p_torr * TORR / (K_B * T_K)
    return scale * n_m3 * 1e-6


def photon_flux(power_W, area_cm2):
    """Photons per cm^2 per s from a laser of given power on a given area."""
    e_photon = H_C / LAMBDA_D1
    return power_W / e_photon / area_cm2


def sigma_eff(total_amagat, laser_ghz, broadening_ghz=CONST["broadening_ghz"][0]):
    """Effective absorption cross-section, cm^2, for a laser whose line is
    wider than the atomic one.  Both are taken as Lorentzians centred
    together; the overlap of two Lorentzians is a Lorentzian of the summed
    width, so the effective cross-section is the peak of that."""
    width = (broadening_ghz * total_amagat + laser_ghz) * 1e9
    return 2.0 * SIGMA_INT / (np.pi * width)


def rho_eff(xe, n2, he):
    """Effective density (amagat) that quenches Rb-Xe van der Waals molecules;
    N2 and He break them up less efficiently than Xe does."""
    return xe + 0.92 * n2 + 0.31 * he


def gamma_sd(n_rb, xe, n2, he, c=None):
    """Rb spin-destruction rate, s^-1."""
    c = c or {k: v[0] for k, v in CONST.items()}
    return (c["sd_xe"] * xe * AMAGAT + c["sd_n2"] * n2 * AMAGAT
            + c["sd_he"] * he * AMAGAT + c["sd_rb"] * n_rb)


def gamma_se(n_rb, xe, n2, he, c=None):
    """Rb -> Xe spin-exchange rate seen by a xenon atom, s^-1."""
    c = c or {k: v[0] for k, v in CONST.items()}
    return n_rb * (c["kappa_se"] + c["kappa_vdw"] / max(rho_eff(xe, n2, he), 1e-6))


def cell_profile(T_K, power_W, area_cm2, length_cm, xe, n2, he,
                 laser_ghz=120.0, c=None, nz=200):
    """Rb polarization along the cell, with the laser attenuated by the
    atoms it has not yet polarized.

    Returns z (cm), P_Rb(z), Phi(z)/Phi(0), and the volume-averaged P_Rb.
    """
    c = c or {k: v[0] for k, v in CONST.items()}
    n_rb = rb_density(T_K, c["rb_scale"])
    tot = xe + n2 + he
    sig = sigma_eff(tot, laser_ghz, c["broadening_ghz"])
    G = gamma_sd(n_rb, xe, n2, he, c)
    phi0 = photon_flux(power_W, area_cm2)

    def rhs(z, y):
        phi = max(y[0], 0.0)
        R = sig * phi
        P = R / (R + G)
        return [-n_rb * sig * phi * (1.0 - P)]

    sol = solve_ivp(rhs, (0, length_cm), [phi0], dense_output=True,
                    rtol=1e-8, atol=phi0 * 1e-12, max_step=length_cm / nz)
    z = np.linspace(0, length_cm, nz)
    phi = np.clip(sol.sol(z)[0], 0, None)
    R = sig * phi
    P = R / (R + G)
    return z, P, phi / phi0, float(np.trapezoid(P, z) / length_cm)


def xe_polarization(P_rb, g_se, g_xe, t):
    """Xenon polarization after a time t in the cell, starting from zero."""
    g = g_se + g_xe
    return P_rb * g_se / g * (1.0 - np.exp(-g * t))


def polarizer(T_K, power_W, flow_slm, xe, n2, he, area_cm2=20.0,
              length_cm=25.0, cell_pressure_atm=None, laser_ghz=120.0,
              c=None):
    """One operating point of a continuous-flow polarizer.

    flow_slm   gas flow in standard litres per minute (all species)
    xe,n2,he   partial densities in amagat inside the cell

    Returns a dict with the cell-averaged Rb polarization, the exchange and
    destruction rates, the residence time, the xenon polarization leaving the
    cell and the production rate of polarization (P_Xe times the xenon flow),
    which is the quantity a hospital actually buys.
    """
    c = c or {k: v[0] for k, v in CONST.items()}
    tot = xe + n2 + he
    volume = area_cm2 * length_cm
    # residence time: the cell holds volume*tot amagat-cm^3 of gas and the
    # flow removes flow_slm*1000/60 amagat-cm^3 per second
    t_res = volume * tot / (flow_slm * 1000.0 / 60.0)
    n_rb = rb_density(T_K, c["rb_scale"])
    z, P, phi, P_rb = cell_profile(T_K, power_W, area_cm2, length_cm, xe, n2,
                                   he, laser_ghz, c)
    g_se = gamma_se(n_rb, xe, n2, he, c)
    g_xe = c["gamma_xe"]
    P_xe = xe_polarization(P_rb, g_se, g_xe, t_res)
    xe_flow = flow_slm * xe / tot                  # standard litres/min of Xe
    return {
        "T_K": T_K, "power_W": power_W, "flow_slm": flow_slm,
        "n_rb": n_rb, "P_rb": P_rb, "laser_transmitted": float(phi[-1]),
        "gamma_se": g_se, "gamma_sd": gamma_sd(n_rb, xe, n2, he, c),
        "gamma_xe": g_xe, "t_res_s": t_res, "P_xe": float(P_xe),
        "xe_slm": xe_flow, "production_slm": float(P_xe * xe_flow),
    }


def sample_constants(rng, n):
    """Draw n sets of rate constants, each log-uniform within its range."""
    out = []
    for _ in range(n):
        c = {}
        for k, (mid, lo, hi) in CONST.items():
            c[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        out.append(c)
    return out


if __name__ == "__main__":
    for T in (90, 110, 130, 150, 170):
        print(f"{T} C  n_Rb = {rb_density(T + 273.15):.2e} cm^-3")
    print(f"sigma_eff(3 amagat, 120 GHz) = {sigma_eff(3, 120):.2e} cm^2")
    print(f"photon flux 50 W / 20 cm^2 = {photon_flux(50, 20):.2e} /cm^2/s")
    r = polarizer(140 + 273.15, 50, 1.0, xe=0.15, n2=0.3, he=2.55)
    for k, v in r.items():
        print(f"  {k:18s} {v:.4g}")
