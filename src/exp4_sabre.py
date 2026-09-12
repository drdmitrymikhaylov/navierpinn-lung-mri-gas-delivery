"""SABRE-SHEATH: which field moves parahydrogen order onto a 15N nucleus.

Parahydrogen carries no polarization -- a singlet has none -- but it carries
order, and order becomes polarization when the two protons are made
inequivalent.  In SABRE the iridium catalyst binds the two hydrides and the
substrate's 15N in one complex; the hydride trans to the nitrogen couples to
it (J about -20 to -25 Hz) and the cis hydride barely does, which is the
asymmetry the singlet needs.  Whether it flows depends on the field: the
singlet-triplet gap set by J_HH has to match the Zeeman splitting between
proton and nitrogen, (gamma_H - gamma_N) B, and that happens in the sub-
microtesla range -- the shield in SHEATH exists to reach it.

This is the smallest model that has the effect: three spins, two hydrides
and one nitrogen, with the complex living for a random exponential time
before the substrate leaves and takes its 15N polarization with it.  The
observable is the 15N polarization carried away per binding event, as a
function of field, the couplings, and the residence time.

The point of computing it is the shape: the matching field, the width of the
resonance, and how both move when the couplings are uncertain -- because the
couplings in a given complex usually are.
"""

import json
import sys

sys.path.insert(0, 'src')

import numpy as np

GAMMA_H = 42.577e6         # Hz/T
GAMMA_N = -4.316e6         # Hz/T  (15N)

# Pauli-like spin-1/2 operators
sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)
I2 = np.eye(2, dtype=complex)


def op(single, k, n):
    mats = [I2] * n
    mats[k] = single
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


class System:
    """Two hydrides (spins 0, 1) and n_N nitrogen-15 nuclei.

    couplings: dict {(i, j): J_Hz}.  A one-substrate complex has one 15N
    (spin 2); a complex with two equatorial substrates has two (spins 2, 3),
    each trans to one hydride."""

    def __init__(self, n_N, couplings):
        self.n = 2 + n_N
        self.n_N = n_N
        self.X = [op(sx, k, self.n) for k in range(self.n)]
        self.Y = [op(sy, k, self.n) for k in range(self.n)]
        self.Z = [op(sz, k, self.n) for k in range(self.n)]
        self.J = couplings

    def dot(self, i, j):
        return self.X[i] @ self.X[j] + self.Y[i] @ self.Y[j] + self.Z[i] @ self.Z[j]

    def hamiltonian(self, B_T):
        """In Hz (times 2 pi for rad/s)."""
        H = -GAMMA_H * B_T * (self.Z[0] + self.Z[1])
        for k in range(2, self.n):
            H = H - GAMMA_N * B_T * self.Z[k]
        for (i, j), J in self.J.items():
            if J != 0.0:
                H = H + J * self.dot(i, j)
        return H

    def initial_state(self):
        """Hydrides in the singlet, every nitrogen unpolarized."""
        s = np.zeros(4, complex)
        s[1], s[2] = 1 / np.sqrt(2), -1 / np.sqrt(2)      # (|ud> - |du>)/sqrt2
        rho = np.outer(s, s.conj())
        for _ in range(self.n_N):
            rho = np.kron(rho, I2 / 2)
        return rho

    def n_polarization_vs_time(self, B_T, times):
        """Mean <2 N_z> over the nitrogen spins, in [-1, 1]."""
        H = 2 * np.pi * self.hamiltonian(B_T)
        w, V = np.linalg.eigh(H)
        rho0 = self.initial_state()
        Nz = sum(self.Z[k] for k in range(2, self.n)) / self.n_N
        rho0_e = V.conj().T @ rho0 @ V
        Nz_e = V.conj().T @ Nz @ V
        out = []
        for t in times:
            ph = np.exp(-1j * w * t)
            rho_t = (ph[:, None] * rho0_e) * ph.conj()[None, :]
            out.append(2 * np.real(np.trace(rho_t @ Nz_e)))
        return np.array(out)

    def transferred(self, B_T, tau_s, n_t=None):
        """15N polarization carried off by a substrate that leaves after an
        exponentially distributed residence time of mean tau.

        Done in closed form: in the eigenbasis each coherence oscillates at
        (w_i - w_j) and its exponential-weighted average is 1/(1 + i dw tau).
        (A time grid was tried first and aliased above 0.7 uT, where the
        Zeeman frequencies pass the grid's Nyquist limit.)"""
        H = 2 * np.pi * self.hamiltonian(B_T)
        w, V = np.linalg.eigh(H)
        rho0_e = V.conj().T @ self.initial_state() @ V
        Nz = sum(self.Z[k] for k in range(2, self.n)) / self.n_N
        Nz_e = V.conj().T @ Nz @ V
        dw = w[:, None] - w[None, :]
        weight = 1.0 / (1.0 + 1j * dw * tau_s)
        return float(2 * np.real(np.sum(rho0_e * weight * Nz_e.T)))

    def level_anticrossing(self, B):
        """Field at which the two levels connected by the singlet come
        closest: the LAC read off the spectrum rather than from a formula."""
        gaps = []
        for b in B:
            w = np.linalg.eigvalsh(self.hamiltonian(b))
            gaps.append(np.min(np.diff(w)))
        return float(B[int(np.argmin(gaps))])


def three_spin(J_HH, J_aN, J_bN):
    return System(1, {(0, 1): J_HH, (0, 2): J_aN, (1, 2): J_bN})


def transferred(B_T, J_HH, J_aN, J_bN, tau_s, n_t=400):
    return three_spin(J_HH, J_aN, J_bN).transferred(B_T, tau_s, n_t)


def main():
    J_HH, J_aN, J_bN = -8.0, -24.0, 0.0            # Hz, typical Ir-hydride/15N
    B = np.linspace(0.0, 2.0e-6, 401)               # T
    taus = [0.05, 0.2, 1.0]                         # s
    out = {"J": {"HH": J_HH, "aN": J_aN, "bN": J_bN}, "B_uT": (B * 1e6).tolist(),
           "curves": {}, "matching": {}}
    for tau in taus:
        p = np.array([transferred(b, J_HH, J_aN, J_bN, tau) for b in B])
        out["curves"][str(tau)] = p.tolist()
        j = int(np.argmax(np.abs(p)))
        half = np.abs(p) > 0.5 * np.abs(p[j])
        out["matching"][str(tau)] = {
            "B_match_uT": float(B[j] * 1e6), "P_max": float(p[j]),
            "fwhm_uT": float((B[half].max() - B[half].min()) * 1e6)}
    print(f"J_HH = {J_HH} Hz, J(H_trans,N) = {J_aN} Hz, J(H_cis,N) = {J_bN} Hz")
    naive = abs(J_HH) / (GAMMA_H - GAMMA_N) * 1e6
    first_order = (-J_HH - (J_aN + J_bN) / 4) / (GAMMA_H - GAMMA_N) * 1e6
    out["B_naive_uT"] = naive
    out["B_first_order_uT"] = first_order
    print(f"naive matching B = |J_HH| / (gamma_H - gamma_N) = {naive:.2f} uT")
    print(f"first-order crossing S,beta <-> T-,alpha: "
          f"(-J_HH - (J_aN+J_bN)/4)/(gamma_H - gamma_N) = {first_order:.2f} uT")
    for tau, m in out["matching"].items():
        print(f"  tau = {float(tau):4.2f} s   B_match = {m['B_match_uT']:.2f} uT   "
              f"P_15N per event = {m['P_max']:+.3f}   FWHM = {m['fwhm_uT']:.2f} uT")

    # uncertainty in the couplings moves the resonance
    rng = np.random.default_rng(0)
    out["coupling_scatter"] = []
    Bc = np.linspace(0.0, 1.5e-6, 301)
    for _ in range(40):
        jhh = rng.uniform(-11.0, -6.0)
        jan = rng.uniform(-28.0, -18.0)
        jbn = rng.uniform(-2.0, 2.0)
        p = np.array([transferred(b, jhh, jan, jbn, 0.2) for b in Bc])
        j = int(np.argmax(np.abs(p)))
        out["coupling_scatter"].append({"J_HH": jhh, "J_aN": jan, "J_bN": jbn,
                                        "B_match_uT": float(Bc[j] * 1e6),
                                        "P_max": float(p[j])})
    bm = np.array([d["B_match_uT"] for d in out["coupling_scatter"]])
    print(f"\ncouplings drawn from their plausible ranges: B_match "
          f"{bm.mean():.2f} +- {bm.std():.2f} uT (range {bm.min():.2f}..{bm.max():.2f})")

    # the cis coupling: what happens when both hydrides couple equally
    p_sym = max(abs(transferred(b, J_HH, -12.0, -12.0, 0.2)) for b in Bc)
    out["symmetric_couplings_P_max"] = p_sym
    print(f"with J_aN = J_bN (no asymmetry): max |P_15N| = {p_sym:.4f}")

    # a second equatorial substrate: two 15N, each trans to one hydride
    two = System(2, {(0, 1): J_HH, (0, 2): J_aN, (1, 3): J_aN,
                     (0, 3): J_bN, (1, 2): J_bN})
    p2 = np.array([two.transferred(b, 0.2) for b in B])
    j = int(np.argmax(np.abs(p2)))
    half = np.abs(p2) > 0.5 * np.abs(p2[j])
    out["two_substrates"] = {"curve": p2.tolist(), "B_match_uT": float(B[j] * 1e6),
                             "P_max_per_N": float(p2[j]),
                             "fwhm_uT": float((B[half].max() - B[half].min()) * 1e6)}
    print(f"two bound 15N substrates: B_match = {B[j]*1e6:.2f} uT, "
          f"P per 15N per event = {p2[j]:+.3f}, FWHM {out['two_substrates']['fwhm_uT']:.2f} uT")

    with open("results/exp4_sabre.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
