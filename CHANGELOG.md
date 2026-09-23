# Changelog

## 2026-09-23

- Section 5 read replicate by replicate: on the laser-starved series the
  P_Rb(T) PINN returns κ below the truth in 5/5 runs (0.81–0.93) and Γ in
  4/5 — the 0.88 is a bias, not scatter; on the right model it straddles
  (3/5). The data χ²/dof that flags the wrong model for least squares (123
  vs 0.98) is 0.10–0.45 on all 20 PINN runs, right model or wrong, and the
  physics residual is *lower* on the wrong model — the repair removes the
  diagnostic. New subsection "Reading the five replicates one by one".
- Section 3: per-minute cost of each stage (cryogenic 0.7 %/min, bag 4.0,
  transport 4.1 — same T1 to 4 % — lung 4.9 %/s); 7 s of breath-hold = 8 min
  of transport. Flip-angle correction: the variable T1-aware schedule's mean
  signal is 6 % *higher* than constant 15° (0.097 vs 0.091), not lower; it
  gives up the first line (0.37×) and summed energy (0.60 vs 0.82). Leaking
  valve row ×0.005 → ×0.004.
- Added `src/replicate_checks.py`, `results/replicate_checks.json`,
  `tests/test_readme_numbers.py` (7 tests).
