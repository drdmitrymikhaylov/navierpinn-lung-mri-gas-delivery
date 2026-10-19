# Changelog

## 2026-10-19

- Section 1's "largest single source of the spread is the rubidium density"
  was a correlation ranking. The same 400 draws re-run with each constant
  held at nominal: the Rb–Xe spin-destruction rate (fourth by ρ) removes
  29 % of the 90 % width and the density factor 24 %, from opposite ends;
  40 % of the variance each; four constants carry 92 % of the width, the
  other five ≤ 1 %. Knowing κ·f to ±10 % (what a build-up series measures)
  removes 32 %. 67 % of draws fall below nominal. Finding 3 amended, new
  subsection "Which measurement would buy the most",
  `results/uncertainty_fixing.json` (script private, with exp1).
- Section 3's one-at-a-time ratios do not multiply: the five compatible
  repairs together deliver 64 % (×2.74), not the ×3.52 product; the two
  worst rows together 3.6 % (×0.15 vs ×0.31). Upstream at its best passes
  96 %; the lung alone caps delivery at 67 % for 8 s, 47 % for 15 s. New
  subsection "All at once" with the floor table, finding 6 amended,
  `src/budget_extremes.py`, `results/budget_extremes.json`.
- `tests/test_second_pass.py` (2 tests) pins both; three tracked `._*`
  files removed and ignored.

## 2026-09-23

- Section 5 read replicate by replicate: on the laser-starved series the
  P_Rb(T) PINN returns κ below the truth in 5/5 runs (0.81–0.93) and Γ in
  4/5 — the 0.88 is a bias, not scatter; on the right model it straddles
  (3/5). The data χ²/dof that flags the wrong model for least squares (123
  vs 0.98) is 0.10–0.45 on all 20 PINN runs, right model or wrong, and the
  physics residual is *lower* on the wrong model — the repair removes the
  diagnostic. New subsection "Reading the five replicates one by one".
- Section 3: per-minute cost of each stage (cryogenic 0.7 %/min, bag 3.9,
  transport 4.1 — same T1 to 4 % — lung 4.9 %/s); 7 s of breath-hold = 8 min
  of transport. Flip-angle correction: the variable T1-aware schedule's mean
  signal is 6 % *higher* than constant 15° (0.097 vs 0.091), not lower; it
  gives up the first line (0.37×) and summed energy (0.60 vs 0.82). Leaking
  valve row ×0.005 → ×0.004.
- Added `src/replicate_checks.py`, `results/replicate_checks.json`,
  `tests/test_readme_numbers.py` (7 tests).
