# URF-Jacobi Letter — Replication Data Archive

**Companion repository to the Letter:**

> **Jacobi instability on $\mathrm{SPD}(n)$ as a precursor of systemic financial transitions**
> Evangelos Papadopoulos, *Physics Letters A*, 2026

This repository hosts the **computed outputs** required to independently verify the empirical claims of the Letter — per-crisis geometric indicator timelines, Lean 4 formal proofs, and the public S&P 500 price series used for the rupture-trigger robustness check.

It does **not** contain the implementation pipeline (the proprietary code that produces the geometric indicators from raw market data). The methodology is fully described in §2–3 of the Letter; what this archive provides is the **machine-readable data + verification harness** so that any reader can confirm the published claims from the data alone.

---

## Contents

```
data/
  forensic_dotcom_2000.json      Per-crisis forensic timeline: D, R, Ric, TSS, FCI, Dp per day + events
  forensic_gfc_2007.json
  forensic_covid_2020.json
  forensic_tradewar_2026.json
  sp500_close.csv                Frozen S&P 500 daily close prices 1998–2026 (Yahoo Finance ^GSPC)
  data_manifest.csv              Per-crisis metadata (window, rupture event, lead days, source)

lean/
  urfTheory.lean                 Suture / Rupture / Lyapunov stability theorems (Thm 10.1, 10.2, 12.1)
  urfWizard.lean                 Auxiliary lemmas

verify.py                        Loads the data archive and asserts every published claim of §4
VERIFY_BASELINE.txt              Reference output of verify.py (for diff-based reproducibility)
requirements.txt                 Python dependencies (numpy, scipy, pandas, scikit-learn, tabulate)

figures/                         Pre-generated PNG figures referenced in the Letter
```

---

## Reproducing the Letter's claims

```bash
# 1. Clone
git clone https://github.com/evangelos55/urf-jacobi-replication.git
cd urf-jacobi-replication

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the verifier
python verify.py
```

`verify.py` loads the data files and asserts the empirical claims of the Letter from §4:

| Test | Reproduces | Letter §       |
|------|------------|----------------|
| 1    | Mean lead time + 95 % bootstrap CI (block bootstrap, 10 000 resamples)              | §4.2   |
| 2    | Bonferroni-corrected Mann–Whitney $U$-test on R(t), TSS(t), Dp(t), FCI(t)           | §4.2   |
| 3    | ROC / AUC analysis — included for transparency, not a §4 claim                       | (ref)  |
| 4    | False-positive rate at the operational $D > 0.3$ threshold                           | §4.4   |
| 5    | Rupture-trigger robustness (alternate price-rupture markers vs baseline lead time)  | §4.4   |
| 6    | Price-side blindness signature — Minsky Singularity (S&P ATH inside $[t^*, t_c]$)    | §4.5   |

A green print-out confirms that every published number can be derived from the
data files alone. Reference output is in `VERIFY_BASELINE.txt`.

---

## Lean 4 formal verification

The mathematical-stability theorems underlying URF-3 / URF-4 (Suture
Convergence, Rupture Divergence, Lyapunov Energy Decay) are formally verified
in Lean 4 / Mathlib. To audit them:

```bash
# Requires Lean 4 + Mathlib via elan
cd lean
lake build
lake env lean --run urfTheory.lean   # prints axiom footprint per theorem
```

Each theorem closes without any `sorry` placeholder. The `#print axioms`
footprint contains only the standard Lean / Mathlib axioms (propositional
extensionality, classical choice, quotient soundness).

---

## What is **not** in this archive

- **Implementation pipeline.** The end-to-end code that ingests raw market
  data, computes the SPD(5) covariance, derives the geometric indicators and
  calibrates the thresholds is **proprietary** to the author and his commercial
  entities.
- **Production deployment.** Real-time inference, dashboards, alerts, and
  multi-tenant deployment of the geometric indicators (the *Econosysmographe™*
  platform) are commercial products, not part of this academic archive.

The Letter's methodology section (§2–3) describes the construction completely;
together with the data files in this archive, every empirical claim of the
Letter is independently verifiable. Re-implementation for academic use is
welcome under the licence terms below.

---

## Citation

If you use this archive in academic work, please cite the Letter:

```bibtex
@article{papadopoulos2026jacobi,
  author  = {Papadopoulos, Evangelos},
  title   = {Jacobi instability on {SPD}($n$) as a precursor of systemic financial transitions},
  journal = {Physics Letters A},
  year    = {2026}
}
```

A citable Zenodo DOI for this archive will be added upon Letter publication.

---

## Licence

**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).**

This archive is released for the **sole purpose of independent scientific
validation** of the Letter's empirical claims. Commercial use of the
methodology, the indicator formulae, the calibrated thresholds, or any derived
production system is reserved to the author. Modifications and redistribution
of modified versions are not permitted. See [`LICENSE`](LICENSE) for the full
terms.

For commercial licensing of the URF / Econosysmographe™ methodology:
**contact@econosysmographe.eu**

---

## Contact

Evangelos Papadopoulos
Independent Researcher, London, United Kingdom
contact@econosysmographe.eu
