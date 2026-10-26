# Changelog

## 1.2.0 — 26 Oct 2026
- v3 correction (`applier/data/rds_trkcalib_2018BParking_v3.json`, configuration `configs/rds2018_v3.yaml`):
  - categories in kappa_eff, the candidate's sensitivity to e relative to A (5 bins x max|eta| x charge = 30), one background
    shape per category. Errors relative to v2: A 0.49-0.81, e 0.39, M 0.89-0.95; corr(A, e) 0.96/0.90/0.74 -> 0.81/0.62/0.33;
  - per-track random numbers from the track's own kinematics (`fit.z_mode: hash`): independent of candidate order;
  - forward a2 fixed at 0 (not determined by the data; -2lnL +0.6 vs free, 1 dof; tying it to the 0.7-1.2 bin: +5.1);
  - variances a2, c2 >= 0 (`model.limits`): a negative variance let a2 + c2 pT^2 reach 0, a kink that made fits invalid.
- New options: `categories.kappa_edges`, `categories.share_bkg_shape`, `categories.eta_edges`, `model.fix`, `model.ties`,
  `model.limits`, `fit.mc_bootstrap_seed`, `toys.template_frac`.
- Invalid minima are retried from their end point; Minuit diagnostics are stored in result.json (`config.minuit`).
- Correction files drop null variations (fixed terms).
- Checked and not adopted: one background shape shared across kappa bins (the wrong-sign background changes with kappa_eff);
  a separate M for 1.4 < |eta| < 2.5 (-2lnL -0.08 for one more parameter).

## 1.1.0 — 25 Oct 2026
- Convention of the CMS W-mass measurement (arXiv:2412.13872, Eqs. 2-3), now the default (`model.convention: cms`):
  k' = k (1 + A - e k + q M/k)(1 + sqrt(a^2 + c^2/k^2) z), k = 1/pT. Terms A, e (epsilon), M, a, c with CMS definitions and
  units; (A, e, M) give data relative to MC, (a, c) the missing resolution. Fit variables a2 = (a/1e-3)^2, c2 = (c/1e-4/GeV)^2.
- The template's curvature fluctuation is -z (z = the momentum-space draw), so v1 and v2 share one smearing realisation.
- a2 lower limit opened to -20 (as c2): a limit at 0 made HESSE invalid when the forward a2 sits near 0.
- v2 correction (`applier/data/rds_trkcalib_2018BParking_v2.json`, schema rds_trkcalib/2): same inputs as v1; minimum equal
  to v1 within 0.02 in -2lnL; every term within 0.5 sigma of the first-order conversion A=-s0, M=-k, e~s1 pT/p, a=r0, c=r1.
- Appliers read schema 2 and 1. New plot fit_components.png (data, total fit, signal, background, pulls).
- `model.convention: v1` keeps the first convention; `configs/rds2018_v1.yaml` still reproduces v1 exactly.

## 1.0.0 — 24 Oct 2026
- First release: produced the v1 correction (`applier/data/rds_trkcalib_2018BParking_v1.json`).
- Model: p(1 + s0 + s1/p + k q pT)(1 + sqrt(v0 1e-6 + v1 1e-8 pT^2) z); s0, k, v0 per |eta| bin (0-0.7-1.2-2.5); s1, v1 shared.
- 18 categories (max |eta| x softest-track pT x Ds charge); cubic-kernel templates and data; one Minuit fit of all parameters.
- Config-driven pipeline (`scalesmear run`), provenance record, input cache, closure toys (`toy`/`toys`), correction export.
- D+ -> K pi pi reflection veto; m(KK) window on data (stored) and MC (corrected).
- Option `fit.z_mode: hash` (order-independent template random numbers); default stays `sequential` for v1.
- Appliers: header-only C++ and numpy Python, identical to 2e-16; hash-based per-track z.
