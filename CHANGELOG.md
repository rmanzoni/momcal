# Changelog

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
