# Changelog

## 1.0.0 — 24 Oct 2026
- First release: produced the v1 correction (`applier/data/rds_trkcalib_2018BParking_v1.json`).
- Model: p(1 + s0 + s1/p + k q pT)(1 + sqrt(v0 1e-6 + v1 1e-8 pT^2) z); s0, k, v0 per |eta| bin (0-0.7-1.2-2.5); s1, v1 shared.
- 18 categories (max |eta| x softest-track pT x Ds charge); cubic-kernel templates and data; one Minuit fit of all parameters.
- Config-driven pipeline (`scalesmear run`), provenance record, input cache, closure toys (`toy`/`toys`), correction export.
- D+ -> K pi pi reflection veto; m(KK) window on data (stored) and MC (corrected).
- Option `fit.z_mode: hash` (order-independent template random numbers); default stays `sequential` for v1.
- Appliers: header-only C++ and numpy Python, identical to 2e-16; hash-based per-track z.
