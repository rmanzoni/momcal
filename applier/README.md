# rds_trkcalib — track momentum calibration for R(Ds) MC

Corrects the momentum of every MC charged track (kaons, pion, muon) so that the MC Ds → φ(KK)π peak matches 2018 BParking data:

```
k' = k (1 + A - e*k + q*M/k) (1 + sqrt(a^2 + c^2/k^2) z),   k = 1/pT
pT' = pT / [ (1 + A - e/pT + q*M*pT) (1 + sqrt(max(a2 + c2*pT^2, 0)) z) ]
```

This is the convention of the CMS W-mass measurement ([arXiv:2412.13872](https://arxiv.org/abs/2412.13872), Eqs. 2–3). The terms are A (field, scale), e = ε (energy loss, GeV), M (sagitta, GeV⁻¹, sign follows the track charge q), a (multiple scattering) and c (hit resolution, GeV⁻¹). Here (A, ε, M) describe data relative to MC, and (a, c) the resolution MC is missing. The file stores a2 = a² and c2 = c² [GeV⁻²], so a resolution variation can cross zero. Files of schema `rds_trkcalib/2` use this convention.

The v1 file (schema `rds_trkcalib/1`) uses the first convention, p' = p (1 + s0 + s1/p + k q pT)(1 + sqrt(v0·1e-6 + v1·1e-8 pT²) z). Both appliers read both schemas and pick the formula from the file.

Only the magnitude of the momentum changes; the direction stays. **Apply to MC only.** Parameters per |η| bin (0–0.7, 0.7–1.2, 1.2–2.5; beyond 2.5 uses the last bin). How the correction was derived is explained in the document *Track momentum calibration from the Ds peak: how it works*.

## Contents

| Path | What |
| --- | --- |
| `data/rds_trkcalib_2018BParking_v2.json` | the current correction (CMS convention): per-bin values, fit values + covariance, 11 eigen-variations, random-number convention, derivation record |
| `data/rds_trkcalib_2018BParking_v1.json` | the first correction, old convention (kept for comparison) |
| `python/rds_trkcalib.py` | Python applier (numpy only), vectorised |
| `python/example_uproot.py` | corrects a flat ntuple and writes a friend tree (tested) |
| `cpp/RDsTrackCalib.h` | C++14 header-only applier, no dependencies (own JSON reader) |
| `cpp/example_standalone.cpp` | one candidate + muon, compiled and run (tested with g++ and clang++) |
| `cpp/example_rdf.C` | ROOT RDataFrame macro (written for ROOT ≥ 6.24; not run here, no ROOT in the test area) |
| `test/crosscheck.py` | builds the C++ test, runs C++ and Python on 200 000 tracks, compares everything |

## Use

Python:
```python
from rds_trkcalib import TrackCalib
cal = TrackCalib("data/rds_trkcalib_2018BParking_v2.json")
z = cal.gauss(run, lumi, event, track_id)                  # fixed per track
f = cal.factor(p, pt, eta, q, z)                           # p'/p ; arrays or scalars
f_up = cal.factor(p, pt, eta, q, z, variation="eig3_up")
```

C++:
```cpp
#include "RDsTrackCalib.h"
rdscalib::TrackCalib cal("rds_trkcalib_2018BParking_v2.json");   // build once, e.g. in the constructor of the producer
double z = cal.gauss(run, lumi, event, trackIndex);
double f = cal.factor(p, pt, eta, q, z);                   // multiply px, py, pz by f, recompute E with the mass hypothesis
```

In a CMSSW ntuplizer the natural key is the index of the track in its input collection, for example:
```cpp
const auto& id = iEvent.id();
double z = cal.gauss(id.run(), id.luminosityBlock(), id.event(), iPFCand);          // packedPFCandidates index
double zLost = cal.gauss(id.run(), id.luminosityBlock(), id.event(), 1000000 + iLost); // offset for lostTracks
```
Any key works as long as the same physical track gets the same key in every candidate of the event. Flat ntuples without a track index can use `track_id_from_kinematics(pt, eta, phi)` / `rdscalib::trackIdFromKinematics` (pT rounded to 1 MeV, η and φ to 1e-4).

## Variations

| Name | Meaning |
| --- | --- |
| `nominal` | the fit result |
| `eig0_up/down` … `eig10_up/down` | ±1σ along each eigenvector of the 11×11 fit covariance; uncorrelated, add the effects in quadrature |
| `scale_only` | scale terms only (A, ε, M), no extra smearing (for checks) |
| `smear_only` | extra smearing only (a, c), no scale change (for checks) |

These are statistical uncertainties of the fit only. Systematic variations (kernel, mass window, background shape, category edges, D⁺-veto dependence of A) are not included yet.

## The muon

Use the same formula with the muon's own pT, charge and |η|. In the validation on the slim ntuples the muon direction was not available and its |η| bin was taken from the Ds direction; at ntuple production use the true muon |η|. The muon is harder than most of the 1–10 GeV hadron tracks that fix the parameters, so its correction is an extrapolation.

## Checks done

- C++ vs Python, both correction files, 200 000 random tracks including bin edges, |η| > 2.5 and negative event numbers: the Gaussian z agrees to 4e-16 and the factor to 2e-16 for all 25 variations; the kinematic track id is identical.
- Applier vs fitter, 3.4 M signal-MC tracks with the fitter's z: factors identical to 2e-16.
- v2 vs v1 on 3.36 M MC tracks: scale factors differ by 2×10⁻⁵ on average; corrected m(KKπ) mean 1967.178 vs 1967.181 MeV, rms 9.655 vs 9.656 MeV (with the D⁺ veto).
- Hash-based z vs the fitter's numpy z on the same MC (v1): corrected m(KKπ) mean 1967.853 vs 1967.854 MeV, rms 9.830 vs 9.830 MeV (1945–1991 MeV window; uncorrected 1968.727 / 9.695).

Run `python test/crosscheck.py` after any change (needs numpy and a C++14 compiler; set `CXX` to choose it).
