# rds_momcalib — track momentum calibration for R(Ds), 2018 BParking

This repository fits, exports and applies a per-track momentum correction for MC. The correction is derived from the Ds → φ(KK)π peak and written in the convention of the CMS W-mass measurement ([arXiv:2412.13872](https://arxiv.org/abs/2412.13872), Eqs. 2–3), on the curvature k = 1/pT:

```
k' = k (1 + A - e*k + q*M/k) (1 + sqrt(a^2 + c^2/k^2) z)
pT' = pT / [ (1 + A - e/pT + q*M*pT) (1 + sqrt(a^2 + c^2 pT^2) z) ]      direction unchanged
```

A (field), e = ε (energy loss, GeV), M (sagitta, 1/GeV), a (multiple scattering), c (hit resolution, 1/GeV), as in CMS. Here (A, ε, M) describe data relative to MC, and (a, c) the resolution MC is missing. A, M and a are per |η| bin (0–0.7–1.2–2.5); ε and c are shared. The first correction (v1) used p' = p (1 + s0 + s1/p + k q pT)(1 + sqrt(r0² + r1² pT²) z), which equals the CMS form to first order with A = −s0, M = −k, ε ≈ s1·pT/p, a = r0, c = r1. It can still be selected with `model.convention: v1`.

| Folder | What |
| --- | --- |
| `src/scalesmear/` | the fitter (Python package, command `scalesmear`) |
| `configs/` | `rds2018_v3.yaml`: the current correction (κ_eff categories, CMS convention); `rds2018_v2.yaml`, `rds2018_v1.yaml`: earlier corrections, kept reproducible; `quick.yaml` checks the full chain in minutes |
| `applier/` | the correction files `rds_trkcalib_2018BParking_v3.json` (current), `_v2.json` and `_v1.json`, plus code that applies them in C++ (header only) and Python. How to use them in an ntuplizer: Section 6 below; details in `applier/README.md` |
| `tests/` | pytest suite: end-to-end on synthetic ntuples, applier = fitter, D⁺ veto, configuration |
| `scripts/` | environment setup, v1 reproduction, result comparison |
| `psi_t3/` | Slurm scripts for the PSI Tier-3 (one fit; toy arrays) |

The method is explained in the document *Track momentum calibration from the Ds peak: how it works*.

## 1. Install (laptop or Tier-3), once

```bash
git clone <this repository> rds_momcalib && cd rds_momcalib
scripts/setup_env.sh                # .venv with the pinned versions (requirements-lock.txt), then runs the tests (~15 s)
source .venv/bin/activate
```

The setup needs Python ≥ 3.11; point to one with `PYTHON=/path/to/python3.11 scripts/setup_env.sh`. Two alternatives:

- **conda / mamba:** `micromamba create -f environment.yml && micromamba activate rds-momcalib && pip install -e . --no-deps`.
- **Older Python (≥ 3.9):** `pip install -e .[test]` installs the minimum versions. Results then agree only to numerical precision, not bit for bit.

## 2. Run

```bash
export RDS_NTUPLES=~/Documents/RDs/understand_comb_slope/ntuples     # folder with rs_/sig_slim_21oct2026.root
scalesmear run configs/quick.yaml          # ~1–2 min smoke test on 150k data / 400k MC candidates (not physics)
scalesmear run configs/rds2018_v3.yaml     # the reference fit: ~30–60 min on one core, ~2 GB memory
```

Change any setting without editing the file: `--set fit.seed=7 --set selection.dplus_veto=0.04 --set output.dir=results/veto40`. `scalesmear config configs/rds2018_v3.yaml` prints the full configuration, including defaults.

Outputs go to `output.dir` (default `results/<name>_<version>/`):

| File | Content |
| --- | --- |
| `rds_trkcalib_<name>_<version>.json` | the correction file read by the appliers |
| `result.json` | all fitted values and errors (including the nuisances), covariance of the 11 terms, χ² per category, configuration, selection efficiencies |
| `fit_components.png`, `lineshape.png`, `terms.png`, `fit_categories.png` | the fit (data, total, signal, background, pulls) per \|η\| region; background-subtracted peak before/after; the 11 terms; all 18 categories with pulls |
| `provenance.json` | package version and git commit, library versions, host, Slurm job, command, input file fingerprints (size, mtime, sha256 of first and last 16 MB) |
| `config_used.yaml`, `log.txt`, `status.json` | the exact configuration, the full log, and a heartbeat with the number of likelihood calls (look here while the fit runs) |
| `cache/` | the selected tracks as `.npz`, reused when inputs and selection are unchanged |

## 3. PSI Tier-3

```bash
cd /work/$USER/rds_momcalib && scripts/setup_env.sh && mkdir -p logs
export RDS_NTUPLES=/work/$USER/rds/ntuples
sbatch psi_t3/fit.sbatch configs/rds2018_v3.yaml
sbatch --array=1-30 psi_t3/toys.sbatch configs/rds2018_v3.yaml        # closure toys, one per task
scalesmear toys configs/rds2018_v3.yaml                               # afterwards: pull means and widths
```

The scripts request partition `standard`, 4 h, 1 core and 8 GB. These values have not been tested on the Tier-3, so check them with `sinfo` and adjust the `#SBATCH` lines. Each toy uses a random 20% of the selected MC as pseudo-data. That part gets the correction listed in `toys.inject` in the configuration, plus 50% exponential background. The other 80% of the MC is the template. Pulls should have mean ≈ 0 and width ≈ 1.

## 4. Reproducibility

- **What fixes the result:** the configuration file, the input files and the package version. The per-track Gaussian numbers of the template come from `fit.seed`, and the MC order follows the file order. Rerunning the same configuration on the same inputs with the pinned versions gives the same numbers.
- **One trap:** with `fit.z_mode: sequential` (v1), the template's Gaussian numbers are drawn in candidate order. Adding or removing even one MC candidate therefore changes the numbers of all later tracks. The fit stays statistically equivalent, but its values move at the level of the MC statistics. Example: applying the selection again to the float32 `.npz` extracts drops 5 of 4.2 M candidates at the cut boundaries. `fit.z_mode: hash` computes each track's z from its own kinematics, the convention the appliers use, so it does not depend on order or selection. It is the better choice for new versions; v1 keeps `sequential` so that it reproduces exactly.
- **What is recorded:** every run writes `provenance.json` with the git commit (and whether the tree had uncommitted changes), the library versions and the input fingerprints. `result.json` carries a `physics_hash` of all settings that change the result.
- **Reproducing v1:** `RDS_NTUPLES=... scripts/reproduce_v1.sh` refits and runs `scripts/compare_results.py` against `applier/data/rds_trkcalib_2018BParking_v1.json`. The script prints each term's difference in σ and fails above 0.01σ.
- **Checks done:** the ROOT ntuples give exactly the candidates the v1 fit used (1 105 575 data and 4 215 253 MC after the selection, same order, same values). Rerunning the pipeline on these inputs gives the v1 values, errors and covariance **identically** (0.0000σ). v2 (CMS convention) reaches the same minimum (Δ(−2 ln L) = 0.02), and every term is within 0.5σ of the converted v1 value. v3 changes the categories and the random numbers, so it is a new fit, not a re-expression: see `CHANGELOG.md`. See `REPRODUCTION.md` and `reference/v1/`, `reference/v2/`, `reference/v3/`.

## 5. Configuration keys

| Key | v3 value | Meaning |
| --- | --- | --- |
| `selection.expression` | cos working point | numpy expression over branch names, applied while reading |
| `selection.kk_window` | 0.020 GeV | \|m(KK) − m(φ)\| window: on stored m(KK) for data; on corrected m(KK) for the MC template |
| `selection.dplus_veto` | 0.050 GeV | removes \|m(Kππ) − m(D⁺)\| < value, in data and MC |
| `model.eta_edges` | 0, 0.7, 1.2, 2.5 | \|η\| bins of the parameters |
| `model.convention` | cms | `cms` (A, e, M, a2, c2) or `v1` (s0, s1, k, v0, v1) |
| `model.terms` | A, e, M, a2, c2 | terms fitted; a2 = (a/1e-3)², c2 = (c/1e-4 GeV⁻¹)² in the fit |
| `model.shared` | e, c2 | terms with one value for all bins |
| `model.start` | A 1e-3, a2 11 | starting values, per term or per parameter (`A_2`) |
| `model.fix` | a2_2 = 0 | terms held fixed (the forward a² is not determined by the data) |
| `model.limits` | a2, c2 ≥ 0 | replaces the default limits (a negative variance creates a kink where a² + c²pT² reaches 0) |
| `model.ties` | — | {parameter: parameter it equals}, e.g. {a2_2: a2_1} |
| `categories.kappa_edges` | −100, 0.15, 0.26, 0.37, 0.52, 100 GeV⁻¹ | bins of κ_eff, the candidate's sensitivity to ε relative to A (quintiles of the signal MC); if unset, `categories.pt_edges` (softest-track pT, v1/v2) is used |
| `categories.share_bkg_shape` | false | one background shape per (\|η\|, charge) instead of per category; wrong for κ categories (the background shape changes with κ_eff) |
| `categories.eta_edges` | (model's) | \|η\| edges of the categories, if different from the parameters' |
| `categories.split_charge` | true | Ds⁺ / Ds⁻ categories (needed for k) |
| `fit.mass_range`, `fit.bin_width` | 1.920–2.015, 0.001 GeV | binned likelihood range and bins |
| `fit.background` | expo2 | exp(λx + κx²) per category |
| `fit.seed` | 12345 | per-track z of the MC template |
| `fit.z_mode` | hash | `hash` (from the track itself; order independent) or `sequential` (numpy, candidate order; v1, v2) |
| `fit.mc_bootstrap_seed` | — | Poisson(1) weights on the MC, for MC-template uncertainty studies |
| `correction.seed` | 20181 | seed of the hash-based z used by the appliers |
| `toys.*` | | closure-toy settings |

## 6. Apply the correction when producing ntuples

The correction is applied to **MC only**, to every charged track the analysis uses: kaons, pion and muon, each with its own pT, |η| and charge. Two appliers give identical numbers (to 2·10⁻¹⁶, checked by `applier/test/crosscheck.py`):

- **Python:** `applier/python/rds_trkcalib.py`, one file, needs only numpy.
- **C++:** `applier/cpp/RDsTrackCalib.h`, header only, C++14, no dependencies (it has its own JSON reader).

Both read the correction file `applier/data/rds_trkcalib_2018BParking_v3.json`. Copy the applier and the JSON file next to the ntuplizer; nothing else from this package is needed.

### 6.1 What to do for each MC track

1. Take the track's pT, p, η and charge q.
2. Get its Gaussian number z from a key that identifies the track in the event (Section 6.2).
3. `f = factor(p, pT, η, q, z)` is p'/p. Multiply px, py and pz by f; the direction does not change.
4. Recompute the energy with the track's mass hypothesis, then every quantity built from the momenta: masses, q², missing mass, etc.

### 6.2 The random-number key

`z = gauss(run, lumi, event, key)`. The same physical track must get the same z in every candidate of the event, so the key identifies the track, not the candidate. Two options:

- **The track's index in its input collection**, for example the `packedPFCandidates` index. Add an offset for each further collection so that keys do not collide, for example 1 000 000 + index for `lostTracks`.
- **The track's kinematics**, `track_id_from_kinematics(pt, eta, phi)` in Python or `rdscalib::trackIdFromKinematics(pt, eta, phi)` in C++. It rounds pT to 1 MeV and η, φ to 10⁻⁴. The fitter (`fit.z_mode: hash`) and `example_uproot.py` use this key.

**Store z per track in the ntuple** (or the factor f for the nominal correction). With the raw momenta and z stored, every variation can be recomputed offline without reprocessing. Recomputing the kinematic key offline from float32 branches can, in rare cases, round differently from the double values seen in CMSSW and give a different z.

### 6.3 Python

```python
import numpy as np
from rds_trkcalib import TrackCalib, track_id_from_kinematics

cal = TrackCalib("rds_trkcalib_2018BParking_v3.json")      # once, before the event loop

# in the event loop, MC only
aux = event.eventAuxiliary()                                # FWLite; any source of run, lumi, event works
run, lumi, evt = aux.run(), aux.luminosityBlock(), aux.event()
for trk, mass in tracks:                                    # e.g. [(k1, M_K), (k2, M_K), (pi, M_PI), (mu, M_MU)]
    pt, eta, phi, q = trk.pt(), trk.eta(), trk.phi(), trk.charge()
    p = pt * np.cosh(eta)
    z = cal.gauss(run, lumi, evt, track_id_from_kinematics(pt, eta, phi))
    f = cal.factor(p, pt, eta, q, z)                        # p'/p, a float
    px, py, pz = trk.px() * f, trk.py() * f, trk.pz() * f
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    # optional, per variation: cal.factor(p, pt, eta, q, z, "eig0_up"), ...
```

- All functions also take numpy arrays, which is much faster for flat ntuples than a loop over tracks.
- `cal.variation_names` lists the 43 variations of v3: `nominal`, `scale_only`, `smear_only`, `eig0_up` … `eig9_down` (statistical), `mctemplate_<term>_up/down` (MC template). `applier/README.md` explains how to combine them.
- `cal.correct_p3(px, py, pz, q, z)` and `cal.correct_ptetaphi(pt, eta, phi, q, z)` return the corrected momentum directly.
- For an existing flat ntuple, `applier/python/example_uproot.py` writes the factors and the corrected m(KKπ), m(KK) to a friend tree. Edit the branch names in `TRACKS` and `MUON` at its top, then:
  ```bash
  python applier/python/example_uproot.py mc.root tree mc_trkcalib.root --variations all
  ```

### 6.4 C++

Standalone test (one candidate and a muon):
```bash
g++ -O2 -std=c++14 -I applier/cpp applier/cpp/example_standalone.cpp -o example
./example applier/data/rds_trkcalib_2018BParking_v3.json
```

In a CMSSW module: put `RDsTrackCalib.h` in `<Package>/interface/` and the JSON file in `<Package>/data/`. `TrackCalib` has no default constructor, so build it in the module's constructor:
```cpp
#include "<Package>/interface/RDsTrackCalib.h"

// member
std::unique_ptr<const rdscalib::TrackCalib> trkCalib_;

// constructor
trkCalib_ = std::make_unique<const rdscalib::TrackCalib>(
    edm::FileInPath(iConfig.getParameter<std::string>("trkCalibFile")).fullPath());
// python config: trkCalibFile = cms.string("<Package>/data/rds_trkcalib_2018BParking_v3.json")

// produce / analyze, MC only
if (!iEvent.isRealData()) {
  const auto& id = iEvent.id();
  for (size_t i = 0; i < pfCands->size(); ++i) {               // the tracks the candidates are built from
    const pat::PackedCandidate& c = (*pfCands)[i];
    const double z = trkCalib_->gauss(id.run(), id.luminosityBlock(), static_cast<int64_t>(id.event()), i);
    const double f = trkCalib_->factor(c.p(), c.pt(), c.eta(), c.charge(), z);
    // corrected momentum: (c.px() * f, c.py() * f, c.pz() * f); energy from the mass hypothesis
    // optional: trkCalib_->factor(c.p(), c.pt(), c.eta(), c.charge(), z, "eig0_up"), ...
  }
}
```

- Every method is `const`, so one instance can be shared by all streams of a multi-threaded job.
- Muons from the muon collection: same call with the muon's pT, η, charge, and a key with its own offset, e.g. 2 000 000 + muon index.
- `trkCalib_->variations()` lists the variation names; `trkCalib_->correctedPt(pt, eta, q, z)` returns the corrected pT directly.

ROOT RDataFrame on a flat ntuple (edit the branch names in the `Define` strings first; this macro has not been run yet, as there is no ROOT in the test area):
```bash
cd applier/cpp
root -l -b -q 'example_rdf.C("mc.root", "tree", "mc_trkcalib.root", "../data/rds_trkcalib_2018BParking_v3.json", true)'
```
The last argument says the ntuple has `run`, `lumi` and `event` branches; with `false` the entry number is used as the event key.

### 6.5 Vertex fits

Scaling the 4-vectors corrects every quantity computed from them. A vertex or kinematic fit made in the ntuplizer, however, reads the track parameters. For the correction to reach the fit results (refitted momenta, constrained masses), scale the track's q/p by 1/f, and the q/p row and column of its covariance by 1/f, before building the transient track. Otherwise the refitted quantities stay uncorrected.

### 6.6 Checks after integration

- Run `python applier/test/crosscheck.py` after any change to the appliers; it needs numpy and a C++14 compiler (set `CXX` to choose it).
- In a few events, print pT, η, q, z and f from the ntuplizer, and recompute f offline with the Python applier from the same inputs and z. They must agree to about 10⁻¹⁵.
- With v3, the corrected MC m(KKπ) peak moves down by about 1 MeV (−1.30 MeV on average in the inclusive Hb MC used for validation).

## Versions

See `CHANGELOG.md`. Release 1.0.0 produced the v1 correction, 1.1.0 the v2 correction (CMS convention), 1.2.0 the v3 correction (κ_eff categories).
