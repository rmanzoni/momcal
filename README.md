# rds_momcalib — track momentum calibration for R(Ds), 2018 BParking

This repository fits, exports and applies a per-track momentum correction for MC. The correction is derived from the Ds → φ(KK)π peak:

```
p' = p * (1 + s0 + s1/p + k*q*pT) * (1 + sqrt(max(v0*1e-6 + v1*1e-8*pT^2, 0)) * z)
```

| Folder | What |
| --- | --- |
| `src/scalesmear/` | the fitter (Python package, command `scalesmear`) |
| `configs/` | `rds2018_v1.yaml` is the reference configuration of the v1 correction; `quick.yaml` checks the full chain in minutes |
| `applier/` | the correction file `rds_trkcalib_2018BParking_v1.json`, plus code that applies it in C++ (header only) and Python. See `applier/README.md` |
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
scalesmear run configs/rds2018_v1.yaml     # the reference fit: ~1 h on one core, ~4 GB memory
```

Change any setting without editing the file: `--set fit.seed=7 --set selection.dplus_veto=0.04 --set output.dir=results/veto40`. `scalesmear config configs/rds2018_v1.yaml` prints the full configuration, including defaults.

Outputs go to `output.dir` (default `results/<name>_<version>/`):

| File | Content |
| --- | --- |
| `rds_trkcalib_<name>_<version>.json` | the correction file read by the appliers |
| `result.json` | all fitted values and errors (including the nuisances), covariance of the 11 terms, χ² per category, configuration, selection efficiencies |
| `lineshape.png`, `terms.png`, `fit_categories.png` | background-subtracted peak before/after per \|η\| region; the 11 terms; all 18 categories with pulls |
| `provenance.json` | package version and git commit, library versions, host, Slurm job, command, input file fingerprints (size, mtime, sha256 of first and last 16 MB) |
| `config_used.yaml`, `log.txt`, `status.json` | the exact configuration, the full log, and a heartbeat with the number of likelihood calls (look here while the fit runs) |
| `cache/` | the selected tracks as `.npz`, reused when inputs and selection are unchanged |

## 3. PSI Tier-3

```bash
cd /work/$USER/rds_momcalib && scripts/setup_env.sh && mkdir -p logs
export RDS_NTUPLES=/work/$USER/rds/ntuples
sbatch psi_t3/fit.sbatch configs/rds2018_v1.yaml
sbatch --array=1-30 psi_t3/toys.sbatch configs/rds2018_v1.yaml        # closure toys, one per task
scalesmear toys configs/rds2018_v1.yaml                               # afterwards: pull means and widths
```

The scripts request partition `standard`, 4 h, 1 core and 8 GB. These values have not been tested on the Tier-3, so check them with `sinfo` and adjust the `#SBATCH` lines. Each toy uses a random 20% of the selected MC as pseudo-data. That part gets the correction listed in `toys.inject` in the configuration, plus 50% exponential background. The other 80% of the MC is the template. Pulls should have mean ≈ 0 and width ≈ 1.

## 4. Reproducibility

- **What fixes the result:** the configuration file, the input files and the package version. The per-track Gaussian numbers of the template come from `fit.seed`, and the MC order follows the file order. Rerunning the same configuration on the same inputs with the pinned versions gives the same numbers.
- **One trap:** with `fit.z_mode: sequential` (v1), the template's Gaussian numbers are drawn in candidate order. Adding or removing even one MC candidate therefore changes the numbers of all later tracks. The fit stays statistically equivalent, but its values move at the level of the MC statistics. Example: applying the selection again to the float32 `.npz` extracts drops 5 of 4.2 M candidates at the cut boundaries. `fit.z_mode: hash` computes each track's z from its own kinematics, the convention the appliers use, so it does not depend on order or selection. It is the better choice for new versions; v1 keeps `sequential` so that it reproduces exactly.
- **What is recorded:** every run writes `provenance.json` with the git commit (and whether the tree had uncommitted changes), the library versions and the input fingerprints. `result.json` carries a `physics_hash` of all settings that change the result.
- **Reproducing v1:** `RDS_NTUPLES=... scripts/reproduce_v1.sh` refits and runs `scripts/compare_results.py` against `applier/data/rds_trkcalib_2018BParking_v1.json`. The script prints each term's difference in σ and fails above 0.01σ.
- **Checks done:** the ROOT ntuples give exactly the candidates the v1 fit used (1 105 575 data and 4 215 253 MC after the selection, same order, same values). The pipeline rerun from these inputs is compared in `REPRODUCTION.md`.

## 5. Configuration keys

| Key | v1 value | Meaning |
| --- | --- | --- |
| `selection.expression` | cos working point | numpy expression over branch names, applied while reading |
| `selection.kk_window` | 0.020 GeV | \|m(KK) − m(φ)\| window: on stored m(KK) for data; on corrected m(KK) for the MC template |
| `selection.dplus_veto` | 0.050 GeV | removes \|m(Kππ) − m(D⁺)\| < value, in data and MC |
| `model.eta_edges` | 0, 0.7, 1.2, 2.5 | \|η\| bins of the parameters |
| `model.shared` | s1, v1 | terms with one value for all bins |
| `model.start` | s0 −1e-3, v0 11 | starting values, per term or per parameter (`s0_2`) |
| `categories.pt_edges` | 0, 1.0, 1.6, 100 GeV | softest-track pT categories |
| `categories.split_charge` | true | Ds⁺ / Ds⁻ categories (needed for k) |
| `fit.mass_range`, `fit.bin_width` | 1.920–2.015, 0.001 GeV | binned likelihood range and bins |
| `fit.background` | expo2 | exp(λx + κx²) per category |
| `fit.seed` | 12345 | per-track z of the MC template |
| `fit.z_mode` | sequential | `sequential` (numpy, candidate order) or `hash` (from the track itself; order independent) |
| `correction.seed` | 20181 | seed of the hash-based z used by the appliers |
| `toys.*` | | closure-toy settings |

## Versions

See `CHANGELOG.md`. Release 1.0.0 produced the v1 correction.
