# Reproduction of the v1 correction

Date: 24 Oct 2026. Package: scalesmear 1.0.0, reference environment (`requirements-lock.txt`, Python 3.13).

1. **Inputs.** Applying `configs/rds2018_v1.yaml` to the ROOT ntuples selects exactly the candidates the original v1 fit used. That fit read pre-extracted `.npz` files.

   | Sample | File | Selected (ROOT) | Original extract |
   | --- | --- | --- | --- |
   | data | rs_slim_21oct2026.root | 1 105 575 | 1 105 575 |
   | MC | sig_slim_21oct2026.root | 4 215 253 | 4 215 253 |

   Order and values are identical: the same first and last momenta, and the same sum of k1_px to 1e-16. The branches are stored as doubles holding float32-precision values.
2. **Refit.** `scalesmear run configs/rds2018_v1.yaml` ran on the same candidates. The refit took 67 min and 46 487 likelihood calls, and the minimum is valid. After the m(KK) window and the D⁺ veto, 910 663 data and 3 364 768 MC candidates remain, as in v1.
3. **Comparison.** `scripts/compare_results.py applier/data/rds_trkcalib_2018BParking_v1.json <new correction file>` shows the 11 values, the errors and the covariance are **identical**: 0.0000σ difference and covariance difference 0. χ² is 1727.59 for 1620 in both.

The outputs of this run are in `reference/v1/`: result.json, provenance, config, log and plots.

Caveat: applying the selection expression to the float32 `.npz` extracts, instead of to the ROOT files, drops 2 data and 5 MC candidates at the cut boundaries. With `fit.z_mode: sequential` this changes the MC random-number realisation (see README, Reproducibility). Run from the ROOT files, or from extracts with an empty selection.
