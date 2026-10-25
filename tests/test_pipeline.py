"""End to end on synthetic ntuples: read ROOT, select, fit, export; the hidden correction is recovered; reruns are identical."""
import json, os
import numpy as np
from scalesmear import pipeline
from conftest import INJECT


def test_run_recovers_injected_and_is_deterministic(small_config):
    r1 = pipeline.run(small_config)
    assert r1.valid
    for n, v in INJECT.items():
        assert abs(r1.values[n] - v) < 4 * r1.errors[n], (n, r1.values[n], r1.errors[n], v)
    out = small_config["output"]["dir"]
    for f in ("result.json", "provenance.json", "config_used.yaml", "log.txt", "lineshape.png", "terms.png", "fit_categories.png",
              "rds_trkcalib_synthetic_test.json"):
        assert os.path.exists(os.path.join(out, f)), f
    eff = r1.config["efficiencies"]
    assert abs(eff["data"]["n_in"] / 80_000 - 0.95) < 0.01          # the selection expression removed the 5% failing rows
    assert 0.5 < eff["data"]["n_out"] / eff["data"]["n_in"] < 1.0      # m(KK) window and D+ veto
    r2 = pipeline.run(small_config)                     # second run reads the cache: must give the identical result
    assert r2.values == r1.values and r2.errors == r1.errors


def test_toys_and_collect(small_config):
    small_config["toys"]["data_frac"] = 0.3
    small_config["toys"]["inject"] = {"A": 1.0e-3, "M": -1.0e-4, "a2": 12.0}
    for s in (1, 2, 3):
        d = pipeline.run_toy(small_config, s)
        assert d["valid"]
        for n in d["values"]:
            assert abs(d["values"][n] - d["injected"][n]) < 5 * d["errors"][n], (n, d)
    summ = pipeline.collect_toys(small_config)
    assert set(summ) == {"A_0", "M_0", "a2_0"} and all(v["n"] == 3 for v in summ.values())
    assert os.path.exists(os.path.join(small_config["output"]["dir"], "toys_pulls.png"))


def test_v1_convention_matches_cms_to_first_order(small_config):
    """The same synthetic data fitted in the v1 convention: s0 = -A, k = -M, v0 = a2 to first order."""
    import copy
    cfg = copy.deepcopy(small_config)
    cfg["model"].update(convention="v1", terms=["s0", "k", "v0"], start={"s0": -1e-3, "v0": 9.0})
    cfg["output"]["dir"] += "_v1"; cfg["output"]["plots"] = False
    r = pipeline.run(cfg)
    assert r.valid
    for v1, cms, sign in (("s0_0", "A_0", -1), ("k_0", "M_0", -1), ("v0_0", "a2_0", +1)):
        assert abs(r.values[v1] - sign * INJECT[cms]) < 4 * r.errors[v1], (v1, r.values[v1], r.errors[v1])
