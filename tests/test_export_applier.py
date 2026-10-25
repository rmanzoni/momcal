"""The exported correction file, read by the Python applier, gives exactly the fitter's momentum factor."""
import json, os, sys
import numpy as np
import pytest
from scalesmear.model import PtVarCorrection, CmsCorrection, MorphedMC
from scalesmear.export import write_correction
from conftest import toy_ds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "applier", "python"))
from rds_trkcalib import TrackCalib


@pytest.mark.parametrize("conv", ["cms", "v1"])
def test_applier_matches_fitter(tmp_path, conv):
    if conv == "cms":
        corr = CmsCorrection([0, 0.7, 1.2, 2.5], shared=("e", "c2"))
        x = [1e-3, 0.9e-3, 1.3e-3, -3e-4, -2e-5, -5e-5, 1.2e-4, 6.3, 12.5, 0.4, 14.5]
    else:
        corr = PtVarCorrection([0, 0.7, 1.2, 2.5], shared=("s1", "v1"))
        x = [-1e-3, -0.9e-3, -1.3e-3, -3e-4, 2e-5, 5e-5, -1.2e-4, 6.3, 12.5, 0.4, 14.5]
    rng = np.random.default_rng(3)
    vals = dict(zip(corr.names, x))
    cov = np.diag(rng.uniform(0.5, 2, len(vals)) * 1e-8)
    res = dict(values=vals, errors={k: 1e-4 for k in vals}, correction=corr.names, correction_cov=cov.tolist(), valid=True,
               chi2=[1.0], ndf=[1], config=dict(eta_edges=[0, 0.7, 1.2, 2.5], convention=conv))
    rj = tmp_path / "result.json"; rj.write_text(json.dumps(res))
    cj = tmp_path / "corr.json"; write_correction(str(rj), str(cj), "t", "v0")
    cal = TrackCalib(str(cj))
    s = toy_ds(20_000, 4); m = MorphedMC(s, corr, seed=5, light=False)
    f_fit = corr.factor(m, np.array([vals[n] for n in corr.names]))
    P = s.p; pt = np.hypot(P[..., 0], P[..., 1]); p = np.sqrt((P**2).sum(-1)); eta = np.arcsinh(P[..., 2] / pt)
    zk = corr.Z_SIGN * m.z if conv == "cms" else m.z           # the cms template smears the curvature with z_k = -z
    f_app = cal.factor(p, pt, eta, s.charge, zk)
    assert np.max(np.abs(f_app - f_fit)) < 1e-14
    assert cal.cms == (conv == "cms")
    # eigen-variations: up and down are symmetric around nominal, in the quantity that is linear in the parameters
    # (cms: the curvature factor 1/f; v1: the momentum factor f)
    up = cal.factor(p, pt, eta, s.charge, 0.0, "eig0_up"); dn = cal.factor(p, pt, eta, s.charge, 0.0, "eig0_down")
    nom = cal.factor(p, pt, eta, s.charge, 0.0)
    if conv == "cms":
        up, dn, nom = 1 / up, 1 / dn, 1 / nom
    assert np.allclose(up - nom, nom - dn, atol=1e-15)
