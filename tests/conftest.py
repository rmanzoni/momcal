"""Synthetic flat ntuples with the branch names of the R(Ds) slim ntuples, for fast end-to-end tests."""
import os
import numpy as np
import pytest
from scalesmear.io import Sample, MK, MPI

MDS, MPHI = 1.96835, 1.019461
INJECT = {"s0_0": -1.5e-3, "k_0": 2.0e-4, "v0_0": 16.0}       # correction hidden in the synthetic "data"


def toy_ds(n, seed):
    """Ds -> phi(KK) pi decays with a 0.4% 'detector' resolution per track; half Ds+, half Ds-."""
    rng = np.random.default_rng(seed)
    def two(M, P, m1, m2):
        q = np.sqrt((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (2 * M)
        c = rng.uniform(-1, 1, len(P)); f = rng.uniform(0, 2 * np.pi, len(P)); s = np.sqrt(1 - c**2)
        d = q[:, None] * np.stack([s * np.cos(f), s * np.sin(f), c], 1)
        E = np.sqrt(M**2 + (P**2).sum(1)); b = P / E[:, None]; g = E / M
        def boost(e, p):
            bp = (b * p).sum(1); b2 = (b * b).sum(1)
            return p + ((g - 1) * bp / b2 + g * e)[:, None] * b
        return boost(np.sqrt(m1**2 + q**2), d), boost(np.sqrt(m2**2 + q**2), -d)
    P = rng.normal(0, 3, (n, 3)) + np.array([0, 0, 6.0]) * rng.choice([-1, 1], n)[:, None]
    phi, pi = two(np.full(n, MDS), P, MPHI, MPI)
    k1, k2 = two(np.full(n, MPHI), phi, MK, MK)
    p = np.stack([k1, k2, pi], 1) * (1 + 0.004 * rng.standard_normal((n, 3)))[..., None]
    q = np.tile([1, -1, 1], (n, 1)); q[n // 2:] *= -1
    return Sample(p, q.astype(np.int8), np.zeros(n), np.zeros(n), "toy")


def masses(p):
    e = np.sqrt((p**2).sum(-1) + np.array([MK, MK, MPI])**2)
    m = np.sqrt(e.sum(1)**2 - (p.sum(1)**2).sum(1)); mkk = np.sqrt(e[:, :2].sum(1)**2 - (p[:, :2].sum(1)**2).sum(1))
    return m, mkk


def to_branches(s, m, mkk, rng, nfail=0):
    n = len(m)
    d = {f"{t}_{c}": s.p[:, i, j] for i, t in enumerate(("k1", "k2", "pi")) for j, c in enumerate(("px", "py", "pz"))}
    d.update({f"{t}_charge": s.charge[:, i].astype(np.int32) for i, t in enumerate(("k1", "k2", "pi"))})
    d.update(phiPi_m=m, kk_m=mkk, dsMu_m=np.full(n, 4.0), tv_prob=np.full(n, 0.5), sv_prob=np.full(n, 0.5),
             ds_vtx_cosine_xyz_pv=np.where(rng.random(n) < nfail, 0.5, 0.999))      # some candidates fail the selection
    return d


@pytest.fixture(scope="session")
def ntuples(tmp_path_factory):
    import uproot
    from scalesmear.model import PtVarCorrection, MorphedMC
    d = tmp_path_factory.mktemp("ntuples"); rng = np.random.default_rng(5)
    mc = toy_ds(160_000, 1)
    m, mkk = masses(mc.p)
    with uproot.recreate(d / "mc.root") as f:
        f["tree"] = to_branches(mc, m, mkk, rng, nfail=0.05)
    sig = toy_ds(50_000, 2)
    corr = PtVarCorrection([0, 2.5], terms=("s0", "k", "v0"))
    f_ = corr.factor(MorphedMC(sig, corr, seed=99, light=False), [INJECT[n] for n in corr.names])
    sig = Sample(sig.p * f_[..., None], sig.charge, sig.m_kkpi, sig.m_kk)
    m, mkk = masses(sig.p)
    nb = 30_000; lam = -2.6; a, b = 1.908, 2.028                                 # exponential background
    x = a + np.log1p(rng.random(nb) * np.expm1(lam * (b - a))) / lam
    j = rng.integers(0, len(sig.p), nb)
    allp = Sample(np.concatenate([sig.p, sig.p[j]]), np.concatenate([sig.charge, sig.charge[j]]), None, None)
    br = to_branches(allp, np.concatenate([m, x]), np.concatenate([mkk, np.full(nb, MPHI)]), rng, nfail=0.05)
    with uproot.recreate(d / "data.root") as f:
        f["tree"] = br
    return d


@pytest.fixture
def small_config(ntuples, tmp_path):
    from scalesmear.config import load_config
    import yaml
    cfg = dict(name="synthetic", version="test",
               inputs=dict(data=[str(ntuples / "data.root")], mc=[str(ntuples / "mc.root")]),
               model=dict(eta_edges=[0.0, 2.5], terms=["s0", "k", "v0"], shared=[], start={"s0": -1e-3, "v0": 9.0}),
               categories=dict(pt_edges=None, split_charge=True),
               output=dict(dir=str(tmp_path / "out"), plots=True))
    p = tmp_path / "cfg.yaml"; p.write_text(yaml.safe_dump(cfg))
    return load_config(str(p))
