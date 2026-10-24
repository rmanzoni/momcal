"""Fast closure test on a synthetic MC sample: injects a correction and checks it is recovered."""
import numpy as np
from scalesmear.io import Sample, MK, MPI
from scalesmear.model import Correction
from scalesmear import closure


def toy_ds(n=300_000, seed=1):  # kept here: older single-bin model test
    rng = np.random.default_rng(seed)
    mds, mphi = 1.96835, 1.019461
    def two(M, P, m1, m2):
        q = np.sqrt((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (2 * M)
        c = rng.uniform(-1, 1, len(P)); f = rng.uniform(0, 2 * np.pi, len(P)); s = np.sqrt(1 - c**2)
        d = q[:, None] * np.stack([s * np.cos(f), s * np.sin(f), c], 1)
        E = np.sqrt(M**2 + (P**2).sum(1)); b = P / E[:, None]; g = E / M
        def boost(e, p):
            bp = (b * p).sum(1); b2 = (b * b).sum(1)
            return p + ((g - 1) * bp / b2 + g * e)[:, None] * b
        return boost(np.sqrt(m1**2 + q**2), d), boost(np.sqrt(m2**2 + q**2), -d)
    P = rng.normal(0, 3, (n, 3)) + np.array([0, 0, 8.0])
    phi, pi = two(np.full(n, mds), P, mphi, MPI)
    k1, k2 = two(np.full(n, mphi), phi, MK, MK)
    p = np.stack([k1, k2, pi], 1) * (1 + 0.004 * rng.standard_normal((n, 3)))[..., None]   # 'detector' resolution
    e = np.sqrt((p**2).sum(-1) + np.array([MK, MK, MPI])**2)
    m = np.sqrt(e.sum(1)**2 - (p.sum(1)**2).sum(1)); mkk = np.sqrt(e[:, :2].sum(1)**2 - (p[:, :2].sum(1)**2).sum(1))
    return Sample(p, np.tile([1, -1, 1], (n, 1)).astype(np.int8), m, mkk, "toy")


def test_global_closure():
    mc = toy_ds()
    corr = Correction([0, 2.5])
    inject = [-8e-4, 6e-3]
    f, r = closure.run(mc, corr, inject, mrange=(1.935, 2.000))
    v = r.correction_values(); e = np.array([r.errors[n] for n in r.corr_names])
    assert r.valid
    assert np.all(np.abs(v - inject) < 4 * e + 1e-5), (v, e)
