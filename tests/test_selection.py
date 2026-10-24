"""D+ -> K- pi+ pi+ decays reconstructed as K+ K- pi+ are removed by the reflection veto; Ds -> phi pi mostly kept."""
import numpy as np
from scalesmear.io import Sample, MK, MPI
from scalesmear.selection import m_kpipi, dplus_veto, MDPLUS
from conftest import toy_ds


def test_dplus_reflection_is_vetoed():
    rng = np.random.default_rng(0); n = 2000
    # D+ -> K- pi+ pi+ at rest, three-body phase space approximated by random directions rescaled to the D+ mass
    p = rng.normal(0, 0.5, (n, 3, 3)); p -= p.mean(1, keepdims=True)
    m = np.array([MK, MPI, MPI])
    for _ in range(60):                                   # rescale momenta until sum E = m(D+)
        e = np.sqrt((p**2).sum(-1) + m**2).sum(1); p *= (1 + 0.5 * (MDPLUS - e) / e)[:, None, None]
    # reconstructed as k1 = pi+ (taken as K+), k2 = K-, pi = pi+   -> the K with the pion's charge is k1
    s = Sample(p[:, [1, 0, 2]], np.tile([1, -1, 1], (n, 1)), np.zeros(n), np.zeros(n))
    assert np.allclose(m_kpipi(s), MDPLUS, atol=1e-6)
    assert not dplus_veto(s, 0.05).any()


def test_ds_mostly_kept():
    s = toy_ds(20_000, 1)
    assert dplus_veto(s, 0.05).mean() > 0.6
