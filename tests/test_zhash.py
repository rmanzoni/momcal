import os, sys
import numpy as np
from scalesmear import zhash
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "applier", "python"))
import rds_trkcalib


def test_same_algorithm_as_applier():
    rng = np.random.default_rng(0); n = 10000
    pt, eta, phi = rng.uniform(0.3, 20, n), rng.uniform(-2.5, 2.5, n), rng.uniform(-3.14, 3.14, n)
    t1 = zhash.track_id_from_kinematics(pt, eta, phi); t2 = rds_trkcalib.track_id_from_kinematics(pt, eta, phi)
    assert np.array_equal(t1, t2)
    k = rng.integers(0, 2**40, n)
    assert np.array_equal(zhash.gauss_from_key(7, 1, 2, k, t1), rds_trkcalib.gauss_from_key(7, 1, 2, k, t1))


def test_hash_z_independent_of_order():
    from scalesmear.model import PtVarCorrection, MorphedMC
    from conftest import toy_ds
    s = toy_ds(5000, 3); corr = PtVarCorrection([0, 2.5])
    z1 = MorphedMC(s, corr, seed=1, z_mode="hash").z
    perm = np.random.default_rng(1).permutation(len(s))
    z2 = MorphedMC(s.subset(perm), corr, seed=1, z_mode="hash").z
    assert np.array_equal(z1[perm], z2)
    assert abs(z1.mean()) < 0.03 and abs(z1.std() - 1) < 0.03
