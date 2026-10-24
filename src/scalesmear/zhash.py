"""Hash-based per-track Gaussian numbers: identical algorithm to applier/python/rds_trkcalib.py (tests/test_zhash.py)."""
import numpy as np

_M64 = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------- deterministic random numbers
def splitmix64(x):
    """splitmix64 finaliser on uint64 (numpy arrays or Python ints). Wraps modulo 2^64."""
    if isinstance(x, (int, np.integer)) and not isinstance(x, np.ndarray):
        z = (int(x) + 0x9E3779B97F4A7C15) & _M64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        return z ^ (z >> 31)
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _u64(x):
    """Integers (possibly negative) to uint64 with two's-complement wrap, as a C++ static_cast<uint64_t> does."""
    a = np.asarray(x)
    if a.dtype.kind == "u":
        return a.astype(np.uint64)
    return a.astype(np.int64).view(np.uint64) if a.ndim else np.array(int(a) & _M64, dtype=np.uint64)


def _unit(h):
    """uint64 -> double in (0, 1): the top 53 bits, centred in their interval."""
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)


def gauss_from_key(seed, run, lumi, event, track_id):
    """Unit Gaussian fixed by (seed, run, lumi, event, track_id). Vectorised; returns float or numpy array."""
    h = splitmix64(_u64(seed))
    for k in (run, lumi, event, track_id):
        h = splitmix64(h ^ _u64(k))
    h2 = splitmix64(h)
    u1, u2 = _unit(h), _unit(h2)
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return float(z) if np.ndim(z) == 0 else z


def track_id_from_kinematics(pt, eta, phi):
    """Integer track identifier from (pT [GeV], eta, phi), for ntuples without a track index.
    pT is rounded to 1 MeV, eta and phi to 1e-4; packed into 58 bits (24 + 17 + 17)."""
    ipt = np.floor(np.asarray(pt, float) * 1000.0 + 0.5).astype(np.int64) & ((1 << 24) - 1)
    ie = np.floor((np.asarray(eta, float) + 6.5) * 1e4 + 0.5).astype(np.int64) & ((1 << 17) - 1)
    ip = np.floor((np.asarray(phi, float) + 3.2) * 1e4 + 0.5).astype(np.int64) & ((1 << 17) - 1)
    out = (ipt << 34) | (ie << 17) | ip
    return int(out) if np.ndim(out) == 0 else out
