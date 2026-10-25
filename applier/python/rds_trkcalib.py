"""Apply the R(Ds) track momentum calibration to MC tracks. Reads correction files of schema rds_trkcalib/2 and /1.

schema 2, CMS W-mass convention (arXiv:2412.13872), on the curvature k = 1/pT:
    k' = k (1 + A - e*k + q*M/k)(1 + sqrt(a^2 + c^2/k^2) z)
    i.e. pT' = pT / [ (1 + A - e/pT + q*M*pT) * (1 + sqrt(max(a2 + c2*pT^2, 0)) * z) ]
schema 1 (the first, v1 convention):
    p' = p * (1 + s0 + s1/p + k*q*pT) * (1 + sqrt(max(v0*1e-6 + v1*1e-8*pT^2, 0)) * z)

Only the momentum magnitude changes; the direction is kept, so p'/p = pT'/pT = factor(). Apply to MC only, to every
charged track (kaons, pion, muon), each with its own pT, charge and |eta|.

Usage
-----
    from rds_trkcalib import TrackCalib
    cal = TrackCalib("rds_trkcalib_2018BParking_v2.json")
    z = cal.gauss(run, lumi, event, track_id)               # fixed per track
    f = cal.factor(p, pt, eta, q, z)                         # numpy arrays or scalars
    px2, py2, pz2 = px * f, py * f, pz * f
    f_up = cal.factor(p, pt, eta, q, z, variation="eig0_up") # statistical variations eig0 ... eig10

Only numpy is needed. The C++ header RDsTrackCalib.h gives identical numbers (see test/).
"""
from __future__ import annotations
import json
import numpy as np

__all__ = ["TrackCalib", "splitmix64", "gauss_from_key", "track_id_from_kinematics"]
TERMS = {"rds_trkcalib/1": ("s0", "s1", "k", "v0", "v1"), "rds_trkcalib/2": ("A", "e", "M", "a2", "c2")}
SCALE_TERMS = {"rds_trkcalib/1": ("s0", "s1", "k"), "rds_trkcalib/2": ("A", "e", "M")}
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


# ---------------------------------------------------------------- the correction
class TrackCalib:
    """Correction read from the JSON file. All methods are vectorised over numpy arrays."""

    def __init__(self, path):
        with open(path) as f:
            self.cfg = json.load(f)
        self.schema = self.cfg.get("schema")
        if self.schema not in TERMS:
            raise ValueError(f"{path}: unknown schema {self.schema!r}")
        self.cms = self.schema == "rds_trkcalib/2"
        self.terms = TERMS[self.schema]
        self.edges = np.asarray(self.cfg["eta_bins"]["edges"], float)
        self.nb = len(self.edges) - 1
        self.seed = int(self.cfg["random"]["seed"])
        self.nominal = {t: np.asarray(self.cfg["per_bin"][t], float) for t in self.terms}
        self.variation_names = ["nominal", "scale_only", "smear_only"] + [
            f"{n}_{d}" for n in self.cfg["variations"]["names"] for d in ("up", "down")]
        self._cache = {}

    def __repr__(self):
        return f"TrackCalib({self.cfg['name']} {self.cfg['version']}, |eta| edges {self.edges.tolist()})"

    # parameters ---------------------------------------------------
    def params(self, variation="nominal"):
        """Per-bin parameters {term: array(nb)} for a variation name."""
        if variation in self._cache:
            return self._cache[variation]
        p = {t: v.copy() for t, v in self.nominal.items()}
        scale_terms = SCALE_TERMS[self.schema]
        if variation == "scale_only":
            for t in self.terms:
                if t not in scale_terms: p[t][:] = 0.0
        elif variation == "smear_only":
            for t in scale_terms: p[t][:] = 0.0
        elif variation != "nominal":
            name, _, d = variation.rpartition("_")
            names = self.cfg["variations"]["names"]
            if name not in names or d not in ("up", "down"):
                raise ValueError(f"unknown variation {variation!r}; known: {self.variation_names}")
            shift = self.cfg["variations"]["shift_per_bin"][names.index(name)]
            sign = 1.0 if d == "up" else -1.0
            for t in self.terms:
                p[t] = p[t] + sign * np.asarray(shift[t], float)
        self._cache[variation] = p
        return p

    def eta_bin(self, eta):
        """|eta| bin index; beyond the last edge -> last bin."""
        return np.clip(np.searchsorted(self.edges, np.abs(eta), side="right") - 1, 0, self.nb - 1)

    # factors -------------------------------------------------------
    def scale(self, p, pt, eta, q, variation="nominal"):
        """Momentum scale factor without smearing: 1/(1 + A - e/pT + q M pT) (schema 2) or 1 + s0 + s1/p + k q pT (schema 1)."""
        P = self.params(variation); b = self.eta_bin(eta)
        pt = np.asarray(pt, float); q = np.asarray(q, float)
        if self.cms:
            return 1.0 / (1.0 + P["A"][b] - P["e"][b] / pt + P["M"][b] * q * pt)
        return 1.0 + P["s0"][b] + P["s1"][b] / np.asarray(p, float) + P["k"][b] * q * pt

    def resolution(self, pt, eta, variation="nominal"):
        """Extra relative resolution: sqrt(a2 + c2 pT^2) (schema 2, on the curvature) or sqrt(v0 1e-6 + v1 1e-8 pT^2)."""
        P = self.params(variation); b = self.eta_bin(eta); pt = np.asarray(pt, float)
        if self.cms:
            return np.sqrt(np.maximum(P["a2"][b] + P["c2"][b] * pt * pt, 0.0))
        return np.sqrt(np.maximum(P["v0"][b] * 1e-6 + P["v1"][b] * 1e-8 * pt * pt, 0.0))

    def factor(self, p, pt, eta, q, z, variation="nominal"):
        """Full factor p'/p (= pT'/pT) for tracks of momentum p, transverse momentum pt, pseudorapidity eta, charge q,
        Gaussian z. Schema 2 smears the curvature (divides by 1 + sigma z); schema 1 the momentum (multiplies)."""
        sm = 1.0 + self.resolution(pt, eta, variation) * np.asarray(z, float)
        sc = self.scale(p, pt, eta, q, variation)
        out = sc / sm if self.cms else sc * sm
        return float(out) if np.ndim(out) == 0 else out

    def gauss(self, run, lumi, event, track_id):
        """The per-track Gaussian z for this file's seed."""
        return gauss_from_key(self.seed, run, lumi, event, track_id)

    # convenience ----------------------------------------------------
    def correct_p3(self, px, py, pz, q, z, variation="nominal"):
        """Corrected (px, py, pz). Recompute energies with the mass hypothesis afterwards."""
        px, py, pz = (np.asarray(a, float) for a in (px, py, pz))
        pt = np.hypot(px, py); p = np.sqrt(pt * pt + pz * pz)
        eta = np.arcsinh(pz / pt)
        f = self.factor(p, pt, eta, q, z, variation)
        return px * f, py * f, pz * f

    def correct_ptetaphi(self, pt, eta, phi, q, z, variation="nominal"):
        """Corrected (pT, eta, phi): only pT changes."""
        pt = np.asarray(pt, float); p = pt * np.cosh(np.asarray(eta, float))
        return pt * self.factor(p, pt, eta, q, z, variation), eta, phi
