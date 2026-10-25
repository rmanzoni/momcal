"""Track-level correction, corrected masses and smooth binned templates."""
from __future__ import annotations
import numpy as np
from .io import MK, MPI, Sample

MPHI = 1.019461


class Correction:
    """Per-track momentum correction p' = p (1 + s_b) (1 + r_b z).

    eta_edges : |eta| bin edges, e.g. [0, 0.9, 1.5, 2.5]
    species   : 'common' (one set for all tracks) or 'kpi' (separate sets for kaons and pions)
    Parameter names: s_{species}{bin}, r_{species}{bin}, e.g. s_all0, r_all0 or s_K1, r_pi2.
    """

    def __init__(self, eta_edges=(0.0, 2.5), species="common"):
        self.eta_edges = np.asarray(eta_edges, float)
        self.nb = len(self.eta_edges) - 1
        self.species = species
        self.groups = ["all"] if species == "common" else ["K", "pi"]
        self.names = [f"{x}_{g}{b}" for x in ("s", "r") for g in self.groups for b in range(self.nb)]

    def track_index(self, sample: Sample):
        """Index of the parameter set for each track, shape (n, 3)."""
        aeta = np.abs(sample.eta())
        b = np.clip(np.digitize(aeta, self.eta_edges[1:-1]), 0, self.nb - 1)
        if self.species == "common":
            return b
        g = np.array([0, 0, 1])[None, :]          # k1, k2 -> K ; pi -> pi
        return g * self.nb + b

    def split(self, values):
        v = np.asarray(values, float)
        n = len(self.groups) * self.nb
        return v[:n], v[n:]

    # interface used by MorphedMC and the fit
    def factor(self, m, values):
        sc, rs = self.split(values)
        return (1.0 + sc[m.idx]) * (1.0 + rs[m.idx] * m.z)

    def limits(self, name):
        return (-0.01, 0.01) if name.startswith("s_") else (0.0, 0.05)

    def start(self, name):
        return 0.0 if name.startswith("s_") else 0.004

    def step(self, name):
        return 2e-4 if name.startswith("s_") else 1e-3


class PtCorrection:
    """Per-track correction with explicit momentum dependence, per |eta| bin b:

        p' = p * (1 + s0_b + s1_b / p + k_b * q * pT) * (1 + sqrt(r0_b^2 + (r1_b * pT)^2) * z)

    s0 [1]      overall momentum scale (magnetic field)
    s1 [GeV]    absolute momentum offset, i.e. energy loss in material (relative effect ~ 1/p)
    k  [1/GeV]  charge-dependent curvature bias, delta(1/pT) = -k*q  (misalignment / sagitta); sign follows the track charge q
    r0 [1]      extra relative resolution, constant (multiple scattering)
    r1 [1/GeV]  extra relative resolution growing with pT (hit-position resolution)
    `terms` selects which terms are free; the others are fixed to 0.
    """
    TERMS = ("s0", "s1", "k", "r0", "r1")
    # r0, r1 enter only squared, so they may take either sign; the free sign keeps a true value of 0 away from a limit
    # (a limit at 0 makes the HESSE error meaningless there). Quote |r0|, |r1|.
    LIM = dict(s0=(-0.01, 0.01), s1=(-0.05, 0.05), k=(-0.01, 0.01), r0=(-0.05, 0.05), r1=(-0.01, 0.01))
    STA = dict(s0=0.0, s1=0.0, k=0.0, r0=0.004, r1=2e-4)        # r1 = 0 is a stationary point by symmetry: do not start there
    STE = dict(s0=5e-5, s1=2e-4, k=1e-5, r0=1e-4, r1=3e-5)        # close to the expected errors: Minuit converges faster

    def __init__(self, eta_edges=(0.0, 2.5), terms=TERMS):
        self.eta_edges = np.asarray(eta_edges, float)
        self.nb = len(self.eta_edges) - 1
        self.terms = tuple(t for t in self.TERMS if t in terms)
        self.species = "common"; self.groups = ["all"]
        self.names = [f"{t}_{b}" for t in self.terms for b in range(self.nb)]

    def track_index(self, sample):
        aeta = np.abs(sample.eta())
        return np.clip(np.digitize(aeta, self.eta_edges[1:-1]), 0, self.nb - 1)

    def values_by_term(self, values):
        v = dict(zip(self.names, np.asarray(values, float)))
        return {t: np.array([v.get(f"{t}_{b}", 0.0) for b in range(self.nb)]) for t in self.TERMS}

    def factor(self, m, values):
        t = self.values_by_term(values)
        if not hasattr(m, "_pt_cache"):                         # per-sample constants, computed once
            m._pt_cache = (1.0 / m.p, m.q * m.pt, m.pt ** 2)
        invp, qpt, pt2 = m._pt_cache
        g = (lambda a: a[0]) if self.nb == 1 else (lambda a: a[m.idx])   # one |eta| bin: plain numbers, no indexing
        scale = 1.0 + g(t["s0"]) + g(t["s1"]) * invp + g(t["k"]) * qpt
        res = np.sqrt(g(t["r0"]) ** 2 + g(t["r1"]) ** 2 * pt2)
        return scale * (1.0 + res * m.z)

    def limits(self, name): return self.LIM[name.rsplit("_", 1)[0]]
    def start(self, name): return self.STA[name.rsplit("_", 1)[0]]
    def step(self, name): return self.STE[name.rsplit("_", 1)[0]]


class PtVarCorrection(PtCorrection):
    """As PtCorrection, with the resolution written through variances and with terms that can be shared by all |eta| bins.

        p' = p * (1 + s0 + s1 / p + k * q * pT) * (1 + sqrt(v0 * 1e-6 + v1 * 1e-8 * pT^2) * z)

    v0 = (r0 / 1e-3)^2 and v1 = (r1 / 1e-4 GeV^-1)^2. The data constrain r0^2 + r1^2 pT^2, which is linear in (v0, v1):
    the likelihood is close to parabolic in them, v1 = 0 is an ordinary point (v1 may go slightly negative), and the errors
    are not distorted by the curved (r0, r1) valley. Report r0 = 1e-3 sqrt(v0), r1 = 1e-4 sqrt(max(v1, 0)).
    shared : terms with one value for all |eta| bins (name "<term>_all"), e.g. ("s1", "v1").
    """
    TERMS = ("s0", "s1", "k", "v0", "v1")
    LIM = dict(s0=(-0.01, 0.01), s1=(-0.05, 0.05), k=(-0.01, 0.01), v0=(0.0, 400.0), v1=(-20.0, 400.0))
    STA = dict(s0=0.0, s1=0.0, k=0.0, v0=16.0, v1=4.0)
    STE = dict(s0=5e-5, s1=2e-4, k=1e-5, v0=0.5, v1=2.0)

    def __init__(self, eta_edges=(0.0, 2.5), terms=None, shared=()):
        self.eta_edges = np.asarray(eta_edges, float)
        self.nb = len(self.eta_edges) - 1
        terms = self.TERMS if terms is None else terms
        self.terms = tuple(t for t in self.TERMS if t in terms)
        self.shared = tuple(t for t in self.terms if t in shared)
        self.species = "common"; self.groups = ["all"]
        self.names = [f"{t}_all" if t in self.shared else f"{t}_{b}" for t in self.terms for b in ((0,) if t in self.shared else range(self.nb))]

    def values_by_term(self, values):
        v = dict(zip(self.names, np.asarray(values, float)))
        return {t: np.full(self.nb, v[f"{t}_all"]) if t in self.shared else np.array([v.get(f"{t}_{b}", 0.0) for b in range(self.nb)])
                for t in self.TERMS}

    def factor(self, m, values):
        t = self.values_by_term(values)
        if not hasattr(m, "_pt_cache"):
            m._pt_cache = (1.0 / m.p, m.q * m.pt, m.pt ** 2)
        invp, qpt, pt2 = m._pt_cache
        g = (lambda a: a[0]) if self.nb == 1 else (lambda a: a[m.idx])
        scale = 1.0 + g(t["s0"]) + g(t["s1"]) * invp + g(t["k"]) * qpt
        res = np.sqrt(np.maximum(g(t["v0"]) * 1e-6 + g(t["v1"]) * 1e-8 * pt2, 0.0))
        return scale * (1.0 + res * m.z)

    @staticmethod
    def to_r(values_dict):
        """r0, r1 from fitted v0, v1 (same bin suffixes)."""
        out = {}
        for n, v in values_dict.items():
            if n.startswith("v0_"): out["r0_" + n[3:]] = 1e-3 * np.sqrt(max(v, 0.0))
            if n.startswith("v1_"): out["r1_" + n[3:]] = 1e-4 * np.sqrt(max(v, 0.0))
        return out


class CmsCorrection(PtVarCorrection):
    """Correction in the convention of the CMS W-mass measurement (arXiv:2412.13872, Eqs. 2-3), acting on the curvature
    k = 1/pT of each MC track so that MC matches data:

        k' = k * (1 + A - e*k + q*M/k) * (1 + sqrt(a^2 + c^2/k^2) * z)
        i.e.  pT' = pT / [ (1 + A - e/pT + q*M*pT) * (1 + sqrt(a^2 + (c*pT)^2) * z) ],  direction unchanged.

    A [1]      magnetic-field (scale) term
    e [GeV]    energy-loss term (CMS: epsilon)
    M [1/GeV]  sagitta (misalignment) term, sign follows the track charge q
    a [1]      extra curvature resolution from multiple scattering
    c [1/GeV]  extra curvature resolution from hit resolution
    (A, e, M) describe data relative to MC, as CMS's describe reconstruction relative to the reference; (a, c) are the
    resolution MC is missing (quadrature difference). CMS's b, d correlation term is not used.
    Fit variables for the resolution: a2 = (a/1e-3)^2, c2 = (c/1e-4 GeV^-1)^2 (linear in the variance, may cross 0).

    The template's per-track Gaussian (MorphedMC.z) is drawn as a momentum fluctuation, as in v1; an upward momentum
    fluctuation is a downward curvature fluctuation, so the curvature uses z_k = -z (Z_SIGN). With the same seed, v1 and
    this convention then share one smearing realisation and differ only by the convention.
    """
    Z_SIGN = -1.0
    TERMS = ("A", "e", "M", "a2", "c2")
    # a2 and c2 may go below 0 (the total variance a2 + c2 pT^2 is clipped at 0): a limit at 0 makes HESSE invalid when the
    # true value is near 0 (seen in the forward |eta| bin)
    LIM = dict(A=(-0.01, 0.01), e=(-0.05, 0.05), M=(-0.01, 0.01), a2=(-20.0, 400.0), c2=(-20.0, 400.0))
    STA = dict(A=0.0, e=0.0, M=0.0, a2=16.0, c2=4.0)
    STE = dict(A=5e-5, e=2e-4, M=1e-5, a2=0.5, c2=2.0)

    def factor(self, m, values):
        """p'/p = pT'/pT per track."""
        t = self.values_by_term(values)
        if not hasattr(m, "_cms_cache"):
            m._cms_cache = (1.0 / m.pt, m.q * m.pt, m.pt ** 2)
        invpt, qpt, pt2 = m._cms_cache
        g = (lambda x: x[0]) if self.nb == 1 else (lambda x: x[m.idx])
        scale = 1.0 + g(t["A"]) - g(t["e"]) * invpt + g(t["M"]) * qpt
        res = np.sqrt(np.maximum(g(t["a2"]) * 1e-6 + g(t["c2"]) * 1e-8 * pt2, 0.0))
        return 1.0 / (scale * (1.0 + res * (self.Z_SIGN * m.z)))

    @staticmethod
    def to_ac(values_dict):
        """a, c from fitted a2, c2 (same bin suffixes)."""
        out = {}
        for n, v in values_dict.items():
            if n.startswith("a2_"): out["a_" + n[3:]] = 1e-3 * np.sqrt(max(v, 0.0))
            if n.startswith("c2_"): out["c_" + n[3:]] = 1e-4 * np.sqrt(max(v, 0.0))
        return out


MODELS = {"cms": CmsCorrection, "v1": PtVarCorrection}
DEFAULT_MODEL = {
    "cms": dict(terms=["A", "e", "M", "a2", "c2"], shared=["e", "c2"], start={"A": 1.0e-3, "a2": 11.0}),
    "v1": dict(terms=["s0", "s1", "k", "v0", "v1"], shared=["s1", "v1"], start={"s0": -1.0e-3, "v0": 11.0}),
}


class MorphedMC:
    """MC sample with fixed per-track random numbers; returns corrected masses for given parameters."""

    def __init__(self, sample: Sample, corr: Correction, seed=12345, light=True, z_mode="sequential"):
        """z_mode 'sequential': z from numpy's generator in candidate order (v1); a change of the selected sample changes the
        realisation of all later tracks. 'hash': z from the track's own kinematics (the appliers' convention with
        run = lumi = event = 0), independent of order and selection."""
        self.s = sample
        self.corr = corr
        self.idx = corr.track_index(sample)
        if z_mode == "sequential":
            self.z = np.random.default_rng(seed).standard_normal(self.idx.shape)
        elif z_mode == "hash":
            from .zhash import gauss_from_key, track_id_from_kinematics
            P = sample.p; pt = np.hypot(P[..., 0], P[..., 1])
            tid = track_id_from_kinematics(pt, np.arcsinh(P[..., 2] / pt), np.arctan2(P[..., 1], P[..., 0]))
            self.z = gauss_from_key(seed, 0, 0, 0, tid)
        else:
            raise ValueError(f"z_mode {z_mode!r}")
        self.p2 = (sample.p ** 2).sum(-1)                       # |p|^2 per track
        self.p = np.sqrt(self.p2); self.pt = np.hypot(sample.p[..., 0], sample.p[..., 1])
        self.q = sample.charge.astype(float)
        self.m2 = np.array([MK, MK, MPI]) ** 2
        # scalar products p_i.p_j of the uncorrected tracks: the corrected ones are f_i f_j p_i.p_j (directions unchanged)
        P = sample.p
        self.d01 = (P[:, 0] * P[:, 1]).sum(-1); self.d02 = (P[:, 0] * P[:, 2]).sum(-1); self.d12 = (P[:, 1] * P[:, 2]).sum(-1)
        self.n = len(sample)
        if light: self.s = None                                 # the (n,3,3) momenta are not needed any more: save memory

    def masses(self, values):
        f = self.corr.factor(self, values)                      # (n, 3) momentum factor
        e = np.sqrt(self.p2 * f * f + self.m2)
        f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]; e0, e1, e2 = e[:, 0], e[:, 1], e[:, 2]
        kk = e0 * e1 - f0 * f1 * self.d01
        m2kk = 2 * self.m2[0] + 2 * kk
        m2 = m2kk + self.m2[2] + 2 * (e0 * e2 - f0 * f2 * self.d02 + e1 * e2 - f1 * f2 * self.d12)
        return np.sqrt(np.maximum(m2, 0.0)), np.sqrt(np.maximum(m2kk, 0.0))


def smooth_hist(x, edges, w=None, kernel="cubic"):
    """Histogram in which each entry is spread over neighbouring bins with a smooth kernel, so that a likelihood built on it
    changes smoothly with the correction parameters.
    kernel="linear"   : triangular interpolation between bin centres (continuous, kinked derivative; rms 0.41 bin)
    kernel="quadratic": quadratic B-spline over three bins (continuous first derivative; rms 0.5 bin)
    kernel="cubic"    : cubic B-spline over four bins (continuous second derivative; rms 0.58 bin). Default: Minuit's HESSE
                        needs smooth second derivatives; with the other kernels the errors scattered by factors 3-9 between
                        toys, with this one they are stable."""
    c = 0.5 * (edges[1:] + edges[:-1]); bw = edges[1] - edges[0]
    u = (x - c[0]) / bw
    w = np.ones_like(x) if w is None else w
    n = len(c); out = np.zeros(n)
    if kernel == "linear":
        i0 = np.floor(u).astype(np.int64); t = u - i0
        parts = ((i0, 1 - t), (i0 + 1, t))
    elif kernel == "quadratic":
        i = np.floor(u + 0.5).astype(np.int64); d = u - i
        parts = ((i - 1, 0.5 * (0.5 - d) ** 2), (i, 0.75 - d * d), (i + 1, 0.5 * (0.5 + d) ** 2))
    else:                                                   # cubic B-spline over four bins: continuous second derivative
        i = np.floor(u).astype(np.int64); t = u - i; t2 = t * t; t3 = t2 * t
        parts = ((i - 1, (1 - t) ** 3 / 6), (i, (3 * t3 - 6 * t2 + 4) / 6), (i + 1, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6), (i + 2, t3 / 6))
    for j, f in parts:
        ok = (j >= 0) & (j < n)
        out += np.bincount(j[ok], weights=(w * f)[ok], minlength=n)
    return out


def expo_shape(c, lam, kap=0.0, m0=1.968):
    y = np.exp(lam * (c - m0) + kap * (c - m0) ** 2)
    return y / y.sum()
