"""Fit channels: one decay, one mass variable, its categories and its templates built from corrected MC tracks.

    kkpi  : Ds -> phi(KK) pi,  m(KK pi)          (wraps the v3 ScaleSmearFit, unchanged)
    mumu  : J/psi -> mu mu,    m(mu mu)          from the B+ -> J/psi K+ events
    jpsik : B+ -> J/psi K+,    m(J/psi_rf K)     the stored refitted J/psi (mass constrained) + the kaon

Every channel gives, per category, the data histogram and normalised templates that depend on the correction values:
the signal template first, then any extra peaking components (B+ -> J/psi pi+ in jpsik). The background is an exponential
of a quadratic per category, centred at the channel's m0. All channels share one Correction object, i.e. one set of
per-track parameters; a track's parameters are chosen by its own |eta|.
"""
from __future__ import annotations
import numpy as np
from .tracks import Tracks
from .model import smooth_hist, smooth_hist_sq
from .zhash import gauss_from_key, track_id_from_kinematics

MJPSI, MBPLUS = 3.0969, 5.27934


class TrackMorph:
    """MC tracks with a fixed Gaussian per track; corrected momenta are f_i p_i with f_i = corr.factor (direction unchanged).
    Same attribute names as model.MorphedMC (idx, pt, q, z), so the Correction classes work on it unchanged."""

    def __init__(self, t: Tracks, corr, seed=12345, z_mode="hash"):
        self.corr = corr
        self.idx = corr.track_index(t)
        P = t.p
        self.pt = np.hypot(P[..., 0], P[..., 1])
        self.p2 = (P ** 2).sum(-1)
        self.q = t.charge.astype(float)
        if z_mode == "hash":
            tid = track_id_from_kinematics(self.pt, np.arcsinh(P[..., 2] / self.pt), np.arctan2(P[..., 1], P[..., 0]))
            self.z = gauss_from_key(seed, 0, 0, 0, tid)
        elif z_mode == "sequential":
            self.z = np.random.default_rng(seed).standard_normal(self.pt.shape)
        else:
            raise ValueError(f"z_mode {z_mode!r}")
        self.m2 = t.masses ** 2
        self.is_mu = np.array([kd == "mu" for kd in t.kinds])[None, :]          # for corrections with muon- or kaon-specific terms
        self.is_K = np.array([kd == "K" for kd in t.kinds])[None, :]
        k = P.shape[1]
        self.dot = {(i, j): (P[:, i] * P[:, j]).sum(-1) for i in range(k) for j in range(i + 1, k)}
        self.n = len(t)

    def factors(self, cv):
        return np.ones_like(self.pt) if cv is None else self.corr.factor(self, np.asarray(cv, float))

    def mass(self, f, tracks):
        """Invariant mass of the listed tracks with momentum factors f (n, k)."""
        e = {i: np.sqrt(self.p2[:, i] * f[:, i] ** 2 + self.m2[i]) for i in tracks}
        m2 = sum(self.m2[i] for i in tracks)
        for a, i in enumerate(tracks):
            for j in tracks[a + 1:]:
                m2 = m2 + 2 * (e[i] * e[j] - f[:, i] * f[:, j] * self.dot[(min(i, j), max(i, j))])
        return np.sqrt(np.maximum(m2, 0.0))


# ---------------------------------------------------------------------------------------------------- categories
def _bin(x, edges):
    edges = np.asarray(edges, float)
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)


def mumu_categories(eta_edges, dpt_edges):
    """|eta| bin of the mu+ x |eta| bin of the mu- x bin of pT(mu+) - pT(mu-). The pT-difference bins separate M (charge-
    dependent, cancels for equal pT) from A."""
    eta_edges = np.asarray(eta_edges, float); dpt_edges = np.asarray(dpt_edges, float)
    ne, nd = len(eta_edges) - 1, len(dpt_edges) - 1
    def f(t: Tracks):
        plus = (t.charge[:, 0] > 0)
        ip, im = np.where(plus, 0, 1), np.where(plus, 1, 0)
        r = np.arange(len(t)); pt = t.pt(); aeta = np.abs(t.eta())
        return (_bin(aeta[r, ip], eta_edges) * ne + _bin(aeta[r, im], eta_edges)) * nd + _bin(pt[r, ip] - pt[r, im], dpt_edges)
    f.n = ne * ne * nd
    f.labels = [f"mu+ |eta| {eta_edges[a]:.1f}-{eta_edges[a+1]:.1f}, mu- {eta_edges[b]:.1f}-{eta_edges[b+1]:.1f}, "
                f"dpT {dpt_edges[d]:g}..{dpt_edges[d+1]:g}" for a in range(ne) for b in range(ne) for d in range(nd)]
    f.eta_edges = eta_edges
    return f


def jpsik_categories(eta_edges, pt_edges, split_charge=True):
    """Kaon |eta| bin x kaon pT bin x kaon charge."""
    eta_edges = np.asarray(eta_edges, float); pt_edges = np.asarray(pt_edges, float)
    ne, npt, nq = len(eta_edges) - 1, len(pt_edges) - 1, (2 if split_charge else 1)
    def f(t: Tracks):
        c = _bin(np.abs(t.eta()[:, 2]), eta_edges) * npt + _bin(t.pt()[:, 2], pt_edges)
        return c * nq + (t.charge[:, 2] < 0) if split_charge else c
    f.n = ne * npt * nq
    f.labels = [f"K |eta| {eta_edges[a]:.1f}-{eta_edges[a+1]:.1f}, pT {pt_edges[b]:g}-{pt_edges[b+1]:g}" + (["", ", K+", ", K-"][q + 1] if split_charge else "")
                for a in range(ne) for b in range(npt) for q in range(nq)]
    f.eta_edges = eta_edges
    return f


# ---------------------------------------------------------------------------------------------------- mass variables
def m_mumu(tm: TrackMorph, f):
    return tm.mass(f, (0, 1))


class JpsiRfK:
    """m(J/psi_rf K): the stored refitted J/psi (dimuon vertex, J/psi mass imposed) + the kaon.

    The refit cannot be redone after correcting the muons. In the 2018 data the stored refitted momentum is, to 0.4% rms,
    the raw dimuon momentum rescaled to the J/psi mass: |p_rf| = (m_J/psi / m_raw(mu mu)) |p1 + p2| (fitted slope 0.96 in
    log-log; direction within 1e-4 rad). The correction is carried over with the same rule:
        p_rf' = p_rf * w * (m_raw / m_raw')^refit_power,   w = |f1 p1 + f2 p2| / |p1 + p2|,
    m_raw' the corrected dimuon mass. A common muon scale therefore cancels (the mass constraint absorbs it), and the
    variable measures the kaon. refit_power = 0 is the naive 'scale by the muons' mean correction'. With f = 1 this is
    exactly the stored rf_mass."""

    def __init__(self, t: Tracks, tm: TrackMorph, refit_power=1.0):
        from .tracks import p3
        x = t.extra
        pJ = p3(x["jpsi_rf_pt"], x["jpsi_rf_eta"], x["jpsi_rf_phi"]); pK = t.p[:, 2]
        self.mJ2 = x["jpsi_rf_mass"] ** 2
        self.pJ2 = (pJ ** 2).sum(-1); self.dJK = (pJ * pK).sum(-1)
        self.mm2 = (tm.p2[:, 0] + tm.p2[:, 1] + 2 * tm.dot[(0, 1)])          # |p1 + p2|^2
        self.mraw = tm.mass(np.ones_like(tm.pt), (0, 1))
        self.tm = tm; self.refit_power = float(refit_power)

    def __call__(self, f):
        tm = self.tm
        w = np.sqrt((f[:, 0] ** 2 * tm.p2[:, 0] + f[:, 1] ** 2 * tm.p2[:, 1] + 2 * f[:, 0] * f[:, 1] * tm.dot[(0, 1)]) / self.mm2)
        if self.refit_power:
            w = w * (self.mraw / np.maximum(tm.mass(f, (0, 1)), 1e-6)) ** self.refit_power
        eJ = np.sqrt(w ** 2 * self.pJ2 + self.mJ2); eK = np.sqrt(f[:, 2] ** 2 * tm.p2[:, 2] + tm.m2[2])
        m2 = self.mJ2 + tm.m2[2] + 2 * (eJ * eK - w * f[:, 2] * self.dJK)
        return np.sqrt(np.maximum(m2, 0.0))


def m_jpsik_delta(tm: TrackMorph, f):
    """Cross-check variable from the raw corrected tracks only: m(mu mu K) - m(mu mu) + m(J/psi)."""
    return tm.mass(f, (0, 1, 2)) - tm.mass(f, (0, 1)) + MJPSI


# ---------------------------------------------------------------------------------------------------- channels
class Channel:
    """Data histograms and MC templates for one decay. Subclasses set the mass function and the categories."""
    name = "?"
    m0 = 0.0
    components = ()                         # extra peaking templates after the signal, e.g. ("jpsipi",)
    background = "expo2"                    # 'expo2': exp(lam x + kap x^2); 'expo': kap fixed at 0
    min_mc_eff = 25.0                       # Barlow-Beeston-lite only in bins with at least this many effective MC events

    def _setup_bins(self, mrange, bw):
        self.mrange = tuple(mrange); self.bw = bw
        self.edges = np.arange(mrange[0], mrange[1] + bw / 2, bw)
        self.c = 0.5 * (self.edges[1:] + self.edges[:-1])

    def _set_data(self, m, cat, smooth=True):
        self.n_raw = np.array([np.histogram(m[cat == i], self.edges)[0] for i in range(self.ncat)], float)
        self.n = np.array([smooth_hist(m[cat == i], self.edges) for i in range(self.ncat)]) if smooth else self.n_raw
        self.n_data = int(len(m))

    def _sorted(self, cat):
        order = np.argsort(cat, kind="stable")
        return order, np.searchsorted(cat[order], np.arange(self.ncat + 1))

    def _hists(self, m, order, bounds, w=None):
        m = m[order]; w = None if w is None else w[order]
        out = np.zeros((self.ncat, len(self.c)))
        for i in range(self.ncat):
            a, b = bounds[i], bounds[i + 1]
            if b > a:
                h = smooth_hist(m[a:b], self.edges, None if w is None else w[a:b]); s = h.sum()
                out[i] = h / s if s > 0 else h
        return out

    def _relvar(self, m, order, bounds, w=None):
        """Relative statistical variance per bin of the (unnormalised) smoothed MC histogram, (ncat, nbins); 0 where empty."""
        m = m[order]; w = None if w is None else w[order]
        out = np.zeros((self.ncat, len(self.c)))
        for i in range(self.ncat):
            a, b = bounds[i], bounds[i + 1]
            if b > a:
                ww = None if w is None else w[a:b]
                h = smooth_hist(m[a:b], self.edges, ww); h2 = smooth_hist_sq(m[a:b], self.edges, ww)
                rv = np.where(h > 0, h2 / np.maximum(h, 1e-300) ** 2, 0.0)
                # Gaussian Barlow-Beeston-lite is only valid where the bin holds enough MC: in sparse tails the factor would
                # inflate the signal template to imitate the background. Bins below min_mc_eff effective MC events: no factor.
                out[i] = np.where(rv * self.min_mc_eff <= 1.0, rv, 0.0)
        return out

    def mc_relvar(self, cv):
        """Relative MC-statistics variance of the signal template per bin, at correction cv (for the Barlow-Beeston-lite
        treatment)."""
        return self._relvar(self.masses(cv), self._order, self._bounds, self.mc_w)

    @property
    def labels(self):
        return [f"{self.name}: {l}" for l in self.cat.labels]

    @property
    def ncat(self):
        return self.cat.n

    def templates(self, cv):
        """(n_components, ncat, nbins), each row normalised to 1 (0 where the component has no MC)."""
        raise NotImplementedError


class MuMuChannel(Channel):
    name = "mumu"; m0 = MJPSI

    def __init__(self, data: Tracks, mc: Tracks, corr, mrange=(2.96, 3.24), bw=0.002, eta_edges=None, dpt_edges=(-1e3, -4, -1, 1, 4, 1e3),
                 seed=12345, z_mode="hash", smooth_data=True, background="expo", bkg_share="channel"):
        self.corr = corr; self.background = background
        self.cat = mumu_categories(eta_edges if eta_edges is not None else corr.eta_edges, dpt_edges)
        nd = len(dpt_edges) - 1
        self.bkg_groups = {"channel": np.zeros(self.cat.n, int), "category": np.arange(self.cat.n),
                           "eta": np.arange(self.cat.n) // nd}[bkg_share]
        self._setup_bins(mrange, bw)
        dm = TrackMorph(data, corr, seed, z_mode)
        self._set_data(m_mumu(dm, dm.factors(None)), self.cat(data), smooth_data); del dm
        self.mc = TrackMorph(mc, corr, seed, z_mode); self.mc_cat = self.cat(mc); self.mc_w = None
        self._order, self._bounds = self._sorted(self.mc_cat)
        self.n_mc = len(mc)

    def masses(self, cv):
        return m_mumu(self.mc, self.mc.factors(cv))

    def templates(self, cv):
        return self._hists(self.masses(cv), self._order, self._bounds, self.mc_w)[None]


MPIPLUS = 0.13957


def argus_gauss(edges, c, sigma, m0=MBPLUS - MPIPLUS, p=0.5, step=2.5e-4):
    """ARGUS (RooArgusBG: m (1 - t)^p exp(c (1 - t)), t = (m/m0)^2, zero above m0) convolved with a Gaussian of width sigma,
    integrated in the bins `edges`, normalised to 1. Partially reconstructed B -> J/psi K X with a lost particle: the endpoint
    m(B+) - m(pi+) and p = 0.5 are fixed, as in the Vcb J/psi K validation (config.py)."""
    lo, hi = edges[0] - 6 * sigma, edges[-1] + 6 * sigma
    x = np.arange(lo, hi + step, step)
    t = np.clip((x / m0) ** 2, 0, 1)
    a = np.where(x < m0, x * (1 - t) ** p * np.exp(c * (1 - t)), 0.0)
    k = np.arange(-6 * sigma, 6 * sigma + step, step); g = np.exp(-0.5 * (k / sigma) ** 2)
    y = np.convolve(a, g / g.sum(), mode="same")
    cum = np.concatenate([[0.0], np.cumsum(y)])
    idx = np.clip(np.searchsorted(x, edges), 0, len(x))
    h = np.diff(cum[idx]); s_ = h.sum()
    return h / s_ if s_ > 0 else h


class JpsiKChannel(Channel):
    name = "jpsik"; m0 = MBPLUS

    def __init__(self, data: Tracks, mc: Tracks, corr, mc_x: Tracks | None = None, mrange=(5.17, 5.45), bw=0.002, eta_edges=None,
                 pt_edges=(2.0, 4.0, 7.0, 1e3), split_charge=True, seed=12345, z_mode="hash", smooth_data=True, refit_power=1.0, background="expo2", bkg_share="category",
                 argus=False):
        self.corr = corr; self.background = background; self.use_argus = bool(argus); self.argus_T = None; self.argus_par = None
        self.cat = jpsik_categories(eta_edges if eta_edges is not None else corr.eta_edges, pt_edges, split_charge)
        per_eta = (len(pt_edges) - 1) * (2 if split_charge else 1)
        self.bkg_groups = {"channel": np.zeros(self.cat.n, int), "category": np.arange(self.cat.n),
                           "eta": np.arange(self.cat.n) // per_eta}[bkg_share]
        self._setup_bins(mrange, bw)
        dm = TrackMorph(data, corr, seed, z_mode)
        self._set_data(JpsiRfK(data, dm)(dm.factors(None)), self.cat(data), smooth_data); del dm
        self.mc = TrackMorph(mc, corr, seed, z_mode); self.mfun = JpsiRfK(mc, self.mc, refit_power); self.mc_w = None
        self._order, self._bounds = self._sorted(self.cat(mc))
        self.n_mc = len(mc)
        self.components = ()
        if mc_x is not None and len(mc_x):
            self.components = ("jpsipi",)
            self.mcx = TrackMorph(mc_x, corr, seed, z_mode); self.mfunx = JpsiRfK(mc_x, self.mcx, refit_power)
            self._orderx, self._boundsx = self._sorted(self.cat(mc_x))
            self.n_mcx = len(mc_x)
        if self.use_argus:
            self.components = self.components + ("argus",)

    def masses(self, cv):
        return self.mfun(self.mc.factors(cv))

    def _mc_templates(self, cv):
        T = [self._hists(self.masses(cv), self._order, self._bounds, self.mc_w)]
        if hasattr(self, "mfunx"):
            T.append(self._hists(self.mfunx(self.mcx.factors(cv)), self._orderx, self._boundsx))
        return T

    def templates(self, cv):
        T = self._mc_templates(cv)
        if self.use_argus:
            if self.argus_T is None:
                raise RuntimeError("jpsik: ARGUS shapes not prepared (call prepare(cv) first)")
            T.append(self.argus_T)
        return np.stack(T)

    def prepare(self, cv, log=None):
        """Fit, per category, the ARGUS (x) Gauss shape (curvature c and Gaussian width) to the data with the correction at cv,
        together with the signal and J/psi pi templates, an exp-of-quadratic background and free yields. The shapes are then
        held fixed; only the ARGUS yield floats in the main fit (the partially reconstructed region, below 5.14 GeV, is far
        from the peak and hardly depends on the momentum correction)."""
        if not self.use_argus:
            return
        from iminuit import Minuit
        T = self._mc_templates(cv); x = self.c - self.m0; out = []; par = []
        for i in range(self.ncat):
            n = self.n[i]; Ts = T[0][i]; Tx = T[1][i] if len(T) > 1 else np.zeros_like(Ts); tot = max(n.sum(), 1.0)
            def nll(Ns, Nb, lam, kap, Npi, Na, c, sig):
                y = np.exp(lam * x + kap * x * x); y = y / y.sum()
                mu = Ns * Ts + Nb * y + Npi * Tx + Na * argus_gauss(self.edges, c, sig)
                mu = np.maximum(mu, 1e-12); return 2 * np.sum(mu - n * np.log(mu))
            m = Minuit(nll, Ns=0.7 * tot, Nb=0.2 * tot, lam=-2.0, kap=0.0, Npi=0.02 * tot, Na=0.08 * tot, c=-10.0, sig=0.02)
            m.errordef = 1; m.print_level = 0
            for k in ("Ns", "Nb", "Npi", "Na"): m.limits[k] = (0, None)
            m.limits["lam"] = (-60, 60); m.limits["kap"] = (-400, 400); m.limits["c"] = (-100, 20); m.limits["sig"] = (0.005, 0.08)
            if self.background == "expo": m.fixed["kap"] = True
            if Tx.sum() == 0: m.values["Npi"] = 0; m.fixed["Npi"] = True
            m.migrad()
            if not m.valid: m.simplex(); m.migrad()
            par.append(dict(c=float(m.values["c"]), sigma=float(m.values["sig"]), Na=float(m.values["Na"]), valid=bool(m.valid)))
            out.append(argus_gauss(self.edges, m.values["c"], m.values["sig"]))
        self.argus_T = np.array(out); self.argus_par = par
        if log is not None:
            log("  jpsik ARGUS (x) Gauss per category: " + ", ".join(f"c {p['c']:.1f} sigma {1e3 * p['sigma']:.0f} MeV" for p in par), flush=True)


class KKPiChannel(Channel):
    """The v3 fit (ScaleSmearFit) as a channel: same data histograms, templates, m(KK) re-cut on the corrected MC."""
    name = "kkpi"; m0 = 1.968

    def __init__(self, fit):
        if getattr(fit, "grouped", False):
            raise ValueError("kkpi channel: shared background shapes (share_bkg_shape) are not supported in the combined fit")
        self.fit = fit; self.corr = fit.corr; self.cat = fit.cat
        fit.mc.is_K = np.array([[True, True, False]]); fit.mc.is_mu = np.array([[False, False, False]])   # tracks k1, k2, pi
        self.edges, self.c = fit.edges, fit.c; self.mrange = (float(fit.edges[0]), float(fit.edges[-1])); self.bw = float(fit.edges[1] - fit.edges[0])
        self.n_raw = np.array(fit.n_raw); self.n = np.array(fit.n); self.n_data = int(fit.config["n_data"]); self.n_mc = int(fit.config["n_mc"])
        if fit.comb != "expo2":
            raise ValueError("kkpi channel: only the expo2 background is supported in the combined fit")

    def mc_relvar(self, cv):
        from .model import MPHI
        f = self.fit; m, mkk = f.mc.masses(np.asarray(cv, float))
        w = (np.abs(mkk - MPHI) < f.kk_window).astype(float) if f.recut_kk else None
        if f.mc_w is not None: w = f.mc_w if w is None else w * f.mc_w
        order = np.argsort(f.mc_cat, kind="stable"); bounds = np.searchsorted(f.mc_cat[order], np.arange(self.ncat + 1))
        return self._relvar(m, order, bounds, w)

    @property
    def mc_w(self):
        return self.fit.mc_w

    @mc_w.setter
    def mc_w(self, w):
        self.fit.mc_w = w

    def templates(self, cv):
        return np.array(self.fit.templates(np.asarray(cv, float)))[None]
