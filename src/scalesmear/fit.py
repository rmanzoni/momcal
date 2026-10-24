"""Binned likelihood fit of the MC track correction to data."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
import numpy as np
from iminuit import Minuit
from .io import Sample
from .model import Correction, MorphedMC, smooth_hist, expo_shape, MPHI


def make_categories(eta_edges, pt_edges=None, split_charge=False):
    """Categories = max |eta| bin of the three tracks  x  pT bin of the softest track (optional)  x  Ds charge (optional,
    from the pion charge). Splitting by charge separates charge-dependent curvature biases."""
    eta_edges = np.asarray(eta_edges, float); pt_edges = None if pt_edges is None else np.asarray(pt_edges, float)
    ne = len(eta_edges) - 1; npt = 1 if pt_edges is None else len(pt_edges) - 1; nq = 2 if split_charge else 1
    def f(sample: Sample):
        c = np.clip(np.digitize(np.abs(sample.eta()).max(1), eta_edges[1:-1]), 0, ne - 1)
        if pt_edges is not None:
            ptmin = np.hypot(sample.p[..., 0], sample.p[..., 1]).min(1)
            c = c * npt + np.clip(np.digitize(ptmin, pt_edges[1:-1]), 0, npt - 1)
        if split_charge:
            c = c * 2 + (sample.charge[:, 2] < 0)
        return c
    f.n = ne * npt * nq
    labels = []
    for i in range(ne):
        for j in range(npt):
            for k in range(nq):
                l = f"max|eta| {eta_edges[i]:.1f}-{eta_edges[i+1]:.1f}"
                if pt_edges is not None: l += f", min pT {pt_edges[j]:.1f}-{pt_edges[j+1]:.1f}"
                if split_charge: l += ", Ds+" if k == 0 else ", Ds-"
                labels.append(l)
    f.labels = labels
    return f


def maxeta_categories(edges):
    """Category = |eta| bin of the most forward of the three tracks."""
    edges = np.asarray(edges, float)
    def f(sample: Sample):
        a = np.abs(sample.eta()).max(1)
        return np.clip(np.digitize(a, edges[1:-1]), 0, len(edges) - 2)
    f.n = len(edges) - 1
    f.labels = [f"max|eta| {edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    return f


@dataclass
class FitResult:
    names: list
    values: dict
    errors: dict
    corr_names: list
    corr_cov: np.ndarray
    valid: bool
    nll: float
    chi2: list
    ndf: list
    labels: list
    config: dict = field(default_factory=dict)

    def correction_values(self):
        return np.array([self.values[n] for n in self.corr_names])

    def summary(self):
        lines = [f"valid minimum: {self.valid}"]
        units = {"s0": (1e3, "1e-3"), "s1": (1e3, "MeV"), "k": (1e4, "1e-4/GeV"), "v0": (1.0, "(1e-3)^2"), "v1": (1.0, "(1e-4/GeV)^2"),
                 "s": (1e3, "1e-3"), "r": (1e3, "1e-3"), "r0": (1e3, "1e-3"), "r1": (1e4, "1e-4/GeV")}
        for n in self.corr_names:
            f, unit = units.get(n.rsplit("_", 1)[0], units.get(n.split("_")[0], (1e3, "1e-3")))
            lines.append(f"  {n:10s} = {f*self.values[n]:+9.4f} +- {f*self.errors[n]:.4f}  [{unit}]")
        lines.append(f"  total chi2/ndf = {sum(self.chi2):.1f}/{sum(self.ndf)}")
        for l, c, d in zip(self.labels, self.chi2, self.ndf):
            lines.append(f"  {l:24s} chi2/ndf = {c:.1f}/{d}")
        return "\n".join(lines)

    def to_json(self, path):
        d = dict(values=self.values, errors=self.errors, correction=self.corr_names,
                 correction_cov=np.asarray(self.corr_cov).tolist(), valid=self.valid, nll=self.nll,
                 chi2=self.chi2, ndf=self.ndf, categories=self.labels, config=self.config)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


class ScaleSmearFit:
    """Fit data m(KK pi) with corrected MC + an exponential combinatorial per category.

    data, mc     : Sample
    corr         : Correction (eta binning and species of the track parameters)
    categories   : callable(Sample) -> int array, with attributes .n and .labels; default: max |eta| of the tracks
    mrange       : fit range in GeV; stay away from the preselection edges (1.908, 2.028)
    bw           : bin width in GeV
    comb         : 'expo' or 'expo2' (curved exponent)
    ws           : optional wrong-sign Sample; if given, the comb slope of each category is Gaussian-constrained to WS
    recut_kk     : re-apply |m(KK) - m_phi| < kk_window to the corrected MC
    fix          : dict of correction parameters to fix, e.g. {'s_all1': 0.0}
    """

    def __init__(self, data: Sample, mc: Sample, corr: Correction, categories=None, mrange=(1.920, 2.015), bw=0.001,
                 comb="expo2", ws: Sample | None = None, recut_kk=True, kk_window=0.020, fix=None, seed=12345, verbose=False, smooth_data=True,
                 z_mode="sequential"):
        self.corr = corr; self.verbose = verbose
        self.cat = categories or maxeta_categories(corr.eta_edges)
        self.edges = np.arange(mrange[0], mrange[1] + bw / 2, bw)
        self.c = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.comb = comb
        self.recut_kk, self.kk_window = recut_kk, kk_window
        self.mc = MorphedMC(mc, corr, seed, z_mode=z_mode)
        self.mc_cat = self.cat(mc)
        self.mc_w = None                     # optional per-candidate MC weights (used by mc_stat_bootstrap)
        self._start0 = None; self.ncalls = 0
        self.progress = None                 # optional callable(ncalls, -2lnL), called after every likelihood evaluation
        dcat = self.cat(data)
        # the data histogram is filled with the same kernel as the template, so both carry the same extra width
        # (an unsmoothed data histogram against a smoothed template biases the fitted resolution low)
        self.smooth_data = smooth_data
        self.n_raw = [np.histogram(data.m_kkpi[dcat == i], self.edges)[0].astype(float) for i in range(self.cat.n)]
        self.n = [smooth_hist(data.m_kkpi[dcat == i], self.edges) for i in range(self.cat.n)] if smooth_data else self.n_raw
        self.ws_con = None
        if ws is not None:
            wcat = self.cat(ws)
            self.ws_con = [self._ws_slope(np.histogram(ws.m_kkpi[wcat == i], self.edges)[0].astype(float)) for i in range(self.cat.n)]
        self.fix = fix or {}
        self.config = dict(eta_edges=list(map(float, corr.eta_edges)), species=corr.species, mrange=list(mrange), bw=bw,
                           comb=comb, recut_kk=recut_kk, kk_window=kk_window, ws_constraint=ws is not None, seed=seed, z_mode=z_mode,
                           n_data=int(len(data)), n_mc=int(len(mc)), categories=self.cat.labels)

    def _ws_slope(self, nw):
        def nll(N, lam):
            mu = N * expo_shape(self.c, lam)
            return 2 * np.sum(mu - nw * np.log(np.maximum(mu, 1e-12)))
        m = Minuit(nll, N=nw.sum(), lam=-2.0); m.errordef = 1; m.migrad(); m.hesse()
        return m.values["lam"], m.errors["lam"]

    def templates(self, cvals):
        m, mkk = self.mc.masses(cvals)
        w = (np.abs(mkk - MPHI) < self.kk_window).astype(float) if self.recut_kk else None
        if self.mc_w is not None:
            w = self.mc_w if w is None else w * self.mc_w
        if not hasattr(self, "_order"):                         # sort the MC by category once: contiguous slices afterwards
            self._order = np.argsort(self.mc_cat, kind="stable")
            self._bounds = np.searchsorted(self.mc_cat[self._order], np.arange(self.cat.n + 1))
        m = m[self._order]; w = None if w is None else w[self._order]
        out = []
        for i in range(self.cat.n):
            a, b = self._bounds[i], self._bounds[i + 1]
            h = smooth_hist(m[a:b], self.edges, None if w is None else w[a:b])
            out.append(h / max(h.sum(), 1e-12))
        return out

    def _unpack(self, a):
        k = len(self.corr.names)
        cv = np.asarray(a[:k]); rest = np.asarray(a[k:]).reshape(self.cat.n, 4)
        return cv, rest

    def model(self, a):
        cv, rest = self._unpack(a)
        T = self.templates(cv)
        mus = []
        for i, (Ns, Nb, lam, kap) in enumerate(rest):
            mus.append(Ns * T[i] + Nb * expo_shape(self.c, lam, kap if self.comb == "expo2" else 0.0))
        return mus, T

    def nll(self, *a):
        mus, _ = self.model(a)
        v = 0.0
        for n, mu in zip(self.n, mus):
            mu = np.maximum(mu, 1e-12); v += 2 * np.sum(mu - n * np.log(mu))
        if self.ws_con is not None:
            _, rest = self._unpack(a)
            for (l0, e0), r in zip(self.ws_con, rest):
                v += ((r[2] - l0) / e0) ** 2
        return v

    # ---- profiled fit: outer minimisation over the correction, inner per-category fit of the nuisances
    def _inner(self, i, T, start):
        n = self.n[i]
        def nll(Ns, Nb, lam, kap):
            mu = np.maximum(Ns * T + Nb * expo_shape(self.c, lam, kap), 1e-12)
            v = 2 * np.sum(mu - n * np.log(mu))
            if self.ws_con is not None:
                l0, e0 = self.ws_con[i]; v += ((lam - l0) / e0) ** 2
            return v
        m = Minuit(nll, *start, name=["Ns", "Nb", "lam", "kap"]); m.errordef = 1; m.tol = 1e-6; m.print_level = 0
        m.limits["Ns"] = (0, None); m.limits["Nb"] = (0, None); m.limits["lam"] = (-15, 5); m.limits["kap"] = (-200, 200)
        if self.comb != "expo2": m.values["kap"] = 0.0; m.fixed["kap"] = True
        m.migrad()
        if not m.valid:                                         # rare: retry from scratch before giving up
            m.values = [0.6 * n.sum(), 0.4 * n.sum(), -2.5, 0.0]; m.simplex(); m.migrad()
        return m.fval, list(m.values)

    def profile_nll(self, cv):
        T = self.templates(cv)
        tot = 0.0
        for i in range(self.cat.n):
            # always start the nuisances from the same point (their best values at the starting correction), so the
            # profiled likelihood depends only on the correction values and not on the path Minuit took to get there
            f, v = self._inner(i, T[i], self._start0[i] if self._start0 is not None else self._start[i]); tot += f; self._start[i] = v
        self.ncalls += 1
        if self.progress is not None: self.progress(self.ncalls, tot)
        if self.verbose and self.ncalls % 50 == 0: print(f'  call {self.ncalls}: -2lnL {tot:.2f}', flush=True)
        return tot

    def full_hesse(self, values, progress=None):
        """Errors from one Minuit fit of ALL parameters (correction + per-category yields and background shapes), started at
        a profiled-fit result. Unlike the profiled likelihood, this function contains no inner minimisation, so it has no
        numerical noise from the inner fits; HESSE then gives stable errors. Templates are cached by correction values, so
        steps in the nuisance directions cost almost nothing. Returns the Minuit object."""
        k = len(self.corr.names); cache = {}
        names = list(self.corr.names) + [f"{q}_{i}" for i in range(self.cat.n) for q in ("Ns", "Nb", "lam", "kap")]
        cnt = [0]
        def nll(*a):
            cnt[0] += 1
            if progress is not None: progress(cnt[0])
            key = tuple(a[:k])
            if key not in cache:
                if len(cache) > 256: cache.clear()
                cache[key] = self.templates(np.asarray(key))
            T = cache[key]; v = 0.0
            for i in range(self.cat.n):
                Ns, Nb, lam, kap = a[k + 4 * i: k + 4 * i + 4]
                mu = np.maximum(Ns * T[i] + Nb * expo_shape(self.c, lam, kap if self.comb == "expo2" else 0.0), 1e-12)
                v += 2 * np.sum(mu - self.n[i] * np.log(mu))
            return v
        m = Minuit(nll, *[values[n] for n in names], name=names); m.errordef = 1; m.tol = 1.0
        for x in self.corr.names:
            m.limits[x] = self.corr.limits(x); m.errors[x] = self.corr.step(x)
        for i in range(self.cat.n):
            m.limits[f"Ns_{i}"] = (0, None); m.limits[f"Nb_{i}"] = (0, None)
            if self.comb != "expo2": m.fixed[f"kap_{i}"] = True
        m.migrad(); m.hesse()
        return m

    def fit_full(self, start=None, progress=None) -> FitResult:
        """Recommended fit: the per-category yields and background shapes are fitted once at the starting correction, then ONE
        Minuit fit of all parameters (correction + nuisances) gives values and errors. In toys this returns the same values and
        errors as the profiled fit followed by full_hesse, in less time, and its errors are correct (pull widths 0.9-1.2).
        chi2 is computed from the raw (unsmoothed) counts, so it measures the fit quality."""
        cn = list(self.corr.names); start = start or {}
        st = {x: start.get(x, self.corr.start(x)) for x in cn}; cv = np.array([st[x] for x in cn])
        self._start = [[0.6 * n.sum(), 0.4 * n.sum(), -2.5, 0.0] for n in self.n]; self._start0 = None; self.ncalls = 0
        self.profile_nll(cv)
        vals = dict(st); vals.update({f"{q}_{i}": float(v) for i, nu in enumerate(self._start) for q, v in zip(("Ns", "Nb", "lam", "kap"), nu)})
        m = self.full_hesse(vals, progress=progress); self.minuit = m
        names = list(m.parameters); k = len(cn)
        mus, _ = self.model([m.values[x] for x in names])
        chi2 = [float(((n - mu) ** 2 / np.maximum(mu, 1)).sum()) for n, mu in zip(self.n_raw, mus)]
        nfree_cat = 4 if self.comb == "expo2" else 3
        ndf = [int(len(self.c) - nfree_cat - round(k / self.cat.n)) for _ in self.n]
        return FitResult(names, {x: float(m.values[x]) for x in names}, {x: float(m.errors[x]) for x in names}, cn,
                         np.array(m.covariance)[:k, :k], bool(m.valid), float(m.fval), chi2, ndf, self.cat.labels, self.config)

    def wide_hessian(self, cv, h):
        """Covariance of the correction from a Hessian of the profiled -2lnL by central differences with steps h (one per
        parameter, of the size of the expected errors). Minuit's HESSE uses much smaller steps, at which the histogram-based
        likelihood has small wiggles; its curvature can then be too large and its errors too small (seen in toys).
        2 n^2 + 1 likelihood calls. Returns the covariance (2 H^-1, since the function is -2lnL)."""
        cv = np.asarray(cv, float); n = len(cv); h = np.asarray(h, float)
        f0 = self.profile_nll(cv); H = np.zeros((n, n))
        def f(d): return self.profile_nll(cv + d)
        for i in range(n):
            e = np.zeros(n); e[i] = h[i]
            H[i, i] = (f(e) - 2 * f0 + f(-e)) / h[i] ** 2
            for j in range(i):
                e2 = np.zeros(n); e2[j] = h[j]
                H[i, j] = H[j, i] = (f(e + e2) - f(e - e2) - f(-e + e2) + f(-e - e2)) / (4 * h[i] * h[j])
        return 2 * np.linalg.inv(H)

    def fit(self, strategy=1, start=None) -> FitResult:
        cn = list(self.corr.names)
        self.ncalls = 0
        self._start = [[0.6 * n.sum(), 0.4 * n.sum(), -2.5, 0.0] for n in self.n]
        p0 = [self.fix.get(x, self.corr.start(x)) if start is None else start.get(x, self.corr.start(x)) for x in cn]
        self._start0 = None
        self.profile_nll(p0)                                    # nuisances at the starting correction
        self._start0 = [list(v) for v in self._start]
        m = Minuit(lambda *a: self.profile_nll(a), *p0, name=cn); m.errordef = 1; m.strategy = strategy
        # stop when the estimated distance to the minimum is below 0.002 in -2lnL (tol=1), i.e. far below the 1-unit change
        # that defines the errors; chasing Minuit's default (0.0002) only costs calls on a likelihood built from a histogram
        m.tol = 1.0
        for x in cn:
            m.limits[x] = self.corr.limits(x)
            m.errors[x] = self.corr.step(x)
            if x in self.fix: m.fixed[x] = True
        if self.verbose: print('  fit start', dict(zip(cn, p0)), flush=True)
        m.migrad()
        if not m.valid: m.migrad()                              # one more pass if the distance to the minimum is not yet small
        m.hesse()
        cv = np.array(m.values)
        self.profile_nll(cv)
        names = cn + [f"{q}_{i}" for i in range(self.cat.n) for q in ("Ns", "Nb", "lam", "kap")]
        vals = list(map(float, cv)) + [float(x) for st in self._start for x in st]
        errs = list(map(float, m.errors)) + [float("nan")] * (4 * self.cat.n)
        mus, _ = self.model(vals)
        chi2 = [float(((n - mu) ** 2 / np.maximum(mu, 1)).sum()) for n, mu in zip(self.n, mus)]
        nfree_cat = 4 if self.comb == "expo2" else 3
        ncorr_free = sum(1 for x in cn if x not in self.fix)
        ndf = [int(len(self.c) - nfree_cat - round(ncorr_free / self.cat.n)) for _ in self.n]
        self.minuit = m; self.edm = float(m.fmin.edm)
        return FitResult(names, dict(zip(names, vals)), dict(zip(names, errs)), cn, np.array(m.covariance),
                         bool(m.valid), float(m.fval), chi2, ndf, self.cat.labels, self.config)


def mc_stat_bootstrap(fit: ScaleSmearFit, n=10, seed=1):
    """Refit n times with Poisson(1) weights on the MC candidates; the spread of the correction parameters is the
    uncertainty from the finite MC sample (not included in the HESSE errors). Returns an (n, n_par) array."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        fit.mc_w = rng.poisson(1.0, fit.mc.n).astype(float)
        out.append(fit.fit().correction_values())
    fit.mc_w = None
    return np.array(out)
