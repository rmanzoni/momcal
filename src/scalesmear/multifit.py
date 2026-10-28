"""One binned likelihood over several channels that share the per-track correction.

Parameters: the correction terms (shared), then per channel and category the nuisances
    Ns (signal yield), Nb (background yield), lam, kap (background shape exp(lam (m - m0) + kap (m - m0)^2)),
    N<component> (yield of each extra peaking template, e.g. Njpsipi).
The fit is the v3 'fit_full': nuisances fitted once per category at the starting correction, then ONE Minuit fit of all
parameters for values and errors. Templates are cached per correction values, so steps in the nuisance directions are cheap.
"""
from __future__ import annotations
import time
import numpy as np
from iminuit import Minuit
from .fit import FitResult


def expo_rows(c, lam, kap, m0):
    """exp(lam (c - m0) + kap (c - m0)^2), each row normalised to 1; lam, kap of shape (ncat,)."""
    x = c[None, :] - m0
    y = np.exp(lam[:, None] * x + kap[:, None] * x * x)
    return y / y.sum(1, keepdims=True)


class MultiFit:
    def __init__(self, channels, corr, fix=None):
        self.ch = list(channels); self.corr = corr; self.fix = dict(fix or {})
        for c in self.ch:
            if c.corr is not corr:
                raise ValueError(f"channel {c.name}: built with a different Correction object")
        self.cn = list(corr.names); k = len(self.cn)
        # per channel: per-category columns (Ns, Nb, N<comp>...) and background-shape columns (lam, kap) that may be shared
        # by several categories (channel attribute bkg_groups: group of each category)
        self.layout = []; names = list(self.cn); pos = k
        for c in self.ch:
            qc = ["Ns", "Nb"] + [f"N{x}" for x in c.components]
            g = np.asarray(getattr(c, "bkg_groups", None) if getattr(c, "bkg_groups", None) is not None else np.arange(c.ncat))
            groups = list(dict.fromkeys(g.tolist())); shared = len(groups) < c.ncat
            ixc = pos + np.arange(c.ncat * len(qc)).reshape(c.ncat, len(qc))
            names += [f"{c.name}.{x}_{i}" for i in range(c.ncat) for x in qc]; pos += c.ncat * len(qc)
            ixg = pos + np.arange(len(groups) * 2).reshape(len(groups), 2)
            sn = ("lamg", "kapg") if shared else ("lam", "kap")
            names += [f"{c.name}.{x}_{gg}" for gg in groups for x in sn]; pos += len(groups) * 2
            gpos = {gg: j for j, gg in enumerate(groups)}
            ix = np.concatenate([ixc[:, :2], np.array([ixg[gpos[gg]] for gg in g]), ixc[:, 2:]], 1)   # Ns, Nb, lam, kap, comps
            self.layout.append(dict(c=c, ix=ix, groups=g, gnames=[f"{c.name}.{x}_{gg}" for gg in groups for x in sn],
                                    shape_names=sn, gpos=gpos, qc=qc))
        self.names = names
        self.nuis_names = names[k:]
        import os
        self.progress = None; self._cache = {}; self.min_entries = 100; self.threads = os.cpu_count() or 1; self.use_grad = False; self.bb = False

    # ------------------------------------------------------------------ model
    def templates(self, cv):
        key = tuple(np.round(np.asarray(cv, float), 15))
        if key not in self._cache:
            if len(self._cache) > 64: self._cache.clear()
            if self.threads > 1 and len(self.ch) > 1:            # numpy releases the GIL in the heavy loops: channels in parallel
                from concurrent.futures import ThreadPoolExecutor
                if not hasattr(self, "_pool"): self._pool = ThreadPoolExecutor(min(self.threads, len(self.ch)))
                self._cache[key] = list(self._pool.map(lambda c: c.templates(np.asarray(key)), self.ch))
            else:
                self._cache[key] = [c.templates(np.asarray(key)) for c in self.ch]
        return self._cache[key]

    def channel_model(self, j, a, T):
        """Expected counts (ncat, nbins) of channel j for the full parameter vector a, and the components."""
        L = self.layout[j]; c, ix = L["c"], L["ix"]; a = np.asarray(a, float)
        Ns, Nb, lam, kap = (a[ix[:, i]] for i in range(4))
        sig = Ns[:, None] * T[0]; bkg = Nb[:, None] * expo_rows(c.c, lam, kap, c.m0)
        comps = [a[ix[:, 4 + i]][:, None] * T[1 + i] for i in range(len(c.components))]
        return sig + bkg + sum(comps), sig, bkg, comps

    @staticmethod
    def _bb_beta(s, r, n, rv):
        """Barlow-Beeston-lite: per bin, the factor beta on the signal template s (relative MC variance rv) that minimises
        2 (beta s + r - n ln(beta s + r)) + (beta - 1)^2 / rv; closed form (positive root of a quadratic)."""
        ok = (rv > 0) & (s > 0)
        rv_ = np.where(ok, rv, 1.0); s_ = np.where(ok, s, 1.0)
        A = s_ / rv_; B = s_ * s_ + (r - s_) / rv_; C = s_ * (r - n) - r / rv_
        beta = (-B + np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))) / (2 * A)
        return np.where(ok, np.maximum(beta, 0.0), 1.0)

    def channel_nll(self, j, a, T):
        """-2lnL of channel j (with the Barlow-Beeston-lite factors profiled when self.bb is on); also returns beta."""
        c = self.ch[j]; mu, sig, _, _ = self.channel_model(j, a, T); beta = None
        rv = getattr(c, "relvar", None) if self.bb else None
        if rv is not None:
            beta = self._bb_beta(sig, mu - sig, c.n, rv); mu = mu + (beta - 1) * sig
        mu = np.maximum(mu, 1e-12); v = 2 * np.sum(mu - c.n * np.log(mu))
        if beta is not None:
            v += np.sum(np.where(rv > 0, (beta - 1) ** 2 / np.where(rv > 0, rv, 1.0), 0.0))
        return v, beta

    def nll(self, *a):
        a = np.asarray(a, float); T = self.templates(a[:len(self.cn)])
        return sum(self.channel_nll(j, a, T[j])[0] for j in range(len(self.ch)))

    def grad(self, *a, fixed=None):
        """Gradient of -2lnL: analytic in the nuisances (yields, background shapes), central differences of the likelihood in
        the correction terms (steps corr.step; the templates of the shifted points are cached)."""
        a = np.asarray(a, float); k = len(self.cn); g = np.zeros_like(a)
        T = self.templates(a[:k])
        for j, L in enumerate(self.layout):
            c, ix = L["c"], L["ix"]
            mu, sig, bkg, comps = self.channel_model(j, a, T[j])
            w = 2 * (1 - c.n / np.maximum(mu, 1e-12)); w[mu <= 1e-12] = 0.0
            x = c.c - c.m0
            Ns, Nb, lam, kap = (a[ix[:, i]] for i in range(4))
            E = expo_rows(c.c, lam, kap, c.m0)
            mx = (E * x).sum(1, keepdims=True); mx2 = (E * x * x).sum(1, keepdims=True)
            np.add.at(g, ix[:, 0], (w * T[j][0]).sum(1))
            np.add.at(g, ix[:, 1], (w * E).sum(1))
            np.add.at(g, ix[:, 2], (w * Nb[:, None] * E * (x - mx)).sum(1))
            np.add.at(g, ix[:, 3], (w * Nb[:, None] * E * (x * x - mx2)).sum(1))
            for i in range(len(c.components)):
                np.add.at(g, ix[:, 4 + i], (w * T[j][1 + i]).sum(1))
        for i, n in enumerate(self.cn):
            if fixed is not None and fixed[i]:
                continue
            h = self.corr.step(n); ap = a.copy(); am = a.copy(); ap[i] += h; am[i] -= h
            g[i] = (self.nll(*ap) - self.nll(*am)) / (2 * h)
        return g

    # ------------------------------------------------------------------ start values
    def _inner_start(self, cv):
        """Per category, nuisances fitted with the correction at cv (as in v3); shared shapes start at the background-weighted
        mean of their categories."""
        T = self.templates(cv); out = {}
        for j, L in enumerate(self.layout):
            c = L["c"]; per = []
            for i in range(c.ncat):
                n = c.n[i]; Ts = [T[j][k][i] for k in range(len(T[j]))]
                x = c.c - c.m0
                def nll(*p):
                    Ns, Nb, lam, kap = p[:4]
                    y = np.exp(lam * x + kap * x * x); y = y / y.sum()
                    mu = Ns * Ts[0] + Nb * y + sum(p[4 + k] * Ts[1 + k] for k in range(len(c.components)))
                    mu = np.maximum(mu, 1e-12); return 2 * np.sum(mu - n * np.log(mu))
                tot = max(n.sum(), 1.0)
                q = ["Ns", "Nb", "lam", "kap"] + [f"N{x}" for x in c.components]
                st = [0.8 * tot, 0.2 * tot, -1.0, 0.0] + [0.02 * tot] * len(c.components)
                m = Minuit(nll, *st, name=q); m.errordef = 1; m.print_level = 0
                m.limits["Ns"] = (0, None); m.limits["Nb"] = (0, None); m.limits["lam"] = (-60, 60); m.limits["kap"] = (-400, 400)
                if getattr(c, "background", "expo2") == "expo": m.values["kap"] = 0.0; m.fixed["kap"] = True
                for k, comp in enumerate(c.components):
                    m.limits[f"N{comp}"] = (0, None)
                    if Ts[1 + k].sum() == 0: m.values[f"N{comp}"] = 0.0; m.fixed[f"N{comp}"] = True
                m.migrad()
                if not m.valid: m.simplex(); m.migrad()
                v = dict(zip(q, map(float, m.values))); per.append(v)
                for x_ in L["qc"]: out[f"{c.name}.{x_}_{i}"] = v[x_]
            for gg, jg in L["gpos"].items():
                ii = np.flatnonzero(L["groups"] == gg); wb = np.array([per[i]["Nb"] for i in ii]) + 1e-9
                for x_, src in zip(L["shape_names"], ("lam", "kap")):
                    out[f"{c.name}.{x_}_{gg}"] = float(np.average([per[i][src] for i in ii], weights=wb))
        return out

    # ------------------------------------------------------------------ profile fit
    def _bounds_and_fixed(self, vals):
        """Bounds of every parameter and which are fixed (empty categories, expo backgrounds, missing components, model.fix)."""
        lo = np.full(len(self.names), -np.inf); hi = np.full(len(self.names), np.inf); fixed = np.zeros(len(self.names), bool)
        pos = {n: i for i, n in enumerate(self.names)}
        for x in self.cn:
            lo[pos[x]], hi[pos[x]] = self.corr.limits(x)
        cv = np.array([vals[x] for x in self.cn]); T0 = self.templates(cv); self.empty = []
        for j, L in enumerate(self.layout):
            c = L["c"]; used = set()
            for i in range(c.ncat):
                tot = float(c.n[i].sum())
                for x in L["qc"]:
                    k = pos[f"{c.name}.{x}_{i}"]; lo[k] = 0.0                    # yields >= 0: the expected counts stay positive
                for kk, comp in enumerate(c.components):
                    if T0[j][1 + kk][i].sum() == 0: fixed[pos[f"{c.name}.N{comp}_{i}"]] = True
                if c.n_raw[i].sum() < self.min_entries:
                    self.empty.append(c.labels[i])
                    for x in L["qc"]: fixed[pos[f"{c.name}.{x}_{i}"]] = True
                else:
                    used.add(L["groups"][i])
            for nm in L["gnames"]:
                k = pos[nm]; gg = int(nm.rsplit("_", 1)[1]) if nm.rsplit("_", 1)[1].lstrip("-").isdigit() else nm.rsplit("_", 1)[1]
                if nm.split(".")[1].startswith("lam"): lo[k], hi[k] = -60, 60
                else:
                    lo[k], hi[k] = -400, 400
                    if getattr(c, "background", "expo2") == "expo": fixed[k] = True
                if gg not in used: fixed[k] = True
        for x in self.fix:
            fixed[pos[x]] = True
        return lo, hi, fixed

    def _nuis_grad(self, a):
        """Analytic gradient of -2lnL in the nuisances (correction entries left at 0)."""
        k = len(self.cn); g = np.zeros_like(a); T = self.templates(a[:k])
        for j, L in enumerate(self.layout):
            c, ix = L["c"], L["ix"]
            mu = self.channel_model(j, a, T[j])[0]
            w = 2 * (1 - c.n / np.maximum(mu, 1e-12)); w[mu <= 1e-12] = 0.0
            x = c.c - c.m0
            Nb, lam, kap = a[ix[:, 1]], a[ix[:, 2]], a[ix[:, 3]]
            E = expo_rows(c.c, lam, kap, c.m0)
            mx = (E * x).sum(1, keepdims=True); mx2 = (E * x * x).sum(1, keepdims=True)
            np.add.at(g, ix[:, 0], (w * T[j][0]).sum(1))
            np.add.at(g, ix[:, 1], (w * E).sum(1))
            np.add.at(g, ix[:, 2], (w * Nb[:, None] * E * (x - mx)).sum(1))
            np.add.at(g, ix[:, 3], (w * Nb[:, None] * E * (x * x - mx2)).sum(1))
            for i in range(len(c.components)):
                np.add.at(g, ix[:, 4 + i], (w * T[j][1 + i]).sum(1))
        g[:k] = 0.0
        return g

    def _channel_derivs(self, j, a, T):
        """mu (ncat, nb) and its first and second derivatives in the channel's nuisances, per category, in the column order
        of layout ix: Ns, Nb, lam, kap, comps. D: (ncat, nq, nb); D2: (ncat, nq, nq, nb)."""
        L = self.layout[j]; c, ix = L["c"], L["ix"]; nq = ix.shape[1]
        Ns, Nb, lam, kap = (a[ix[:, i]] for i in range(4))
        E = expo_rows(c.c, lam, kap, c.m0); x = c.c - c.m0
        phi = np.stack([np.broadcast_to(x, E.shape), np.broadcast_to(x * x, E.shape)], 1)          # (ncat, 2, nb)
        mphi = (E[:, None] * phi).sum(-1, keepdims=True); d = phi - mphi                           # centred statistics
        cov = (E[:, None, None] * d[:, :, None] * d[:, None, :]).sum(-1, keepdims=True)          # (ncat, 2, 2, 1)
        dE = E[:, None] * d                                                                        # dE/d(lam, kap)
        d2E = E[:, None, None] * (d[:, :, None] * d[:, None, :] - cov)
        mu = Ns[:, None] * T[0] + Nb[:, None] * E + sum(a[ix[:, 4 + i]][:, None] * T[1 + i] for i in range(len(c.components)))
        D = np.zeros((c.ncat, nq, len(c.c))); D2 = np.zeros((c.ncat, nq, nq, len(c.c)))
        D[:, 0] = T[0]; D[:, 1] = E; D[:, 2:4] = Nb[:, None, None] * dE
        for i in range(len(c.components)): D[:, 4 + i] = T[1 + i]
        D2[:, 1, 2:4] = dE; D2[:, 2:4, 1] = dE; D2[:, 2:4, 2:4] = Nb[:, None, None, None] * d2E
        return mu, D, D2

    def _newton_channel(self, j, a, maxit=200):
        """Levenberg-Marquardt on the nuisances of channel j (correction fixed), exact gradient and Hessian; respects the
        bounds and fixed flags. Converges to machine precision, so the profiled likelihood is smooth."""
        L = self.layout[j]; c, ix = L["c"], L["ix"]
        T = self.templates(a[:len(self.cn)])[j]
        cols = np.unique(ix); free = cols[~self._fixed[cols]]
        if not len(free): return a
        pos = {p: i for i, p in enumerate(free)}; lo, hi = self._lo[free], self._hi[free]
        def nll_ch(b):
            return self.channel_nll(j, b, T)[0]
        f = nll_ch(a); lam_lm = 1e-3
        m = np.array([[pos.get(p, -1) for p in row] for row in ix]); ok = m >= 0               # (ncat, nq) -> free index
        def derivs(b):
            mu, D, D2 = self._channel_derivs(j, b, T)
            if self.bb and getattr(c, "relvar", None) is not None:      # beta at its optimum: envelope theorem (approximate Hessian)
                beta = self.channel_nll(j, b, T)[1]
                mu = mu + (beta - 1) * b[ix[:, 0]][:, None] * T[0]; D = D.copy(); D[:, 0] = D[:, 0] * beta
            mu = np.maximum(mu, 1e-12); r = c.n / mu
            gq = 2 * ((1 - r)[:, None] * D).sum(-1)                                                   # (ncat, nq)
            Hq = 2 * ((r / mu)[:, None, None] * D[:, :, None] * D[:, None, :] + (1 - r)[:, None, None] * D2).sum(-1)
            g = np.zeros(len(free)); H = np.zeros((len(free), len(free)))
            np.add.at(g, m[ok], gq[ok])
            for q1 in range(ix.shape[1]):
                for q2 in range(ix.shape[1]):
                    sel = ok[:, q1] & ok[:, q2]
                    if sel.any(): np.add.at(H, (m[sel, q1], m[sel, q2]), Hq[sel, q1, q2])
            return g, H
        for it in range(maxit):
            g, H = derivs(a); x0 = a[free]
            act = ((x0 <= lo + 1e-12) & (g > 0)) | ((x0 >= hi - 1e-12) & (g < 0))                     # active bounds
            fr = ~act; dg = np.abs(np.diag(H)); dg = np.maximum(dg, 1e-9 * max(dg.max(), 1e-12))
            if np.all(np.abs(g[fr]) <= 1e-7 * np.sqrt(dg[fr]) + 1e-10):                                 # converged: |g| << curvature
                break
            moved = False
            for _ in range(40):
                Hd = H + lam_lm * np.diag(dg); step = np.zeros_like(x0)
                try:
                    step[fr] = -np.linalg.solve(Hd[np.ix_(fr, fr)], g[fr])
                except np.linalg.LinAlgError:
                    lam_lm *= 10; continue
                xn = np.clip(x0 + step, lo, hi); b = a.copy(); b[free] = xn; fn = nll_ch(b)
                if fn <= f:
                    moved = True; break
                lam_lm *= 10
            if not moved:
                break
            small = np.all(np.abs(xn - x0) <= 1e-12 * (1 + np.abs(x0)))
            a, f = b, fn; lam_lm = max(lam_lm / 10, 1e-10)
            if small:
                break
        g, H = derivs(a); x0 = a[free]
        act = ((x0 <= lo + 1e-12) & (g > 0)) | ((x0 >= hi - 1e-12) & (g < 0)); dg = np.maximum(np.abs(np.diag(H)), 1e-12)
        if np.any(np.abs(g[~act]) > 1e-4 * np.sqrt(dg[~act]) + 1e-6):           # not converged: L-BFGS on scaled variables
            from scipy.optimize import minimize
            sc = 1 / np.sqrt(dg)
            def fun(z):
                b = a.copy(); b[free] = z * sc; gg, _ = derivs(b)
                return nll_ch(b), gg * sc
            r = minimize(fun, x0 / sc, jac=True, method="L-BFGS-B", bounds=list(zip(lo / sc, hi / sc)),
                         options=dict(maxiter=20000, ftol=1e-16, gtol=1e-9, maxcor=50))
            if r.fun <= f:
                a = a.copy(); a[free] = r.x * sc
            self.n_fallback = getattr(self, "n_fallback", 0) + 1
        return a

    def profile(self, cv, a0=None):
        """min over the nuisances of -2lnL at correction cv: Levenberg-Marquardt per channel, started from the last solution
        (or a0). Returns (-2lnL, full parameter vector)."""
        k = len(self.cn); a = np.array(self._a_last if a0 is None else a0, float); a[:k] = cv
        ref = getattr(self, "_a_ref", None)
        for j in range(len(self.ch)):
            a = self._newton_channel(j, a)
            if ref is not None and a0 is None:
                # the warm start can drift into a minimum with a background yield stuck at 0: also solve from the reference
                # start (the nuisances at the starting correction) and keep the lower of the two
                b = ref.copy(); b[:k] = cv
                for i in np.unique(self.layout[j]["ix"]): b[i] = ref[i]
                b = self._newton_channel(j, np.where(np.isin(np.arange(len(a)), np.unique(self.layout[j]["ix"])), b, a))
                T = self.templates(cv)[j]
                if self.channel_nll(j, b, T)[0] < self.channel_nll(j, a, T)[0] - 1e-9:
                    a = b
        self._a_last = a; v = self.nll(*a); self.ncalls += 1
        if self.progress is not None: self.progress(self.ncalls, v)
        return v, a

    def fit_profile(self, start=None, progress=None, log=print):
        """Recommended: Minuit over the correction terms only, the nuisances re-minimised at every step (profile likelihood);
        errors from HESSE of the profiled -2lnL, which equal those of the full covariance. One template evaluation per step."""
        start = start or {}
        st = {x: float(start.get(x, self.corr.start(x))) for x in self.cn}; st.update({x: v for x, v in self.fix.items() if x in st})
        cv0 = np.array([st[x] for x in self.cn]); t0 = time.time()
        for c in self.ch:                                             # per-channel preparation at the starting correction (ARGUS shapes)
            if hasattr(c, "prepare"): c.prepare(cv0, log=log)
        self._cache.clear()
        if self.bb:                                                   # MC-statistics variance of the signal templates, at the start
            for c in self.ch: c.relvar = c.mc_relvar(cv0)
        vals = dict(st); vals.update(self._inner_start(cv0))
        self._lo, self._hi, self._fixed = self._bounds_and_fixed(vals)
        a0 = np.array([vals[n] for n in self.names]); a0 = np.clip(a0, self._lo, self._hi); self._a_last = a0; self.ncalls = 0
        # where the background is close to 0 its slope is barely constrained and the likelihood in it can have several minima:
        # at the starting correction, try several starting slopes per channel and keep the best; later steps follow that
        # minimum from the previous solution
        a0[:len(self.cn)] = cv0
        for j, L in enumerate(self.layout):
            best = None
            for lam0 in (None, -10.0, -5.0, -2.0, 0.0, 2.0):
                b = a0.copy()
                if lam0 is not None:
                    for nm in L["gnames"]:
                        k_ = self.names.index(nm)
                        if nm.split(".")[1].startswith("lam") and not self._fixed[k_]: b[k_] = lam0
                b = self._newton_channel(j, b); v = self.nll(*b)
                if best is None or v < best[0] - 1e-9: best = (v, b)
            for k_ in np.unique(L["ix"]): a0[k_] = best[1][k_]
        self._a_last = a0; self._a_ref = None
        self.profile(cv0); self._a_ref = self._a_last.copy()
        log(f"  start values of {len(self.nuis_names)} nuisances in {time.time() - t0:.0f} s", flush=True)
        if self.empty:
            log(f"  {len(self.empty)} categories with fewer than {self.min_entries} data entries, nuisances fixed: {self.empty}", flush=True)
        self.progress = progress and (lambda n, v: progress(n))
        m = Minuit(lambda *c: self.profile(np.asarray(c))[0], *cv0, name=self.cn); m.errordef = 1; m.tol = 1.0
        for x in self.cn:
            m.limits[x] = self.corr.limits(x); m.errors[x] = self.corr.step(x)
            if x in self.fix: m.fixed[x] = True
        m.migrad(); m.hesse()
        for _ in range(2):
            if m.valid: break
            m.migrad(); m.hesse()
        # a term on a physical limit (e.g. a variance at 0) makes the covariance singular: fix it there and redo HESSE
        lim = [x for x in self.cn if not m.fixed[x] and m.limits[x] is not None and any(
            abs(m.values[x] - L_) < 1e-6 * (1 + abs(L_)) for L_ in m.limits[x] if L_ is not None and abs(L_) != float("inf"))]
        if lim:
            log(f"  at a limit, fixed there for HESSE: {lim}", flush=True)
            for x in lim: m.fixed[x] = True
            m.hesse()
        self.at_limit_fixed = lim
        self.minuit = m
        cv = np.array(m.values); fval, a = self.profile(cv)
        return self._result(m, a, np.array(m.covariance) if m.covariance is not None else np.full((len(cv), len(cv)), np.nan),
                            {x: float(m.errors[x]) for x in self.cn}, fixed=self._fixed, nfree_corr=sum(1 for x in self.cn if not m.fixed[x]))

    def _result(self, m, a, cov, cerr, fixed, nfree_corr):
        T = self.templates(a[:len(self.cn)])
        chi2, ndf, labels = [], [], []; ncat_tot = sum(c.ncat for c in self.ch)
        for j, L in enumerate(self.layout):
            c = L["c"]; mu = self.channel_model(j, a, T[j])[0]
            live = c.n_raw.sum(1) >= self.min_entries                     # empty categories count for neither chi2 nor ndf
            chi2 += [float(((n - u) ** 2 / np.maximum(u, 1)).sum()) if ok else 0.0 for n, u, ok in zip(c.n_raw, mu, live)]
            nfree_ch = sum(1 for i, x in enumerate(self.names) if x.startswith(c.name + ".") and not fixed[i])
            ncat_live = sum(int((L_["c"].n_raw.sum(1) >= self.min_entries).sum()) for L_ in self.layout)
            ndf += [int(round(len(c.c) - nfree_ch / max(live.sum(), 1) - nfree_corr / ncat_live)) if ok else 0 for ok in live]
            labels += c.labels
        fm = m.fmin
        config = dict(eta_edges=list(map(float, self.corr.eta_edges)), species=getattr(self.corr, "species", "common"),
                      channels={c.name: dict(mrange=list(map(float, c.mrange)), bw=float(c.bw), m0=c.m0, ncat=c.ncat,
                                             components=list(c.components), n_data=getattr(c, "n_data", None),
                                             n_mc=getattr(c, "n_mc", None), categories=c.cat.labels) for c in self.ch},
                      categories=labels, n_data=sum(int(getattr(c, "n_data", 0)) for c in self.ch),
                      n_mc=sum(int(getattr(c, "n_mc", 0)) for c in self.ch), comb="expo2",
                      minuit=dict(edm=float(fm.edm), above_max_edm=bool(fm.is_above_max_edm), call_limit=bool(fm.has_reached_call_limit),
                                  hesse_failed=bool(fm.hesse_failed), posdef_covar=bool(fm.has_posdef_covar),
                                  covar_accurate=bool(fm.has_accurate_covar), nfcn=int(m.nfcn), empty_categories=self.empty,
                                  at_limit=[x for x in self.cn if not m.fixed[x] and m.limits[x] is not None and any(
                                      abs(m.values[x] - L_) < 1e-6 * (1 + abs(L_)) for L_ in m.limits[x] if L_ is not None and abs(L_) != float("inf"))]))
        vals = {x: float(v) for x, v in zip(self.names, a)}
        errs = {x: float("nan") for x in self.names}; errs.update(cerr)
        k = len(self.cn)
        return FitResult(self.names, vals, errs, self.cn, np.asarray(cov)[:k, :k], bool(m.valid), float(m.fval), chi2, ndf, labels, config)

    # ------------------------------------------------------------------ fit
    def fit_full(self, start=None, progress=None, log=print):
        start = start or {}
        st = {x: float(start.get(x, self.corr.start(x))) for x in self.cn}; st.update({x: v for x, v in self.fix.items() if x in st})
        cv = np.array([st[x] for x in self.cn]); t0 = time.time()
        for c in self.ch:
            if hasattr(c, "prepare"): c.prepare(cv, log=log)
        self._cache.clear()
        vals = dict(st); vals.update(self._inner_start(cv))
        log(f"  start values of {len(self.nuis_names)} nuisances in {time.time() - t0:.0f} s", flush=True)
        cnt = [0]
        def f(*a):
            cnt[0] += 1
            if progress is not None: progress(cnt[0])
            return self.nll(*a)
        fixed = np.zeros(len(self.names), bool)
        def gr(*a):
            return self.grad(*a, fixed=fixed)
        m = Minuit(f, *[vals[n] for n in self.names], name=self.names, grad=gr if self.use_grad else None); m.errordef = 1; m.tol = 1.0
        for x in self.cn:
            m.limits[x] = self.corr.limits(x); m.errors[x] = self.corr.step(x)
        T0 = self.templates(cv)
        for j, L in enumerate(self.layout):
            c = L["c"]
            for i in range(c.ncat):
                for x in L["qc"]:
                    # Nb may go a little below 0: a yield sitting on a limit makes the covariance singular (HESSE fails) in
                    # categories with almost no background; the total stays positive because Ns dominates
                    nm = f"{c.name}.{x}_{i}"; tot = float(c.n[i].sum())
                    m.limits[nm] = (-0.02 * tot - 5.0, None) if x == "Nb" else (0, None)
                    m.errors[nm] = max(1.0, 0.01 * abs(vals[nm]), 1e-3 * tot)
                for k, comp in enumerate(c.components):
                    if T0[j][1 + k][i].sum() == 0: m.values[f"{c.name}.N{comp}_{i}"] = 0.0; m.fixed[f"{c.name}.N{comp}_{i}"] = True
            for nm in L["gnames"]:
                if nm.split(".")[1].startswith("lam"): m.limits[nm] = (-60, 60)
                else:
                    m.limits[nm] = (-400, 400)
                    if getattr(c, "background", "expo2") == "expo": m.values[nm] = 0.0; m.fixed[nm] = True
        # categories with (almost) no data carry no information and leave their yields undetermined: their nuisances are
        # fixed at the start values, and a background shape used only by such categories is fixed too
        self.empty = []
        for j, L in enumerate(self.layout):
            c = L["c"]; used = set()
            for i in range(c.ncat):
                if c.n_raw[i].sum() < self.min_entries:
                    self.empty.append(c.labels[i])
                    for x in L["qc"]: m.fixed[f"{c.name}.{x}_{i}"] = True
                else:
                    used.add(L["groups"][i])
            for gg in L["gpos"]:
                if gg not in used:
                    for x in L["shape_names"]: m.fixed[f"{c.name}.{x}_{gg}"] = True
        if self.empty:
            log(f"  {len(self.empty)} categories with fewer than {self.min_entries} data entries, nuisances fixed: {self.empty}", flush=True)
        for x, v in self.fix.items():
            m.values[x] = v; m.fixed[x] = True
        fixed[:] = [m.fixed[x] for x in self.names]
        m.migrad(); m.hesse()
        for _ in range(2):
            if m.valid: break
            m.migrad(); m.hesse()
        self.minuit = m
        a = np.array([m.values[x] for x in self.names]); T = self.templates(a[:len(self.cn)])
        chi2, ndf, labels = [], [], []
        nfree_corr = sum(1 for x in self.cn if not m.fixed[x]); ncat_tot = sum(c.ncat for c in self.ch)
        nfree_nuis = sum(1 for x in self.nuis_names if not m.fixed[x])
        for j, L in enumerate(self.layout):
            c = L["c"]; mu = self.channel_model(j, a, T[j])[0]
            chi2 += [float(((n - u) ** 2 / np.maximum(u, 1)).sum()) for n, u in zip(c.n_raw, mu)]
            nfree_ch = sum(1 for x in self.nuis_names if x.startswith(c.name + ".") and not m.fixed[x])
            ndf += [int(round(len(c.c) - nfree_ch / c.ncat - nfree_corr / ncat_tot))] * c.ncat
            labels += c.labels
        k = len(self.cn); fm = m.fmin
        cov = np.array(m.covariance)[:k, :k] if m.covariance is not None else np.full((k, k), np.nan)
        config = dict(eta_edges=list(map(float, self.corr.eta_edges)), species=getattr(self.corr, "species", "common"),
                      channels={c.name: dict(mrange=list(map(float, c.mrange)), bw=float(c.bw), m0=c.m0, ncat=c.ncat,
                                             components=list(c.components), n_data=getattr(c, "n_data", None),
                                             n_mc=getattr(c, "n_mc", None), categories=c.cat.labels) for c in self.ch},
                      categories=labels, n_data=sum(int(getattr(c, "n_data", 0)) for c in self.ch),
                      n_mc=sum(int(getattr(c, "n_mc", 0)) for c in self.ch), comb="expo2",
                      minuit=dict(edm=float(fm.edm), above_max_edm=bool(fm.is_above_max_edm), call_limit=bool(fm.has_reached_call_limit),
                                  hesse_failed=bool(fm.hesse_failed), posdef_covar=bool(fm.has_posdef_covar),
                                  covar_accurate=bool(fm.has_accurate_covar), nfcn=int(m.nfcn), empty_categories=self.empty,
                                  at_limit=[x for x in self.names if not m.fixed[x] and m.limits[x] is not None and any(
                                      abs(m.values[x] - L) < 1e-6 * (1 + abs(L)) for L in m.limits[x] if L is not None and abs(L) != float("inf"))]))
        return FitResult(self.names, {x: float(m.values[x]) for x in self.names}, {x: float(m.errors[x]) for x in self.names},
                         self.cn, cov, bool(m.valid), float(m.fval), chi2, ndf, labels, config)

    # ------------------------------------------------------------------ helpers for plots and checks
    def components_at(self, res, uncorrected=False):
        """Per channel: dict(n_raw, total, sig, bkg, comps) at the fitted values; with uncorrected=True the templates are
        evaluated with the correction set to zero (same yields and background)."""
        a = np.array([res.values[x] for x in self.names])
        cv = a[:len(self.cn)].copy()
        if uncorrected: cv[:] = 0.0
        T = self.templates(cv); out = []
        for j, c in enumerate(self.ch):
            tot, sig, bkg, comps = self.channel_model(j, a, T[j])
            out.append(dict(channel=c, n_raw=c.n_raw, total=tot, sig=sig, bkg=bkg, comps=comps))
        return out
