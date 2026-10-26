"""Diagnostic plots of a fit."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .model import expo_shape

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE = "#c0392b", "#2471a3"


def _models(fit, res):
    a = [res.values[n] for n in res.names]
    mus, T = fit.model(a)
    a0 = list(a); k = len(fit.corr.names); a0[:k] = [0.0] * k
    mus0, T0 = fit.model(a0)
    return a, mus, T, mus0, T0


def plot_fit(fit, res, path, ncol=6):
    """Every category: data, corrected MC + background, uncorrected MC + background (same yields), pulls."""
    _, mus, _, mus0, _ = _models(fit, res)
    nc = fit.cat.n; ncol = min(ncol, nc); nrow = -(-nc // ncol)
    fig = plt.figure(figsize=(3.6 * ncol, 4.2 * nrow))
    gs = fig.add_gridspec(2 * nrow, ncol, height_ratios=[3, 1.2] * nrow, hspace=0.08, wspace=0.28)
    X = 1e3 * fit.c
    for i in range(nc):
        r, c = divmod(i, ncol)
        a0 = fig.add_subplot(gs[2 * r, c]); a1 = fig.add_subplot(gs[2 * r + 1, c], sharex=a0)
        n = fit.n_raw[i]
        a0.errorbar(X, n, np.sqrt(np.maximum(n, 1)), fmt="o", ms=1.8, color="k", lw=0.6, label="data")
        a0.plot(X, mus0[i], color=BLUE, lw=1.0, ls="--", label="MC, no correction")
        a0.plot(X, mus[i], color=RED, lw=1.3, label="MC, fitted correction")
        a0.set_title(f"{fit.cat.labels[i]}\nχ² {res.chi2[i]:.0f} / {res.ndf[i]}", fontsize=8)
        plt.setp(a0.get_xticklabels(), visible=False)
        a1.plot(X, (n - mus[i]) / np.sqrt(np.maximum(mus[i], 1)), "o", ms=1.8, color=RED)
        a1.axhspan(-2, 2, color="0.92", zorder=0); a1.axhline(0, color="0.5", lw=0.6); a1.set_ylim(-5, 5)
        if r == nrow - 1: a1.set_xlabel("m(KKπ) [MeV]")
        if c == 0: a0.set_ylabel("candidates / bin"); a1.set_ylabel("pull")
        if i == 0: a0.legend(fontsize=7, frameon=False)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_lineshape(fit, res, path):
    """Background-subtracted data vs signal MC before / after the correction, categories grouped by the |eta| bin of the most
    forward track (the categories are ordered |eta|-major)."""
    _, _, T, _, T0 = _models(fit, res)
    edges = np.asarray(getattr(fit.cat, "eta_edges", fit.corr.eta_edges)); ne = len(edges) - 1
    per = fit.cat.n // ne; c = fit.c; X = 1e3 * c
    def stats(y, lo=1.945, hi=1.991):
        w = (c > lo) & (c < hi); y = np.clip(y[w], 0, None); x = 1e3 * c[w]; m = (x * y).sum() / y.sum()
        return m, np.sqrt(((x - m) ** 2 * y).sum() / y.sum())
    fig = plt.figure(figsize=(5 * ne, 6.4)); gs = fig.add_gridspec(2, ne, height_ratios=[3, 1.2], hspace=0.06, wspace=0.22)
    for e in range(ne):
        s = np.zeros_like(c); v = np.zeros_like(c); m1 = np.zeros_like(c); m0 = np.zeros_like(c)
        nv = fit.nuisances_by_category(res.values)
        for k in range(e * per, (e + 1) * per):
            Ns, Nb, lam, kap = (nv[f"{q}_{k}"] for q in ("Ns", "Nb", "lam", "kap"))
            s += fit.n_raw[k] - Nb * expo_shape(c, lam, kap if fit.comb == "expo2" else 0.0)
            v += fit.n_raw[k]; m1 += Ns * T[k]; m0 += Ns * T0[k]
        er = np.sqrt(np.maximum(v, 1))
        a0 = fig.add_subplot(gs[0, e]); a1 = fig.add_subplot(gs[1, e], sharex=a0)
        a0.errorbar(X, s, er, fmt="o", ms=2.2, color="k", lw=0.6, label="data − fitted background")
        a0.plot(X, m0, color=BLUE, ls="--", lw=1.3, label="MC, no correction")
        a0.plot(X, m1, color=RED, lw=1.5, label="MC, fitted correction")
        (ms, ws), (q0, w0), (q1, w1) = stats(s), stats(m0), stats(m1)
        a0.text(0.02, 0.97, f"max |η| {edges[e]:.1f}–{edges[e + 1]:.1f}\nmean / rms, 1945–1991 MeV\n data       {ms:.2f} / {ws:.2f}"
                f"\n no corr.   {q0:.2f} / {w0:.2f}\n corrected  {q1:.2f} / {w1:.2f}", transform=a0.transAxes, va="top",
                fontsize=8, family="monospace")
        a0.set_ylim(top=1.55 * max(m0.max(), s.max())); plt.setp(a0.get_xticklabels(), visible=False)
        a1.plot(X, (s - m1) / er, "o", ms=2.2, color=RED); a1.axhspan(-2, 2, color="0.92", zorder=0)
        a1.axhline(0, color="0.5", lw=0.6); a1.set_ylim(-5, 5); a1.set_xlabel("m(KKπ) [MeV]")
        cs = sum(res.chi2[e * per:(e + 1) * per]); nd = sum(res.ndf[e * per:(e + 1) * per])
        a1.text(0.02, 0.06, f"χ² {cs:.0f} / {nd} ({per} categories)", transform=a1.transAxes, fontsize=8)
        if e == 0:
            a0.set_ylabel("signal candidates / bin"); a1.set_ylabel("(data − corr.) / σ"); H, L = a0.get_legend_handles_labels()
    fig.legend(H, L, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97), fontsize=9)
    fig.suptitle("Background-subtracted data vs MC, categories grouped by the |η| of the most forward track", fontsize=10, y=1.0)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_fit_components_arrays(c, n_raw, T, values, comb, eta_edges, chi2, ndf, path, title=""):
    """The fit to data itself, categories grouped by |eta| of the most forward track: data, total fit, corrected-MC signal
    and background components, and pulls of data against the total. Arrays: bin centres c [GeV]; per category the raw data
    counts n_raw, the normalised template T, the fitted values Ns_i, Nb_i, lam_i, kap_i (in `values`), chi2, ndf."""
    ne = len(eta_edges) - 1; per = len(n_raw) // ne; X = 1e3 * c
    fig = plt.figure(figsize=(5 * ne, 6.4)); gs = fig.add_gridspec(2, ne, height_ratios=[3, 1.2], hspace=0.06, wspace=0.22)
    for e in range(ne):
        n = np.zeros_like(c); sig = np.zeros_like(c); bkg = np.zeros_like(c)
        for k in range(e * per, (e + 1) * per):
            Ns, Nb, lam, kap = (values[f"{q}_{k}"] for q in ("Ns", "Nb", "lam", "kap"))
            n += n_raw[k]; sig += Ns * T[k]; bkg += Nb * expo_shape(c, lam, kap if comb == "expo2" else 0.0)
        tot = sig + bkg; er = np.sqrt(np.maximum(n, 1))
        a0 = fig.add_subplot(gs[0, e]); a1 = fig.add_subplot(gs[1, e], sharex=a0)
        a0.fill_between(X, 0, bkg, color="0.85", lw=0, label="background (exp. of quadratic)")
        a0.plot(X, sig, color=RED, lw=1.3, ls="--", label="signal: corrected MC")
        a0.plot(X, tot, color=BLUE, lw=1.6, label="total fit")
        a0.errorbar(X, n, er, fmt="o", ms=2.0, color="k", lw=0.6, label="data")
        cs = sum(chi2[e * per:(e + 1) * per]); nd = sum(ndf[e * per:(e + 1) * per])
        a0.text(0.02, 0.97, f"max |η| {eta_edges[e]:.1f}–{eta_edges[e + 1]:.1f}\n{per} categories summed\n"
                f"signal {sig.sum():,.0f}\nbackground {bkg.sum():,.0f}", transform=a0.transAxes, va="top", fontsize=8)
        a0.set_ylim(0, 1.3 * n.max()); plt.setp(a0.get_xticklabels(), visible=False)
        a1.plot(X, (n - tot) / er, "o", ms=2.2, color=BLUE); a1.axhspan(-2, 2, color="0.92", zorder=0)
        a1.axhline(0, color="0.5", lw=0.6); a1.set_ylim(-5, 5); a1.set_xlabel("m(KKπ) [MeV]")
        a1.text(0.02, 0.06, f"χ² {cs:.0f} / {nd} (per category, summed)", transform=a1.transAxes, fontsize=8)
        if e == 0:
            a0.set_ylabel("candidates / MeV"); a1.set_ylabel("(data − fit) / σ"); H, L = a0.get_legend_handles_labels()
    fig.legend(H[::-1], L[::-1], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97), fontsize=9)
    fig.suptitle(title or "Fit to data: signal (corrected MC) + background, categories grouped by the |η| of the most forward track",
                 fontsize=10, y=1.0)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_fit_components(fit, res, path, title=""):
    _, _, T, _, _ = _models(fit, res)
    plot_fit_components_arrays(fit.c, fit.n_raw, T, fit.nuisances_by_category(res.values), fit.comb,
                               getattr(fit.cat, "eta_edges", fit.corr.eta_edges),
                               res.chi2, res.ndf, path, title)


def term_values(res, term, nb):
    """(value, error) per |eta| bin of a term, in display units; v0, v1 shown as r0 [1e-3], r1 [1e-4/GeV]."""
    out = []
    for b in range(nb):
        n = f"{term}_{b}" if f"{term}_{b}" in res.values else f"{term}_all"
        ties = getattr(res, "config", {}).get("ties", {}) if hasattr(res, "config") else {}
        if n not in res.values and f"{term}_{b}" in ties:
            n = ties[f"{term}_{b}"]
        x, dx = res.values[n], res.errors[n]
        if term in ("v0", "v1", "a2", "c2"):             # sqrt of the fitted variance: 1e-3 (v0, a2) or 1e-4/GeV (v1, c2)
            r = np.sqrt(max(x, 0.0)); dr = dx / (2 * r) if r > 0.3 * np.sqrt(dx) else np.sqrt(dx)
            out.append((r, dr))
        else:
            f = {"s0": 1e3, "s1": 1e3, "k": 1e4, "A": 1e3, "e": 1e3, "M": 1e4}[term]; out.append((f * x, f * dx))
    return np.array(out)


def plot_terms(res, path, truth=None):
    """The five terms per |eta| bin; a shared term is a band."""
    nb = len(res.config["eta_edges"]) - 1; e = res.config["eta_edges"]
    U = {"A": "A  [10⁻³]", "e": "ε  [MeV]", "M": "M  [10⁻⁴ GeV⁻¹]", "a2": "a  [10⁻³]", "c2": "c  [10⁻⁴ GeV⁻¹]",
         "s0": "s0  [10⁻³]", "s1": "s1  [MeV]", "k": "k  [10⁻⁴ GeV⁻¹]", "v0": "r0  [10⁻³]", "v1": "r1  [10⁻⁴ GeV⁻¹]"}
    terms = [t for t in U if any(n.startswith(t + "_") for n in res.corr_names)]
    fig, ax = plt.subplots(1, len(terms), figsize=(3.3 * len(terms), 3.4), squeeze=False); ax = ax[0]; xb = np.arange(nb)
    for j, t in enumerate(terms):
        v = term_values(res, t, nb)
        if f"{t}_all" in res.values:
            ax[j].axhspan(v[0, 0] - v[0, 1], v[0, 0] + v[0, 1], color=RED, alpha=0.18); ax[j].axhline(v[0, 0], color=RED, lw=1.5)
        else:
            ax[j].errorbar(xb, v[:, 0], v[:, 1], fmt="o", color=RED, capsize=3)
        if truth is not None:
            class R: pass
            tr = R(); tr.values = truth; tr.errors = {k: 0.0 for k in truth}
            ax[j].plot(xb, term_values(tr, t, nb)[:, 0], "_", ms=20, color="k")
        ax[j].set_xticks(xb, [f"{e[b]:.1f}–{e[b + 1]:.1f}" for b in range(nb)], fontsize=8); ax[j].set_xlabel("|η| bin")
        ax[j].set_title(U[t], fontsize=10); ax[j].axhline(0, color="0.6", lw=0.7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pulls(pulls, path, title=""):
    """pulls: {name: array of (fitted - injected)/error over toys}. Mean and width with their errors, per parameter."""
    names = list(pulls); y = np.arange(len(names))[::-1]
    mu = np.array([np.mean(pulls[n]) for n in names]); sd = np.array([np.std(pulls[n], ddof=1) for n in names])
    n = np.array([len(pulls[k]) for k in names])
    fig, ax = plt.subplots(1, 2, figsize=(10, 0.36 * len(names) + 1.4), sharey=True)
    ax[0].axvspan(-0.3, 0.3, color="0.93"); ax[0].axvline(0, color="0.5", lw=0.7)
    ax[0].errorbar(mu, y, xerr=sd / np.sqrt(n), fmt="o", color=BLUE, capsize=3); ax[0].set_xlabel("pull mean")
    ax[1].axvspan(0.8, 1.2, color="0.93"); ax[1].axvline(1, color="0.5", lw=0.7)
    ax[1].errorbar(sd, y, xerr=sd / np.sqrt(2 * (n - 1)), fmt="o", color=BLUE, capsize=3); ax[1].set_xlabel("pull width (1 = correct errors)")
    ax[0].set_yticks(y, names, fontsize=8)
    fig.suptitle(title or f"{n.max()} toys", fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# kept for the older command-line paths
def plot_params(res, path, truth=None):
    plot_terms(res, path, truth)
