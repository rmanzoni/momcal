"""Plots of a combined fit: per channel, every category, and the summed background-subtracted peak."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, GREEN, GREY = "#c0392b", "#2471a3", "#1e8449", "0.55"
XLAB = {"kkpi": "m(KKπ) [MeV]", "mumu": "m(μμ) [MeV]", "jpsik": "m(J/ψ$_{rf}$K) [MeV]"}


def plot_categories(comp, comp0, chi2, ndf, path, ncol=6):
    ch = comp["channel"]; nc = ch.ncat; ncol = min(ncol, nc); nrow = -(-nc // ncol)
    fig = plt.figure(figsize=(3.6 * ncol, 4.2 * nrow))
    gs = fig.add_gridspec(2 * nrow, ncol, height_ratios=[3, 1.2] * nrow, hspace=0.08, wspace=0.28)
    X = 1e3 * ch.c
    for i in range(nc):
        r, c = divmod(i, ncol)
        a0 = fig.add_subplot(gs[2 * r, c]); a1 = fig.add_subplot(gs[2 * r + 1, c], sharex=a0)
        n = comp["n_raw"][i]; mu = comp["total"][i]
        a0.errorbar(X, n, np.sqrt(np.maximum(n, 1)), fmt="o", ms=1.6, color="k", lw=0.5, label="data")
        a0.plot(X, comp0["total"][i], color=BLUE, lw=1.0, ls="--", label="MC, no correction")
        a0.plot(X, mu, color=RED, lw=1.3, label="MC, fitted correction")
        a0.plot(X, comp["bkg"][i], color=GREY, lw=0.8, label="background")
        for cm in comp["comps"]:
            a0.plot(X, cm[i], color=GREEN, lw=0.8, label="J/ψπ")
        a0.set_title(f"{ch.cat.labels[i]}\nχ² {chi2[i]:.0f} / {ndf[i]}", fontsize=7.5)
        plt.setp(a0.get_xticklabels(), visible=False)
        a1.plot(X, (n - mu) / np.sqrt(np.maximum(mu, 1)), "o", ms=1.6, color=RED)
        a1.axhspan(-2, 2, color="0.92", zorder=0); a1.axhline(0, color="0.5", lw=0.6); a1.set_ylim(-5, 5)
        if r == nrow - 1: a1.set_xlabel(XLAB.get(ch.name, "mass [MeV]"))
        if c == 0: a0.set_ylabel("candidates / bin"); a1.set_ylabel("pull")
        if i == 0: a0.legend(fontsize=6.5, frameon=False)
    fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)


def _stats(c, y, lo, hi):
    w = (c > lo) & (c < hi); y = np.clip(y[w], 0, None); x = 1e3 * c[w]; m = (x * y).sum() / y.sum()
    return m, np.sqrt(((x - m) ** 2 * y).sum() / y.sum())


def plot_peak(comp, comp0, path):
    """Background-subtracted data (summed over categories) against the signal MC before and after the correction, with the
    mean and rms in a window around the peak; bottom: data / MC."""
    ch = comp["channel"]; c = ch.c; X = 1e3 * c
    d = (comp["n_raw"] - comp["bkg"] - sum(comp["comps"]) if comp["comps"] else comp["n_raw"] - comp["bkg"]).sum(0)
    s1 = comp["sig"].sum(0); s0 = comp0["sig"].sum(0)
    rms = _stats(c, s1, c[0], c[-1])[1] / 1e3; ctr = _stats(c, s1, c[0], c[-1])[0] / 1e3
    lo, hi = ctr - 2 * rms, ctr + 2 * rms
    fig = plt.figure(figsize=(6.4, 6.0)); gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.06)
    a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharex=a0)
    a0.errorbar(X, d, np.sqrt(np.maximum(comp["n_raw"].sum(0), 1)), fmt="o", ms=2, color="k", lw=0.6, label="data − background")
    a0.plot(X, s0, color=BLUE, ls="--", lw=1.1, label="signal MC, no correction")
    a0.plot(X, s1, color=RED, lw=1.4, label="signal MC, fitted correction")
    txt = [f"{lab}: mean {m:.2f} MeV, rms {r:.2f} MeV" for lab, y in (("data", d), ("MC no corr.", s0), ("MC corr.", s1))
           for m, r in [_stats(c, y, lo, hi)]]
    a0.text(0.02, 0.97, f"window {1e3*lo:.0f}–{1e3*hi:.0f} MeV\n" + "\n".join(txt), transform=a0.transAxes, va="top", fontsize=7.5)
    a0.legend(fontsize=7.5, frameon=False, loc="upper right"); a0.set_ylabel("candidates / bin")
    a0.set_title(f"{ch.name}: all {ch.ncat} categories", fontsize=10); plt.setp(a0.get_xticklabels(), visible=False)
    ok = s1 > 0.02 * s1.max()
    a1.plot(X[ok], (d / np.where(s0 > 0, s0, np.nan))[ok], color=BLUE, lw=1.0, ls="--")
    a1.plot(X[ok], (d / np.where(s1 > 0, s1, np.nan))[ok], color=RED, lw=1.3)
    a1.axhline(1, color="0.5", lw=0.6); a1.set_ylim(0.8, 1.2); a1.set_ylabel("data / MC"); a1.set_xlabel(XLAB.get(ch.name, "mass [MeV]"))
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_all(fit, res, out):
    comps = fit.components_at(res); comps0 = fit.components_at(res, uncorrected=True)
    i0 = 0
    for comp, comp0 in zip(comps, comps0):
        ch = comp["channel"]; n = ch.ncat
        plot_categories(comp, comp0, res.chi2[i0:i0 + n], res.ndf[i0:i0 + n], os.path.join(out, f"fit_categories_{ch.name}.png"))
        plot_peak(comp, comp0, os.path.join(out, f"peak_{ch.name}.png"))
        i0 += n
