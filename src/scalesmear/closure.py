"""Closure test: apply a known correction to half of the MC, add an exponential background, fit with the other half."""
from __future__ import annotations
import numpy as np
from .io import Sample
from .model import Correction, MorphedMC
from .fit import ScaleSmearFit


def pseudo_data(mc: Sample, corr: Correction, inject, bkg_frac=0.4, lam=-3.0, mrange=(1.908, 2.028), seed=7, kk_window=0.020):
    """Return a Sample whose m(KK pi), m(KK) are the corrected masses of `mc`, plus uniform-in-exponential background candidates
    (background candidates reuse random MC tracks only to define the category; their mass is drawn from the exponential)."""
    rng = np.random.default_rng(seed)
    m, mkk = MorphedMC(mc, corr, seed=seed + 1).masses(np.asarray(inject, float))
    keep = np.abs(mkk - 1.019461) < kk_window          # the data selection is applied to measured m(KK)
    mc = mc.subset(keep); m, mkk = m[keep], mkk[keep]
    nb = int(bkg_frac / (1 - bkg_frac) * len(mc))
    u = rng.random(nb); a, b = mrange
    x = a + np.log1p(u * np.expm1(lam * (b - a))) / lam
    j = rng.integers(0, len(mc), nb)
    p = np.concatenate([mc.p, mc.p[j]]); q = np.concatenate([mc.charge, mc.charge[j]])
    return Sample(p, q, np.concatenate([m, x]), np.concatenate([mkk, np.full(nb, 1.019461)]), "pseudo-data")


def run(mc: Sample, corr: Correction, inject, data_frac=0.3, split_seed=11, full=True, **kw):
    """Split the MC (fraction data_frac becomes pseudo-data), inject the correction, fit with the rest as template.
    Different split_seed values give independent closure tests; their spread includes the MC statistics of both parts."""
    a, b = mc.split(data_frac, seed=split_seed)
    pdata = pseudo_data(a, corr, inject, seed=7 + split_seed)
    f = ScaleSmearFit(pdata, b, corr, **kw)
    return f, (f.fit_full() if full else f.fit())
