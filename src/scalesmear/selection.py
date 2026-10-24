"""Candidate-level cuts applied after reading: the m(KK) window and the D+ -> K- pi+ pi+ reflection veto."""
from __future__ import annotations
import numpy as np
from .io import MK, MPI, Sample
from .model import MPHI

MDPLUS = 1.86966


def m_kpipi(s: Sample):
    """m(K pi pi) with the pion mass given to the kaon that has the same charge as the Ds pion (D+ -> K- pi+ pi+ hypothesis)."""
    same = s.charge[:, 0] == s.charge[:, 2]
    m = np.where(same[:, None], [MPI, MK, MPI], [MK, MPI, MPI])
    e = np.sqrt((s.p ** 2).sum(-1) + m ** 2)
    P = s.p.sum(1)
    return np.sqrt(np.maximum(e.sum(1) ** 2 - (P ** 2).sum(1), 0.0))


def dplus_veto(s: Sample, width):
    """True for candidates kept: |m(K pi pi) - m(D+)| > width. width <= 0 keeps everything."""
    if not width or width <= 0:
        return np.ones(len(s), bool)
    return np.abs(m_kpipi(s) - MDPLUS) > width


def kk_window(s: Sample, width):
    """True for candidates with |m(KK) - m(phi)| < width (stored m(KK))."""
    if not width or width <= 0:
        return np.ones(len(s), bool)
    return np.abs(s.m_kk - MPHI) < width


def apply(s: Sample, kk=0.020, veto=0.050):
    """Apply both cuts; returns (new sample, dict of efficiencies)."""
    a = kk_window(s, kk); b = dplus_veto(s, veto)
    eff = dict(n_in=int(len(s)), kk_window=float(a.mean()) if len(s) else 1.0,
               dplus_veto=float(b[a].mean()) if a.any() else 1.0)
    out = s.subset(a & b); eff["n_out"] = int(len(out))
    return out, eff
