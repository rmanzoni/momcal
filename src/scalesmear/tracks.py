"""Candidates made of n tracks, each with its own mass hypothesis (muons, kaons, pions), and the readers for the
B+ -> J/psi K+ skims (slim_hadd_jpsik.py output). The Ds -> phi pi reader stays in io.py (Sample, 3 fixed tracks)."""
from __future__ import annotations
import glob, os
from dataclasses import dataclass, field
import numpy as np
from .io import MK, MPI, _read_file, _variables

MMU = 0.1056583755
MASS = {"mu": MMU, "K": MK, "pi": MPI}


@dataclass
class Tracks:
    """p: (n, k, 3) track momenta [GeV]; charge: (n, k); kinds: particle type of each of the k tracks (mass hypothesis);
    extra: per-candidate arrays (stored masses, the refitted J/psi, ...)."""
    p: np.ndarray
    charge: np.ndarray
    kinds: tuple
    extra: dict = field(default_factory=dict)
    name: str = ""

    def __len__(self):
        return len(self.p)

    @property
    def masses(self):
        return np.array([MASS[k] for k in self.kinds])

    def pt(self):
        return np.hypot(self.p[..., 0], self.p[..., 1])

    def eta(self):
        return np.arcsinh(self.p[..., 2] / np.maximum(self.pt(), 1e-9))

    def phi(self):
        return np.arctan2(self.p[..., 1], self.p[..., 0])

    def subset(self, mask):
        return Tracks(self.p[mask], self.charge[mask], self.kinds, {k: v[mask] for k, v in self.extra.items()}, self.name)

    def split(self, frac=0.5, seed=0):
        r = np.random.default_rng(seed).random(len(self)) < frac
        return self.subset(r), self.subset(~r)

    def concat(self, other):
        assert self.kinds == other.kinds
        return Tracks(np.concatenate([self.p, other.p]), np.concatenate([self.charge, other.charge]), self.kinds,
                      {k: np.concatenate([v, other.extra[k]]) for k, v in self.extra.items()}, self.name)


def p3(pt, eta, phi):
    pt = np.asarray(pt, np.float64)
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(np.asarray(eta, np.float64))], -1)


def four(p, m):
    """(E, p) with E from |p| and the mass."""
    return np.sqrt((p ** 2).sum(-1) + np.asarray(m) ** 2)


# B+ -> J/psi K+ skims: tracks (mu1, mu2, k); the refitted J/psi four-vector is kept for m(J/psi_rf K)
JPSIK_TRACKS = (("mu1", "mu"), ("mu2", "mu"), ("k", "K"))
JPSIK_EXTRA = ["jpsi_mass", "rf_mass", "mass", "jpsi_rf_pt", "jpsi_rf_eta", "jpsi_rf_phi", "jpsi_rf_mass"]


def load_jpsik(paths, selection="", tree="tree", max_events=None, name="", step=1_000_000, extra=()):
    """Read B+ -> J/psi K+ skims into Tracks (mu1, mu2, k). `selection` is a numpy expression over branch names."""
    if isinstance(paths, str):
        paths = [paths]
    paths = [f for p in paths for f in (sorted(glob.glob(p)) or [p])]
    if not paths:
        raise FileNotFoundError("no input files")
    trk = [f"{t}_{c}" for t, _ in JPSIK_TRACKS for c in ("pt", "eta", "phi", "charge")]
    branches = list(dict.fromkeys(trk + JPSIK_EXTRA + _variables(selection) + list(extra)))
    keep = {b: [] for b in branches}; n = 0
    for f in paths:
        for a in _read_file(f, branches, tree, step):
            m = np.ones(len(a["mu1_pt"]), bool)
            if selection:
                m &= eval(selection, {"__builtins__": {}, "abs": np.abs, "np": np}, a)
            for b in branches:
                keep[b].append(np.asarray(a[b])[m])
            n += int(m.sum())
            if max_events and n >= max_events:
                break
        if max_events and n >= max_events:
            break
    d = {b: np.concatenate(v) for b, v in keep.items()}
    if max_events:
        d = {b: v[:max_events] for b, v in d.items()}
    p = np.stack([p3(d[f"{t}_pt"], d[f"{t}_eta"], d[f"{t}_phi"]) for t, _ in JPSIK_TRACKS], 1)
    q = np.stack([d[f"{t}_charge"] for t, _ in JPSIK_TRACKS], 1).astype(np.int8)
    ex = {b: d[b].astype(np.float64) for b in JPSIK_EXTRA + list(extra)}
    return Tracks(p, q, tuple(k for _, k in JPSIK_TRACKS), ex, name or os.path.basename(paths[0]))
