"""Reading the flat ntuples (ROOT via uproot, or .npz extracts) into a Sample."""
from __future__ import annotations
import glob, os
from dataclasses import dataclass, field
import numpy as np

MK, MPI = 0.493677, 0.13957
TRACKS = ("k1", "k2", "pi")
MOMENTA = [f"{t}_{c}" for t in TRACKS for c in ("px", "py", "pz")]
CHARGES = [f"{t}_charge" for t in TRACKS]
BASE = MOMENTA + CHARGES + ["phiPi_m", "kk_m"]

# Cumulative working points used in the R(Ds) slope study (AN-24-173 style offline selection on top of the skim)
WORKING_POINTS = {
    "skim": "",
    "tv": "(dsMu_m < 5.37) & (tv_prob > 0.05)",
    "sv": "(dsMu_m < 5.37) & (tv_prob > 0.05) & (sv_prob > 0.05)",
    "cos": "(dsMu_m < 5.37) & (tv_prob > 0.05) & (sv_prob > 0.05) & (ds_vtx_cosine_xyz_pv > 0.99)",
    "iso": "(dsMu_m < 5.37) & (tv_prob > 0.05) & (sv_prob > 0.05) & (ds_vtx_cosine_xyz_pv > 0.99) & (rel_iso_03_pv < 0.1)",
    "pipt": "(dsMu_m < 5.37) & (tv_prob > 0.05) & (sv_prob > 0.05) & (ds_vtx_cosine_xyz_pv > 0.99) & (rel_iso_03_pv < 0.1) & (pi_pt > 1.5)",
}

@dataclass
class Sample:
    """Tracks of KK pi candidates. p has shape (n, 3, 3): candidate, track (k1, k2, pi), component (px, py, pz)."""
    p: np.ndarray
    charge: np.ndarray            # (n, 3) int
    m_kkpi: np.ndarray            # stored m(KK pi) [GeV]
    m_kk: np.ndarray              # stored m(KK) [GeV]
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.m_kkpi)

    @property
    def masses(self):
        return np.array([MK, MK, MPI])

    def eta(self):
        pt = np.hypot(self.p[..., 0], self.p[..., 1])
        return np.arcsinh(self.p[..., 2] / np.maximum(pt, 1e-9))

    def subset(self, mask):
        return Sample(self.p[mask], self.charge[mask], self.m_kkpi[mask], self.m_kk[mask], self.name,
                      {k: v[mask] for k, v in self.extra.items()})

    def split(self, frac=0.5, seed=0):
        r = np.random.default_rng(seed).random(len(self)) < frac
        return self.subset(r), self.subset(~r)


def _variables(expr):
    import re
    return sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr or "")) - {"and", "or", "not", "abs"})


def _read_file(path, branches, tree, step):
    if path.endswith(".npz"):
        z = np.load(path)
        missing = [b for b in branches if b not in z.files]
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        yield {b: z[b] for b in branches}
        return
    import uproot
    t = uproot.open(path)[tree]
    for a in t.iterate(branches, step_size=step, library="np"):
        yield a


def load(paths, selection: str = "", tree: str = "tree", extra=(), max_events=None, name="", step=1_000_000) -> Sample:
    """Read ROOT ntuples or .npz extracts. `paths` is a file, a glob or a list.
    `selection` is a numpy expression over branch names, or a key of WORKING_POINTS."""
    if isinstance(paths, str):
        paths = [paths]
    paths = [f for p in paths for f in (sorted(glob.glob(p)) or [p])]
    selection = WORKING_POINTS.get(selection, selection)
    branches = list(dict.fromkeys(BASE + _variables(selection) + list(extra)))
    keep = {b: [] for b in branches}
    n = 0
    for f in paths:
        for a in _read_file(f, branches, tree, step):
            m = np.ones(len(a["phiPi_m"]), bool)
            if selection:
                m &= eval(selection, {"__builtins__": {}, "abs": np.abs, "np": np}, a)
            for b in branches:
                keep[b].append(np.asarray(a[b])[m])
            n += m.sum()
            if max_events and n >= max_events:
                break
        if max_events and n >= max_events:
            break
    d = {b: np.concatenate(v) for b, v in keep.items()}
    if max_events:
        d = {b: v[:max_events] for b, v in d.items()}
    p = np.stack([np.stack([d[f"{t}_{c}"] for c in ("px", "py", "pz")], -1) for t in TRACKS], 1).astype(np.float64)
    q = np.stack([d[c] for c in CHARGES], 1).astype(np.int8)
    return Sample(p, q, d["phiPi_m"].astype(np.float64), d["kk_m"].astype(np.float64), name or os.path.basename(str(paths[0])),
                  {b: d[b] for b in extra})
