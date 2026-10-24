"""Run configuration: a YAML file, merged over DEFAULTS. Environment variables in strings (${VAR}) are expanded.

Every key that changes the result lives here, so a configuration file plus the package version fully defines a run.
"""
from __future__ import annotations
import copy, hashlib, json, os, re

DEFAULTS = {
    "name": "RDs_2018BParking_DsPhiPi_trkcalib",
    "version": "dev",
    "inputs": {
        "data": [],                       # ROOT files or .npz extracts; globs allowed
        "mc": [],
        "tree": "tree",
        "max_data": None,                 # use at most this many selected candidates (quick tests)
        "max_mc": None,
    },
    "selection": {
        # numpy expression over branch names, applied while reading (here: the 'cos' working point of the R(Ds) study)
        "expression": "(dsMu_m < 5.37) & (tv_prob > 0.05) & (sv_prob > 0.05) & (ds_vtx_cosine_xyz_pv > 0.99)",
        "kk_window": 0.020,               # |m(KK) - m(phi)| < kk_window [GeV]: data on stored m(KK), MC after the correction
        "dplus_veto": 0.050,              # remove |m(K pi pi) - m(D+)| < dplus_veto [GeV], data and MC; 0 = no veto
    },
    "model": {
        "eta_edges": [0.0, 0.7, 1.2, 2.5],
        "terms": ["s0", "s1", "k", "v0", "v1"],
        "shared": ["s1", "v1"],
        "start": {"s0": -1.0e-3, "v0": 11.0},   # per term (all bins) or per parameter name (e.g. s0_2)
    },
    "categories": {
        "pt_edges": [0.0, 1.0, 1.6, 100.0],   # softest-track pT [GeV]
        "split_charge": True,
    },
    "fit": {
        "mass_range": [1.920, 2.015],
        "bin_width": 0.001,
        "background": "expo2",
        "smooth_data": True,
        "seed": 12345,                    # per-track Gaussian z of the MC template
        "z_mode": "sequential",           # 'sequential' (v1: numpy generator, candidate order) or 'hash' (from the track itself)
    },
    "correction": {                        # what goes into the exported correction file
        "seed": 20181,                    # seed of the hash-based z used by the appliers
        "dataset": "",
        "mc": "",
        "notes": [],
    },
    "toys": {                              # closure toys: part of the MC becomes pseudo-data with a known correction
        "data_frac": 0.2,                 # fraction of the MC used as pseudo-data (the rest is the template)
        "bkg_frac": 0.5,                  # background fraction added to the pseudo-data
        "bkg_slope": -2.6,                # exponential slope of that background [1/GeV]
        "inject": {"s0": -1.0e-3, "s1": -0.3e-3, "k_0": 2.0e-5, "k_1": 5.0e-5, "k_2": -1.2e-4,
                   "v0_0": 6.0, "v0_1": 12.0, "v0_2": 2.0, "v1": 14.0},
    },
    "output": {
        "dir": "results/${name}_${version}",
        "cache": True,                    # keep the selected tracks as .npz in <dir>/cache for fast reruns
        "plots": True,
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise KeyError(f"unknown configuration key '{k}'")
        out[k] = _merge(out[k], v) if isinstance(out[k], dict) and isinstance(v, dict) and k not in ("start", "inject") else v
    return out


_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _numify(x):
    """PyYAML (YAML 1.1) reads 1e-3 without a dot as a string: turn such strings into floats."""
    if isinstance(x, str) and _FLOAT.match(x.strip()):
        return float(x)
    if isinstance(x, list):
        return [_numify(v) for v in x]
    if isinstance(x, dict):
        return {k: _numify(v) for k, v in x.items()}
    return x


def _expand(x, ctx):
    if isinstance(x, str):
        x = re.sub(r"\$\{(name|version)\}", lambda m: str(ctx[m.group(1)]), x)
        return os.path.expandvars(os.path.expanduser(x))
    if isinstance(x, list):
        return [_expand(v, ctx) for v in x]
    if isinstance(x, dict):
        return {k: _expand(v, ctx) for k, v in x.items()}
    return x


def set_key(cfg, dotted, value):
    """--set fit.seed=7 style override; the value is parsed as YAML."""
    import yaml
    keys = dotted.split("."); d = cfg
    for k in keys[:-1]:
        d = d[k]
    if keys[-1] not in d and keys[-2:-1] not in (["start"], ["inject"]):
        raise KeyError(f"unknown configuration key '{dotted}'")
    d[keys[-1]] = _numify(yaml.safe_load(value))


def load_config(path=None, overrides=()):
    import yaml
    user = {}
    if path:
        with open(path) as f:
            user = _numify(yaml.safe_load(f) or {})
    cfg = _merge(DEFAULTS, user)
    for o in overrides:
        k, _, v = o.partition("=")
        set_key(cfg, k, v)
    cfg = _expand(cfg, cfg)
    if not cfg["inputs"]["data"] or not cfg["inputs"]["mc"]:
        raise ValueError("inputs.data and inputs.mc must be given")
    for k in ("data", "mc"):
        if isinstance(cfg["inputs"][k], str):
            cfg["inputs"][k] = [cfg["inputs"][k]]
    return cfg


def physics_hash(cfg):
    """Hash of everything that changes the fit result (not output paths)."""
    keep = {k: cfg[k] for k in ("selection", "model", "categories", "fit")}
    keep["inputs"] = {k: cfg["inputs"][k] for k in ("tree", "max_data", "max_mc")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]
