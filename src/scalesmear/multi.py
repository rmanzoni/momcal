"""Combined run over the kkpi, mumu and jpsik channels: configuration, reading, fit, outputs, closure toys.

    scalesmear multi  CONFIG [--set key=value ...]
    scalesmear multi-toy CONFIG --seed N [--set ...]
    scalesmear multi-toys CONFIG

The configuration is a YAML file merged over MULTI_DEFAULTS. 'model' is the same block as in the single-channel runs.
"""
from __future__ import annotations
import copy, json, os, sys, time
import numpy as np

from .config import _merge, _numify, _expand, set_key, TOY_INJECT

MULTI_DEFAULTS = {
    "name": "RDs_2018BParking_trkcalib",
    "version": "dev",
    "model": {
        "convention": "cms",
        "eta_edges": [0.0, 0.7, 1.2, 2.5],
        "terms": ["A", "e", "M", "a2", "c2"],
        "shared": ["e", "c2"],
        "start": {"A": 1.0e-3, "a2": 11.0},
        "fix": {},
        "ties": {},
        "limits": {"a2": [0.0, 400.0], "c2": [0.0, 400.0]},
        "muon_terms": [],                  # terms with separate values for muon tracks, e.g. [a2, c2] (own resolution)
        "kaon_terms": [],                  # terms with separate values for kaon tracks (Ds and B kaons), e.g. [e]
    },
    "fit": {
        "seed": 12345,                     # per-track Gaussian of the MC templates (hash of the track kinematics)
        "z_mode": "hash",
        "smooth_data": True,
        "start_from": None,                # result.json of an earlier run: its correction values are the starting point
        "mc_stat": False,                  # Barlow-Beeston-lite: per-bin factor on the signal template, constrained by its MC statistics
        "method": "profile",               # 'profile': Minuit over the correction, nuisances re-minimised (L-BFGS) at each step;
                                           # 'full': one Minuit over all parameters (v3's fit_full; slow with many categories)
    },
    "channels": {
        "kkpi": {
            "enabled": True,
            "data": [], "mc": [], "tree": "tree", "max_data": None, "max_mc": None,
            "selection": "(tv_prob > 0.05) & (fv_prob > 0.05) & (ds_vtx_cosine_xyz_pv > 0.995) & (dsMu_m < 5.37) & (sv_prob > 0.05)",
            "kk_window": 0.020, "dplus_veto": 0.050,
            "mass_range": [1.920, 2.015], "bin_width": 0.001,
            "categories": {"pt_edges": None, "split_charge": True, "eta_edges": None,
                           "kappa_edges": [-100.0, 0.15, 0.26, 0.37, 0.52, 100.0], "share_bkg_shape": False},
        },
        "mumu": {
            "enabled": True,
            "data": [], "mc": [], "tree": "tree", "max_data": None, "max_mc": None,
            # B+ -> J/psi K+ events near the B peak: the MC is truth-matched B+ -> J/psi K+, the data should be too
            "selection": "(rf_mass > 5.20) & (rf_mass < 5.36)",
            "mass_range": [2.96, 3.24], "bin_width": 0.002, "background": "expo", "bkg_share": "channel",
            "categories": {"eta_edges": None, "dpt_edges": [-1000.0, -4.0, -1.0, 1.0, 4.0, 1000.0]},
        },
        "jpsik": {
            "enabled": True,
            "data": [], "mc": [], "mc_jpsipi": [], "tree": "tree", "max_data": None, "max_mc": None,
            "selection": "(abs(jpsi_mass - 3.0969) < 0.15)",
            "mass_range": [5.17, 5.45], "bin_width": 0.002, "background": "expo2", "bkg_share": "category",
            "refit_power": 1.0,            # carry-over of the muon correction through the J/psi refit (see channels.JpsiRfK)
            "argus": False,                # ARGUS (x) Gauss for partially reconstructed B decays (shape fitted once per category)
            "categories": {"eta_edges": None, "pt_edges": [2.0, 4.0, 7.0, 1000.0], "split_charge": True},
        },
    },
    "toys": {
        "data_frac": 0.3, "bkg_frac": 0.1, "bkg_slope": -2.0, "jpsipi_frac": 0.03,
        "inject": None,
    },
    "correction": {"seed": 20181, "dataset": "", "mc": "", "notes": []},
    "output": {"dir": "results/${name}_${version}", "cache": True, "plots": True},
}


def load_multi_config(path=None, overrides=()):
    import yaml
    user = {}
    if path:
        with open(path) as f:
            user = _numify(yaml.safe_load(f) or {})
    cfg = _merge(MULTI_DEFAULTS, user)
    for o in overrides:
        set_key(cfg, *o.split("=", 1))
    if cfg["toys"]["inject"] is None:
        cfg["toys"]["inject"] = copy.deepcopy(TOY_INJECT[cfg["model"]["convention"]])
    cfg = _expand(cfg, cfg)
    for name, ch in cfg["channels"].items():
        for k in ("data", "mc", "mc_jpsipi"):
            if isinstance(ch.get(k), str):
                ch[k] = [ch[k]]
        if ch["enabled"] and (not ch["data"] or not ch["mc"]):
            raise ValueError(f"channels.{name}: data and mc must be given (or enabled: false)")
    if not any(ch["enabled"] for ch in cfg["channels"].values()):
        raise ValueError("no channel enabled")
    return cfg


def physics_hash(cfg):
    import hashlib
    keep = {k: cfg[k] for k in ("model", "fit")}
    keep["channels"] = {n: {k: v for k, v in c.items() if k not in ("data", "mc", "mc_jpsipi")} for n, c in cfg["channels"].items() if c["enabled"]}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def make_corr(cfg):
    from .model import MODELS, CmsSpeciesCorrection
    m = cfg["model"]
    if m.get("muon_terms") or m.get("kaon_terms"):
        if m["convention"] != "cms":
            raise ValueError("model.muon_terms / kaon_terms need convention 'cms'")
        return CmsSpeciesCorrection(m["eta_edges"], terms=tuple(m["terms"]), shared=tuple(m["shared"]), ties=m.get("ties") or {},
                                    limits=m.get("limits") or {}, muon_terms=tuple(m.get("muon_terms") or ()),
                                    kaon_terms=tuple(m.get("kaon_terms") or ()))
    return MODELS[m["convention"]](m["eta_edges"], terms=tuple(m["terms"]), shared=tuple(m["shared"]), ties=m.get("ties") or {},
                                   limits=m.get("limits") or {})


def _fix_and_start(cfg, corr):
    m = cfg["model"]
    fix = {n: float(m["fix"].get(n, m["fix"].get(n.rsplit("_", 1)[0]))) for n in corr.names
           if n in m["fix"] or n.rsplit("_", 1)[0] in m["fix"]}
    term = lambda n: n.rsplit("_", 1)[0]
    base = lambda n: corr._base(n).rsplit("_", 1)[0] if hasattr(corr, "_base") else term(n)
    start = {n: float(m["start"].get(n, m["start"].get(term(n), m["start"].get(base(n), corr.start(n))))) for n in corr.names}
    return fix, start


# ------------------------------------------------------------------------------------------------ reading
def read_inputs(cfg, outdir, log=print):
    """{channel: dict(data=..., mc=..., ...)} for the enabled channels. The two J/psi K channels read the same files once."""
    from .tracks import load_jpsik
    C = cfg["channels"]; out = {}
    if C["kkpi"]["enabled"]:
        from .pipeline import _read
        from . import selection as sel
        k = C["kkpi"]
        shim = {"inputs": {"data": k["data"], "mc": k["mc"], "tree": k["tree"], "max_data": k["max_data"], "max_mc": k["max_mc"]},
                "selection": {"expression": k["selection"]}, "output": {"cache": cfg["output"]["cache"]}}
        d = _read(shim, "data", outdir); m = _read(shim, "mc", outdir)
        d, ed = sel.apply(d, kk=k["kk_window"], veto=k["dplus_veto"]); m, em = sel.apply(m, kk=0, veto=k["dplus_veto"])
        log(f"kkpi : data {len(d)}, MC {len(m)} after the m(KK) window (data) and the D+ veto", flush=True)
        out["kkpi"] = dict(data=d, mc=m, eff=dict(data=ed, mc=em))
    for name in ("mumu", "jpsik"):
        c = C[name]
        if not c["enabled"]:
            continue
        t = time.time()
        d = load_jpsik(c["data"], c["selection"], c["tree"], c["max_data"], name="data")
        m = load_jpsik(c["mc"], c["selection"], c["tree"], c["max_mc"], name="mc")
        x = load_jpsik(c["mc_jpsipi"], c["selection"], c["tree"], c["max_mc"], name="mc_jpsipi") if name == "jpsik" and c.get("mc_jpsipi") else None
        log(f"{name:5s}: data {len(d)}, MC {len(m)}" + (f", J/psi pi MC {len(x)}" if x is not None else "") +
            f" after '{c['selection']}' in {time.time() - t:.0f} s", flush=True)
        out[name] = dict(data=d, mc=m, mc_x=x)
    return out


def build_channels(cfg, inputs, corr):
    from .channels import MuMuChannel, JpsiKChannel, KKPiChannel
    from .fit import ScaleSmearFit, make_categories
    C = cfg["channels"]; F = cfg["fit"]; chans = []
    if "kkpi" in inputs:
        k = C["kkpi"]; c = k["categories"]
        cats = make_categories(c.get("eta_edges") or cfg["model"]["eta_edges"], c["pt_edges"], c["split_charge"],
                               kappa_edges=c["kappa_edges"], share_bkg_shape=c["share_bkg_shape"])
        f = ScaleSmearFit(inputs["kkpi"]["data"], inputs["kkpi"]["mc"], corr, categories=cats, mrange=tuple(k["mass_range"]),
                          bw=k["bin_width"], comb="expo2", kk_window=k["kk_window"], seed=F["seed"], smooth_data=F["smooth_data"],
                          z_mode=F["z_mode"])
        chans.append(KKPiChannel(f))
    if "mumu" in inputs:
        k = C["mumu"]; c = k["categories"]
        chans.append(MuMuChannel(inputs["mumu"]["data"], inputs["mumu"]["mc"], corr, mrange=tuple(k["mass_range"]), bw=k["bin_width"],
                                 eta_edges=c["eta_edges"], dpt_edges=c["dpt_edges"], seed=F["seed"], z_mode=F["z_mode"], smooth_data=F["smooth_data"],
                                 background=k["background"], bkg_share=k["bkg_share"]))
    if "jpsik" in inputs:
        k = C["jpsik"]; c = k["categories"]
        chans.append(JpsiKChannel(inputs["jpsik"]["data"], inputs["jpsik"]["mc"], corr, mc_x=inputs["jpsik"]["mc_x"],
                                  mrange=tuple(k["mass_range"]), bw=k["bin_width"], eta_edges=c["eta_edges"], pt_edges=c["pt_edges"],
                                  split_charge=c["split_charge"], seed=F["seed"], z_mode=F["z_mode"], smooth_data=F["smooth_data"],
                                  refit_power=k["refit_power"], background=k["background"], bkg_share=k["bkg_share"], argus=k["argus"]))
    return chans


# ------------------------------------------------------------------------------------------------ run
class _Tee:
    def __init__(self, path):
        self.f = open(path, "a"); self.out = sys.stdout
    def write(self, s):
        self.f.write(s); self.out.write(s)
    def flush(self):
        self.f.flush(); self.out.flush()


def run(cfg):
    from . import provenance
    from .multifit import MultiFit
    out = cfg["output"]["dir"]; os.makedirs(out, exist_ok=True)
    sys.stdout = _Tee(os.path.join(out, "log.txt"))
    try:
        import yaml
        yaml.safe_dump(cfg, open(os.path.join(out, "config_used.yaml"), "w"), sort_keys=False)
        prov = provenance.collect({"inputs": {"data": [f for c in cfg["channels"].values() if c["enabled"] for f in c["data"]],
                                              "mc": [f for c in cfg["channels"].values() if c["enabled"] for f in c["mc"]]}})
        prov["physics_hash"] = physics_hash(cfg)
        json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=1)
        print(f"scalesmear {prov['scalesmear']} | git {prov['git']['commit']} | physics hash {prov['physics_hash']} | "
              f"channels {[n for n, c in cfg['channels'].items() if c['enabled']]}", flush=True)
        corr = make_corr(cfg); fix, start = _fix_and_start(cfg, corr)
        if cfg["fit"]["start_from"]:
            prev = json.load(open(cfg["fit"]["start_from"]))["values"]
            start.update({n: float(prev[n]) for n in corr.names if n in prev and n not in fix})
            print(f"start values from {cfg['fit']['start_from']}", flush=True)
        inputs = read_inputs(cfg, out)
        t = time.time(); chans = build_channels(cfg, inputs, corr); del inputs
        print(f"templates set up in {time.time() - t:.0f} s: " + ", ".join(f"{c.name} {c.ncat} categories" for c in chans), flush=True)
        fit = MultiFit(chans, corr, fix=fix); fit.bb = bool(cfg["fit"]["mc_stat"])
        t0 = time.time(); status = os.path.join(out, "status.json")
        def beat(n):
            if n % 200 == 0:
                json.dump(dict(calls=n, elapsed_s=round(time.time() - t0)), open(status, "w"))
                if n % 2000 == 0: print(f"  {n} likelihood calls, {(time.time() - t0) / 60:.1f} min", flush=True)
        print(f"fit ({cfg['fit']['method']}): {len(corr.names)} correction terms ({len(fix)} fixed) + {len(fit.nuis_names)} nuisances", flush=True)
        res = (fit.fit_profile if cfg["fit"]["method"] == "profile" else fit.fit_full)(start, progress=beat)
        dt = time.time() - t0
        print(res.summary(), flush=True)
        print(f"fit time {dt / 60:.1f} min, {fit.minuit.nfcn} calls, valid {res.valid}", flush=True)
        res.config.update(convention=cfg["model"]["convention"], ties=dict(corr.ties), fit_time_s=round(dt), physics_hash=prov["physics_hash"],
                          fix=fix, mrange=None, bw=None, seed=cfg["fit"]["seed"], z_mode=cfg["fit"]["z_mode"])
        rpath = os.path.join(out, "result.json"); res.to_json(rpath)
        from . import export
        cc = cfg["correction"]
        notes = list(cc["notes"]) + [f"Produced by scalesmear {prov['scalesmear']} (git {prov['git']['commit']}), physics hash {prov['physics_hash']}."]
        chan_txt = {"kkpi": "Ds -> phi(KK) pi", "mumu": "J/psi -> mu mu (in B+ -> J/psi K+)", "jpsik": "B+ -> J/psi K+ (refitted J/psi + K)"}
        used = [chan_txt[c.name] for c in chans]
        sel = "; ".join(f"{n}: {c['selection']}" for n, c in cfg["channels"].items() if c["enabled"])
        cpath = os.path.join(out, f"rds_trkcalib_{cfg['name']}_{cfg['version']}.json")
        export.write_correction(rpath, cpath, name=cfg["name"], version=cfg["version"], dataset=cc["dataset"], mc=cc["mc"], selection=sel,
                                notes=notes, seed=cc["seed"],
                                description="Per-track momentum scale and extra resolution for MC, fitted jointly to the mass peaks of "
                                            + ", ".join(used) + " in data.")
        print("correction file:", cpath, flush=True)
        if cfg["output"]["plots"]:
            from . import plots, plots_multi
            plots.plot_terms(res, os.path.join(out, "terms.png"))
            plots_multi.plot_all(fit, res, out)
        print("done:", out, flush=True)
        return res
    finally:
        if isinstance(sys.stdout, _Tee):
            sys.stdout.f.close(); sys.stdout = sys.stdout.out


# ------------------------------------------------------------------------------------------------ toys
def _values(corr, spec):
    return {n: float(spec.get(n, spec.get(n.rsplit("_", 1)[0], 0.0 if n.split("_")[0] in ("A", "e", "M") else corr.start(n))))
            for n in corr.names}


def run_toy(cfg, seed):
    """Closure toy for the J/psi channels (and kkpi if enabled): a fraction of each MC sample, corrected with the injected
    values and with its own random numbers, becomes pseudo-data, plus an exponential background (and J/psi pi in jpsik);
    the rest of the MC is the template. Writes <dir>/toys/toy_<seed>.json."""
    from .multifit import MultiFit
    from .channels import TrackMorph, JpsiRfK, m_mumu, MuMuChannel, JpsiKChannel, KKPiChannel
    out = os.path.join(cfg["output"]["dir"], "toys"); os.makedirs(out, exist_ok=True)
    T = cfg["toys"]; corr = make_corr(cfg); fix, start = _fix_and_start(cfg, corr)
    inj = _values(corr, T["inject"]); iv = np.array([inj[n] for n in corr.names])
    inputs = read_inputs(cfg, cfg["output"]["dir"]); rng = np.random.default_rng(1000 + seed)

    def bkg_masses(n, ch):
        a, b = ch["mass_range"]; lam = T["bkg_slope"]
        return a + np.log1p(rng.random(n) * np.expm1(lam * (b - a))) / lam

    pseudo = {}
    for name, d in inputs.items():
        ch = cfg["channels"][name]
        if name == "kkpi":
            from .closure import pseudo_data
            a, b = d["mc"].split(T["data_frac"], seed=seed)
            lo, hi = ch["mass_range"]
            pseudo[name] = (pseudo_data(a, corr, iv, bkg_frac=T["bkg_frac"], lam=T["bkg_slope"], mrange=(lo - 0.012, hi + 0.013),
                                        seed=1000 + seed, kk_window=ch["kk_window"]), b)
            continue
        a, b = d["mc"].split(T["data_frac"], seed=seed)
        tm = TrackMorph(a, corr, seed=777 + seed, z_mode="hash"); f = tm.factors(iv)
        m = m_mumu(tm, f) if name == "mumu" else JpsiRfK(a, tm, ch["refit_power"])(f)
        extra = []
        if name == "jpsik" and d["mc_x"] is not None and T["jpsipi_frac"] > 0:
            xa, xb = d["mc_x"].split(0.5, seed=seed)
            nx = min(int(T["jpsipi_frac"] * len(a)), len(xa)); xa = xa.subset(rng.permutation(len(xa)) < nx)
            tx = TrackMorph(xa, corr, seed=778 + seed, z_mode="hash"); extra.append((xa, JpsiRfK(xa, tx, ch["refit_power"])(tx.factors(iv))))
            d = dict(d, mc_x=xb)
        nb = int(T["bkg_frac"] / (1 - T["bkg_frac"]) * len(a)); j = rng.integers(0, len(a), nb)
        bk = a.subset(j)
        trk = a; mass = m
        for t_, m_ in extra + [(bk, bkg_masses(nb, ch))]:
            trk = trk.concat(t_); mass = np.concatenate([mass, m_])
        pseudo[name] = (trk, mass, b, d.get("mc_x"))

    F = cfg["fit"]; chans = []
    if "kkpi" in pseudo:
        from .fit import ScaleSmearFit, make_categories
        k = cfg["channels"]["kkpi"]; c = k["categories"]
        cats = make_categories(c.get("eta_edges") or cfg["model"]["eta_edges"], c["pt_edges"], c["split_charge"], kappa_edges=c["kappa_edges"],
                               share_bkg_shape=c["share_bkg_shape"])
        chans.append(KKPiChannel(ScaleSmearFit(pseudo["kkpi"][0], pseudo["kkpi"][1], corr, categories=cats, mrange=tuple(k["mass_range"]),
                                               bw=k["bin_width"], kk_window=k["kk_window"], seed=F["seed"], z_mode=F["z_mode"])))
    for name in ("mumu", "jpsik"):
        if name not in pseudo:
            continue
        trk, mass, tmpl, mcx = pseudo[name]; k = cfg["channels"][name]; c = k["categories"]
        if name == "mumu":
            ch = MuMuChannel(tmpl, tmpl, corr, mrange=tuple(k["mass_range"]), bw=k["bin_width"], eta_edges=c["eta_edges"],
                             dpt_edges=c["dpt_edges"], seed=F["seed"], z_mode=F["z_mode"], background=k["background"],
                             bkg_share=k["bkg_share"])
        else:
            ch = JpsiKChannel(tmpl, tmpl, corr, mc_x=mcx, mrange=tuple(k["mass_range"]), bw=k["bin_width"], eta_edges=c["eta_edges"],
                              pt_edges=c["pt_edges"], split_charge=c["split_charge"], seed=F["seed"], z_mode=F["z_mode"], refit_power=k["refit_power"],
                              background=k["background"], bkg_share=k["bkg_share"], argus=k["argus"])
        ch._set_data(mass, ch.cat(trk), F["smooth_data"])            # replace the data histograms by the pseudo-data
        chans.append(ch)
    fit = MultiFit(chans, corr, fix=fix); fit.bb = bool(cfg["fit"]["mc_stat"]); t0 = time.time()
    res = (fit.fit_profile if cfg["fit"]["method"] == "profile" else fit.fit_full)(start, log=lambda *a, **k: None)
    d = dict(seed=seed, valid=res.valid, injected=inj, values={n: res.values[n] for n in corr.names},
             errors={n: res.errors[n] for n in corr.names}, fixed=fix, chi2=sum(res.chi2), ndf=sum(res.ndf),
             channels=[c.name for c in chans], fit_time_s=round(time.time() - t0), physics_hash=physics_hash(cfg))
    path = os.path.join(out, f"toy_{seed:04d}.json"); json.dump(d, open(path, "w"), indent=1)
    print(f"toy {seed}: valid {res.valid}, chi2 {d['chi2']:.0f}/{d['ndf']}, {d['fit_time_s'] / 60:.1f} min -> {path}", flush=True)
    return d


def collect_toys(cfg):
    """Pull means and widths over the toys of this configuration (fixed terms left out)."""
    import glob
    out = cfg["output"]["dir"]; files = sorted(glob.glob(os.path.join(out, "toys", "toy_*.json")))
    T = [json.load(open(f)) for f in files]; T = [t for t in T if t["valid"]]
    if not T:
        raise FileNotFoundError(f"no valid toys in {out}/toys")
    names = [n for n in T[0]["values"] if n not in T[0].get("fixed", {})]
    pulls = {n: np.array([(t["values"][n] - t["injected"][n]) / t["errors"][n] for t in T]) for n in names}
    summ = {n: dict(mean=float(p.mean()), width=float(p.std(ddof=1)) if len(p) > 1 else float("nan"), n=len(p)) for n, p in pulls.items()}
    json.dump(dict(n_files=len(files), n_valid=len(T), pulls=summ), open(os.path.join(out, "toys_summary.json"), "w"), indent=1)
    from . import plots
    if len(T) > 1:
        plots.plot_pulls(pulls, os.path.join(out, "toys_pulls.png"), f"{len(T)} valid toys of {len(files)}")
    for n, s in summ.items():
        print(f"  {n:8s} pull mean {s['mean']:+.2f}  width {s['width']:.2f}  ({s['n']} toys)")
    return summ
