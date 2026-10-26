"""One configured run: read and select -> fit -> result.json, correction file, plots, provenance."""
from __future__ import annotations
import hashlib, json, os, sys, time
import numpy as np
from .io import load, Sample
from .fit import ScaleSmearFit, make_categories
from . import selection, provenance, export
from .config import physics_hash


class _Tee:
    def __init__(self, path):
        self.f = open(path, "a"); self.out = sys.stdout
    def write(self, s):
        self.out.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.out.flush(); self.f.flush()


def _read(cfg, which, outdir):
    """Selected Sample for 'data' or 'mc', from the cache when the inputs and selection are unchanged."""
    files = provenance.expand(cfg["inputs"][which])
    if not files:
        raise FileNotFoundError(f"no {which} files match {cfg['inputs'][which]}")
    key = json.dumps(dict(files=[provenance.file_fingerprint(f) for f in files], sel=cfg["selection"]["expression"],
                          tree=cfg["inputs"]["tree"], max=cfg["inputs"][f"max_{which}"]), sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache = os.path.join(outdir, "cache", f"{which}_{h}.npz")
    if cfg["output"]["cache"] and os.path.exists(cache):
        z = np.load(cache)
        print(f"{which}: {len(z['m_kkpi'])} candidates from cache {cache}", flush=True)
        return Sample(z["p"], z["charge"], z["m_kkpi"], z["m_kk"], which)
    t = time.time()
    s = load(files, cfg["selection"]["expression"], cfg["inputs"]["tree"], max_events=cfg["inputs"][f"max_{which}"], name=which)
    print(f"{which}: {len(s)} candidates after '{cfg['selection']['expression']}' from {len(files)} file(s) in {time.time() - t:.0f} s", flush=True)
    if cfg["output"]["cache"]:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = f"{cache[:-4]}.{os.getpid()}.tmp.npz"            # atomic: parallel toy jobs may write the same cache
        np.savez(tmp, p=s.p, charge=s.charge, m_kkpi=s.m_kkpi, m_kk=s.m_kk); os.replace(tmp, cache)
    return s


def make_corr(cfg):
    from .model import MODELS
    m = cfg["model"]
    return MODELS[m["convention"]](m["eta_edges"], terms=tuple(m["terms"]), shared=tuple(m["shared"]), ties=m.get("ties") or {},
                                   limits=m.get("limits") or {})


def build_fit(cfg, data, mc):
    m = cfg["model"]; c = cfg["categories"]; f = cfg["fit"]
    corr = make_corr(cfg)
    cats = make_categories(c.get("eta_edges") or m["eta_edges"], c["pt_edges"], c["split_charge"], kappa_edges=c["kappa_edges"],
                           share_bkg_shape=c["share_bkg_shape"])
    fix = {n: float(m["fix"].get(n, m["fix"].get(n.rsplit("_", 1)[0]))) for n in corr.names
           if n in m["fix"] or n.rsplit("_", 1)[0] in m["fix"]}
    fit = ScaleSmearFit(data, mc, corr, categories=cats, fix=fix, mrange=tuple(f["mass_range"]), bw=f["bin_width"], comb=f["background"],
                        kk_window=cfg["selection"]["kk_window"], seed=f["seed"], smooth_data=f["smooth_data"],
                        z_mode=f["z_mode"])
    if f.get("mc_bootstrap_seed") is not None:            # resample the MC: each candidate weighted by Poisson(1)
        fit.mc_w = np.random.default_rng(int(f["mc_bootstrap_seed"])).poisson(1.0, fit.mc.n).astype(float)
    start = {}
    for n in corr.names:
        t = n.rsplit("_", 1)[0]
        start[n] = float(m["start"].get(n, m["start"].get(t, corr.start(n))))
    return fit, corr, start


def run(cfg):
    out = cfg["output"]["dir"]; os.makedirs(out, exist_ok=True)
    sys.stdout = _Tee(os.path.join(out, "log.txt"))
    try:
        import yaml
        with open(os.path.join(out, "config_used.yaml"), "w") as fo:
            yaml.safe_dump(cfg, fo, sort_keys=False)
        prov = provenance.collect(cfg); prov["physics_hash"] = physics_hash(cfg)
        json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=1)
        print(f"scalesmear {prov['scalesmear']} | git {prov['git']['commit']} | numpy {prov['libraries']['numpy']} iminuit "
              f"{prov['libraries']['iminuit']} | physics hash {prov['physics_hash']}", flush=True)

        data = _read(cfg, "data", out); mc = _read(cfg, "mc", out)
        sel = cfg["selection"]
        data, eff_d = selection.apply(data, kk=sel["kk_window"], veto=sel["dplus_veto"])
        mc, eff_m = selection.apply(mc, kk=0, veto=sel["dplus_veto"])      # MC: m(KK) window applied after the correction
        print(f"after m(KK) window and D+ veto: data {eff_d}, MC {eff_m}", flush=True)

        fit, corr, start = build_fit(cfg, data, mc); del mc
        t0 = time.time(); status = os.path.join(out, "status.json")
        def beat(n):
            if n % 200 == 0:
                json.dump(dict(calls=n, elapsed_s=round(time.time() - t0)), open(status, "w"))
                if n % 2000 == 0: print(f"  {n} likelihood calls, {(time.time() - t0) / 60:.1f} min", flush=True)
        print(f"fit: {len(corr.names)} correction terms + {4 * fit.cat.n} nuisances, {fit.cat.n} categories", flush=True)
        res = fit.fit_full(start, progress=beat)
        dt = time.time() - t0
        print(res.summary(), flush=True)
        print(f"fit time {dt / 60:.1f} min, {fit.minuit.nfcn} calls, valid {res.valid}", flush=True)
        res.config.update(convention=cfg["model"]["convention"], ties=dict(corr.ties), selection=sel, efficiencies=dict(data=eff_d, mc=eff_m), fit_time_s=round(dt), nfcn=int(fit.minuit.nfcn),
                          physics_hash=prov["physics_hash"])
        rpath = os.path.join(out, "result.json"); res.to_json(rpath)
        cc = cfg["correction"]
        notes = list(cc["notes"]) + [f"Produced by scalesmear {prov['scalesmear']} (git {prov['git']['commit']}), physics hash {prov['physics_hash']}."]
        cpath = os.path.join(out, f"rds_trkcalib_{cfg['name']}_{cfg['version']}.json")
        export.write_correction(rpath, cpath, name=cfg["name"], version=cfg["version"], dataset=cc["dataset"], mc=cc["mc"],
                                selection=f"{sel['expression']}; |m(KK)-m(phi)| < {sel['kk_window']}; D+ veto {sel['dplus_veto']} GeV",
                                notes=notes, seed=cc["seed"])
        print("correction file:", cpath, flush=True)
        if cfg["output"]["plots"]:
            from . import plots
            plots.plot_fit(fit, res, os.path.join(out, "fit_categories.png"))
            plots.plot_lineshape(fit, res, os.path.join(out, "lineshape.png"))
            plots.plot_fit_components(fit, res, os.path.join(out, "fit_components.png"))
            plots.plot_terms(res, os.path.join(out, "terms.png"))
        print("done:", out, flush=True)
        return res
    finally:
        if isinstance(sys.stdout, _Tee):
            sys.stdout.f.close(); sys.stdout = sys.stdout.out


def _values(corr, spec, default):
    """Per-parameter values from a {term or parameter name: value} dict."""
    return {n: float(spec.get(n, spec.get(n.rsplit("_", 1)[0], default(n)))) for n in corr.names}


def run_toy(cfg, seed):
    """One closure toy: a fraction of the selected MC (chosen with `seed`) gets the injected correction and an exponential
    background and is fitted as data, with the rest of the MC as template. Writes <dir>/toys/toy_<seed>.json."""
    from .closure import pseudo_data
    out = os.path.join(cfg["output"]["dir"], "toys"); os.makedirs(out, exist_ok=True)
    mc = _read(cfg, "mc", cfg["output"]["dir"])
    mc, _ = selection.apply(mc, kk=0, veto=cfg["selection"]["dplus_veto"])
    T = cfg["toys"]
    a, b = mc.split(T["data_frac"], seed=seed); del mc
    if T["template_frac"] < 1.0:
        b, _ = b.split(T["template_frac"], seed=seed + 7919)
    corr = make_corr(cfg)
    inj = _values(corr, T["inject"], corr.start)
    lo, hi = cfg["fit"]["mass_range"]
    pdata = pseudo_data(a, corr, [inj[n] for n in corr.names], bkg_frac=T["bkg_frac"], lam=T["bkg_slope"],
                        mrange=(lo - 0.012, hi + 0.013), seed=1000 + seed, kk_window=cfg["selection"]["kk_window"])
    del a
    fit, corr, start = build_fit(cfg, pdata, b); del b
    t0 = time.time()
    res = fit.fit_full(start)
    d = dict(seed=seed, valid=res.valid, injected=inj, values={n: res.values[n] for n in corr.names},
             errors={n: res.errors[n] for n in corr.names}, chi2=sum(res.chi2), ndf=sum(res.ndf), n_data=len(pdata),
             fit_time_s=round(time.time() - t0), physics_hash=physics_hash(cfg))
    path = os.path.join(out, f"toy_{seed:04d}.json"); json.dump(d, open(path, "w"), indent=1)
    print(f"toy {seed}: valid {res.valid}, chi2 {d['chi2']:.0f}/{d['ndf']}, {d['fit_time_s'] / 60:.1f} min -> {path}", flush=True)
    return d


def collect_toys(cfg):
    """Pull means and widths over all toys of this configuration; writes toys_summary.json and toys_pulls.png."""
    import glob
    out = cfg["output"]["dir"]; files = sorted(glob.glob(os.path.join(out, "toys", "toy_*.json")))
    T = [json.load(open(f)) for f in files]; T = [t for t in T if t["valid"]]
    if not T:
        raise FileNotFoundError(f"no valid toys in {out}/toys")
    names = list(T[0]["values"])
    pulls = {n: np.array([(t["values"][n] - t["injected"][n]) / t["errors"][n] for t in T]) for n in names}
    summ = {n: dict(mean=float(p.mean()), width=float(p.std(ddof=1)), n=len(p)) for n, p in pulls.items()}
    json.dump(dict(n_files=len(files), n_valid=len(T), pulls=summ), open(os.path.join(out, "toys_summary.json"), "w"), indent=1)
    from . import plots
    plots.plot_pulls(pulls, os.path.join(out, "toys_pulls.png"), f"{len(T)} valid toys of {len(files)}")
    for n, s in summ.items():
        print(f"  {n:8s} pull mean {s['mean']:+.2f}  width {s['width']:.2f}  ({s['n']} toys)")
    return summ
