#!/usr/bin/env python3
"""Skim and prune the B+ -> J/psi K+ ntuples (Bmmm, bparking2018_jpsi_tk) for the track/muon
momentum calibration. Samples, selection, mass range, arbitration and extra branches are read
from config.py; one output file per sample in config.SKIMDIR.

  python3 slim_hadd_jpsik.py                         # all samples in config.SAMPLES
  python3 slim_hadd_jpsik.py data mc_jpsik -j 32     # some of them
  python3 slim_hadd_jpsik.py data --fraction 0.3     # random 30% of the data candidates
  python3 slim_hadd_jpsik.py mc_jpsik --max-files 5  # quick test

Per file, in parallel workers:
  selection = config.SELECTION & sample truth (MC) & MASS_RANGE on MASS_BRANCH
  derived   = config.DERIVED flags (bool)
  arbitration: one candidate per (run, lumi, event), highest ARBITRATION_BRANCH
               (events do not span files, so this is exact)
The main process concatenates and writes large blocks, as slim_hadd.py does.
"""
import argparse, ast, glob, importlib.util, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import uproot

# Needed by the calibration: raw track momenta and charges, the three mass variables,
# the refitted J/psi four-vector, event ids (they seed the per-track smearing).
CALIB = [
    "mu1_pt", "mu1_eta", "mu1_phi", "mu1_charge",
    "mu2_pt", "mu2_eta", "mu2_phi", "mu2_charge",
    "k_pt", "k_eta", "k_phi", "k_charge",
    "jpsi_mass", "mass", "rf_mass",
    "jpsi_rf_pt", "jpsi_rf_eta", "jpsi_rf_phi", "jpsi_rf_mass",
    "run", "lumi", "event",
]
# MC only: generator match, used to check the templates
CALIB_MC = [f"{t}_gen_{v}" for t in ("mu1", "mu2", "k") for v in ("pt", "eta", "phi", "pdgid", "charge", "dr")]
# Kept if present, otherwise dropped with a warning
OPTIONAL = ["ncands", "npv", "sv_prob", "cos2d"]


def names_in(expr):
    """Branch names used in a numpy expression (ast-based, so '1e4' is not taken for a name)."""
    if not expr:
        return []
    return sorted({n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)} - {"abs", "np"})


def evaluate(expr, arr):
    return np.asarray(eval(expr, {"__builtins__": {}, "abs": np.abs, "np": np}, arr))


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def find_tree(path, tree):
    with uproot.open(path) as fh:
        if tree:
            return tree, set(fh[tree].keys())
        for k, cls in fh.classnames().items():
            if cls == "TTree":
                name = k.split(";")[0]
                return name, set(fh[name].keys())
        raise RuntimeError(f"no TTree in {path}: {fh.classnames()}")


def arbitrate(run, lumi, event, score):
    """Index of the highest-score candidate in each (run, lumi, event)."""
    if len(run) == 0:
        return np.zeros(0, int)
    o = np.lexsort((-score, event, lumi, run))
    r, l, e = run[o], lumi[o], event[o]
    first = np.ones(len(o), bool)
    first[1:] = (r[1:] != r[:-1]) | (l[1:] != l[:-1]) | (e[1:] != e[:-1])
    return np.sort(o[first])


def read_one(job):
    f, tree, read, keep, cut, derived, arb, fraction, seed = job
    try:
        with uproot.open(f, object_cache=None, array_cache=None) as fh:
            if tree not in fh:
                return f, None, (0, 0), "no tree"
            t = fh[tree]
            if t.num_entries == 0:
                return f, {}, (0, 0), None
            a = t.arrays(read, library="np")
    except Exception as e:
        return f, None, (0, 0), repr(e)
    n = len(a[read[0]])
    m = evaluate(cut, a).astype(bool)
    a = {b: v[m] for b, v in a.items()}
    if arb:
        i = arbitrate(a["run"], a["lumi"], a["event"], a[arb])
        a = {b: v[i] for b, v in a.items()}
    if fraction < 1:      # after arbitration (whole events); seeded by file name, so reruns give the same subsample
        r = np.random.default_rng([seed, zlib.crc32(os.path.basename(f).encode())]).random(len(a[read[0]]))
        a = {b: v[r < fraction] for b, v in a.items()}
    out = {b: a[b] for b in keep}
    for name, expr in derived.items():
        out[name] = evaluate(expr, a).astype(bool)
    return f, out, (n, int(m.sum())), None


class Bar:
    def __init__(self, total, width=40):
        self.total, self.width, self.t0, self.last = total, width, time.time(), 0.0
        self.tty = sys.stderr.isatty()

    def update(self, done, nin, nsel, nout, bad, force=False):
        now = time.time()
        if not force and now - self.last < 0.5:
            return
        self.last = now
        el = now - self.t0
        frac = done / self.total
        eta = el / done * (self.total - done) if done else 0
        fill = int(self.width * frac)
        line = (f"[{'#' * fill}{'.' * (self.width - fill)}] {done}/{self.total} {100*frac:5.1f}%  "
                f"read {nin:,}  selected {nsel:,}  kept {nout:,}  {done/el if el else 0:5.1f} f/s  "
                f"elapsed {fmt(el)}  ETA {fmt(eta)}" + (f"  skipped {bad}" if bad else ""))
        if self.tty:
            sys.stderr.write("\r" + line); sys.stderr.flush()
        elif force or done % max(1, self.total // 50) == 0:
            print(line, flush=True)

    def close(self):
        if self.tty:
            sys.stderr.write("\n")


def fmt(s):
    s = int(s); return f"{s//3600:d}:{s%3600//60:02d}:{s%60:02d}"


def skim(cfg, sample, a):
    spec = cfg.SAMPLES[sample]
    is_mc = spec.get("truth") is not None
    files = sorted(glob.glob(spec["pattern"]))
    if a.max_files:
        files = files[: a.max_files]
    if not files:
        print(f"[{sample}] no files match {spec['pattern']}"); return
    prefix = getattr(cfg, "XROOTD_PREFIX", "") or ""
    files = [prefix + f for f in files]

    tree, have = None, None
    for f in files[:20]:
        try:
            tree, have = find_tree(f, getattr(cfg, "TREE", None)); break
        except Exception as e:
            print(f"[{sample}] skip {f}: {e}")
    if have is None:
        print(f"[{sample}] no readable file"); return

    # the cut: selection & truth & mass range
    lo, hi = cfg.MASS_RANGE
    parts = [f"({cfg.SELECTION})", f"({cfg.MASS_BRANCH} > {lo}) & ({cfg.MASS_BRANCH} < {hi})"]
    if is_mc:
        parts.append(f"({spec['truth']})")
    cut = " & ".join(parts)
    derived = dict(getattr(cfg, "DERIVED", {}))
    arb = None if a.no_arbitration else getattr(cfg, "ARBITRATION_BRANCH", None)

    # branches: required ones stop the job if missing, the rest are dropped with a warning
    required = list(dict.fromkeys(CALIB + (CALIB_MC if is_mc else []) + names_in(cut)
                                  + [n for e in derived.values() for n in names_in(e)] + ([arb] if arb else [])))
    trig = list(getattr(cfg, "TRIGGERS", []))
    trig += [t.replace("mu1_HLT", "mu2_HLT") for t in trig]
    plot = [n for v in getattr(cfg, "PLOT_VARS", {}).values() for n in names_in(v[0])]
    optional = (OPTIONAL + trig + list(getattr(cfg, "KEEP_BRANCHES", [])) + plot
                + (list(getattr(cfg, "MC_EXTRA_BRANCHES", [])) if is_mc else []) + a.extra)
    missing = [b for b in required if b not in have]
    if missing:
        print(f"[{sample}] MISSING required branches, stopping: {missing}"); return
    dropped = sorted({b for b in optional if b not in have})
    if dropped:
        print(f"[{sample}] WARNING missing optional branches (dropped): {dropped}")
    # stored: calibration + trigger flags + config keep/plot/extra; selection-only variables are stored too
    keep = list(dict.fromkeys([b for b in required + optional if b in have]))
    read = keep
    print(f"[{sample}] {len(files)} files, tree '{tree}', {len(keep)} branches stored, "
          f"{a.jobs} workers{', arbitration on ' + arb if arb else ''}"
          f"{f', fraction {a.fraction}' if a.fraction < 1 else ''}")
    print(f"[{sample}] cut: {cut}")

    os.makedirs(a.outdir, exist_ok=True)
    output = os.path.join(a.outdir, f"{sample}{a.suffix}.root")
    try:
        import lz4  # noqa: F401
        comp = uproot.LZ4(4)
    except Exception:
        comp = uproot.ZLIB(1)

    dtypes, buf, nbuf, bad = None, [], 0, []
    nin = nsel = nout = done = 0
    cols = keep + list(derived)
    bar = Bar(len(files))

    def flush(out):
        nonlocal buf, nbuf
        if buf:
            block = {b: np.concatenate([c[b] for c in buf]) for b in cols}
            if "tree" not in out:   # mktree: plain assignment writes an RNTuple in recent uproot
                out.mktree("tree", {b: v.dtype for b, v in block.items()})
            out["tree"].extend(block)
            buf, nbuf = [], 0

    todo = iter(files)
    inflight = set()
    with ProcessPoolExecutor(a.jobs) as ex, uproot.recreate(output, compression=comp) as out:
        def submit():
            for f in todo:
                inflight.add(ex.submit(read_one, (f, tree, read, keep, cut, derived, arb, a.fraction, a.seed)))
                if len(inflight) >= 4 * a.jobs:
                    return
        submit()
        while inflight:
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                inflight.discard(fut)
                f, arr, (n_read, n_sel), err = fut.result()
                done += 1; nin += n_read; nsel += n_sel
                if err:
                    bad.append((f, err))
                elif arr:
                    k = len(arr[cols[0]])
                    if k:
                        if dtypes is None:
                            dtypes = {b: arr[b].dtype for b in cols}
                        buf.append({b: arr[b].astype(dtypes[b], copy=False) for b in cols})
                        nbuf += k; nout += k
                        if nbuf >= a.flush:
                            flush(out)
            submit()
            bar.update(done, nin, nsel, nout, len(bad))
        flush(out)
    bar.update(done, nin, nsel, nout, len(bad), force=True); bar.close()
    for f, e in bad[:20]:
        print(f"[{sample}] skipped {f}: {e}")
    if len(bad) > 20:
        print(f"[{sample}] ... and {len(bad) - 20} more")
    print(f"[{sample}] done: read {nin:,}, selected {nsel:,}, kept {nout:,} -> {output}"
          + (f"  ({len(bad)} files skipped)" if bad else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="*", help="keys of config.SAMPLES (default: all)")
    ap.add_argument("-c", "--config", default="config.py")
    ap.add_argument("-o", "--outdir", default=None, help="default: config.SKIMDIR")
    ap.add_argument("--suffix", default="_calib", help="output name: <outdir>/<sample><suffix>.root")
    ap.add_argument("--extra", nargs="*", default=[], help="additional branches")
    ap.add_argument("--fraction", type=float, default=1.0, help="random fraction of the selected events, after arbitration (data subsample)")
    ap.add_argument("--seed", type=int, default=20181)
    ap.add_argument("--no-arbitration", action="store_true", help="keep all selected candidates")
    ap.add_argument("-j", "--jobs", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--flush", type=int, default=1_000_000, help="entries per write")
    ap.add_argument("--max-files", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    a.outdir = a.outdir or getattr(cfg, "SKIMDIR", "skims")
    samples = a.samples or list(cfg.SAMPLES)
    unknown = [s for s in samples if s not in cfg.SAMPLES]
    if unknown:
        sys.exit(f"unknown samples {unknown}; config has {list(cfg.SAMPLES)}")
    for s in samples:
        skim(cfg, s, a)


if __name__ == "__main__":
    main()
