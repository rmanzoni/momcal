"""Command line.

    scalesmear run     CONFIG [--set key=value ...]      fit data, write result.json, the correction file and plots
    scalesmear toy     CONFIG --seed N [--set ...]       one closure toy (for batch arrays)
    scalesmear toys    CONFIG                            collect the toys: pull means and widths
    scalesmear export  RESULT.json OUT.json --name N --version V      correction file from an existing result
    scalesmear config  [CONFIG]                          print the full configuration with defaults
    scalesmear multi       CONFIG [--set ...]            combined fit of the kkpi, mumu and jpsik channels (v4)
    scalesmear multi-toy   CONFIG --seed N [--set ...]   one closure toy of the combined fit
    scalesmear multi-toys  CONFIG                        collect those toys
"""
import argparse, json, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scalesmear", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def cfgarg(p):
        p.add_argument("config"); p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                                                 help="override a configuration key, e.g. --set fit.seed=7 --set inputs.max_mc=200000")
    cfgarg(sub.add_parser("run", help="fit data"))
    t = sub.add_parser("toy", help="one closure toy"); cfgarg(t); t.add_argument("--seed", type=int, required=True)
    cfgarg(sub.add_parser("toys", help="collect closure toys"))
    e = sub.add_parser("export", help="correction file from a result.json")
    e.add_argument("result"); e.add_argument("out"); e.add_argument("--name", required=True); e.add_argument("--version", required=True)
    e.add_argument("--seed", type=int, default=20181)
    c = sub.add_parser("config", help="print the configuration with all defaults"); c.add_argument("config", nargs="?")
    c.add_argument("--set", action="append", default=[])
    cfgarg(sub.add_parser("multi", help="combined fit of several channels"))
    t = sub.add_parser("multi-toy", help="one closure toy of the combined fit"); cfgarg(t); t.add_argument("--seed", type=int, required=True)
    cfgarg(sub.add_parser("multi-toys", help="collect the combined-fit toys"))
    a = ap.parse_args(argv)

    if a.cmd.startswith("multi"):
        from . import multi
        cfg = multi.load_multi_config(a.config, a.set)
        if a.cmd == "multi":
            res = multi.run(cfg); sys.exit(0 if res.valid else 2)
        if a.cmd == "multi-toy":
            multi.run_toy(cfg, a.seed); return
        multi.collect_toys(cfg); return

    from .config import load_config, DEFAULTS
    if a.cmd == "config":
        import yaml
        if a.config:
            cfg = load_config(a.config, a.set)
        else:
            cfg = DEFAULTS
        yaml.safe_dump(cfg, sys.stdout, sort_keys=False); return
    if a.cmd == "export":
        from .export import write_correction
        write_correction(a.result, a.out, a.name, a.version, seed=a.seed); print("written", a.out); return
    cfg = load_config(a.config, a.set)
    from . import pipeline
    if a.cmd == "run":
        res = pipeline.run(cfg); sys.exit(0 if res.valid else 2)
    if a.cmd == "toy":
        pipeline.run_toy(cfg, a.seed)
    if a.cmd == "toys":
        pipeline.collect_toys(cfg)


if __name__ == "__main__":
    main()
