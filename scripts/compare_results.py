"""Compare two correction files (or two result.json files) term by term.

    python scripts/compare_results.py REFERENCE.json NEW.json

Prints each fitted parameter, the difference and the difference in units of the reference error.
Exit code 0 if every difference is below 0.01 sigma (a rerun with the same inputs and versions gives exactly 0)."""
import json, sys


def load(path):
    d = json.load(open(path))
    if "fit" in d:                                  # correction file
        f = d["fit"]; return dict(zip(f["names"], f["values"])), dict(zip(f["names"], f["errors"]))
    return {n: d["values"][n] for n in d["correction"]}, {n: d["errors"][n] for n in d["correction"]}


(rv, re), (nv, ne) = load(sys.argv[1]), load(sys.argv[2])
worst = 0.0
print(f"{'parameter':10s} {'reference':>14s} {'new':>14s} {'diff/sigma':>11s} {'err ratio':>10s}")
for n in rv:
    d = (nv[n] - rv[n]) / re[n]; worst = max(worst, abs(d))
    print(f"{n:10s} {rv[n]:14.6e} {nv[n]:14.6e} {d:+11.4f} {ne[n] / re[n]:10.4f}")
print(f"largest difference: {worst:.4f} sigma")
sys.exit(0 if worst < 0.01 else 1)
