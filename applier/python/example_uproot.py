"""Example: correct the tracks of an MC flat ntuple with uproot and write the corrected quantities to a friend tree.

    python example_uproot.py mc.root tree_name mc_trkcalib.root [--variations eig0_up,eig0_down]

The output tree has one entry per input entry (same order), so it can be used as a ROOT friend:
    t = f.Get("tree_name"); t.AddFriend("trkcalib", "mc_trkcalib.root")

Written columns, per variation tag (nominal has no suffix):
    <trk>_f        momentum factor p'/p for trk in TRACKS (and the muon if its branches exist)
    kkpi_m         m(K K pi) recomputed from the corrected tracks
    kk_m           m(K K)    recomputed
Edit the branch names in BRANCHES to match your ntuple.
"""
import argparse, os, sys
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rds_trkcalib import TrackCalib, track_id_from_kinematics

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "..", "data", "rds_trkcalib_2018BParking_v1.json")
MK, MPI = 0.493677, 0.13957
TRACKS = {"k1": MK, "k2": MK, "pi": MPI}           # prefix -> mass hypothesis; branches <prefix>_px/_py/_pz/_charge
MUON = dict(pt="mu_pt", eta="mu_eta", phi="mu_phi", charge="mu_charge")
EVENT_ID = ("run", "lumi", "event")                 # used for the random numbers if present


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile"); ap.add_argument("tree"); ap.add_argument("outfile")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--variations", default="", help="comma-separated, e.g. eig0_up,eig0_down or 'all'")
    ap.add_argument("--step", default="200 MB", help="uproot chunk size")
    a = ap.parse_args()
    cal = TrackCalib(a.json)
    vars_ = ["nominal"] + ([v for v in cal.variation_names if v != "nominal"] if a.variations == "all"
                           else [v for v in a.variations.split(",") if v])
    tin = uproot.open(a.infile)[a.tree]
    have = set(tin.keys())
    use_mu = all(MUON[k] in have for k in MUON)
    use_evt = all(k in have for k in EVENT_ID)
    if not use_mu:
        print(f"note: muon not corrected, branches {list(MUON.values())} not all present")
    if not use_evt:
        print("note: no run/lumi/event branches: the entry number is used as event key")
    need = [f"{t}_{c}" for t in TRACKS for c in ("px", "py", "pz", "charge")]
    need += list(MUON.values()) if use_mu else []
    need += list(EVENT_ID) if use_evt else []
    out = uproot.recreate(a.outfile); first = True; offset = 0
    for arr in tin.iterate(need, library="np", step_size=a.step):
        n = len(arr[need[0]])
        run, lumi, evt = ((arr["run"], arr["lumi"], arr["event"]) if use_evt
                          else (np.zeros(n, np.int64), np.zeros(n, np.int64), np.arange(offset, offset + n)))
        res = {}
        trk = {}
        for t in TRACKS:
            px, py, pz = (arr[f"{t}_{c}"].astype(float) for c in ("px", "py", "pz"))
            pt = np.hypot(px, py); p = np.hypot(pt, pz); eta = np.arcsinh(pz / pt); phi = np.arctan2(py, px)
            # same track in several candidates -> same z: the key is the event plus the track's own kinematics
            z = cal.gauss(run, lumi, evt, track_id_from_kinematics(pt, eta, phi))
            trk[t] = (px, py, pz, pt, p, eta, arr[f"{t}_charge"], z)
        for v in vars_:
            sfx = "" if v == "nominal" else "_" + v
            f = {t: cal.factor(x[4], x[3], x[5], x[6], x[7], v) for t, x in trk.items()}
            for t in TRACKS:
                res[f"{t}_f{sfx}"] = f[t]
            def p4(t):
                px, py, pz = (c * f[t] for c in trk[t][:3])
                return np.sqrt(px * px + py * py + pz * pz + TRACKS[t] ** 2), px, py, pz
            k1, k2, pi = p4("k1"), p4("k2"), p4("pi")
            inv = lambda *v4: np.sqrt(np.maximum(sum(x[0] for x in v4) ** 2 - sum((sum(x[i] for x in v4)) ** 2 for i in (1, 2, 3)), 0))
            res[f"kkpi_m{sfx}"] = inv(k1, k2, pi)
            res[f"kk_m{sfx}"] = inv(k1, k2)
            if use_mu:
                mpt, meta, mphi, mq = (arr[MUON[k]] for k in ("pt", "eta", "phi", "charge"))
                zm = cal.gauss(run, lumi, evt, track_id_from_kinematics(mpt, meta, mphi))
                res[f"mu_f{sfx}"] = cal.factor(mpt * np.cosh(meta), mpt, meta, mq, zm, v)
        if first:
            out["trkcalib"] = res; first = False
        else:
            out["trkcalib"].extend(res)
        offset += n
        print(f"{offset} entries", flush=True)
    out.close()


if __name__ == "__main__":
    main()
