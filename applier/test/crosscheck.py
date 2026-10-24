"""C++ vs Python cross-check of rds_trkcalib. Run from this directory:  python crosscheck.py
Builds crosscheck.cpp with g++ (or $CXX), runs both on the same 200k tracks, compares every number."""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from rds_trkcalib import TrackCalib, track_id_from_kinematics

JSON = os.path.join(HERE, "..", "data", "rds_trkcalib_2018BParking_v1.json")
cal = TrackCalib(JSON)
rng = np.random.default_rng(7)
n = 200_000
pt = rng.exponential(2.5, n) + 0.3
eta = rng.uniform(-2.7, 2.7, n)
eta[:8] = [0.0, 0.7, -0.7, 1.2, -1.2, 2.5, 2.6, -3.0]       # bin edges and beyond the last edge
phi = rng.uniform(-np.pi, np.pi, n)
p = pt * np.cosh(eta)
q = rng.choice([-1, 1], n)
run = rng.integers(315000, 326000, n); lumi = rng.integers(1, 3000, n)
event = rng.integers(0, 2**40, n); event[:4] = [0, 1, 2**62, -5]   # also extreme / negative values
tid = rng.integers(0, 2000, n)
csv = os.path.join(HERE, "tracks.csv")
with open(csv, "w") as f:
    f.write("run,lumi,event,track_id,p,pt,eta,q,pt_raw,eta_raw,phi_raw\n")
    for row in zip(run, lumi, event, tid, p, pt, eta, q, pt, eta, phi):
        f.write("%d,%d,%d,%d,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g\n" % row)

exe = os.path.join(HERE, "crosscheck")
cxx = os.environ.get("CXX", "g++")
subprocess.run([cxx, "-O2", "-std=c++14", "-Wall", "-I" + os.path.join(HERE, "..", "cpp"), os.path.join(HERE, "crosscheck.cpp"), "-o", exe], check=True)
out = os.path.join(HERE, "cpp_out.csv")
subprocess.run([exe, JSON, csv, out], check=True)

cols = open(out).readline().strip().split(",")
C = np.loadtxt(out, delimiter=",", skiprows=1)
ok = True
tk = track_id_from_kinematics(pt, eta, phi)
if not np.array_equal(C[:, 0].astype(np.int64), tk):
    # loadtxt reads doubles: ids above 2^53 are not exact there, so compare as text instead
    tk_cpp = np.array([int(l.split(",")[0]) for l in open(out).read().splitlines()[1:]])
    ok &= np.array_equal(tk_cpp, tk)
print("track_id_from_kinematics identical:", ok)
z = cal.gauss(run, lumi, event, tid)
dz = np.max(np.abs(C[:, 1] - z))
print(f"z: max |C++ - Python| = {dz:.2e}; mean {z.mean():+.4f}, rms {z.std():.4f} (expect 0, 1)")
ok &= dz < 1e-12
worst = 0.0
for j, v in enumerate(cols[2:], start=2):
    f = cal.factor(p, pt, eta, q, z, variation=v)
    d = np.max(np.abs(C[:, j] - f)); worst = max(worst, d)
print(f"factor: {len(cols) - 2} variations, max |C++ - Python| = {worst:.2e}")
ok &= worst < 1e-12
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
