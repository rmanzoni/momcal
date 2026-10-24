"""Write a fit result as the correction file read by the C++ and Python appliers (schema rds_trkcalib/1)."""
from __future__ import annotations
import datetime, json
import numpy as np

SCHEMA = "rds_trkcalib/1"
TERMS = ("s0", "s1", "k", "v0", "v1")
FORMULA = "p' = p * (1 + s0 + s1/p + k*q*pT) * (1 + sqrt(max(v0*1e-6 + v1*1e-8*pT^2, 0)) * z)"
DEFAULT_SEED = 20181


def per_bin(names, values, nb):
    """Expand fitted values to one value per |eta| bin and term. Terms not fitted are 0."""
    v = dict(zip(names, values))
    return {t: [float(v.get(f"{t}_all", v.get(f"{t}_{b}", 0.0))) for b in range(nb)] for t in TERMS}


def eigen_variations(cov):
    """Eigenvectors of the covariance, scaled to 1 sigma: nominal +- shift[i] are uncorrelated variations."""
    cov = np.asarray(cov, float)
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]                   # largest first
    lam, vec = lam[order], vec[:, order]
    lam = np.clip(lam, 0.0, None)
    shifts = (vec * np.sqrt(lam)).T                 # row i = shift vector of variation i
    for i in range(len(shifts)):                    # fix the arbitrary sign: largest component positive
        if shifts[i][np.argmax(np.abs(shifts[i]))] < 0: shifts[i] = -shifts[i]
    return lam, shifts


def write_correction(fit_json, out_path, name, version, dataset="", mc="", selection="", notes=None, seed=DEFAULT_SEED):
    """fit_json: a FitResult.to_json file (keys values, errors, correction, correction_cov, chi2, ndf, config)."""
    d = json.load(open(fit_json))
    names = list(d["correction"])
    vals = np.array([d["values"][n] for n in names]); errs = np.array([d["errors"][n] for n in names])
    cov = np.array(d["correction_cov"], float)
    edges = [float(x) for x in d["config"]["eta_edges"]]; nb = len(edges) - 1
    lam, shifts = eigen_variations(cov)
    from . import __version__
    out = {
        "schema": SCHEMA,
        "name": name,
        "version": version,
        "created": datetime.date.today().isoformat(),
        "producer": f"scalesmear {__version__}",
        "description": "Per-track momentum scale and extra resolution for MC, fitted to the Ds -> phi(KK) pi mass peak in data.",
        "apply_to": "MC only, every charged track used in the analysis (kaons, pion, muon). Data are not corrected.",
        "formula": FORMULA,
        "variables": {
            "p": "track momentum magnitude [GeV]", "pT": "track transverse momentum [GeV]", "q": "track charge (+1 or -1)",
            "eta": "track pseudorapidity; the |eta| bin selects the parameters",
            "z": "unit Gaussian random number, fixed per track (see 'random')"},
        "units": {"s0": "1", "s1": "GeV", "k": "1/GeV", "v0": "(1e-3)^2", "v1": "(1e-4 / GeV)^2"},
        "resolution_in_r": "r0 = 1e-3*sqrt(v0) (constant relative resolution), r1 = 1e-4*sqrt(v1) [1/GeV] (relative resolution per GeV of pT)",
        "eta_bins": {"variable": "abs(eta)", "edges": edges, "outside": "values beyond the last edge use the last bin"},
        "per_bin": per_bin(names, vals, nb),
        "fit": {"names": names, "values": vals.tolist(), "errors": errs.tolist(), "covariance": cov.tolist(),
                "note": "names <term>_<bin> are per |eta| bin, <term>_all are shared by all bins; errors are statistical only"},
        "variations": {
            "method": "eigenvectors of the fit covariance; eigN_up = nominal + shift[N], eigN_down = nominal - shift[N]; "
                      "add the N effects in quadrature",
            "names": [f"eig{i}" for i in range(len(shifts))],
            "eigenvalues": lam.tolist(),
            "shift_per_bin": [per_bin(names, s, nb) for s in shifts]},
        "random": {
            "algorithm": "splitmix64 chain over (seed, run, lumi, event, track_id), then Box-Muller (cos branch)",
            "seed": int(seed),
            "track_id": "any integer that is the same for the same track in all candidates of an event, e.g. the index in the "
                        "input track collection. Without an index use track_id_from_kinematics(pt, eta, phi).",
            "note": "the fit used numpy random numbers; any unit Gaussian fixed per track is statistically equivalent"},
        "derivation": {
            "dataset": dataset, "mc": mc, "selection": selection,
            "categories": d["config"].get("categories"), "mass_range_GeV": d["config"].get("mrange"),
            "bin_width_GeV": d["config"].get("bw"), "background": d["config"].get("comb"),
            "n_data": d["config"].get("n_data"), "n_mc": d["config"].get("n_mc"), "fit_seed": d["config"].get("seed"),
            "chi2": float(np.sum(d["chi2"])), "ndf": int(np.sum(d["ndf"])), "valid": bool(d.get("valid", True))},
        "notes": notes or [],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out
