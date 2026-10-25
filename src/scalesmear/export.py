"""Write a fit result as the correction file read by the C++ and Python appliers.

schema rds_trkcalib/2  (convention 'cms', default):
    pT' = pT / [ (1 + A - e/pT + q*M*pT) * (1 + sqrt(max(a2 + c2*pT^2, 0)) * z) ]     (k' = k (1 + A - e k + q M/k)(1 + sigma z))
schema rds_trkcalib/1  (convention 'v1'):
    p'  = p * (1 + s0 + s1/p + k*q*pT) * (1 + sqrt(max(v0*1e-6 + v1*1e-8*pT^2, 0)) * z)
The direction is unchanged in both: p' / p = pT' / pT.
"""
from __future__ import annotations
import datetime, json
import numpy as np

DEFAULT_SEED = 20181
CONVENTIONS = {
    "cms": dict(
        schema="rds_trkcalib/2", terms=("A", "e", "M", "a2", "c2"),
        # fit variables a2 = (a/1e-3)^2, c2 = (c/1e-4 GeV^-1)^2  ->  file: a2 = a^2, c2 = c^2 [GeV^-2]
        to_file={"A": 1.0, "e": 1.0, "M": 1.0, "a2": 1e-6, "c2": 1e-8},
        formula="pT' = pT / [ (1 + A - e/pT + q*M*pT) * (1 + sqrt(max(a2 + c2*pT^2, 0)) * z) ]; "
                "equivalently k' = k (1 + A - e*k + q*M/k)(1 + sqrt(a^2 + c^2/k^2) z) with k = 1/pT; direction unchanged",
        units={"A": "1", "e": "GeV", "M": "1/GeV", "a2": "1 (a^2)", "c2": "GeV^-2 (c^2)"},
        meaning={"A": "magnetic-field scale (CMS A)", "e": "energy loss (CMS epsilon)", "M": "sagitta / misalignment (CMS M)",
                 "a2": "a^2: extra curvature resolution, multiple scattering (CMS a)",
                 "c2": "c^2: extra curvature resolution, hit resolution (CMS c)"},
        reference="CMS W-mass measurement, arXiv:2412.13872, Eqs. 2-3 (A, epsilon, M; a, c). (A, e, M) give data relative to MC; "
                  "(a, c) the resolution MC is missing. CMS's b, d term is not used."),
    "v1": dict(
        schema="rds_trkcalib/1", terms=("s0", "s1", "k", "v0", "v1"),
        to_file={"s0": 1.0, "s1": 1.0, "k": 1.0, "v0": 1.0, "v1": 1.0},
        formula="p' = p * (1 + s0 + s1/p + k*q*pT) * (1 + sqrt(max(v0*1e-6 + v1*1e-8*pT^2, 0)) * z); direction unchanged",
        units={"s0": "1", "s1": "GeV", "k": "1/GeV", "v0": "(1e-3)^2", "v1": "(1e-4 / GeV)^2"},
        meaning={"s0": "scale", "s1": "energy loss (absolute, in p)", "k": "charge-dependent curvature bias",
                 "v0": "(r0/1e-3)^2, constant extra resolution", "v1": "(r1/1e-4 GeV^-1)^2, extra resolution growing with pT"},
        reference="first convention of this analysis; superseded by 'cms'"),
}


def convention_of(names):
    return "cms" if any(n.split("_")[0] in ("A", "M", "a2", "c2") for n in names) else "v1"


def per_bin(names, values, nb, conv):
    """Expand fitted values to one value per |eta| bin and term, in the units of the file. Terms not fitted are 0."""
    c = CONVENTIONS[conv]; v = dict(zip(names, values))
    return {t: [float(c["to_file"][t] * v.get(f"{t}_all", v.get(f"{t}_{b}", 0.0))) for b in range(nb)] for t in c["terms"]}


def eigen_variations(cov):
    """Eigenvectors of the covariance, scaled to 1 sigma: nominal +- shift[i] are uncorrelated variations."""
    cov = np.asarray(cov, float)
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vec = np.clip(lam[order], 0.0, None), vec[:, order]
    shifts = (vec * np.sqrt(lam)).T
    for i in range(len(shifts)):                    # fix the arbitrary sign: largest component positive
        if shifts[i][np.argmax(np.abs(shifts[i]))] < 0: shifts[i] = -shifts[i]
    return lam, shifts


def write_correction(fit_json, out_path, name, version, dataset="", mc="", selection="", notes=None, seed=DEFAULT_SEED):
    """fit_json: a FitResult.to_json file (keys values, errors, correction, correction_cov, chi2, ndf, config)."""
    d = json.load(open(fit_json))
    names = list(d["correction"])
    conv = d["config"].get("convention") or convention_of(names); C = CONVENTIONS[conv]
    vals = np.array([d["values"][n] for n in names]); errs = np.array([d["errors"][n] for n in names])
    cov = np.array(d["correction_cov"], float)
    edges = [float(x) for x in d["config"]["eta_edges"]]; nb = len(edges) - 1
    lam, shifts = eigen_variations(cov)
    pb = per_bin(names, vals, nb, conv)
    from . import __version__
    out = {
        "schema": C["schema"],
        "convention": conv,
        "name": name,
        "version": version,
        "created": datetime.date.today().isoformat(),
        "producer": f"scalesmear {__version__}",
        "description": "Per-track momentum scale and extra resolution for MC, fitted to the Ds -> phi(KK) pi mass peak in data.",
        "apply_to": "MC only, every charged track used in the analysis (kaons, pion, muon). Data are not corrected.",
        "formula": C["formula"],
        "reference": C["reference"],
        "variables": {
            "pT": "track transverse momentum [GeV]", "p": "track momentum [GeV]", "q": "track charge (+1 or -1)",
            "eta": "track pseudorapidity; the |eta| bin selects the parameters",
            "z": "unit Gaussian random number, fixed per track (see 'random')"},
        "terms": {t: dict(unit=C["units"][t], meaning=C["meaning"][t]) for t in C["terms"]},
        "eta_bins": {"variable": "abs(eta)", "edges": edges, "outside": "values beyond the last edge use the last bin"},
        "per_bin": pb,
        "fit": {"names": names, "values": vals.tolist(), "errors": errs.tolist(), "covariance": cov.tolist(),
                "note": "fit variables: <term>_<bin> per |eta| bin, <term>_all shared by all bins; statistical errors only. "
                        + ("a2, c2 in the fit are (a/1e-3)^2 and (c/1e-4 GeV^-1)^2; per_bin holds a^2 and c^2." if conv == "cms" else "")},
        "variations": {
            "method": "eigenvectors of the fit covariance; eigN_up = nominal + shift[N], eigN_down = nominal - shift[N]; "
                      "add the N effects in quadrature",
            "names": [f"eig{i}" for i in range(len(shifts))],
            "eigenvalues": lam.tolist(),
            "shift_per_bin": [per_bin(names, s, nb, conv) for s in shifts]},
        "random": {
            "algorithm": "splitmix64 chain over (seed, run, lumi, event, track_id), then Box-Muller (cos branch)",
            "seed": int(seed),
            "track_id": "any integer that is the same for the same track in all candidates of an event, e.g. the index in the "
                        "input track collection. Without an index use track_id_from_kinematics(pt, eta, phi).",
            "note": "the fit's own template random numbers are statistically equivalent; see the fit config z_mode"},
        "derivation": {
            "dataset": dataset, "mc": mc, "selection": selection,
            "categories": d["config"].get("categories"), "mass_range_GeV": d["config"].get("mrange"),
            "bin_width_GeV": d["config"].get("bw"), "background": d["config"].get("comb"),
            "n_data": d["config"].get("n_data"), "n_mc": d["config"].get("n_mc"), "fit_seed": d["config"].get("seed"),
            "z_mode": d["config"].get("z_mode", "sequential"),
            "chi2": float(np.sum(d["chi2"])), "ndf": int(np.sum(d["ndf"])), "valid": bool(d.get("valid", True))},
        "notes": notes or [],
    }
    if conv == "cms":
        out["per_bin_readable"] = {"a": [float(np.sqrt(max(x, 0))) for x in pb["a2"]],
                                   "c [1/GeV]": [float(np.sqrt(max(x, 0))) for x in pb["c2"]]}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out
