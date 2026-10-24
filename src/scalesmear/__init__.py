"""scalesmear: fit a per-track momentum scale and extra resolution of MC to data with the Ds -> phi(KK) pi mass peak.

    p' = p (1 + s0 + s1/p + k q pT)(1 + sqrt(v0 1e-6 + v1 1e-8 pT^2) z)

Entry points: the `scalesmear` command (cli.py), pipeline.run(config), ScaleSmearFit (fit.py).
"""
from .io import Sample, load, WORKING_POINTS
from .model import Correction, PtCorrection, PtVarCorrection
from .fit import ScaleSmearFit, FitResult, mc_stat_bootstrap, make_categories
__all__ = ["Sample", "load", "WORKING_POINTS", "Correction", "PtCorrection", "PtVarCorrection", "ScaleSmearFit", "FitResult",
           "mc_stat_bootstrap", "make_categories"]
__version__ = "1.0.0"
