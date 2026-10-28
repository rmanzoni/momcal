"""The multi-channel pieces: n-track masses, the J/psi refit carry-over, the kkpi wrapper, and a dimuon closure."""
import numpy as np
from conftest import toy_ds
from scalesmear.tracks import Tracks, MMU, MASS
from scalesmear.channels import TrackMorph, JpsiRfK, m_mumu, MuMuChannel, KKPiChannel, mumu_categories
from scalesmear.model import CmsCorrection
from scalesmear.fit import ScaleSmearFit, make_categories
from scalesmear.multifit import MultiFit

MJ = 3.0969


def _two(rng, M, P, m1, m2):
    q = np.sqrt((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (2 * M)
    c = rng.uniform(-1, 1, len(P)); f = rng.uniform(0, 2 * np.pi, len(P)); s = np.sqrt(1 - c**2)
    d = q[:, None] * np.stack([s * np.cos(f), s * np.sin(f), c], 1)
    E = np.sqrt(M**2 + (P**2).sum(1)); b = P / E[:, None]; g = E / M
    def boost(e, p):
        bp = (b * p).sum(1); b2 = (b * b).sum(1)
        return p + ((g - 1) * bp / b2 + g * e)[:, None] * b
    return boost(np.sqrt(m1**2 + q**2), d), boost(np.sqrt(m2**2 + q**2), -d)


def toy_jpsik(n, seed, res=0.008):
    """B+ -> J/psi(mu mu) K+ with a per-track resolution; the 'refitted' J/psi is the dimuon rescaled to m(J/psi), as in
    the 2018 ntuples."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 4, (n, 3)) + np.array([0, 0, 10.0]) * rng.choice([-1, 1], n)[:, None]
    jp, k = _two(rng, np.full(n, 5.27934), P, MJ, MASS["K"])
    m1, m2 = _two(rng, np.full(n, MJ), jp, MMU, MMU)
    p = np.stack([m1, m2, k], 1) * (1 + res * rng.standard_normal((n, 3)))[..., None]
    q = np.tile([1, -1, 1], (n, 1)); q[n // 2:] *= -1
    e = np.sqrt((p[:, :2] ** 2).sum(-1) + MMU**2); mraw = np.sqrt(e.sum(1) ** 2 - (p[:, :2].sum(1) ** 2).sum(1))
    prf = p[:, :2].sum(1) * (MJ / mraw)[:, None]
    pt = np.hypot(prf[:, 0], prf[:, 1])
    ex = dict(jpsi_rf_pt=pt, jpsi_rf_eta=np.arcsinh(prf[:, 2] / pt), jpsi_rf_phi=np.arctan2(prf[:, 1], prf[:, 0]),
              jpsi_rf_mass=np.full(n, MJ), jpsi_mass=mraw)
    return Tracks(p, q.astype(np.int8), ("mu", "mu", "K"), ex)


def test_masses_and_refit_carry_over():
    t = toy_jpsik(20_000, 1)
    corr = CmsCorrection([0, 1.2, 2.5], terms=("A", "e", "M", "a2", "c2"), shared=("e", "c2"))
    tm = TrackMorph(t, corr); one = np.ones_like(tm.pt)
    assert np.allclose(m_mumu(tm, one), t.extra["jpsi_mass"], atol=1e-9)
    F = JpsiRfK(t, tm)
    e = np.sqrt((t.p ** 2).sum(-1) + np.array([MMU, MMU, MASS["K"]]) ** 2)
    rf = t.p[:, :2].sum(1) * (MJ / t.extra["jpsi_mass"])[:, None]
    ej = np.sqrt((rf ** 2).sum(1) + MJ ** 2)
    m_rf = np.sqrt((ej + e[:, 2]) ** 2 - ((rf + t.p[:, 2]) ** 2).sum(1))
    assert np.allclose(F(one), m_rf, atol=1e-7)
    # a common muon scale is absorbed by the J/psi mass constraint (up to the muon mass: exact only for massless muons);
    # a kaon scale is not
    f = one.copy(); f[:, :2] *= 1.002
    d = F(f) - F(one)
    assert np.abs(np.median(d)) < 3e-5 and np.abs(d).max() < 1e-3
    f = one.copy(); f[:, 2] *= 1.002
    assert np.median(F(f) - F(one)) > 1e-3
    # refit_power 0 is the naive carry-over: then the muon scale does move the mass
    f = one.copy(); f[:, :2] *= 1.002
    assert np.median(JpsiRfK(t, tm, refit_power=0)(f) - F(one)) > 5e-4


def test_kkpi_channel_same_likelihood_as_v3():
    data, mc = toy_ds(30_000, 3), toy_ds(60_000, 4)
    from conftest import masses
    from scalesmear.io import Sample
    m, mkk = masses(data.p); data = Sample(data.p, data.charge, m, mkk)
    corr = CmsCorrection([0, 2.5], terms=("A", "M", "a2"))
    cats = make_categories([0, 2.5], None, True)
    v3 = ScaleSmearFit(data, mc, corr, categories=cats, z_mode="hash")
    mf = MultiFit([KKPiChannel(v3)], corr)
    cv = [1e-3, 1e-5, 9.0]
    nu = {0: (1e4, 2e3, -2.0, 0.5), 1: (1e4, 2e3, -2.5, 0.3)}                 # Ns, Nb, lam, kap per Ds charge
    vals = dict(zip(corr.names, cv))
    vals.update({f"kkpi.{q}_{i}": v for i, t in nu.items() for q, v in zip(("Ns", "Nb", "lam", "kap"), t)})
    v3_layout = cv + [x for i in (0, 1) for x in nu[i]]
    assert np.isclose(mf.nll(*[vals[n] for n in mf.names]), v3.nll(*v3_layout), rtol=1e-10)


def test_mumu_closure():
    """Inject A, M, a into one half of a synthetic dimuon sample; the fit with the other half as template recovers them."""
    t = toy_jpsik(240_000, 7)
    corr = CmsCorrection([0, 2.5], terms=("A", "M", "a2"))
    inj = np.array([1.5e-3, -1.0e-4, 25.0])
    d, mc = t.split(0.4, seed=3)
    tm = TrackMorph(d, corr, seed=999); f = tm.factors(inj)
    d = Tracks(d.p * f[..., None], d.charge, d.kinds, d.extra)
    ch = MuMuChannel(d, mc, corr, mrange=(2.97, 3.23), dpt_edges=(-1e3, -2, 2, 1e3), bkg_share="channel")
    r = MultiFit([ch], corr).fit_full({"A_0": 0.0, "M_0": 0.0, "a2_0": 9.0}, log=lambda *a, **k: None)
    v = r.correction_values(); e = np.array([r.errors[n] for n in r.corr_names])
    assert r.valid
    assert np.all(np.abs(v - inj) < 4 * e), (v, e)


def test_profile_fit_matches_full_fit():
    """The profile fit (Minuit over the correction, L-BFGS over the nuisances) finds the same minimum as the full fit."""
    t = toy_jpsik(160_000, 11)
    corr = CmsCorrection([0, 2.5], terms=("A", "M", "a2"))
    inj = np.array([1.0e-3, 5.0e-5, 16.0])
    d, mc = t.split(0.4, seed=5)
    tm = TrackMorph(d, corr, seed=998); d = Tracks(d.p * tm.factors(inj)[..., None], d.charge, d.kinds, d.extra)
    st = {"A_0": 0.0, "M_0": 0.0, "a2_0": 9.0}; quiet = lambda *a, **k: None
    ch = MuMuChannel(d, mc, corr, mrange=(2.97, 3.23), dpt_edges=(-1e3, -2, 2, 1e3), bkg_share="channel")
    rf = MultiFit([ch], corr).fit_full(st, log=quiet)
    rp = MultiFit([ch], corr).fit_profile(st, log=quiet)
    assert rf.valid and rp.valid
    for n in corr.names:
        assert abs(rf.values[n] - rp.values[n]) < 0.5 * rp.errors[n], n
        assert abs(rf.errors[n] / rp.errors[n] - 1) < 0.15, n


def test_barlow_beeston_lite_beta():
    """The closed-form per-bin factor minimises 2 (beta s + r - n ln(beta s + r)) + (beta - 1)^2 / rv."""
    from scipy.optimize import minimize_scalar
    rng = np.random.default_rng(2)
    s = rng.uniform(0.1, 50, 100); r = rng.uniform(0, 10, 100); n = rng.poisson(1.1 * s + r).astype(float); rv = rng.uniform(1e-3, 0.2, 100)
    b = MultiFit._bb_beta(s, r, n, rv)
    for i in range(100):
        f = lambda x: 2 * (x * s[i] + r[i] - n[i] * np.log(x * s[i] + r[i])) + (x - 1) ** 2 / rv[i]
        x = minimize_scalar(f, bounds=(1e-6, 10), method="bounded", options=dict(xatol=1e-9)).x
        assert abs(b[i] - x) < 1e-5
    assert np.all(MultiFit._bb_beta(s, r, n, np.zeros_like(rv)) == 1.0)       # no MC variance: no freedom


def test_argus_gauss_shape():
    """ARGUS (x) Gauss: normalised, no events far above the endpoint, the edge smeared by sigma."""
    from scalesmear.channels import argus_gauss, MBPLUS, MPIPLUS
    e = np.arange(5.05, 6.0 + 1e-9, 0.002); c = 0.5 * (e[1:] + e[:-1])
    for sig in (0.01, 0.03):
        h = argus_gauss(e, -30.0, sig)
        assert abs(h.sum() - 1) < 1e-12 and np.all(h >= 0)
        assert h[c > MBPLUS - MPIPLUS + 6 * sig].sum() < 1e-6
    assert argus_gauss(e, -30.0, 0.03)[c > MBPLUS - MPIPLUS].sum() > argus_gauss(e, -30.0, 0.01)[c > MBPLUS - MPIPLUS].sum()


def test_species_terms():
    """Muon- and kaon-specific terms act only on their own tracks; with equal values the result equals the shared model."""
    from scalesmear.model import CmsSpeciesCorrection
    class M: pass
    rng = np.random.default_rng(1); m = M()
    m.pt = rng.uniform(1, 20, (50, 3)); m.q = np.sign(rng.normal(size=(50, 3))); m.z = rng.normal(size=(50, 3))
    m.idx = rng.integers(0, 3, (50, 3)); m.is_mu = np.array([[True, True, False]]); m.is_K = np.array([[False, False, True]])
    base = CmsCorrection([0, 0.7, 1.2, 2.5], terms=("A", "e", "M", "a2", "c2"), shared=("e", "c2"))
    sp = CmsSpeciesCorrection([0, 0.7, 1.2, 2.5], terms=("A", "e", "M", "a2", "c2"), shared=("e", "c2"), muon_terms=("a2", "c2"), kaon_terms=("e",))
    assert sp.names[-5:] == ["a2mu_0", "a2mu_1", "a2mu_2", "c2mu_all", "eK_all"]
    v0 = np.array([1e-3, 1e-3, 1e-3, -1e-3, 0, 0, 0, 10, 15, 20, 2.0])
    v1 = np.concatenate([v0, [10, 15, 20, 2.0], [-1e-3]])
    m2 = M(); m2.__dict__ = dict(m.__dict__)
    assert np.allclose(base.factor(m, v0), sp.factor(m2, v1))
    v2 = v1.copy(); v2[-1] = -3e-3; m3 = M(); m3.__dict__ = dict(m.__dict__)
    f1, f2 = sp.factor(m2, v1), sp.factor(m3, v2)
    assert np.allclose(f1[:, :2], f2[:, :2]) and not np.allclose(f1[:, 2], f2[:, 2])
