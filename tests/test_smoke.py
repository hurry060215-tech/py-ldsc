"""Pure-Python smoke tests for pyldsc on tiny synthetic data.

These tests do not require the bundled ``ldsc`` reference data; they verify
the internal consistency of the building blocks (jackknife, IRWLS, the
regression estimators) using analytic identities lifted from the original
``ldsc/test/test_*.py`` suite.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyldsc
from pyldsc import jackknife as jk
from pyldsc import regressions as reg
from pyldsc.ldscore import block_left_to_right, getBlockLefts


# --------------------------------------------------------------------------
# package surface
# --------------------------------------------------------------------------
def test_import_and_all():
    assert pyldsc.__version__ == "0.1.0"
    for name in pyldsc.__all__:
        assert hasattr(pyldsc, name), name


# --------------------------------------------------------------------------
# jackknife
# --------------------------------------------------------------------------
def test_lstsq_jackknife_fast_matches_slow():
    rng = np.random.RandomState(0)
    x = rng.normal(size=(120, 2))
    y = rng.normal(size=(120, 1))
    fast = jk.LstsqJackknifeFast(x, y, n_blocks=10)
    slow = jk.LstsqJackknifeSlow(x, y, n_blocks=10)
    np.testing.assert_allclose(fast.est, slow.est, rtol=1e-9)
    np.testing.assert_allclose(fast.jknife_se, slow.jknife_se, rtol=1e-9)
    np.testing.assert_allclose(
        fast.delete_values, slow.delete_values, rtol=1e-9
    )


def test_jackknife_pseudovalues_roundtrip():
    delete_values = np.array([[1.0], [2.0], [3.0], [4.0]])
    est = np.array([[2.5]])
    pv = jk.Jackknife.delete_values_to_pseudovalues(delete_values, est)
    # mean of pseudovalues == n*est - (n-1)*mean(delete) is the jackknife est
    j_est, _, _, _ = jk.Jackknife.jknife(pv)
    assert j_est.shape == (1, 1)


def test_get_separators_even():
    s = jk.Jackknife.get_separators(100, 10)
    assert s[0] == 0
    assert s[-1] == 100
    assert len(s) == 11


# --------------------------------------------------------------------------
# regressions: analytic identities (from ldsc test_regressions.py)
# --------------------------------------------------------------------------
def test_h2_obs_to_liab():
    # conversion for a balanced study of a 1% phenotype is about 1/2
    x = reg.h2_obs_to_liab(1, 0.5, 0.01)
    np.testing.assert_almost_equal(x, 0.551907298063)
    with pytest.raises(ValueError):
        reg.h2_obs_to_liab(1, 1, 0.5)


def test_gencov_obs_to_liab():
    assert reg.gencov_obs_to_liab(1, None, None, None, None) == 1
    x = reg.gencov_obs_to_liab(1, 0.5, 0.5, 0.01, 0.01)
    np.testing.assert_almost_equal(x, 0.551907298063)


def test_p_z_norm():
    p, z = reg.p_z_norm(10, 1)
    assert z == 10
    np.testing.assert_almost_equal(p * 1e23, 1.523971, decimal=6)
    p, z = reg.p_z_norm(10, 0)
    assert p == 0
    assert np.isinf(z)


def test_hsq_weights_bounds():
    ld = np.ones((4, 1))
    w_ld = np.ones((4, 1))
    N = 9 * np.ones((4, 1))
    M = 7
    # out-of-bounds h2 is clamped to [0, 1]
    np.testing.assert_allclose(
        reg.Hsq.weights(ld, w_ld, N, M, 1),
        reg.Hsq.weights(ld, w_ld, N, M, 2),
    )
    np.testing.assert_allclose(
        reg.Hsq.weights(ld, w_ld, N, M, 0),
        reg.Hsq.weights(ld, w_ld, N, M, -1),
    )


def test_hsq_aggregate():
    chisq = np.ones((10, 1)) * 3 / 2
    ld = np.ones((10, 1)) * 100
    N = np.ones((10, 1)) * 100000
    M = 1e7
    np.testing.assert_almost_equal(reg.Hsq.aggregate(chisq, ld, N, M), 0.5)
    np.testing.assert_almost_equal(
        reg.Hsq.aggregate(chisq, ld, N, M, intercept=1.5), 0
    )


def test_hsq_recovers_simulated_coefficients():
    """Hsq must recover h2 = (0.2, 0.7) exactly from a noiseless design."""
    rng = np.random.RandomState(1)
    hsq1, hsq2 = 0.2, 0.7
    ld = (np.abs(rng.normal(size=800)) + 1).reshape((400, 2))
    N = np.ones((400, 1)) * 1e5
    M = np.ones((1, 2)) * 1e7 / 2.0
    chisq = 1 + 1e5 * (
        ld[:, 0] * hsq1 / M[0, 0] + ld[:, 1] * hsq2 / M[0, 1]
    ).reshape((400, 1))
    w_ld = np.ones_like(chisq)
    for kw in ({"intercept": 1}, {}):
        h = pyldsc.Hsq(chisq, ld, w_ld, N, M, n_blocks=3, **kw)
        np.testing.assert_array_almost_equal(np.ravel(h.cat), [hsq1, hsq2])
        np.testing.assert_almost_equal(h.tot, hsq1 + hsq2)
        d = hsq1 + hsq2
        np.testing.assert_array_almost_equal(
            np.ravel(h.prop), [hsq1 / d, hsq2 / d]
        )
    h_int = pyldsc.Hsq(chisq, ld, w_ld, N, M, n_blocks=3)
    np.testing.assert_almost_equal(h_int.intercept, 1)
    np.testing.assert_almost_equal(h_int.ratio, 0)


def test_rg_self_correlation_is_one():
    """rg of a trait with itself is +/-1 up to noise."""
    rng = np.random.RandomState(2)
    ld = np.abs(rng.normal(size=100).reshape((50, 2))) + 2
    z1 = (np.sum(ld, axis=1) * 10).reshape((50, 1))
    w_ld = rng.normal(size=50).reshape((50, 1))
    N1 = 9 * np.ones((50, 1))
    M = np.array([[700, 222]])
    rg = pyldsc.RG(z1, -z1, ld, w_ld, N1, N1, M, 1.0, 1.0, 0, n_blocks=20)
    assert abs(rg.rg_ratio + 1) < 0.01


# --------------------------------------------------------------------------
# ldscore helpers
# --------------------------------------------------------------------------
def test_get_block_lefts():
    assert np.all(getBlockLefts(np.arange(1, 6), 5) == np.zeros(5))
    assert np.all(getBlockLefts(np.arange(1, 6), 0) == np.arange(0, 5))
    assert np.all(
        getBlockLefts((1, 4, 6, 7, 7, 8), 2) == (0, 1, 1, 2, 2, 2)
    )


def test_block_left_to_right():
    assert np.all(
        block_left_to_right((0, 0, 0, 0, 0)) == (5, 5, 5, 5, 5)
    )
    assert np.all(
        block_left_to_right((0, 1, 2, 3, 4, 5)) == (1, 2, 3, 4, 5, 6)
    )


# --------------------------------------------------------------------------
# munge_sumstats on a tiny synthetic GWAS
# --------------------------------------------------------------------------
def test_munge_sumstats_tiny(tmp_path):
    import pandas as pd

    raw = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "A1": ["A", "C", "G", "T"],
            "A2": ["G", "T", "A", "C"],
            "N": [10000, 10000, 10000, 10000],
            "P": [0.5, 1e-8, 0.2, 0.9],
            "BETA": [0.1, -0.3, 0.05, -0.02],
        }
    )
    fh = tmp_path / "raw.txt"
    raw.to_csv(fh, sep="\t", index=False)
    out = pyldsc.munge_sumstats(str(fh), write=False)
    assert list(out.columns) == ["SNP", "A1", "A2", "N", "Z"]
    assert len(out) == 4
    # sign of Z follows sign of BETA
    assert (np.sign(out.Z) == np.sign(raw.BETA)).all()
