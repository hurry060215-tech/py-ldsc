"""Numerical parity tests against the original ``ldsc`` reference suite.

These tests drive the test data bundled with the upstream ``ldsc`` repository
(``/tmp/ldsc_ref/test/``) through ``pyldsc`` and assert that:

1. The deterministic regressions (h2 / intercept / rg / jackknife SE) on
   the bundled ``simulate_test`` data reproduce stable reference values
   (these are exact-arithmetic results, so they are reproducible to
   floating-point tolerance).
2. The statistical-property checks encoded in the original
   ``ldsc/test/test_sumstats.py`` (``Test_H2_Statistical`` etc.) hold --
   i.e. averaged over many simulated GWAS the estimator is unbiased
   (mean h2 ~= 0.9, mean per-category h2 ~= (0.3, 0.6), intercept ~= 1).
3. The exact-equality identities from the original suite hold
   (``test_twostep_h2``, ``test_h2_M``, ``test_rg_M``, ``test_read_annot``,
   PLINK ``.bed`` reading).

If the upstream test data is not available the whole module is skipped.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import pyldsc
from pyldsc import parse as ps
from pyldsc import ldscore as ldsc_mod

REF = "/tmp/ldsc_ref/test"
SIM = os.path.join(REF, "simulate_test")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(SIM),
    reason="upstream ldsc reference test data not available",
)

ONELD = os.path.join(SIM, "ldscore", "oneld_onefile")
TWOLD = os.path.join(SIM, "ldscore", "twold_onefile")
WLD = os.path.join(SIM, "ldscore", "w")


def _ss(i):
    return os.path.join(SIM, "sumstats", str(i))


# --------------------------------------------------------------------------
# 1. deterministic reference values (exact-arithmetic regressions)
# --------------------------------------------------------------------------
def test_h2_oneld_reference_values():
    """h2 / intercept / ratio on simulate_test/sumstats/1 (oneld)."""
    h = pyldsc.estimate_h2(_ss(1), ONELD, WLD, log=None)
    # reference values produced by pyldsc on the bundled data; the
    # block jackknife is exact arithmetic so these are reproducible.
    np.testing.assert_allclose(h.tot, 0.7366013073, rtol=1e-6)
    np.testing.assert_allclose(h.tot_se, 0.0664862807, rtol=1e-6)
    np.testing.assert_allclose(h.intercept, 2.9081244790, rtol=1e-6)
    np.testing.assert_allclose(h.intercept_se, 0.4117854250, rtol=1e-6)
    np.testing.assert_allclose(h.mean_chisq, 10.641855644, rtol=1e-6)
    np.testing.assert_allclose(h.lambda_gc, 8.653016047, rtol=1e-6)
    np.testing.assert_allclose(h.ratio, 0.1979001293, rtol=1e-6)


def test_partitioned_h2_twold_reference_values():
    """Partitioned h2 on simulate_test/sumstats/1 (twold, 2 categories)."""
    h = pyldsc.estimate_h2(_ss(1), TWOLD, WLD, chisq_max=99999, log=None)
    np.testing.assert_allclose(h.tot, 0.9672111215, rtol=1e-6)
    np.testing.assert_allclose(
        np.ravel(h.cat), [0.3583055613, 0.6089055602], rtol=1e-6
    )
    np.testing.assert_allclose(
        np.ravel(h.coef), [4.5856890915e-06, 7.8320205378e-06], rtol=1e-6
    )


def test_rg_self_reference_values():
    """rg of simulate_test/sumstats/1 with itself."""
    rg = pyldsc.estimate_rg([_ss(1), _ss(1)], ONELD, WLD, log=None)[0]
    np.testing.assert_allclose(rg.rg_ratio, 1.0221663983, rtol=1e-6)
    np.testing.assert_allclose(rg.rg_se, 0.0118002239, rtol=1e-6)
    np.testing.assert_allclose(rg.rg_jknife, 1.0218475010, rtol=1e-6)


# --------------------------------------------------------------------------
# 2. exact-equality identities (from ldsc test_sumstats.Test_Estimate)
# --------------------------------------------------------------------------
def test_h2_M_flag_equality():
    """test_h2_M: passing --M explicitly must not change the estimate."""
    a = pyldsc.estimate_h2(_ss(1), ONELD, WLD, log=None)
    Mval = open(ONELD + ".l2.M_5_50").read().strip()
    b = pyldsc.estimate_h2(_ss(1), ONELD, WLD, M=Mval, log=None)
    np.testing.assert_almost_equal(a.tot, b.tot)
    np.testing.assert_almost_equal(a.tot_se, b.tot_se)


def test_twostep_h2_equality():
    """test_twostep_h2: two-step estimate is stable across cutoffs."""
    x = pyldsc.estimate_h2(_ss(1), ONELD, WLD, chisq_max=9999999,
                           two_step=999, log=None)
    y = pyldsc.estimate_h2(_ss(1), ONELD, WLD, chisq_max=9999,
                           two_step=99999, log=None)
    np.testing.assert_allclose(x.tot, y.tot, atol=1e-5)


def test_rg_M_flag_equality():
    """test_rg_M: passing --M explicitly must not change rg."""
    x = pyldsc.estimate_rg([_ss(1), _ss(1)], ONELD, WLD, log=None)[0]
    Mval = open(ONELD + ".l2.M_5_50").read().strip()
    y = pyldsc.estimate_rg([_ss(1), _ss(1)], ONELD, WLD, M=Mval, log=None)[0]
    np.testing.assert_almost_equal(x.rg_ratio, y.rg_ratio)
    np.testing.assert_almost_equal(x.rg_se, y.rg_se)


# --------------------------------------------------------------------------
# 3. statistical-property checks (ldsc Test_H2_Statistical / Test_RG_*)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_rep", [120])
def test_h2_statistical_unbiased(n_rep):
    """Test_H2_Statistical: mean h2 ~= 0.9, mean cat ~= (0.3, 0.6)."""
    tots, cats, ints = [], [], []
    for i in range(n_rep):
        h = pyldsc.estimate_h2(_ss(i), TWOLD, WLD, chisq_max=99999, log=None)
        tots.append(h.tot)
        cats.append(np.ravel(h.cat))
        ints.append(h.intercept)
    np.testing.assert_allclose(np.nanmean(tots), 0.9, atol=0.05)
    np.testing.assert_allclose(
        np.nanmean(cats, axis=0), [0.3, 0.6], atol=0.05
    )
    np.testing.assert_allclose(np.nanmean(ints), 1.0, atol=0.1)


@pytest.mark.parametrize("n_rep", [120])
def test_h2_noint_statistical_unbiased(n_rep):
    """Test_H2_Statistical with constrained intercept."""
    tots, cats = [], []
    for i in range(n_rep):
        h = pyldsc.estimate_h2(_ss(i), TWOLD, WLD, chisq_max=99999,
                               intercept_h2=1, log=None)
        tots.append(h.tot)
        cats.append(np.ravel(h.cat))
    np.testing.assert_allclose(np.nanmean(tots), 0.9, atol=0.05)
    np.testing.assert_allclose(
        np.nanmean(cats, axis=0), [0.3, 0.6], atol=0.05
    )


# --------------------------------------------------------------------------
# 4. parser parity (ldsc test_sumstats.test_read_annot)
# --------------------------------------------------------------------------
def test_read_annot_overlap_matrix():
    """test_read_annot: overlap matrix from annot_test/test.annot."""
    overlap, M_tot = ps.annot([os.path.join(REF, "annot_test", "test")])
    np.testing.assert_array_equal(
        overlap, [[1, 0, 0], [0, 2, 2], [0, 2, 2]]
    )
    assert M_tot == 3


def test_read_annot_with_frqfile():
    """test_read_annot: frqfile filtering of the overlap matrix."""
    overlap, M_tot = ps.annot(
        [os.path.join(REF, "annot_test", "test")],
        frqfile=os.path.join(REF, "annot_test", "test1"),
    )
    np.testing.assert_array_equal(
        overlap, [[1, 0, 0], [0, 1, 1], [0, 1, 1]]
    )
    assert M_tot == 2


# --------------------------------------------------------------------------
# 5. PLINK .bed reader parity (ldsc test_ldscore.test_bed)
# --------------------------------------------------------------------------
def test_plink_bed_reader():
    """test_bed: 8 SNPs / 5 indivs, 3 monomorphic SNPs dropped."""
    plink = os.path.join(REF, "plink_test", "plink")
    if not os.path.exists(plink + ".bed"):
        pytest.skip("plink_test data missing")
    bim = ps.PlinkBIMFile(plink + ".bim")
    bed = ldsc_mod.PlinkBEDFile(plink + ".bed", 5, bim)
    assert bed.m == 4
    assert bed.n == 5
    assert len(bed.geno) == 64
    np.testing.assert_allclose(bed.freq, [0.6, 0.6, 0.625, 0.625])


def test_ldscore_estimation_runs():
    """LD Score estimation from reference_test PLINK genotypes."""
    plink = os.path.join(REF, "reference_test", "plink")
    if not os.path.exists(plink + ".bed"):
        pytest.skip("reference_test data missing")
    res = pyldsc.estimate_ldscore(
        plink, ld_wind_kb=1000.0, yes_really=True, log=None
    )
    assert res["ldscore"].shape[0] == 10
    assert "L2" in res["ldscore"].columns
    assert res["M"][0] == 10
    # LD Scores are non-negative bias-corrected sums of r^2
    assert (res["ldscore"]["L2"] > 0).all()
