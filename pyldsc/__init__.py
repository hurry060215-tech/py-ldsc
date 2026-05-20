"""pyldsc: a modern-Python reimplementation of LDSC (LD Score Regression).

A faithful, cleanly-rewritten port of ``ldsc`` (Bulik-Sullivan et al.,
*Nature Genetics* 2015, "LD Score regression distinguishes confounding from
polygenicity in genome-wide association studies"; and the stratified /
cell-type extension, Finucane et al., *Nature Genetics* 2015 & 2018).

The original ``ldsc`` is a Python-2-era command-line tool.  ``pyldsc`` is a
pure, importable, modern-Python (numpy / scipy / pandas / bitarray)
reimplementation that matches it numerically.

Numerical parity with the original ``ldsc`` is the design priority -- the
block jackknife is exact arithmetic and the weighted regressions are
deterministic, so estimates agree to floating-point tolerance.

Core capabilities
-----------------
* **LD Score estimation** -- windowed, bias-corrected per-SNP LD Scores from
  PLINK ``.bed/.bim/.fam`` reference genotypes (:func:`estimate_ldscore`).
* **Heritability** -- univariate and partitioned SNP-heritability with a
  free or constrained intercept, block-jackknife SEs, the two-step
  estimator and the ratio (:func:`estimate_h2`, :class:`Hsq`).
* **Partitioned / stratified h2** -- per-category h2, enrichment and
  coefficient z-scores with the ``--overlap-annot`` correction
  (:func:`partitioned_h2`).
* **Genetic correlation** -- cross-trait LDSC: genetic covariance and
  genetic correlation ``rg`` with jackknife SEs (:func:`estimate_rg`,
  :class:`Gencov`, :class:`RG`).
* **LDSC-SEG** -- cell-type / tissue-specific heritability enrichment with
  one-sided coefficient P-values (:func:`ldsc_seg`).
* **munge_sumstats** -- harmonisation of raw GWAS summary statistics into
  the ``.sumstats`` format (:func:`munge_sumstats`).

Quick-start
-----------
>>> import pyldsc
>>> # estimate LD Scores from a PLINK reference panel
>>> res = pyldsc.estimate_ldscore("ref", ld_wind_cm=1.0)
>>> # SNP-heritability from GWAS summary statistics
>>> h2 = pyldsc.estimate_h2("trait.sumstats", "ref", "weights")
>>> h2.tot, h2.intercept
>>> # genetic correlation between two traits
>>> rg = pyldsc.estimate_rg(["t1.sumstats", "t2.sumstats"], "ref", "weights")
>>> rg[0].rg_ratio

Building blocks
---------------
* :class:`LstsqJackknifeFast`, :class:`RatioJackknife` -- block jackknives.
* :class:`IRWLS` -- iteratively re-weighted least squares.
* :class:`PlinkBEDFile` -- in-memory PLINK ``.bed`` reader.
"""
from __future__ import annotations

from .api import (
    Logger,
    estimate_h2,
    estimate_ldscore,
    estimate_rg,
    ldsc_seg,
    partitioned_h2,
)
from .irwls import IRWLS
from .jackknife import (
    Jackknife,
    LstsqJackknifeFast,
    LstsqJackknifeSlow,
    RatioJackknife,
)
from .ldscore import PlinkBEDFile, block_left_to_right, getBlockLefts
from .munge import munge_sumstats
from .regressions import (
    RG,
    Gencov,
    Hsq,
    LD_Score_Regression,
    gencov_obs_to_liab,
    h2_obs_to_liab,
    p_z_norm,
)

__version__ = "0.1.0"

__all__ = [
    # high-level workflows
    "estimate_ldscore",
    "estimate_h2",
    "partitioned_h2",
    "estimate_rg",
    "ldsc_seg",
    "munge_sumstats",
    "Logger",
    # regression estimators
    "LD_Score_Regression",
    "Hsq",
    "Gencov",
    "RG",
    "h2_obs_to_liab",
    "gencov_obs_to_liab",
    "p_z_norm",
    # numerical building blocks
    "Jackknife",
    "LstsqJackknifeFast",
    "LstsqJackknifeSlow",
    "RatioJackknife",
    "IRWLS",
    # LD Score estimation
    "PlinkBEDFile",
    "getBlockLefts",
    "block_left_to_right",
]
