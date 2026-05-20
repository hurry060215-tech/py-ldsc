"""High-level, importable LDSC workflows.

This module re-implements the data-marshalling logic of
``ldsc/ldscore/sumstats.py`` and ``ldsc/ldsc.py`` as plain functions, so the
full LDSC pipeline can be driven from Python without the legacy CLI.

Public functions
----------------
* :func:`estimate_ldscore` -- per-SNP LD Scores from PLINK genotypes.
* :func:`estimate_h2`      -- univariate / partitioned SNP-heritability.
* :func:`partitioned_h2`   -- partitioned heritability (alias with overlap).
* :func:`estimate_rg`      -- genetic correlation between traits.
* :func:`ldsc_seg`         -- LDSC-SEG cell-type-specific analysis.
"""
from __future__ import annotations

import itertools as it
import os

import numpy as np
import pandas as pd
from scipy import stats

from . import ldscore as ld
from . import parse as ps
from . import regressions as reg

__all__ = [
    "estimate_ldscore",
    "estimate_h2",
    "partitioned_h2",
    "estimate_rg",
    "ldsc_seg",
    "Logger",
]

_N_CHR = 22

# --- allele bookkeeping (ported from ldsc.sumstats) ----------------------
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
BASES = list(COMPLEMENT.keys())
STRAND_AMBIGUOUS = {
    "".join(x): x[0] == COMPLEMENT[x[1]]
    for x in it.product(BASES, BASES)
    if x[0] != x[1]
}
VALID_SNPS = {
    "".join(y)
    for y in it.product(BASES, BASES)
    if y[0] != y[1] and not STRAND_AMBIGUOUS["".join(y)]
}
MATCH_ALLELES = {
    x
    for x in ("".join(y) for y in it.product(VALID_SNPS, VALID_SNPS))
    if ((x[0] == x[2]) and (x[1] == x[3]))
    or ((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]]))
    or ((x[0] == x[3]) and (x[1] == x[2]))
    or ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))
}
FLIP_ALLELES = {
    "".join(x): (
        ((x[0] == x[3]) and (x[1] == x[2]))
        or ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))
    )
    for x in MATCH_ALLELES
}


class Logger(object):
    """Minimal logger: writes to a file (optional) and optionally to stdout."""

    def __init__(self, fh=None, verbose=True):
        self.log_fh = open(fh, "w") if fh else None
        self.verbose = verbose
        self.lines = []

    def log(self, msg):
        msg = str(msg)
        self.lines.append(msg)
        if self.log_fh:
            self.log_fh.write(msg + "\n")
        if self.verbose:
            print(msg)

    def close(self):
        if self.log_fh:
            self.log_fh.close()


class _NullLog(object):
    """A logger that swallows everything (used when ``log=None``)."""

    def log(self, msg):  # noqa: D102
        pass


def _coerce_log(log):
    return log if log is not None else _NullLog()


# --------------------------------------------------------------------------
# LD Score estimation
# --------------------------------------------------------------------------
def estimate_ldscore(bfile, ld_wind_cm=None, ld_wind_kb=None,
                     ld_wind_snps=None, annot=None, maf=None, chunk_size=50,
                     out=None, keep_snps=None, keep_indivs=None,
                     yes_really=False, log=None):
    """Estimate per-SNP LD Scores from PLINK ``.bed/.bim/.fam`` genotypes.

    Parameters
    ----------
    bfile : str
        PLINK fileset prefix (expects ``bfile.bed``, ``.bim``, ``.fam``).
    ld_wind_cm, ld_wind_kb, ld_wind_snps : float, optional
        Window width.  Exactly one must be given.
    annot : np.ndarray, shape (n_snp, n_annot), optional
        SNP annotation matrix for partitioned LD Scores.  If ``None``, a
        single all-ones annotation is used (standard univariate LD Scores).
    maf : float, optional
        Minimum minor-allele frequency filter.
    chunk_size : int, optional
        Genotype chunk size (``c`` in the windowed algorithm).
    out : str, optional
        If given, write ``out.l2.ldscore.gz``, ``out.l2.M`` and
        ``out.l2.M_5_50``.

    Returns
    -------
    dict
        ``{'ldscore': DataFrame, 'M': ndarray, 'M_5_50': ndarray,
        'colnames': list}``.
    """
    log = _coerce_log(log)
    array_snps = ps.PlinkBIMFile(bfile + ".bim")
    array_indivs = ps.PlinkFAMFile(bfile + ".fam")
    n = len(array_indivs.IDList)
    log.log("Read {m} SNPs and {n} individuals.".format(
        m=len(array_snps.IDList), n=n))

    geno_array = ld.PlinkBEDFile(
        bfile + ".bed", n, array_snps, keep_snps=keep_snps,
        keep_indivs=keep_indivs, mafMin=maf,
    )
    annot_matrix = annot
    if annot_matrix is not None:
        annot_matrix = np.asarray(annot_matrix, dtype=float)
        annot_matrix = annot_matrix[geno_array.kept_snps, :]

    x = np.array((ld_wind_snps, ld_wind_kb, ld_wind_cm), dtype=bool)
    if np.sum(x) != 1:
        raise ValueError("Must specify exactly one --ld-wind option")
    if ld_wind_snps:
        max_dist = ld_wind_snps
        coords = np.array(range(geno_array.m))
    elif ld_wind_kb:
        max_dist = ld_wind_kb * 1000
        coords = np.array(array_snps.df["BP"])[geno_array.kept_snps]
    else:
        max_dist = ld_wind_cm
        coords = np.array(array_snps.df["CM"])[geno_array.kept_snps]

    block_left = ld.getBlockLefts(coords, max_dist)
    if block_left[len(block_left) - 1] == 0 and not yes_really:
        raise ValueError(
            "Whole-chromosome LD Score requested; pass yes_really=True if "
            "this is intentional."
        )

    log.log("Estimating LD Score.")
    lN = geno_array.ldScoreVarBlocks(block_left, chunk_size, annot=annot_matrix)

    if annot_matrix is None:
        ldscore_colnames = ["L2"]
        M = np.array([geno_array.m])
        M_5_50 = np.array([np.sum(geno_array.maf > 0.05)])
    else:
        n_annot = annot_matrix.shape[1]
        ldscore_colnames = ["ANNOT{}_L2".format(i) for i in range(n_annot)]
        M = np.atleast_1d(np.squeeze(np.asarray(np.sum(annot_matrix, axis=0))))
        ii = geno_array.maf > 0.05
        M_5_50 = np.atleast_1d(
            np.squeeze(np.asarray(np.sum(annot_matrix[ii, :], axis=0)))
        )

    df = pd.DataFrame(geno_array.df[:, :4], columns=["CHR", "SNP", "BP", "CM"])
    for i, cname in enumerate(ldscore_colnames):
        df[cname] = lN[:, i]
    out_df = df.drop(["CM"], axis=1)

    if out is not None:
        out_df.to_csv(out + ".l2.ldscore.gz", sep="\t", index=False,
                      float_format="%.3f", compression="gzip")
        with open(out + ".l2.M", "w") as fh:
            fh.write("\t".join(str(x) for x in M) + "\n")
        with open(out + ".l2.M_5_50", "w") as fh:
            fh.write("\t".join(str(x) for x in M_5_50) + "\n")
        log.log("Wrote LD Scores to {O}.l2.ldscore.gz".format(O=out))

    return {
        "ldscore": out_df,
        "M": np.atleast_1d(M),
        "M_5_50": np.atleast_1d(M_5_50),
        "colnames": ldscore_colnames,
    }


# --------------------------------------------------------------------------
# shared helpers for h2 / rg estimation
# --------------------------------------------------------------------------
def _splitp(fstr):
    flist = fstr.split(",")
    return [os.path.expanduser(os.path.expandvars(x)) for x in flist]


def _smart_merge(x, y):
    """Fast concat when SNP columns match; otherwise an inner merge."""
    if (
        len(x) == len(y)
        and (np.asarray(x.index) == np.asarray(y.index)).all()
        and (np.asarray(x.SNP) == np.asarray(y.SNP)).all()
    ):
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True).drop("SNP", axis=1)
        return pd.concat([x, y], axis=1)
    return pd.merge(x, y, how="inner", on="SNP")


def _check_variance(M_annot, ref_ld):
    """Drop zero-variance LD Score columns."""
    ii = ref_ld.iloc[:, 1:].var() == 0
    if ii.all():
        raise ValueError("All LD Scores have zero variance.")
    ii_snp = np.array([True] + list(~ii))
    ii_m = np.array(~ii)
    ref_ld = ref_ld.loc[:, ii_snp]
    M_annot = M_annot[:, ii_m]
    return M_annot, ref_ld, ii


def _read_ld_sumstats(sumstats_fh, ref_ld, w_ld, M=None, not_M_5_50=False,
                      alleles=False, dropna=True, log=None):
    """Read sumstats + ref LD + weight LD and merge them (ldsc parity)."""
    log = _coerce_log(log)
    ss = ps.sumstats(sumstats_fh, alleles=alleles, dropna=dropna)
    ss = ss.drop_duplicates(subset="SNP")
    log.log("Read summary statistics for {N} SNPs.".format(N=len(ss)))

    ref_paths = _splitp(ref_ld)
    ref_ld_df = ps.ldscore_fromlist(ref_paths)
    n_annot = len(ref_ld_df.columns) - 1

    if M is not None:
        try:
            M_annot = np.array(
                [float(x) for x in _splitp(M)]
            ).reshape((1, -1))
        except ValueError as e:
            raise ValueError("Could not cast M to float: " + str(e.args))
    else:
        M_annot = ps.M_fromlist(ref_paths, common=(not not_M_5_50))
    try:
        M_annot = np.array(M_annot).reshape((1, n_annot))
    except ValueError as e:
        raise ValueError(
            "# terms in M must match # of LD Scores in ref_ld.\n" + str(e.args)
        )

    M_annot, ref_ld_df, novar_cols = _check_variance(M_annot, ref_ld_df)

    w_paths = _splitp(w_ld)
    if len(w_paths) > 1:
        raise ValueError("w_ld must point to a single fileset.")
    w_ld_df = ps.ldscore_fromlist(w_paths)
    if len(w_ld_df.columns) != 2:
        raise ValueError("w_ld may only have one LD Score column.")
    w_ld_df.columns = ["SNP", "LD_weights"]

    merged = _smart_merge(ref_ld_df, ss)
    merged = _smart_merge(merged, w_ld_df)
    if len(merged) == 0:
        raise ValueError("No SNPs remain after merging.")
    log.log("After merging, {N} SNPs remain.".format(N=len(merged)))

    w_ld_cname = merged.columns[-1]
    ref_ld_cnames = list(ref_ld_df.columns[1:])
    return M_annot, w_ld_cname, ref_ld_cnames, merged, novar_cols


# --------------------------------------------------------------------------
# heritability
# --------------------------------------------------------------------------
def estimate_h2(sumstats, ref_ld, w_ld, M=None, intercept_h2=None,
                no_intercept=False, n_blocks=200, chisq_max=None,
                two_step=None, not_M_5_50=False, samp_prev=None,
                pop_prev=None, overlap_annot=False, frqfile=None,
                print_coefficients=False, out=None, log=None):
    """Estimate SNP-heritability (and partitioned h2) from GWAS sumstats.

    Parameters
    ----------
    sumstats : str
        Path to a ``.sumstats(.gz)`` file.
    ref_ld : str
        Reference-panel LD Score fileset prefix(es), comma-separated.
    w_ld : str
        Regression-weight LD Score fileset prefix.
    M : str, optional
        Override the number of SNPs (comma-separated).
    intercept_h2 : float, optional
        Constrain the LD Score regression intercept.
    no_intercept : bool, optional
        Equivalent to ``intercept_h2=1``.
    n_blocks : int, optional
        Number of block-jackknife blocks (default 200).
    chisq_max : float, optional
        Drop SNPs with chi^2 above this threshold.
    two_step : float, optional
        Two-step estimator cutoff (default 30 for single-annotation,
        free-intercept regressions).
    overlap_annot : bool, optional
        Apply the ``--overlap-annot`` correction for overlapping categories.
    frqfile : str, optional
        Allele-frequency fileset prefix used with ``overlap_annot``.

    Returns
    -------
    :class:`pyldsc.regressions.Hsq`
        The fitted heritability object.  Also carries ``.ref_ld_cnames`` and,
        if ``overlap_annot``, ``.overlap_results`` (a DataFrame).
    """
    log = _coerce_log(log)
    if no_intercept:
        intercept_h2 = 1
    elif intercept_h2 is not None:
        intercept_h2 = float(intercept_h2)

    M_annot, w_ld_cname, ref_ld_cnames, merged, novar_cols = _read_ld_sumstats(
        sumstats, ref_ld, w_ld, M=M, not_M_5_50=not_M_5_50, log=log
    )
    ref_ld_arr = np.array(merged[ref_ld_cnames])
    n_snp = len(merged)
    n_blocks = min(n_snp, n_blocks)
    n_annot = len(ref_ld_cnames)

    old_weights = False
    if n_annot == 1:
        if two_step is None and intercept_h2 is None:
            two_step = 30
    else:
        old_weights = True
        if chisq_max is None:
            chisq_max = max(0.001 * merged.N.max(), 80)

    def col(x):
        return np.array(x).reshape((n_snp, 1))

    chisq = col(merged.Z ** 2)
    if chisq_max is not None:
        ii = np.ravel(chisq < chisq_max)
        merged = merged[ii]
        log.log(
            "Removed {M} SNPs with chi^2 > {C} ({N} SNPs remain).".format(
                C=chisq_max, N=np.sum(ii), M=n_snp - np.sum(ii)
            )
        )
        n_snp = int(np.sum(ii))
        ref_ld_arr = np.array(merged[ref_ld_cnames])
        chisq = chisq[ii].reshape((n_snp, 1))

    def col2(x):
        return np.array(x).reshape((n_snp, 1))

    hsqhat = reg.Hsq(
        chisq, ref_ld_arr, col2(merged[w_ld_cname]), col2(merged.N), M_annot,
        n_blocks=n_blocks, intercept=intercept_h2, twostep=two_step,
        old_weights=old_weights,
    )
    hsqhat.ref_ld_cnames = ref_ld_cnames
    log.log(hsqhat.summary(ref_ld_cnames, P=samp_prev, K=pop_prev,
                           overlap=overlap_annot))

    if overlap_annot:
        ref_paths = _splitp(ref_ld)
        overlap_matrix, M_tot = ps.annot(ref_paths, frqfile=frqfile)
        df_results = hsqhat._overlap_output(
            ref_ld_cnames, overlap_matrix, M_annot, M_tot, print_coefficients
        )
        hsqhat.overlap_results = df_results
        if out is not None:
            df_results.to_csv(out + ".results", sep="\t", index=False)
    return hsqhat


def partitioned_h2(sumstats, ref_ld, w_ld, frqfile=None, overlap_annot=True,
                   M=None, intercept_h2=None, no_intercept=False,
                   n_blocks=200, chisq_max=None, not_M_5_50=False,
                   print_coefficients=True, out=None, log=None):
    """Partitioned heritability with the ``--overlap-annot`` correction.

    Thin wrapper around :func:`estimate_h2` defaulting ``overlap_annot=True``
    (the standard stratified-LDSC workflow).
    """
    return estimate_h2(
        sumstats, ref_ld, w_ld, M=M, intercept_h2=intercept_h2,
        no_intercept=no_intercept, n_blocks=n_blocks, chisq_max=chisq_max,
        two_step=None, not_M_5_50=not_M_5_50, overlap_annot=overlap_annot,
        frqfile=frqfile, print_coefficients=print_coefficients, out=out,
        log=log,
    )


# --------------------------------------------------------------------------
# genetic correlation
# --------------------------------------------------------------------------
def _filter_alleles(alleles):
    return alleles.apply(lambda y: y in MATCH_ALLELES)


def _align_alleles(z, alleles):
    """Align trait-2 Z-scores to trait-1's ref allele (handles strand flip)."""
    try:
        z = z * (-1) ** alleles.apply(lambda y: FLIP_ALLELES[y])
    except KeyError as e:
        raise KeyError(
            "Incompatible alleles in .sumstats files: %s. "
            "Did you forget to use merge_alleles with munge_sumstats?"
            % str(e.args)
        )
    return z


def estimate_rg(sumstats_list, ref_ld, w_ld, M=None, intercept_h2=None,
                intercept_gencov=None, no_intercept=False, n_blocks=200,
                chisq_max=None, two_step=None, not_M_5_50=False,
                no_check_alleles=False, samp_prev=None, pop_prev=None,
                log=None):
    """Estimate genetic correlation between trait 1 and a list of traits.

    Parameters
    ----------
    sumstats_list : list of str or comma-separated str
        Two or more ``.sumstats`` paths.  The first is trait 1.
    ref_ld, w_ld : str
        LD Score filesets (as for :func:`estimate_h2`).
    intercept_h2 : list of float or None, optional
        Per-trait constrained h2 intercepts.
    intercept_gencov : list of float or None, optional
        Per-trait constrained genetic-covariance intercepts.
    no_intercept : bool, optional
        Constrain all intercepts (h2 = 1, gencov = 0).

    Returns
    -------
    list of :class:`pyldsc.regressions.RG`
        One genetic-correlation result per trait-2.
    """
    log = _coerce_log(log)
    if isinstance(sumstats_list, str):
        rg_paths = _splitp(sumstats_list)
    else:
        rg_paths = list(sumstats_list)
    n_pheno = len(rg_paths)
    if n_pheno < 2:
        raise ValueError("Must specify at least two phenotypes for rg.")

    def _norm(x):
        if x is None:
            return [None] * n_pheno
        if isinstance(x, (list, tuple)):
            return list(x)
        return [float(v) for v in str(x).split(",")]

    intercept_h2 = _norm(intercept_h2)
    intercept_gencov = _norm(intercept_gencov)
    if no_intercept:
        intercept_h2 = [1] * n_pheno
        intercept_gencov = [0] * n_pheno

    M_annot, w_ld_cname, ref_ld_cnames, sumstats, _ = _read_ld_sumstats(
        rg_paths[0], ref_ld, w_ld, M=M, not_M_5_50=not_M_5_50, alleles=True,
        dropna=True, log=log,
    )
    n_annot = M_annot.shape[1]
    if n_annot == 1 and two_step is None and all(
        i is None for i in intercept_h2
    ):
        two_step = 30

    out = []
    for i, p2 in enumerate(rg_paths[1:n_pheno]):
        loop = ps.sumstats(p2, alleles=True, dropna=False)
        loop = loop.drop_duplicates(subset="SNP")
        s1 = sumstats.rename(columns={"N": "N1", "Z": "Z1"})
        s2 = loop.rename(
            columns={"A1": "A1x", "A2": "A2x", "N": "N2", "Z": "Z2"}
        )
        merged = _smart_merge(s1, s2)
        merged = merged.dropna(how="any")
        alleles = merged.A1 + merged.A2 + merged.A1x + merged.A2x
        if not no_check_alleles:
            keep = _filter_alleles(alleles)
            merged = merged[keep].reset_index(drop=True)
            alleles = alleles[keep].reset_index(drop=True)
            merged["Z2"] = _align_alleles(merged.Z2, alleles)
        merged = merged.drop(["A1", "A1x", "A2", "A2x"], axis=1)

        n_snp = len(merged)
        if chisq_max is not None:
            ii = merged.Z1 ** 2 * merged.Z2 ** 2 < chisq_max ** 2
            merged = merged[ii]
            n_snp = int(np.sum(ii))
        nb = min(n_blocks, n_snp)

        def col(x, nn=n_snp):
            return np.array(x).reshape((nn, 1))

        ref_ld_arr = np.asarray(merged[ref_ld_cnames])
        rghat = reg.RG(
            col(merged.Z1), col(merged.Z2), ref_ld_arr,
            col(merged[w_ld_cname]), col(merged.N1), col(merged.N2), M_annot,
            intercept_hsq1=intercept_h2[0], intercept_hsq2=intercept_h2[i + 1],
            intercept_gencov=intercept_gencov[i + 1], n_blocks=nb,
            twostep=two_step,
        )
        out.append(rghat)
        log.log("rg for {P} = {R}".format(P=p2, R=rghat.rg_ratio))
    return out


# --------------------------------------------------------------------------
# LDSC-SEG (cell-type-specific analysis)
# --------------------------------------------------------------------------
def ldsc_seg(sumstats, ref_ld_cts, ref_ld, w_ld, M=None, intercept_h2=None,
             no_intercept=False, n_blocks=200, chisq_max=None,
             not_M_5_50=False, out=None, log=None):
    """LDSC-SEG cell-type / tissue-specific heritability enrichment.

    Reproduces ``ldsc.py --h2-cts``.  For each cell-type LD Score set, fits a
    heritability regression with the cell-type annotation on top of the
    baseline model and reports the cell-type coefficient with a one-sided
    P-value.

    Parameters
    ----------
    sumstats : str
        ``.sumstats(.gz)`` path.
    ref_ld_cts : str or list of (name, ld_prefix) tuples
        Either a path to a ``--ref-ld-chr-cts`` file (whitespace-delimited,
        two columns: name and comma-separated LD Score prefixes) or a list of
        ``(name, ld_prefix)`` tuples.
    ref_ld : str
        Baseline-model LD Score fileset prefix(es).
    w_ld : str
        Regression-weight LD Score fileset prefix.

    Returns
    -------
    pandas.DataFrame
        Columns ``Name``, ``Coefficient``, ``Coefficient_std_error``,
        ``Coefficient_P_value`` sorted by ascending P-value.
    """
    log = _coerce_log(log)
    if no_intercept:
        intercept_h2 = 1
    elif intercept_h2 is not None:
        intercept_h2 = float(intercept_h2)

    M_annot_baseline, w_ld_cname, ref_ld_cnames, merged, _ = \
        _read_ld_sumstats(sumstats, ref_ld, w_ld, M=M, not_M_5_50=not_M_5_50,
                          log=log)
    n_snp = len(merged)
    n_blocks = min(n_snp, n_blocks)
    if chisq_max is None:
        chisq_max = max(0.001 * merged.N.max(), 80)
    ii = np.ravel(merged.Z ** 2 < chisq_max)
    merged = merged[ii]
    n_snp = int(np.sum(ii))
    log.log("Using {N} SNPs after chi^2 filter.".format(N=n_snp))

    ref_ld_baseline = np.array(merged[ref_ld_cnames]).reshape((n_snp, -1))
    chisq = np.array(merged.Z ** 2)
    keep_snps = merged[["SNP"]]

    if isinstance(ref_ld_cts, str):
        with open(ref_ld_cts) as fh:
            cts_entries = [line.split() for line in fh if line.strip()]
    else:
        cts_entries = [list(x) for x in ref_ld_cts]

    def col(x):
        return np.array(x).reshape((n_snp, 1))

    rows = []
    for name, ct_ld in cts_entries:
        ct_paths = _splitp(ct_ld)
        ref_ld_cts_all = ps.ldscore_fromlist(ct_paths)
        ref_ld_ct = np.array(
            pd.merge(keep_snps, ref_ld_cts_all, on="SNP", how="left").iloc[
                :, 1:
            ]
        )
        if np.any(np.isnan(ref_ld_ct)):
            raise ValueError(
                "Missing LD scores from cts files. Ensure all SNPs in "
                "ref_ld are also in ref_ld_cts."
            )
        ref_ld_full = np.hstack([ref_ld_ct, ref_ld_baseline])
        M_cts = ps.M_fromlist(ct_paths, common=(not not_M_5_50))
        M_annot = np.hstack([M_cts, M_annot_baseline])
        hsqhat = reg.Hsq(
            col(chisq), ref_ld_full, col(merged[w_ld_cname]), col(merged.N),
            M_annot, n_blocks=n_blocks, intercept=intercept_h2, twostep=None,
            old_weights=True,
        )
        coef, coef_se = hsqhat.coef[0], hsqhat.coef_se[0]
        rows.append((name, coef, coef_se, stats.norm.sf(coef / coef_se)))

    df = pd.DataFrame(
        rows,
        columns=["Name", "Coefficient", "Coefficient_std_error",
                 "Coefficient_P_value"],
    )
    df = df.sort_values(by="Coefficient_P_value").reset_index(drop=True)
    if out is not None:
        df.to_csv(out + ".cell_type_results.txt", sep="\t", index=False)
        log.log("Results written to {O}.cell_type_results.txt".format(O=out))
    return df
