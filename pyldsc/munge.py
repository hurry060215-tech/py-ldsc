"""Harmonise raw GWAS summary statistics into the LDSC ``.sumstats`` format.

Faithful modern-Python port of ``ldsc/munge_sumstats.py``
(Bulik-Sullivan & Finucane, 2014).

The single public entry point :func:`munge_sumstats` performs column-name
recognition, allele alignment, INFO / MAF / N / P filtering, signed-statistic
handling and conversion of P-values to signed Z-scores.
"""
from __future__ import annotations

import bz2
import gzip

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .api import VALID_SNPS, MATCH_ALLELES

__all__ = ["munge_sumstats", "DEFAULT_CNAMES"]

NULL_VALUES = {"LOG_ODDS": 0, "BETA": 0, "OR": 1, "Z": 0}

DEFAULT_CNAMES = {
    # rs number
    "SNP": "SNP", "MARKERNAME": "SNP", "SNPID": "SNP", "RS": "SNP",
    "RSID": "SNP", "RS_NUMBER": "SNP", "RS_NUMBERS": "SNP",
    # number of studies
    "NSTUDY": "NSTUDY", "N_STUDY": "NSTUDY", "NSTUDIES": "NSTUDY",
    "N_STUDIES": "NSTUDY",
    # p-value
    "P": "P", "PVALUE": "P", "P_VALUE": "P", "PVAL": "P", "P_VAL": "P",
    "GC_PVALUE": "P",
    # allele 1
    "A1": "A1", "ALLELE1": "A1", "ALLELE_1": "A1", "EFFECT_ALLELE": "A1",
    "REFERENCE_ALLELE": "A1", "INC_ALLELE": "A1", "EA": "A1",
    # allele 2
    "A2": "A2", "ALLELE2": "A2", "ALLELE_2": "A2", "OTHER_ALLELE": "A2",
    "NON_EFFECT_ALLELE": "A2", "DEC_ALLELE": "A2", "NEA": "A2",
    # N
    "N": "N", "NCASE": "N_CAS", "CASES_N": "N_CAS", "N_CASE": "N_CAS",
    "N_CASES": "N_CAS", "N_CONTROLS": "N_CON", "N_CAS": "N_CAS",
    "N_CON": "N_CON", "NCONTROL": "N_CON", "CONTROLS_N": "N_CON",
    "N_CONTROL": "N_CON", "WEIGHT": "N",
    # signed statistics
    "ZSCORE": "Z", "Z-SCORE": "Z", "GC_ZSCORE": "Z", "Z": "Z", "OR": "OR",
    "B": "BETA", "BETA": "BETA", "LOG_ODDS": "LOG_ODDS", "EFFECTS": "BETA",
    "EFFECT": "BETA", "SIGNED_SUMSTAT": "SIGNED_SUMSTAT",
    # info / maf
    "INFO": "INFO", "EAF": "FRQ", "FRQ": "FRQ", "MAF": "FRQ", "FRQ_U": "FRQ",
    "F_U": "FRQ",
}

NUMERIC_COLS = ["P", "N", "N_CAS", "N_CON", "Z", "OR", "BETA", "LOG_ODDS",
                "INFO", "FRQ", "SIGNED_SUMSTAT", "NSTUDY"]


def _get_compression(fh):
    if fh.endswith("gz"):
        return gzip.open, "gzip"
    elif fh.endswith("bz2"):
        return bz2.BZ2File, "bz2"
    return open, None


def _clean_header(header):
    """Uppercase, replace dashes/dots with underscores, strip newlines."""
    return header.upper().replace("-", "_").replace(".", "_").replace("\n", "")


def _read_header(fh):
    openfunc, _ = _get_compression(fh)
    with openfunc(fh, "rt") if openfunc is not open else open(fh) as f:
        return [x.rstrip("\n") for x in f.readline().split()]


def _p_to_z(P, N):
    """Convert P-value to standardized (unsigned) beta / Z-score."""
    return np.sqrt(chi2.isf(P, 1))


def munge_sumstats(sumstats, out=None, N=None, N_cas=None, N_con=None,
                   info_min=0.9, maf_min=0.01, n_min=None, no_alleles=False,
                   merge_alleles=None, signed_sumstats=None, a1_inc=False,
                   snp=None, N_col=None, a1=None, a2=None, p=None, frq=None,
                   info=None, ignore=None, keep_maf=False, daner=False,
                   chunksize=5_000_000, log=None, write=True):
    """Convert a raw GWAS summary-statistics file to ``.sumstats`` format.

    Parameters
    ----------
    sumstats : str
        Path to the raw GWAS file (optionally ``.gz`` / ``.bz2``).
    out : str, optional
        Output prefix.  When given and ``write`` is True, writes
        ``out.sumstats.gz``.
    N, N_cas, N_con : float, optional
        Sample size / case / control counts (used when the file lacks them).
    info_min : float, optional
        Minimum INFO score (default 0.9).
    maf_min : float, optional
        Minimum minor-allele frequency (default 0.01).
    n_min : float, optional
        Minimum N.  Default is ``(90th-percentile N) / 1.5``.
    no_alleles : bool, optional
        Do not require allele columns (h2-only workflows).
    merge_alleles : str, optional
        Path to a SNP/A1/A2 file; output is matched/aligned to it.
    signed_sumstats : str, optional
        ``"COLUMN,null_value"`` -- explicit signed-statistic column.
    a1_inc : bool, optional
        Treat A1 as the trait-increasing allele (unsigned input).
    snp, N_col, a1, a2, p, frq, info : str, optional
        Override column-name detection.
    ignore : str, optional
        Comma-separated column names to ignore.

    Returns
    -------
    pandas.DataFrame
        Harmonised sumstats with columns ``SNP``, ``Z``, ``N`` and
        (unless ``no_alleles``) ``A1``, ``A2``.
    """
    from .api import _NullLog

    log = log if log is not None else _NullLog()

    file_cnames = _read_header(sumstats)

    # column-name flags
    flag_cnames = {}
    for value, internal in [
        (snp, "SNP"), (N_col, "N"), (a1, "A1"), (a2, "A2"), (p, "P"),
        (frq, "FRQ"), (info, "INFO"),
    ]:
        if value is not None:
            flag_cnames[_clean_header(value)] = internal

    signed_sumstat_null = None
    if signed_sumstats is not None:
        cname, null_value = signed_sumstats.split(",")
        signed_sumstat_null = float(null_value)
        flag_cnames[_clean_header(cname)] = "SIGNED_SUMSTAT"

    ignore_cnames = (
        [_clean_header(x) for x in ignore.split(",")] if ignore else []
    )

    if signed_sumstats is not None or a1_inc:
        mod_default = {
            k: v for k, v in DEFAULT_CNAMES.items() if v not in NULL_VALUES
        }
    else:
        mod_default = DEFAULT_CNAMES

    cname_map = {k: v for k, v in flag_cnames.items() if k not in ignore_cnames}
    for k, v in mod_default.items():
        if k not in ignore_cnames and k not in flag_cnames:
            cname_map[k] = v

    cname_translation = {
        x: cname_map[_clean_header(x)]
        for x in file_cnames
        if _clean_header(x) in cname_map
    }

    # determine the signed-statistic column
    if signed_sumstats is None and not a1_inc:
        sign_cnames = [
            x for x in cname_translation if cname_translation[x] in NULL_VALUES
        ]
        if len(sign_cnames) > 1:
            raise ValueError(
                "Too many signed sumstat columns. Use ignore= to drop one."
            )
        if len(sign_cnames) == 0:
            raise ValueError("Could not find a signed summary statistic.")
        sign_cname = sign_cnames[0]
        signed_sumstat_null = NULL_VALUES[cname_translation[sign_cname]]
        cname_translation[sign_cname] = "SIGNED_SUMSTAT"
    else:
        sign_cname = "SIGNED_SUMSTAT"

    req_cols = ["SNP", "P"] if a1_inc else ["SNP", "P", "SIGNED_SUMSTAT"]
    for c in req_cols:
        if c not in cname_translation.values():
            raise ValueError("Could not find {C} column.".format(C=c))

    if (
        N is None
        and not (N_cas and N_con)
        and "N" not in cname_translation.values()
        and any(
            x not in cname_translation.values() for x in ["N_CAS", "N_CON"]
        )
    ):
        raise ValueError("Could not determine N.")
    if not no_alleles and not all(
        x in cname_translation.values() for x in ["A1", "A2"]
    ):
        raise ValueError("Could not find A1/A2 columns.")

    # merge-alleles file
    ma = None
    if merge_alleles is not None:
        openfunc, comp = _get_compression(merge_alleles)
        ma = pd.read_csv(merge_alleles, compression=comp, header=0,
                         sep=r"\s+", na_values=".")
        if any(x not in ma.columns for x in ["SNP", "A1", "A2"]):
            raise ValueError("merge_alleles must have columns SNP, A1, A2.")
        ma["MA"] = (ma.A1 + ma.A2).apply(lambda y: y.upper())
        ma = ma[["SNP", "MA"]]

    # read the file
    _, comp = _get_compression(sumstats)
    signed_cols = [
        k for k, v in cname_translation.items() if v == "SIGNED_SUMSTAT"
    ]
    dat = pd.read_csv(
        sumstats, sep=r"\s+", header=0, compression=comp,
        usecols=list(cname_translation.keys()), na_values=[".", "NA"],
        dtype={c: np.float64 for c in signed_cols},
    )
    tot_snps = len(dat)
    dat = dat.dropna(
        axis=0, how="any",
        subset=[c for c in dat.columns if c != "INFO"],
    ).reset_index(drop=True)
    dat.columns = [cname_translation[c] for c in dat.columns]

    if merge_alleles is not None:
        dat = dat[dat.SNP.isin(ma.SNP)].reset_index(drop=True)

    if "INFO" in dat.columns:
        info_col = dat["INFO"]
        dat = dat[info_col >= info_min].reset_index(drop=True)
    if "FRQ" in dat.columns:
        fr = np.minimum(dat["FRQ"], 1 - dat["FRQ"])
        dat = dat[(fr > maf_min) & ~((dat["FRQ"] < 0) | (dat["FRQ"] > 1))]
        dat = dat.reset_index(drop=True)
    drop_cols = ["INFO"] if keep_maf else ["INFO", "FRQ"]
    dat = dat.drop([c for c in drop_cols if c in dat.columns], axis=1)

    dat = dat[(dat.P > 0) & (dat.P <= 1)].reset_index(drop=True)

    if not no_alleles:
        dat.A1 = dat.A1.str.upper()
        dat.A2 = dat.A2.str.upper()
        dat = dat[(dat.A1 + dat.A2).isin(VALID_SNPS)].reset_index(drop=True)

    dat = dat.drop_duplicates(subset="SNP").reset_index(drop=True)

    # N handling
    if all(c in dat.columns for c in ["N_CAS", "N_CON"]):
        Nc = dat.N_CAS + dat.N_CON
        P = dat.N_CAS / Nc
        dat["N"] = Nc * P / P[Nc == Nc.max()].mean()
        dat = dat.drop(["N_CAS", "N_CON"], axis=1)
    if "N" in dat.columns:
        nmin = n_min if n_min else dat.N.quantile(0.9) / 1.5
        dat = dat[dat.N >= nmin].reset_index(drop=True)
    elif "NSTUDY" in dat.columns:
        dat = dat[dat.NSTUDY >= dat.NSTUDY.max()].drop(
            ["NSTUDY"], axis=1
        ).reset_index(drop=True)
    if "N" not in dat.columns:
        if N is not None:
            dat["N"] = N
        elif N_cas and N_con:
            dat["N"] = N_cas + N_con
        else:
            raise ValueError("Cannot determine N.")

    # P -> signed Z
    dat["Z"] = _p_to_z(dat.P, dat.N)
    dat = dat.drop("P", axis=1)
    if not a1_inc:
        dat["Z"] *= (-1) ** (dat.SIGNED_SUMSTAT < signed_sumstat_null)
        dat = dat.drop("SIGNED_SUMSTAT", axis=1)

    # allele merge
    if merge_alleles is not None:
        dat = pd.merge(ma, dat, how="left", on="SNP", sort=False).reset_index(
            drop=True
        )
        ii = dat.A1.notnull()
        a1234 = dat.A1[ii] + dat.A2[ii] + dat.MA[ii]
        match = a1234.apply(lambda y: y in MATCH_ALLELES)
        jj = pd.Series(np.zeros(len(dat), dtype=bool))
        jj[ii] = match.astype(bool).values
        dat.loc[~jj.astype(bool), [c for c in dat.columns if c != "SNP"]] = (
            float("nan")
        )
        dat = dat.drop(["MA"], axis=1)

    print_colnames = [
        c for c in dat.columns if c in ["SNP", "N", "Z", "A1", "A2"]
    ]
    if keep_maf and "FRQ" in dat.columns:
        print_colnames.append("FRQ")
    dat = dat[print_colnames]

    log.log(
        "Munged {tot} input SNPs to {n} output SNPs.".format(
            tot=tot_snps, n=len(dat)
        )
    )

    if out is not None and write:
        dat.to_csv(out + ".sumstats.gz", sep="\t", index=False,
                   float_format="%.3f", compression="gzip")
        log.log("Wrote {O}.sumstats.gz".format(O=out))

    return dat
