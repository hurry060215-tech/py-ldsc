"""File parsers for the LDSC-defined file formats.

Faithful modern-Python port of ``ldsc/ldscore/parse.py``
(Bulik-Sullivan & Finucane, 2014).

Readers/writers for ``.sumstats(.gz)``, ``.l2.ldscore(.gz)``,
``.l{N}.M`` / ``.M_5_50``, ``.annot(.gz)``, ``.bim`` / ``.fam`` and
``--frqfile`` files.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

__all__ = [
    "series_eq",
    "read_csv",
    "sub_chr",
    "which_compression",
    "get_compression",
    "sumstats",
    "ldscore",
    "ldscore_fromlist",
    "M",
    "M_fromlist",
    "annot",
    "frq_parser",
    "read_cts",
    "PlinkBIMFile",
    "PlinkFAMFile",
    "FilterFile",
    "AnnotFile",
    "ThinAnnotFile",
]

_N_CHR = 22


def series_eq(x, y):
    """Compare two series; return False if lengths differ."""
    return len(x) == len(y) and (np.asarray(x) == np.asarray(y)).all()


def read_csv(fh, **kwargs):
    """Whitespace-delimited ``pd.read_csv`` with ``'.'`` treated as NaN."""
    return pd.read_csv(fh, sep=r"\s+", na_values=".", **kwargs)


def sub_chr(s, chrom):
    """Substitute ``chrom`` for ``@``; if no ``@``, append ``chrom``."""
    if "@" not in s:
        s += "@"
    return s.replace("@", str(chrom))


def get_present_chrs(fh, num):
    """Return the chromosomes for which ``sub_chr(fh, chrom).*`` exists."""
    chrs = []
    for chrom in range(1, num):
        if glob.glob(sub_chr(fh, chrom) + ".*"):
            chrs.append(chrom)
    return chrs


def which_compression(fh):
    """Given a file prefix, figure out which compression suffix to use."""
    if os.access(fh + ".bz2", 4):
        return ".bz2", "bz2"
    elif os.access(fh + ".gz", 4):
        return ".gz", "gzip"
    elif os.access(fh, 4):
        return "", None
    else:
        raise IOError("Could not open {F}[./gz/bz2]".format(F=fh))


def get_compression(fh):
    """Infer the compression of ``fh`` from its suffix."""
    if fh.endswith("gz"):
        return "gzip"
    elif fh.endswith("bz2"):
        return "bz2"
    return None


def read_cts(fh, match_snps):
    """Read a ``--cts-bin`` continuous-annotation file."""
    compression = get_compression(fh)
    cts = read_csv(fh, compression=compression, header=None,
                   names=["SNP", "ANNOT"])
    if not series_eq(cts.SNP, match_snps):
        raise ValueError(
            "--cts-bin and the .bim file must have identical SNP columns."
        )
    return cts.ANNOT.values


def sumstats(fh, alleles=False, dropna=True):
    """Parse a ``.sumstats`` file into a DataFrame (SNP, Z, N[, A1, A2])."""
    dtype_dict = {"SNP": str, "Z": float, "N": float, "A1": str, "A2": str}
    compression = get_compression(fh)
    usecols = ["SNP", "Z", "N"]
    if alleles:
        usecols += ["A1", "A2"]
    try:
        x = read_csv(fh, usecols=usecols, dtype=dtype_dict,
                     compression=compression)
    except (AttributeError, ValueError) as e:
        raise ValueError("Improperly formatted sumstats file: " + str(e.args))
    if dropna:
        x = x.dropna(how="any")
    return x


def ldscore_fromlist(flist, num=None):
    """Side-by-side concatenation of a list of LD Score files."""
    ldscore_array = []
    for i, fh in enumerate(flist):
        y = ldscore(fh, num)
        if i > 0:
            if not series_eq(y.SNP, ldscore_array[0].SNP):
                raise ValueError(
                    "LD Scores for concatenation must have identical SNP "
                    "columns."
                )
            else:  # keep SNP column from only the first file
                y = y.drop(["SNP"], axis=1)
        new_col_dict = {
            c: c + "_" + str(i) for c in y.columns if c != "SNP"
        }
        y = y.rename(columns=new_col_dict)
        ldscore_array.append(y)
    return pd.concat(ldscore_array, axis=1)


def l2_parser(fh, compression):
    """Parse a single ``.l2.ldscore`` file."""
    x = read_csv(fh, header=0, compression=compression)
    if "MAF" in x.columns and "CM" in x.columns:  # backwards compat. v<1.0.0
        x = x.drop(["MAF", "CM"], axis=1)
    return x


def annot_parser(fh, compression, frqfile_full=None, compression_frq=None):
    """Parse a single ``.annot`` file (drop SNP/CHR/BP/CM columns)."""
    df_annot = read_csv(fh, header=0, compression=compression).drop(
        ["SNP", "CHR", "BP", "CM"], axis=1, errors="ignore"
    ).astype(float)
    if frqfile_full is not None:
        df_frq = frq_parser(frqfile_full, compression_frq)
        df_annot = df_annot[(0.95 > df_frq.FRQ) & (df_frq.FRQ > 0.05)]
    return df_annot


def frq_parser(fh, compression):
    """Parse a PLINK ``.frq`` allele-frequency file."""
    df = read_csv(fh, header=0, compression=compression)
    if "MAF" in df.columns:
        df = df.rename(columns={"MAF": "FRQ"})
    return df[["SNP", "FRQ"]]


def ldscore(fh, num=None):
    """Parse ``.l2.ldscore`` file(s), optionally split across chromosomes."""
    suffix = ".l2.ldscore"
    if num is not None:  # num files, e.g., one per chromosome
        chrs = get_present_chrs(fh, num + 1)
        first_fh = sub_chr(fh, chrs[0]) + suffix
        s, compression = which_compression(first_fh)
        chr_ld = [
            l2_parser(sub_chr(fh, i) + suffix + s, compression) for i in chrs
        ]
        x = pd.concat(chr_ld)  # automatically sorted by chromosome
    else:  # just one file
        s, compression = which_compression(fh + suffix)
        x = l2_parser(fh + suffix + s, compression)

    x = x.sort_values(by=["CHR", "BP"])  # SEs will be wrong unless sorted
    x = x.drop(["CHR", "BP"], axis=1).drop_duplicates(subset="SNP")
    return x


def M(fh, num=None, N=2, common=False):
    """Parse ``.l{N}.M`` file(s), optionally split across chromosomes."""
    parsefunc = lambda y: [
        float(z) for z in open(y, "r").readline().split()
    ]
    suffix = ".l" + str(N) + ".M"
    if common:
        suffix += "_5_50"
    if num is not None:
        x = np.sum(
            [parsefunc(sub_chr(fh, i) + suffix)
             for i in get_present_chrs(fh, num + 1)],
            axis=0,
        )
    else:
        x = parsefunc(fh + suffix)
    return np.array(x).reshape((1, len(x)))


def M_fromlist(flist, num=None, N=2, common=False):
    """Read a list of ``.M*`` files and concatenate side-by-side."""
    return np.hstack([M(fh, num, N, common) for fh in flist])


def annot(fh_list, num=None, frqfile=None):
    """Parse ``.annot`` file(s) and return an overlap matrix and ``M_tot``."""
    annot_suffix = [".annot" for _ in fh_list]
    annot_compression = []
    if num is not None:  # files split per chromosome
        chrs = get_present_chrs(fh_list[0], num + 1)
        for i, fh in enumerate(fh_list):
            first_fh = sub_chr(fh, chrs[0]) + annot_suffix[i]
            annot_s, annot_comp_single = which_compression(first_fh)
            annot_suffix[i] += annot_s
            annot_compression.append(annot_comp_single)

        if frqfile is not None:
            frq_suffix = ".frq"
            first_frqfile = sub_chr(frqfile, 1) + frq_suffix
            frq_s, frq_compression = which_compression(first_frqfile)
            frq_suffix += frq_s

        y = []
        M_tot = 0
        for chrom in chrs:
            if frqfile is not None:
                df_annot_chr_list = [
                    annot_parser(
                        sub_chr(fh, chrom) + annot_suffix[i],
                        annot_compression[i],
                        sub_chr(frqfile, chrom) + frq_suffix, frq_compression,
                    )
                    for i, fh in enumerate(fh_list)
                ]
            else:
                df_annot_chr_list = [
                    annot_parser(
                        sub_chr(fh, chrom) + annot_suffix[i],
                        annot_compression[i],
                    )
                    for i, fh in enumerate(fh_list)
                ]
            annot_matrix_chr_list = [
                np.asarray(df) for df in df_annot_chr_list
            ]
            annot_matrix_chr = np.hstack(annot_matrix_chr_list)
            y.append(np.dot(annot_matrix_chr.T, annot_matrix_chr))
            M_tot += len(df_annot_chr_list[0])
        x = sum(y)
    else:  # just one file
        for i, fh in enumerate(fh_list):
            annot_s, annot_comp_single = which_compression(
                fh + annot_suffix[i]
            )
            annot_suffix[i] += annot_s
            annot_compression.append(annot_comp_single)

        if frqfile is not None:
            frq_suffix = ".frq"
            frq_s, frq_compression = which_compression(frqfile + frq_suffix)
            frq_suffix += frq_s
            df_annot_list = [
                annot_parser(
                    fh + annot_suffix[i], annot_compression[i],
                    frqfile + frq_suffix, frq_compression,
                )
                for i, fh in enumerate(fh_list)
            ]
        else:
            df_annot_list = [
                annot_parser(fh + annot_suffix[i], annot_compression[i])
                for i, fh in enumerate(fh_list)
            ]
        annot_matrix_list = [np.asarray(y) for y in df_annot_list]
        annot_matrix = np.hstack(annot_matrix_list)
        x = np.dot(annot_matrix.T, annot_matrix)
        M_tot = len(df_annot_list[0])

    return x, M_tot


def __ID_List_Factory__(colnames, keepcol, fname_end, header=None,
                        usecols=None):
    """Factory producing simple ID-list container classes (.bim/.fam/...)."""

    class IDContainer(object):
        def __init__(self, fname):
            self.__usecols__ = usecols
            self.__colnames__ = colnames
            self.__keepcol__ = keepcol
            self.__fname_end__ = fname_end
            self.__header__ = header
            self.__read__(fname)
            self.n = len(self.df)

        def __read__(self, fname):
            end = self.__fname_end__
            if end and not fname.endswith(end):
                raise ValueError(
                    "{f} filename must end in {f}".format(f=end)
                )
            comp = get_compression(fname)
            self.df = pd.read_csv(
                fname, header=self.__header__, usecols=self.__usecols__,
                sep=r"\s+", compression=comp,
            )
            if self.__colnames__:
                self.df.columns = self.__colnames__
            if self.__keepcol__ is not None:
                self.IDList = self.df.iloc[:, [self.__keepcol__]].astype(
                    "object"
                )

        def loj(self, externalDf):
            """Indices of ``self.IDList`` entries appearing in ``externalDf``."""
            r = externalDf.columns[0]
            l = self.IDList.columns[0]
            merge_df = externalDf.iloc[:, [0]].copy()
            merge_df["keep"] = True
            z = pd.merge(
                self.IDList, merge_df, how="left", left_on=l, right_on=r,
                sort=False,
            )
            ii = z["keep"] == True  # noqa: E712
            return np.nonzero(np.asarray(ii))[0]

    return IDContainer


PlinkBIMFile = __ID_List_Factory__(
    ["CHR", "SNP", "CM", "BP", "A1", "A2"], 1, ".bim",
    usecols=[0, 1, 2, 3, 4, 5],
)
PlinkFAMFile = __ID_List_Factory__(["IID"], 0, ".fam", usecols=[1])
FilterFile = __ID_List_Factory__(["ID"], 0, None, usecols=[0])
AnnotFile = __ID_List_Factory__(None, 2, None, header=0, usecols=None)
ThinAnnotFile = __ID_List_Factory__(None, None, None, header=0, usecols=None)
