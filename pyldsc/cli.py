"""Thin command-line interface mirroring the original ``ldsc.py`` flags.

Sub-commands
------------
``pyldsc l2``      -- estimate LD Scores (``--l2``).
``pyldsc h2``      -- univariate / partitioned heritability (``--h2``).
``pyldsc rg``      -- genetic correlation (``--rg``).
``pyldsc h2-cts``  -- LDSC-SEG cell-type analysis (``--h2-cts``).
``pyldsc munge``   -- harmonise raw GWAS sumstats (``munge_sumstats.py``).
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .api import Logger, estimate_h2, estimate_ldscore, estimate_rg, ldsc_seg
from .munge import munge_sumstats

__all__ = ["main", "build_parser"]


def build_parser():
    """Construct the top-level ``pyldsc`` argument parser."""
    p = argparse.ArgumentParser(
        prog="pyldsc",
        description="py-ldsc: modern-Python LD Score Regression.",
    )
    p.add_argument("--version", action="version",
                   version="pyldsc " + __version__)
    sub = p.add_subparsers(dest="cmd")

    # --- l2 -------------------------------------------------------------
    l2 = sub.add_parser("l2", help="estimate LD Scores from PLINK genotypes")
    l2.add_argument("--bfile", required=True, help="PLINK fileset prefix")
    l2.add_argument("--out", required=True, help="output prefix")
    l2.add_argument("--ld-wind-cm", type=float, default=None)
    l2.add_argument("--ld-wind-kb", type=float, default=None)
    l2.add_argument("--ld-wind-snps", type=int, default=None)
    l2.add_argument("--maf", type=float, default=None)
    l2.add_argument("--chunk-size", type=int, default=50)
    l2.add_argument("--yes-really", action="store_true", default=False)

    # --- h2 -------------------------------------------------------------
    h2 = sub.add_parser("h2", help="estimate (partitioned) heritability")
    h2.add_argument("--h2", required=True, help=".sumstats path")
    h2.add_argument("--ref-ld", required=True)
    h2.add_argument("--w-ld", required=True)
    h2.add_argument("--out", default=None)
    h2.add_argument("--M", default=None)
    h2.add_argument("--n-blocks", type=int, default=200)
    h2.add_argument("--chisq-max", type=float, default=None)
    h2.add_argument("--two-step", type=float, default=None)
    h2.add_argument("--intercept-h2", default=None)
    h2.add_argument("--no-intercept", action="store_true", default=False)
    h2.add_argument("--not-M-5-50", action="store_true", default=False)
    h2.add_argument("--overlap-annot", action="store_true", default=False)
    h2.add_argument("--frqfile", default=None)
    h2.add_argument("--print-coefficients", action="store_true", default=False)
    h2.add_argument("--samp-prev", type=float, default=None)
    h2.add_argument("--pop-prev", type=float, default=None)

    # --- rg -------------------------------------------------------------
    rg = sub.add_parser("rg", help="estimate genetic correlation")
    rg.add_argument("--rg", required=True, help="comma-separated .sumstats")
    rg.add_argument("--ref-ld", required=True)
    rg.add_argument("--w-ld", required=True)
    rg.add_argument("--out", default=None)
    rg.add_argument("--M", default=None)
    rg.add_argument("--n-blocks", type=int, default=200)
    rg.add_argument("--chisq-max", type=float, default=None)
    rg.add_argument("--two-step", type=float, default=None)
    rg.add_argument("--intercept-h2", default=None)
    rg.add_argument("--intercept-gencov", default=None)
    rg.add_argument("--no-intercept", action="store_true", default=False)
    rg.add_argument("--not-M-5-50", action="store_true", default=False)
    rg.add_argument("--no-check-alleles", action="store_true", default=False)

    # --- h2-cts ---------------------------------------------------------
    cts = sub.add_parser("h2-cts", help="LDSC-SEG cell-type analysis")
    cts.add_argument("--h2-cts", required=True, help=".sumstats path")
    cts.add_argument("--ref-ld-chr-cts", required=True)
    cts.add_argument("--ref-ld", required=True)
    cts.add_argument("--w-ld", required=True)
    cts.add_argument("--out", default=None)
    cts.add_argument("--n-blocks", type=int, default=200)
    cts.add_argument("--chisq-max", type=float, default=None)
    cts.add_argument("--intercept-h2", default=None)
    cts.add_argument("--no-intercept", action="store_true", default=False)
    cts.add_argument("--not-M-5-50", action="store_true", default=False)

    # --- munge ----------------------------------------------------------
    mg = sub.add_parser("munge", help="harmonise raw GWAS sumstats")
    mg.add_argument("--sumstats", required=True)
    mg.add_argument("--out", required=True)
    mg.add_argument("--N", type=float, default=None)
    mg.add_argument("--N-cas", type=float, default=None)
    mg.add_argument("--N-con", type=float, default=None)
    mg.add_argument("--info-min", type=float, default=0.9)
    mg.add_argument("--maf-min", type=float, default=0.01)
    mg.add_argument("--n-min", type=float, default=None)
    mg.add_argument("--no-alleles", action="store_true", default=False)
    mg.add_argument("--merge-alleles", default=None)
    mg.add_argument("--signed-sumstats", default=None)
    mg.add_argument("--a1-inc", action="store_true", default=False)
    mg.add_argument("--ignore", default=None)
    mg.add_argument("--keep-maf", action="store_true", default=False)

    return p


def main(argv=None):
    """CLI entry point."""
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0

    log = Logger(args.out + ".log" if getattr(args, "out", None) else None)

    if args.cmd == "l2":
        estimate_ldscore(
            args.bfile, ld_wind_cm=args.ld_wind_cm,
            ld_wind_kb=args.ld_wind_kb, ld_wind_snps=args.ld_wind_snps,
            maf=args.maf, chunk_size=args.chunk_size, out=args.out,
            yes_really=args.yes_really, log=log,
        )
    elif args.cmd == "h2":
        estimate_h2(
            args.h2, args.ref_ld, args.w_ld, M=args.M,
            intercept_h2=args.intercept_h2, no_intercept=args.no_intercept,
            n_blocks=args.n_blocks, chisq_max=args.chisq_max,
            two_step=args.two_step, not_M_5_50=args.not_M_5_50,
            overlap_annot=args.overlap_annot, frqfile=args.frqfile,
            print_coefficients=args.print_coefficients,
            samp_prev=args.samp_prev, pop_prev=args.pop_prev, out=args.out,
            log=log,
        )
    elif args.cmd == "rg":
        estimate_rg(
            args.rg, args.ref_ld, args.w_ld, M=args.M,
            intercept_h2=args.intercept_h2,
            intercept_gencov=args.intercept_gencov,
            no_intercept=args.no_intercept, n_blocks=args.n_blocks,
            chisq_max=args.chisq_max, two_step=args.two_step,
            not_M_5_50=args.not_M_5_50,
            no_check_alleles=args.no_check_alleles, log=log,
        )
    elif args.cmd == "h2-cts":
        ldsc_seg(
            args.h2_cts, args.ref_ld_chr_cts, args.ref_ld, args.w_ld,
            intercept_h2=args.intercept_h2, no_intercept=args.no_intercept,
            n_blocks=args.n_blocks, chisq_max=args.chisq_max,
            not_M_5_50=args.not_M_5_50, out=args.out, log=log,
        )
    elif args.cmd == "munge":
        munge_sumstats(
            args.sumstats, out=args.out, N=args.N, N_cas=args.N_cas,
            N_con=args.N_con, info_min=args.info_min, maf_min=args.maf_min,
            n_min=args.n_min, no_alleles=args.no_alleles,
            merge_alleles=args.merge_alleles,
            signed_sumstats=args.signed_sumstats, a1_inc=args.a1_inc,
            ignore=args.ignore, keep_maf=args.keep_maf, log=log,
        )

    log.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
