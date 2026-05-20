"""Benchmark the core py-ldsc workflows on the bundled ldsc test data.

Run with::

    python examples/benchmark.py

The script uses the test data shipped with the upstream ``ldsc`` repository
(expected under ``/tmp/ldsc_ref/test``); pass a different location as the
first command-line argument if needed.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import pyldsc

REF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ldsc_ref/test"
SIM = os.path.join(REF, "simulate_test")
ONELD = os.path.join(SIM, "ldscore", "oneld_onefile")
TWOLD = os.path.join(SIM, "ldscore", "twold_onefile")
WLD = os.path.join(SIM, "ldscore", "w")


def _ss(i):
    return os.path.join(SIM, "sumstats", str(i))


def timed(label, fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    print("  {:<34s} {:8.3f} s".format(label, dt))
    return out


def main():
    if not os.path.isdir(SIM):
        print("ldsc reference data not found at", SIM)
        print("clone https://github.com/bulik/ldsc to /tmp/ldsc_ref first.")
        return

    print("py-ldsc benchmark (data: {})".format(SIM))
    print("-" * 56)

    # --- LD Score estimation --------------------------------------------
    plink = os.path.join(REF, "reference_test", "plink")
    if os.path.exists(plink + ".bed"):
        res = timed(
            "estimate_ldscore (PLINK .bed)",
            lambda: pyldsc.estimate_ldscore(
                plink, ld_wind_kb=1000.0, yes_really=True, log=None
            ),
        )
        print("      -> {} SNPs, M = {}".format(
            len(res["ldscore"]), res["M"]))

    # --- univariate heritability ----------------------------------------
    h = timed(
        "estimate_h2 (univariate)",
        lambda: pyldsc.estimate_h2(_ss(1), ONELD, WLD, log=None),
    )
    print("      -> h2 = {:.4f} ({:.4f}), intercept = {:.4f}".format(
        h.tot, h.tot_se, h.intercept))

    # --- partitioned heritability ---------------------------------------
    hp = timed(
        "estimate_h2 (2-category partitioned)",
        lambda: pyldsc.estimate_h2(_ss(1), TWOLD, WLD, chisq_max=99999,
                                   log=None),
    )
    print("      -> total h2 = {:.4f}, cat = {}".format(
        hp.tot, np.round(np.ravel(hp.cat), 4)))

    # --- genetic correlation --------------------------------------------
    rg = timed(
        "estimate_rg (self-correlation)",
        lambda: pyldsc.estimate_rg([_ss(1), _ss(1)], ONELD, WLD, log=None),
    )
    print("      -> rg = {:.4f} ({:.4f})".format(
        rg[0].rg_ratio, rg[0].rg_se))

    # --- LDSC-SEG --------------------------------------------------------
    seg = timed(
        "ldsc_seg (cell-type enrichment)",
        lambda: pyldsc.ldsc_seg(_ss(1), [("CT_oneld", ONELD)], TWOLD, WLD,
                                log=None),
    )
    print("      -> {}".format(seg.iloc[0].to_dict()))

    # --- batch heritability over many simulated GWAS --------------------
    def batch():
        return [
            pyldsc.estimate_h2(_ss(i), TWOLD, WLD, chisq_max=99999, log=None)
            for i in range(50)
        ]

    hs = timed("estimate_h2 x 50 (statistical check)", batch)
    print("      -> mean h2 = {:.4f} (target 0.9)".format(
        np.nanmean([x.tot for x in hs])))
    print("-" * 56)
    print("done.")


if __name__ == "__main__":
    main()
