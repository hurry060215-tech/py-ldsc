# py-ldsc

**py-ldsc** (`pyldsc`) is a faithful, cleanly-rewritten **modern-Python**
reimplementation of [**LDSC**](https://github.com/bulik/ldsc) — LD Score
Regression.

It ports the full LDSC toolkit:

* **LD Score Regression** — Bulik-Sullivan et al., *Nature Genetics* 2015,
  *"LD Score regression distinguishes confounding from polygenicity in
  genome-wide association studies"*.
* **Partitioned heritability / stratified LDSC** — Finucane et al.,
  *Nature Genetics* 2015.
* **LDSC-SEG** (cell-type / tissue-specific enrichment) — Finucane et al.,
  *Nature Genetics* 2018.

The original `ldsc` is a Python-2-era command-line tool. `pyldsc` is a pure,
importable, modern-Python package (numpy / scipy / pandas / bitarray — no
rpy2) that **matches the original numerically**. The block jackknife is exact
arithmetic and the weighted regressions are deterministic, so estimates agree
with `ldsc` to floating-point tolerance.

## Installation

```bash
pip install pyldsc
```

From source:

```bash
git clone https://github.com/omicverse/py-ldsc
cd py-ldsc
pip install -e .
```

Requires Python ≥ 3.9 and `numpy`, `scipy`, `pandas`, `bitarray`.

## What it covers

| Capability | Function / class | Original LDSC flag |
|---|---|---|
| LD Score estimation | `estimate_ldscore` | `--l2` |
| SNP-heritability | `estimate_h2`, `Hsq` | `--h2` |
| Partitioned / stratified h2 | `partitioned_h2` | `--h2 --overlap-annot` |
| Genetic correlation | `estimate_rg`, `RG`, `Gencov` | `--rg` |
| LDSC-SEG cell-type enrichment | `ldsc_seg` | `--h2-cts` |
| Munge raw GWAS sumstats | `munge_sumstats` | `munge_sumstats.py` |
| Block jackknife | `LstsqJackknifeFast`, `RatioJackknife` | — |
| IRWLS | `IRWLS` | — |
| PLINK `.bed` reader | `PlinkBEDFile` | — |

All the LDSC machinery is reproduced: windowed bias-corrected LD Scores
(`--ld-wind-cm/-kb/-snps`), the regression weights, the two-step estimator,
chi-square filtering (`--chisq-max`), block-jackknife standard errors, the
intercept / ratio, constrained intercepts (`--intercept-h2`,
`--intercept-gencov`, `--no-intercept`), the `--overlap-annot` correction and
per-category enrichment / coefficient z-scores.

## Python API

```python
import pyldsc

# 1. Estimate LD Scores from a PLINK reference panel
res = pyldsc.estimate_ldscore("reference", ld_wind_cm=1.0, out="reference")
res["ldscore"]   # DataFrame: CHR, SNP, BP, L2
res["M"], res["M_5_50"]

# 2. SNP-heritability from GWAS summary statistics
h2 = pyldsc.estimate_h2("trait.sumstats", ref_ld="baseline", w_ld="weights")
h2.tot, h2.tot_se          # heritability +/- jackknife SE
h2.intercept, h2.ratio     # LD Score regression intercept and ratio
print(h2.summary())

# 3. Partitioned / stratified heritability (--overlap-annot)
part = pyldsc.partitioned_h2("trait.sumstats", ref_ld="baseline.",
                             w_ld="weights.", frqfile="frq.")
part.overlap_results       # per-category enrichment / coefficient table

# 4. Genetic correlation between traits
rg = pyldsc.estimate_rg(["trait1.sumstats", "trait2.sumstats"],
                        ref_ld="baseline", w_ld="weights")
rg[0].rg_ratio, rg[0].rg_se, rg[0].p

# 5. LDSC-SEG cell-type / tissue enrichment
seg = pyldsc.ldsc_seg("trait.sumstats", ref_ld_cts="celltypes.ldcts",
                      ref_ld="baseline.", w_ld="weights.")
seg     # DataFrame: Name, Coefficient, Coefficient_std_error, P_value

# 6. Harmonise raw GWAS summary statistics
ss = pyldsc.munge_sumstats("raw_gwas.txt", out="trait", N=100000,
                           merge_alleles="w_hm3.snplist")
```

## Command-line interface

A thin `pyldsc` CLI mirrors the original `ldsc.py` flags:

```bash
pyldsc l2     --bfile reference --ld-wind-cm 1 --out reference
pyldsc h2     --h2 trait.sumstats --ref-ld baseline --w-ld weights --out trait
pyldsc rg     --rg t1.sumstats,t2.sumstats --ref-ld baseline --w-ld weights
pyldsc h2-cts --h2-cts trait.sumstats --ref-ld-chr-cts celltypes.ldcts \
              --ref-ld baseline. --w-ld weights.
pyldsc munge  --sumstats raw_gwas.txt --N 100000 --out trait
```

## Numerical parity

`tests/test_parity.py` drives the test data bundled with the upstream `ldsc`
repository through `pyldsc` and asserts numerical agreement:

* deterministic h2 / intercept / rg / jackknife-SE values on the
  `simulate_test` data reproduce to `rtol = 1e-6`;
* the exact-equality identities from the original suite hold
  (`test_twostep_h2`, `test_h2_M`, `test_rg_M`, `test_read_annot`, PLINK
  `.bed` parsing);
* the statistical-property checks of `ldsc/test/test_sumstats.py`
  (`Test_H2_Statistical`) hold — averaged over simulated GWAS the estimator
  is unbiased: mean h2 ≈ 0.9, mean per-category h2 ≈ (0.3, 0.6),
  intercept ≈ 1.

```bash
pytest tests/ -q
```

## Examples

* `examples/benchmark.py` — timing of the LDSC workflows on the bundled data.
* `examples/compare_reference.ipynb` — side-by-side comparison of `pyldsc`
  against the original LDSC reference values, with a regression-fit plot and
  an LDSC-SEG cell-type enrichment bar plot.

## License

GPL-3.0, matching the upstream `ldsc`. The original LDSC was written by
Brendan Bulik-Sullivan and Hilary Finucane; `pyldsc` is an independent
modern-Python port for the [omicverse](https://github.com/Starlitnightly/omicverse)
project.
