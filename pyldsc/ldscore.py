"""LD Score estimation from PLINK ``.bed/.bim/.fam`` reference genotypes.

Faithful modern-Python port of ``ldsc/ldscore/ldscore.py``
(Bulik-Sullivan & Finucane, 2014).

The :class:`PlinkBEDFile` class reads a PLINK ``.bed`` file into memory and
computes windowed, bias-corrected per-SNP LD Scores via
:meth:`PlinkBEDFile.ldScoreVarBlocks`.
"""
from __future__ import annotations

import bitarray as ba
import numpy as np

__all__ = [
    "getBlockLefts",
    "block_left_to_right",
    "PlinkBEDFile",
]


def getBlockLefts(coords, max_dist):
    """Convert sorted coordinates + a window width to per-SNP block lefts.

    ``block_left[j] = min{k : dist(j, k) < max_dist}``.
    """
    M = len(coords)
    j = 0
    block_left = np.zeros(M)
    for i in range(M):
        while j < M and abs(coords[j] - coords[i]) > max_dist:
            j += 1
        block_left[i] = j
    return block_left


def block_left_to_right(block_left):
    """Convert block lefts to block rights.

    ``block_right[j] = max{k : block_left[k] <= j}``.
    """
    M = len(block_left)
    j = 0
    block_right = np.zeros(M)
    for i in range(M):
        while j < M and block_left[j] <= i:
            j += 1
        block_right[i] = j
    return block_right


class __GenotypeArrayInMemory__(object):
    """Parent class for in-memory genotype matrices (e.g. PLINK ``.bed``)."""

    def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None,
                 mafMin=None):
        self.m = len(snp_list.IDList)
        self.n = n
        self.keep_snps = keep_snps
        self.keep_indivs = keep_indivs
        self.df = np.array(snp_list.df[["CHR", "SNP", "BP", "CM"]])
        self.colnames = ["CHR", "SNP", "BP", "CM"]
        self.mafMin = mafMin if mafMin is not None else 0
        self._currentSNP = 0
        (self.nru, self.geno) = self.__read__(fname, self.m, n)

        if keep_indivs is not None:
            keep_indivs = np.array(keep_indivs, dtype="int")
            if np.any(keep_indivs > self.n):
                raise ValueError("keep_indivs indices out of bounds")
            (self.geno, self.m, self.n) = self.__filter_indivs__(
                self.geno, keep_indivs, self.m, self.n
            )
            if self.n > 0:
                print("After filtering, {n} individuals remain".format(n=self.n))
            else:
                raise ValueError("After filtering, no individuals remain")

        if keep_snps is not None:
            keep_snps = np.array(keep_snps, dtype="int")
            if np.any(keep_snps > self.m):
                raise ValueError("keep_snps indices out of bounds")

        (self.geno, self.m, self.n, self.kept_snps, self.freq) = \
            self.__filter_snps_maf__(
                self.geno, self.m, self.n, self.mafMin, keep_snps
            )

        if self.m > 0:
            print("After filtering, {m} SNPs remain".format(m=self.m))
        else:
            raise ValueError("After filtering, no SNPs remain")

        self.df = self.df[self.kept_snps, :]
        self.maf = np.minimum(self.freq, np.ones(self.m) - self.freq)
        self.sqrtpq = np.sqrt(self.freq * (np.ones(self.m) - self.freq))
        self.df = np.c_[self.df, self.maf]
        self.colnames.append("MAF")

    def __read__(self, fname, m, n):
        raise NotImplementedError

    def __filter_indivs__(self, geno, keep_indivs, m, n):
        raise NotImplementedError

    def __filter_snps_maf__(self, geno, m, n, mafMin, keep_snps):
        raise NotImplementedError

    def ldScoreVarBlocks(self, block_left, c, annot=None):
        """Compute an unbiased estimate of L2(j) for each SNP j."""
        func = lambda x: self.__l2_unbiased__(x, self.n)
        snp_getter = self.nextSNPs
        return self.__corSumVarBlocks__(block_left, c, func, snp_getter, annot)

    def __l2_unbiased__(self, x, n):
        """Apply the LDSC bias correction to squared correlations."""
        denom = n - 2 if n > 2 else n  # allow n<2 for testing
        sq = np.square(x)
        return sq - (1 - sq) / denom

    def __corSumVarBlocks__(self, block_left, c, func, snp_getter, annot=None):
        """Windowed sum of (bias-corrected) squared correlations."""
        m, n = self.m, self.n
        block_sizes = np.array(np.arange(m) - block_left)
        block_sizes = np.ceil(block_sizes / c) * c
        if annot is None:
            annot = np.ones((m, 1))
        else:
            annot_m = annot.shape[0]
            if annot_m != self.m:
                raise ValueError("Incorrect number of SNPs in annot")

        n_a = annot.shape[1]
        cor_sum = np.zeros((m, n_a))
        b = np.nonzero(block_left > 0)
        if np.any(b):
            b = b[0][0]
        else:
            b = m
        b = int(np.ceil(b / c) * c)
        if b > m:
            c = 1
            b = m
        l_A = 0
        A = snp_getter(b)
        rfuncAB = np.zeros((b, c))
        rfuncBB = np.zeros((c, c))
        for l_B in range(0, b, c):
            B = A[:, l_B:l_B + c]
            np.dot(A.T, B / n, out=rfuncAB)
            rfuncAB = func(rfuncAB)
            cor_sum[l_A:l_A + b, :] += np.dot(rfuncAB, annot[l_B:l_B + c, :])
        b0 = b
        md = int(c * np.floor(m / c))
        end = md + 1 if md != m else md
        for l_B in range(b0, end, c):
            old_b = b
            b = int(block_sizes[l_B])
            if l_B > b0 and b > 0:
                A = np.hstack((A[:, old_b - b + c:old_b], B))
                l_A += old_b - b + c
            elif l_B == b0 and b > 0:
                A = A[:, b0 - b:b0]
                l_A = b0 - b
            elif b == 0:
                A = np.array(()).reshape((n, 0))
                l_A = l_B
            if l_B == md:
                c = m - md
                rfuncAB = np.zeros((b, c))
                rfuncBB = np.zeros((c, c))
            if b != old_b:
                rfuncAB = np.zeros((b, c))

            B = snp_getter(c)
            p1 = np.all(annot[l_A:l_A + b, :] == 0)
            p2 = np.all(annot[l_B:l_B + c, :] == 0)
            if p1 and p2:
                continue

            np.dot(A.T, B / n, out=rfuncAB)
            rfuncAB = func(rfuncAB)
            cor_sum[l_A:l_A + b, :] += np.dot(rfuncAB, annot[l_B:l_B + c, :])
            cor_sum[l_B:l_B + c, :] += np.dot(
                annot[l_A:l_A + b, :].T, rfuncAB
            ).T
            np.dot(B.T, B / n, out=rfuncBB)
            rfuncBB = func(rfuncBB)
            cor_sum[l_B:l_B + c, :] += np.dot(rfuncBB, annot[l_B:l_B + c, :])

        return cor_sum


class PlinkBEDFile(__GenotypeArrayInMemory__):
    """In-memory reader for the PLINK ``.bed`` genotype format."""

    def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None,
                 mafMin=None):
        self._bedcode = {
            2: ba.bitarray("11"),
            9: ba.bitarray("10"),
            1: ba.bitarray("01"),
            0: ba.bitarray("00"),
        }
        __GenotypeArrayInMemory__.__init__(
            self, fname, n, snp_list, keep_snps=keep_snps,
            keep_indivs=keep_indivs, mafMin=mafMin,
        )

    def __read__(self, fname, m, n):
        if not fname.endswith(".bed"):
            raise ValueError(".bed filename must end in .bed")

        fh = open(fname, "rb")
        magicNumber = ba.bitarray(endian="little")
        magicNumber.fromfile(fh, 2)
        bedMode = ba.bitarray(endian="little")
        bedMode.fromfile(fh, 1)
        e = (4 - n % 4) if n % 4 != 0 else 0
        nru = n + e
        self.nru = nru
        if magicNumber != ba.bitarray("0011011011011000"):
            raise IOError("Magic number from Plink .bed file not recognized")
        if bedMode != ba.bitarray("10000000"):
            raise IOError("Plink .bed file must be in default SNP-major mode")

        self.geno = ba.bitarray(endian="little")
        self.geno.fromfile(fh)
        fh.close()
        self.__test_length__(self.geno, self.m, self.nru)
        return (self.nru, self.geno)

    def __test_length__(self, geno, m, nru):
        exp_len = 2 * m * nru
        real_len = len(geno)
        if real_len != exp_len:
            s = "Plink .bed file has {n1} bits, expected {n2}"
            raise IOError(s.format(n1=real_len, n2=exp_len))

    def __filter_indivs__(self, geno, keep_indivs, m, n):
        n_new = len(keep_indivs)
        e = (4 - n_new % 4) if n_new % 4 != 0 else 0
        nru_new = n_new + e
        nru = self.nru
        z = ba.bitarray(m * 2 * nru_new, endian="little")
        z.setall(0)
        for e, i in enumerate(keep_indivs):
            z[2 * e::2 * nru_new] = geno[2 * i::2 * nru]
            z[2 * e + 1::2 * nru_new] = geno[2 * i + 1::2 * nru]
        self.nru = nru_new
        return (z, m, n_new)

    def __filter_snps_maf__(self, geno, m, n, mafMin, keep_snps):
        """Filter SNPs on MAF (Chris Chang / PLINK2 bit-counting algorithm)."""
        nru = self.nru
        m_poly = 0
        y = ba.bitarray()
        if keep_snps is None:
            keep_snps = range(m)
        kept_snps = []
        freq = []
        for e, j in enumerate(keep_snps):
            z = geno[2 * nru * j:2 * nru * (j + 1)]
            A = z[0::2]
            a = A.count()
            B = z[1::2]
            b = B.count()
            c = (A & B).count()
            major_ct = b + c
            n_nomiss = n - a + c
            f = major_ct / (2 * n_nomiss) if n_nomiss > 0 else 0
            het_miss_ct = a + b - 2 * c
            if np.minimum(f, 1 - f) > mafMin and het_miss_ct < n:
                freq.append(f)
                y += z
                m_poly += 1
                kept_snps.append(j)
        return (y, m_poly, n, kept_snps, freq)

    def nextSNPs(self, b, minorRef=None):
        """Return an ``(n, b)`` matrix of normalised genotypes for ``b`` SNPs."""
        try:
            b = int(b)
            if b <= 0:
                raise ValueError("b must be > 0")
        except TypeError:
            raise TypeError("b must be an integer")

        if self._currentSNP + b > self.m:
            s = "{b} SNPs requested, {k} SNPs remain"
            raise ValueError(s.format(b=b, k=(self.m - self._currentSNP)))

        c = self._currentSNP
        n = self.n
        nru = self.nru
        sl = self.geno[2 * c * nru:2 * (c + b) * nru]
        X = np.array(list(sl.decode(self._bedcode)), dtype="float64").reshape(
            (b, nru)
        ).T
        X = X[0:n, :]
        Y = np.zeros(X.shape)
        for j in range(0, b):
            newsnp = X[:, j]
            ii = newsnp != 9
            avg = np.mean(newsnp[ii])
            newsnp[np.logical_not(ii)] = avg
            denom = np.std(newsnp)
            if denom == 0:
                denom = 1
            if minorRef is not None and self.freq[self._currentSNP + j] > 0.5:
                denom = denom * -1
            Y[:, j] = (newsnp - avg) / denom

        self._currentSNP += b
        return Y
