"""Block jackknife estimators.

Faithful modern-Python port of ``ldsc/ldscore/jackknife.py``
(Bulik-Sullivan & Finucane, 2014).

Everything in this module deals with 2D numpy arrays.  1D data are
represented as arrays with shape ``(N, 1)`` or ``(1, N)`` -- the first
dimension is the number of data points (or blocks), the second is the
dimensionality of the data.

Classes
-------
* :class:`Jackknife`           -- base class.
* :class:`LstsqJackknifeSlow`  -- delete-block least squares (incl. NNLS).
* :class:`LstsqJackknifeFast`  -- fast block jackknife for linear regression.
* :class:`RatioJackknife`      -- block jackknife of a ratio estimator.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

__all__ = [
    "Jackknife",
    "LstsqJackknifeSlow",
    "LstsqJackknifeFast",
    "RatioJackknife",
]


def _check_shape(x, y):
    """Check that ``x`` and ``y`` have the correct shapes for regression."""
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError("x and y must be 2D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Number of datapoints in x != number of datapoints in y.")
    if y.shape[1] != 1:
        raise ValueError("y must have shape (n_snp, 1)")
    n, p = x.shape
    if p > n:
        raise ValueError("More dimensions than datapoints.")
    return (n, p)


def _check_shape_block(xty_block_values, xtx_block_values):
    """Check that the per-block ``X'y`` / ``X'X`` arrays have correct shapes."""
    if xtx_block_values.shape[0:2] != xty_block_values.shape:
        raise ValueError(
            "Shape of xty_block_values must equal shape of first two "
            "dimensions of xtx_block_values."
        )
    if len(xtx_block_values.shape) < 3:
        raise ValueError("xtx_block_values must be a 3D array.")
    if xtx_block_values.shape[1] != xtx_block_values.shape[2]:
        raise ValueError("Last two axes of xtx_block_values must have same dimension.")
    return xtx_block_values.shape[0:2]


class Jackknife(object):
    """Base class for jackknife objects.

    Parameters
    ----------
    x : np.ndarray, shape (n, p)
        Independent variable.
    y : np.ndarray, shape (n, 1)
        Dependent variable.
    n_blocks : int, optional
        Number of jackknife blocks.
    separators : list of int, optional
        Explicit block boundaries (overrides ``n_blocks``).
    """

    def __init__(self, x, y, n_blocks=None, separators=None):
        self.N, self.p = _check_shape(x, y)
        if separators is not None:
            if max(separators) != self.N:
                raise ValueError(
                    "Max(separators) must be equal to number of data points."
                )
            if min(separators) != 0:
                raise ValueError("Min(separators) must be equal to 0.")
            self.separators = sorted(separators)
            self.n_blocks = len(separators) - 1
        elif n_blocks is not None:
            self.n_blocks = n_blocks
            self.separators = self.get_separators(self.N, self.n_blocks)
        else:
            raise ValueError("Must specify either n_blocks or separators.")

        if self.n_blocks > self.N:
            raise ValueError("More blocks than data points.")

    @classmethod
    def jknife(cls, pseudovalues):
        """Convert pseudovalues to jackknife estimate, variance, SE and cov."""
        n_blocks = pseudovalues.shape[0]
        jknife_cov = np.atleast_2d(np.cov(pseudovalues.T, ddof=1) / n_blocks)
        jknife_var = np.atleast_2d(np.diag(jknife_cov))
        jknife_se = np.atleast_2d(np.sqrt(jknife_var))
        jknife_est = np.atleast_2d(np.mean(pseudovalues, axis=0))
        return (jknife_est, jknife_var, jknife_se, jknife_cov)

    @classmethod
    def delete_values_to_pseudovalues(cls, delete_values, est):
        """Convert whole-data estimate and delete values to pseudovalues."""
        n_blocks, p = delete_values.shape
        if est.shape != (1, p):
            raise ValueError(
                "Different number of parameters in delete_values than in est."
            )
        return n_blocks * est - (n_blocks - 1) * delete_values

    @classmethod
    def get_separators(cls, N, n_blocks):
        """Define (approximately) evenly-spaced block boundaries."""
        return np.floor(np.linspace(0, N, n_blocks + 1)).astype(int)


class LstsqJackknifeSlow(Jackknife):
    """Slow linear-regression block jackknife.

    Computes delete values directly by deleting one block at a time.
    Useful for testing and for non-negative least squares (which does not
    admit a fast block jackknife algorithm).
    """

    def __init__(self, x, y, n_blocks=None, nn=False, separators=None):
        Jackknife.__init__(self, x, y, n_blocks, separators)
        if nn:  # non-negative least squares
            func = lambda xx, yy: np.atleast_2d(nnls(xx, np.array(yy).T[0])[0])
        else:
            func = lambda xx, yy: np.atleast_2d(
                np.linalg.lstsq(xx, np.array(yy).T[0], rcond=None)[0]
            )

        self.est = func(x, y)
        self.delete_values = self._delete_values(x, y, func, self.separators)
        self.pseudovalues = self.delete_values_to_pseudovalues(
            self.delete_values, self.est
        )
        (
            self.jknife_est,
            self.jknife_var,
            self.jknife_se,
            self.jknife_cov,
        ) = self.jknife(self.pseudovalues)

    @classmethod
    def _delete_values(cls, x, y, func, s):
        """Compute delete values by deleting one block at a time."""
        _check_shape(x, y)
        d = [
            func(
                np.vstack([x[0:s[i], ...], x[s[i + 1]:, ...]]),
                np.vstack([y[0:s[i], ...], y[s[i + 1]:, ...]]),
            )
            for i in range(len(s) - 1)
        ]
        return np.concatenate(d, axis=0)


class LstsqJackknifeFast(Jackknife):
    """Fast block jackknife for linear regression.

    Builds per-block ``X'y`` / ``X'X`` matrices and forms delete values from
    block totals, avoiding repeated full least-squares solves.
    """

    def __init__(self, x, y, n_blocks=None, separators=None):
        Jackknife.__init__(self, x, y, n_blocks, separators)
        xty, xtx = self.block_values(x, y, self.separators)
        self.est = self.block_values_to_est(xty, xtx)
        self.delete_values = self.block_values_to_delete_values(xty, xtx)
        self.pseudovalues = self.delete_values_to_pseudovalues(
            self.delete_values, self.est
        )
        (
            self.jknife_est,
            self.jknife_var,
            self.jknife_se,
            self.jknife_cov,
        ) = self.jknife(self.pseudovalues)

    @classmethod
    def block_values(cls, x, y, s):
        """Compute per-block ``X'y`` and ``X'X``."""
        n, p = _check_shape(x, y)
        n_blocks = len(s) - 1
        xtx_block_values = np.zeros((n_blocks, p, p))
        xty_block_values = np.zeros((n_blocks, p))
        for i in range(n_blocks):
            xty_block_values[i, ...] = np.dot(
                x[s[i]:s[i + 1], ...].T, y[s[i]:s[i + 1], ...]
            ).reshape((1, p))
            xtx_block_values[i, ...] = np.dot(
                x[s[i]:s[i + 1], ...].T, x[s[i]:s[i + 1], ...]
            )
        return (xty_block_values, xtx_block_values)

    @classmethod
    def block_values_to_est(cls, xty_block_values, xtx_block_values):
        """Convert block values to the whole-data regression estimate."""
        n_blocks, p = _check_shape_block(xty_block_values, xtx_block_values)
        xty = np.sum(xty_block_values, axis=0)
        xtx = np.sum(xtx_block_values, axis=0)
        return np.linalg.solve(xtx, xty).reshape((1, p))

    @classmethod
    def block_values_to_delete_values(cls, xty_block_values, xtx_block_values):
        """Convert block values to jackknife delete values."""
        n_blocks, p = _check_shape_block(xty_block_values, xtx_block_values)
        delete_values = np.zeros((n_blocks, p))
        xty_tot = np.sum(xty_block_values, axis=0)
        xtx_tot = np.sum(xtx_block_values, axis=0)
        for j in range(n_blocks):
            delete_xty = xty_tot - xty_block_values[j]
            delete_xtx = xtx_tot - xtx_block_values[j]
            delete_values[j, ...] = np.linalg.solve(delete_xtx, delete_xty).reshape(
                (1, p)
            )
        return delete_values


class RatioJackknife(Jackknife):
    """Block jackknife of a ratio estimator.

    Parameters
    ----------
    est : np.ndarray, shape (1, p)
        Whole-data ratio estimate.
    numer_delete_values : np.ndarray, shape (n_blocks, p)
        Delete values for the numerator.
    denom_delete_values : np.ndarray, shape (n_blocks, p)
        Delete values for the denominator.
    """

    def __init__(self, est, numer_delete_values, denom_delete_values):
        if numer_delete_values.shape != denom_delete_values.shape:
            raise ValueError(
                "numer_delete_values.shape != denom_delete_values.shape."
            )
        if len(numer_delete_values.shape) != 2:
            raise ValueError("Delete values must be matrices.")
        if (
            len(est.shape) != 2
            or est.shape[0] != 1
            or est.shape[1] != numer_delete_values.shape[1]
        ):
            raise ValueError("Shape of est does not match shape of delete values.")

        self.n_blocks = numer_delete_values.shape[0]
        self.est = est
        self.pseudovalues = self.delete_values_to_pseudovalues(
            self.est, denom_delete_values, numer_delete_values
        )
        (
            self.jknife_est,
            self.jknife_var,
            self.jknife_se,
            self.jknife_cov,
        ) = self.jknife(self.pseudovalues)

    @classmethod
    def delete_values_to_pseudovalues(cls, est, denom, numer):
        """Convert ratio delete values to pseudovalues."""
        n_blocks, p = denom.shape
        pseudovalues = np.zeros((n_blocks, p))
        for j in range(n_blocks):
            pseudovalues[j, ...] = (
                n_blocks * est - (n_blocks - 1) * numer[j, ...] / denom[j, ...]
            )
        return pseudovalues
