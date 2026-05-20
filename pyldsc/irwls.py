"""Iteratively re-weighted least squares.

Faithful modern-Python port of ``ldsc/ldscore/irwls.py``
(Bulik-Sullivan & Finucane, 2015).

The :class:`IRWLS` class performs two rounds of re-weighting (matching the
original ``ldsc`` behaviour exactly) and then runs a block jackknife with the
final weights to obtain standard errors.
"""
from __future__ import annotations

import numpy as np

from . import jackknife as jk

__all__ = ["IRWLS"]


class IRWLS(object):
    """Iteratively re-weighted least squares (FWLS).

    Parameters
    ----------
    x : np.ndarray, shape (n, p)
        Independent variable.
    y : np.ndarray, shape (n, 1)
        Dependent variable.
    update_func : callable
        Transforms the output of ``np.linalg.lstsq`` into new weights.
    n_blocks : int
        Number of jackknife blocks.
    w : np.ndarray, shape (n, 1), optional
        Initial regression weights (inverse-CVF scale).  Default is ones.
    slow : bool, optional
        Use the slow (delete-block) jackknife.  Mostly for testing.
    separators : list of int, optional
        Explicit block boundaries.
    """

    def __init__(self, x, y, update_func, n_blocks, w=None, slow=False,
                 separators=None):
        n, p = jk._check_shape(x, y)
        if w is None:
            w = np.ones_like(y)
        if w.shape != (n, 1):
            raise ValueError(
                "w has shape {S}. w must have shape ({N}, 1).".format(S=w.shape, N=n)
            )

        jknife = self.irwls(
            x, y, update_func, n_blocks, w, slow=slow, separators=separators
        )
        self.est = jknife.est
        self.jknife_se = jknife.jknife_se
        self.jknife_est = jknife.jknife_est
        self.jknife_var = jknife.jknife_var
        self.jknife_cov = jknife.jknife_cov
        self.delete_values = jknife.delete_values
        self.separators = jknife.separators

    @classmethod
    def irwls(cls, x, y, update_func, n_blocks, w, slow=False, separators=None):
        """Run two rounds of re-weighting, then a final block jackknife."""
        (n, p) = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                "y has shape {S}. y must have shape ({N}, 1).".format(S=y.shape, N=n)
            )
        if w.shape != (n, 1):
            raise ValueError(
                "w has shape {S}. w must have shape ({N}, 1).".format(S=w.shape, N=n)
            )

        w = np.sqrt(w)
        for _ in range(2):  # two re-weighting iterations (matches ldsc)
            new_w = np.sqrt(update_func(cls.wls(x, y, w)))
            if new_w.shape != w.shape:
                raise ValueError("New weights must have same shape.")
            w = new_w

        x = cls._weight(x, w)
        y = cls._weight(y, w)
        if slow:
            jknife = jk.LstsqJackknifeSlow(x, y, n_blocks, separators=separators)
        else:
            jknife = jk.LstsqJackknifeFast(x, y, n_blocks, separators=separators)
        return jknife

    @classmethod
    def wls(cls, x, y, w):
        """Weighted least squares; returns the ``np.linalg.lstsq`` tuple."""
        (n, p) = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                "y has shape {S}. y must have shape ({N}, 1).".format(S=y.shape, N=n)
            )
        if w.shape != (n, 1):
            raise ValueError(
                "w has shape {S}. w must have shape ({N}, 1).".format(S=w.shape, N=n)
            )
        x = cls._weight(x, w)
        y = cls._weight(y, w)
        coef = np.linalg.lstsq(x, y, rcond=None)
        return coef

    @classmethod
    def _weight(cls, x, w):
        """Weight ``x`` row-wise by ``w`` (normalised to sum 1)."""
        if np.any(w <= 0):
            raise ValueError("Weights must be > 0")
        (n, p) = x.shape
        if w.shape != (n, 1):
            raise ValueError(
                "w has shape {S}. w must have shape (n, 1).".format(S=w.shape)
            )
        w = w / float(np.sum(w))
        x_new = np.multiply(x, w)
        return x_new
