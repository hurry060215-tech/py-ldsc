"""LD Score regression estimators of heritability and genetic correlation.

Faithful modern-Python port of ``ldsc/ldscore/regressions.py``
(Bulik-Sullivan & Finucane, 2015; Finucane et al., 2015/2018).

Shape convention is ``(n_snp, n_annot)`` for all classes; the last column of
the design matrix is the intercept (when a free intercept is fit).

Classes
-------
* :class:`LD_Score_Regression` -- shared base class.
* :class:`Hsq`                 -- univariate / partitioned SNP-heritability.
* :class:`Gencov`              -- genetic covariance.
* :class:`RG`                  -- genetic correlation.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.stats import t as tdist

from . import jackknife as jk
from .irwls import IRWLS

__all__ = [
    "LD_Score_Regression",
    "Hsq",
    "Gencov",
    "RG",
    "h2_obs_to_liab",
    "gencov_obs_to_liab",
    "p_z_norm",
    "append_intercept",
    "remove_intercept",
    "remove_brackets",
    "update_separators",
]


def remove_brackets(x):
    """Strip brackets and trailing whitespace from a numpy array printout."""
    return x.replace("[", "").replace("]", "").strip()


def _s(x):
    """Pretty-print a numpy array (mirrors ldsc's ``s`` lambda)."""
    return remove_brackets(str(np.atleast_2d(np.asarray(x))))


def update_separators(s, ii):
    """Map masked separators ``s`` back onto the unmasked index space."""
    maplist = np.arange(len(ii))[np.squeeze(ii)]
    t = np.array([maplist[i] for i in s[1:-1]])
    t = np.hstack(((0), t, (len(ii))))
    return t


def p_z_norm(est, se):
    """Convert estimate and SE to a Z-score and a (two-sided) P-value."""
    try:
        Z = est / se
    except (FloatingPointError, ZeroDivisionError):
        Z = float("inf")
    if se == 0:
        Z = float("inf")
    P = chi2.sf(Z ** 2, 1, loc=0, scale=1)  # 0 if Z = inf
    return P, Z


def append_intercept(x):
    """Append a column of ones (intercept term) to a design matrix."""
    n_row = x.shape[0]
    intercept = np.ones((n_row, 1))
    return np.concatenate((x, intercept), axis=1)


def remove_intercept(x):
    """Drop the last column of a matrix."""
    n_col = x.shape[1]
    return x[:, 0:n_col - 1]


def gencov_obs_to_liab(gencov_obs, P1, P2, K1, K2):
    """Convert observed-scale genetic covariance to the liability scale."""
    c1 = 1
    c2 = 1
    if P1 is not None and K1 is not None:
        c1 = np.sqrt(h2_obs_to_liab(1, P1, K1))
    if P2 is not None and K2 is not None:
        c2 = np.sqrt(h2_obs_to_liab(1, P2, K2))
    return gencov_obs * c1 * c2


def h2_obs_to_liab(h2_obs, P, K):
    """Convert observed-scale heritability to the liability scale.

    Parameters
    ----------
    h2_obs : float
        Heritability on the observed scale in an ascertained sample.
    P : float in (0, 1)
        Prevalence of the phenotype in the sample.
    K : float in (0, 1)
        Prevalence of the phenotype in the population.
    """
    if np.isnan(P) and np.isnan(K):
        return h2_obs
    if K <= 0 or K >= 1:
        raise ValueError("K must be in the range (0,1)")
    if P <= 0 or P >= 1:
        raise ValueError("P must be in the range (0,1)")

    thresh = norm.isf(K)
    conversion_factor = (
        K ** 2 * (1 - K) ** 2 / (P * (1 - P) * norm.pdf(thresh) ** 2)
    )
    return h2_obs * conversion_factor


class LD_Score_Regression(object):
    """Shared machinery for univariate / partitioned LD Score regression."""

    def __init__(self, y, x, w, N, M, n_blocks, intercept=None, slow=False,
                 step1_ii=None, old_weights=False):
        for i in [y, x, w, M, N]:
            try:
                if len(i.shape) != 2:
                    raise TypeError("Arguments must be 2D arrays.")
            except AttributeError:
                raise TypeError("Arguments must be arrays.")

        n_snp, self.n_annot = x.shape
        if any(i.shape != (n_snp, 1) for i in [y, w, N]):
            raise ValueError(
                "N, weights and response (z1z2 or chisq) must have shape "
                "(n_snp, 1)."
            )
        if M.shape != (1, self.n_annot):
            raise ValueError("M must have shape (1, n_annot).")

        M_tot = float(np.sum(M))
        x_tot = np.sum(x, axis=1).reshape((n_snp, 1))
        self.constrain_intercept = intercept is not None
        self.intercept = intercept
        self.n_blocks = n_blocks
        tot_agg = self.aggregate(y, x_tot, N, M_tot, intercept)
        initial_w = self._update_weights(x_tot, w, N, M_tot, tot_agg, intercept)
        Nbar = np.mean(N)  # keep condition number low
        x = np.multiply(N, x) / Nbar
        if not self.constrain_intercept:
            x, x_tot = append_intercept(x), append_intercept(x_tot)
            yp = y
        else:
            yp = y - intercept
            self.intercept_se = "NA"
        del y
        self.twostep_filtered = None
        if step1_ii is not None and self.constrain_intercept:
            raise ValueError("twostep is not compatible with constrain_intercept.")
        elif step1_ii is not None and self.n_annot > 1:
            raise ValueError(
                "twostep not compatible with partitioned LD Score yet."
            )
        elif step1_ii is not None:
            n1 = np.sum(step1_ii)
            self.twostep_filtered = n_snp - n1
            x1 = x[np.squeeze(step1_ii), :]
            yp1, w1, N1, initial_w1 = map(
                lambda a: a[step1_ii].reshape((n1, 1)),
                (yp, w, N, initial_w),
            )
            update_func1 = lambda a: self._update_func(
                a, x1, w1, N1, M_tot, Nbar, ii=step1_ii
            )
            step1_jknife = IRWLS(
                x1, yp1, update_func1, n_blocks, slow=slow, w=initial_w1
            )
            step1_int, _ = self._intercept(step1_jknife)
            yp = yp - step1_int
            x = remove_intercept(x)
            x_tot = remove_intercept(x_tot)
            update_func2 = lambda a: self._update_func(
                a, x_tot, w, N, M_tot, Nbar, step1_int
            )
            s = update_separators(step1_jknife.separators, step1_ii)
            step2_jknife = IRWLS(
                x, yp, update_func2, n_blocks, slow=slow, w=initial_w, separators=s
            )
            c = np.sum(np.multiply(initial_w, x)) / np.sum(
                np.multiply(initial_w, np.square(x))
            )
            jknife = self._combine_twostep_jknives(
                step1_jknife, step2_jknife, M_tot, c, Nbar
            )
        elif old_weights:
            initial_w = np.sqrt(initial_w)
            x = IRWLS._weight(x, initial_w)
            y = IRWLS._weight(yp, initial_w)
            jknife = jk.LstsqJackknifeFast(x, y, n_blocks)
        else:
            update_func = lambda a: self._update_func(
                a, x_tot, w, N, M_tot, Nbar, intercept
            )
            jknife = IRWLS(x, yp, update_func, n_blocks, slow=slow, w=initial_w)

        self.coef, self.coef_cov, self.coef_se = self._coef(jknife, Nbar)
        self.cat, self.cat_cov, self.cat_se = self._cat(
            jknife, M, Nbar, self.coef, self.coef_cov
        )
        self.tot, self.tot_cov, self.tot_se = self._tot(self.cat, self.cat_cov)
        self.prop, self.prop_cov, self.prop_se = self._prop(
            jknife, M, Nbar, self.cat, self.tot
        )
        self.enrichment, self.M_prop = self._enrichment(
            M, M_tot, self.cat, self.tot
        )
        if not self.constrain_intercept:
            self.intercept, self.intercept_se = self._intercept(jknife)

        self.jknife = jknife
        self.tot_delete_values = self._delete_vals_tot(jknife, Nbar, M)
        self.part_delete_values = self._delete_vals_part(jknife, Nbar, M)
        if not self.constrain_intercept:
            self.intercept_delete_values = jknife.delete_values[:, self.n_annot]

        self.M = M

    @classmethod
    def aggregate(cls, y, x, N, M, intercept=None):
        """Aggregate (method-of-moments) estimate used to seed the weights."""
        if intercept is None:
            intercept = cls.__null_intercept__
        num = M * (np.mean(y) - intercept)
        denom = np.mean(np.multiply(x, N))
        return num / denom

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None,
                     ii=None):
        raise NotImplementedError

    def _delete_vals_tot(self, jknife, Nbar, M):
        """Delete values for total h2 / gencov."""
        n_annot = self.n_annot
        tot_delete_vals = jknife.delete_values[:, 0:n_annot]
        tot_delete_vals = np.dot(tot_delete_vals, M.T) / Nbar
        return tot_delete_vals

    def _delete_vals_part(self, jknife, Nbar, M):
        """Delete values for partitioned h2 / gencov."""
        n_annot = self.n_annot
        return jknife.delete_values[:, 0:n_annot] / Nbar

    def _coef(self, jknife, Nbar):
        """Coefficient estimates and covariance from the jackknife."""
        n_annot = self.n_annot
        coef = jknife.est[0, 0:n_annot] / Nbar
        coef_cov = jknife.jknife_cov[0:n_annot, 0:n_annot] / Nbar ** 2
        coef_se = np.sqrt(np.diag(coef_cov))
        return coef, coef_cov, coef_se

    def _cat(self, jknife, M, Nbar, coef, coef_cov):
        """Convert coefficients to per-category h2 / gencov."""
        cat = np.multiply(M, coef)
        cat_cov = np.multiply(np.dot(M.T, M), coef_cov)
        cat_se = np.sqrt(np.diag(cat_cov))
        return cat, cat_cov, cat_se

    def _tot(self, cat, cat_cov):
        """Convert per-category h2 to total h2 / gencov."""
        tot = np.sum(cat)
        tot_cov = np.sum(cat_cov)
        tot_se = np.sqrt(tot_cov)
        return tot, tot_cov, tot_se

    def _prop(self, jknife, M, Nbar, cat, tot):
        """Convert h2 estimates to per-category proportions of h2."""
        n_annot = self.n_annot
        n_blocks = jknife.delete_values.shape[0]
        numer_delete_vals = (
            np.multiply(M, jknife.delete_values[:, 0:n_annot]) / Nbar
        )
        denom_delete_vals = np.sum(numer_delete_vals, axis=1).reshape(
            (n_blocks, 1)
        )
        denom_delete_vals = np.dot(denom_delete_vals, np.ones((1, n_annot)))
        prop = jk.RatioJackknife(
            cat / tot, numer_delete_vals, denom_delete_vals
        )
        return prop.est, prop.jknife_cov, prop.jknife_se

    def _enrichment(self, M, M_tot, cat, tot):
        """Compute per-category enrichment for h2 / gencov."""
        M_prop = M / M_tot
        enrichment = np.divide(cat, M) / (tot / M_tot)
        return enrichment, M_prop

    def _intercept(self, jknife):
        """Extract intercept and intercept SE from the block jackknife."""
        n_annot = self.n_annot
        intercept = jknife.est[0, n_annot]
        intercept_se = jknife.jknife_se[0, n_annot]
        return intercept, intercept_se

    def _combine_twostep_jknives(self, step1_jknife, step2_jknife, M_tot, c,
                                 Nbar=1):
        """Combine free- and constrained-intercept jackknives (--two-step)."""
        n_blocks, n_annot = step1_jknife.delete_values.shape
        n_annot -= 1
        if n_annot > 2:
            raise ValueError(
                "twostep not yet implemented for partitioned LD Score."
            )
        step1_int, _ = self._intercept(step1_jknife)
        est = np.hstack((step2_jknife.est, np.array(step1_int).reshape((1, 1))))
        delete_values = np.zeros((n_blocks, n_annot + 1))
        delete_values[:, n_annot] = step1_jknife.delete_values[:, n_annot]
        delete_values[:, 0:n_annot] = step2_jknife.delete_values - c * (
            step1_jknife.delete_values[:, n_annot] - step1_int
        ).reshape((n_blocks, n_annot))
        pseudovalues = jk.Jackknife.delete_values_to_pseudovalues(
            delete_values, est
        )
        jknife_est, jknife_var, jknife_se, jknife_cov = jk.Jackknife.jknife(
            pseudovalues
        )
        jknife = namedtuple(
            "jknife",
            ["est", "jknife_se", "jknife_est", "jknife_var", "jknife_cov",
             "delete_values"],
        )
        return jknife(
            est, jknife_se, jknife_est, jknife_var, jknife_cov, delete_values
        )


class Hsq(LD_Score_Regression):
    """Univariate / partitioned SNP-heritability via LD Score regression."""

    __null_intercept__ = 1

    def __init__(self, y, x, w, N, M, n_blocks=200, intercept=None, slow=False,
                 twostep=None, old_weights=False):
        step1_ii = None
        if twostep is not None:
            step1_ii = y < twostep

        LD_Score_Regression.__init__(
            self, y, x, w, N, M, n_blocks, intercept=intercept, slow=slow,
            step1_ii=step1_ii, old_weights=old_weights,
        )
        self.mean_chisq, self.lambda_gc = self._summarize_chisq(y)
        if not self.constrain_intercept:
            self.ratio, self.ratio_se = self._ratio(
                self.intercept, self.intercept_se, self.mean_chisq
            )

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None,
                     ii=None):
        """Update function for IRWLS (heritability)."""
        hsq = M * x[0][0] / Nbar
        if intercept is None:
            intercept = max(x[0][1])  # divide-by-zero error if intercept < 0
        else:
            if ref_ld_tot.shape[1] > 1:
                raise ValueError(
                    "Design matrix has intercept column for constrained "
                    "intercept regression!"
                )
        ld = ref_ld_tot[:, 0].reshape(w_ld.shape)  # remove intercept
        w = self.weights(ld, w_ld, N, M, hsq, intercept, ii)
        return w

    def _summarize_chisq(self, chisq):
        """Compute mean chi^2 and lambda_GC."""
        mean_chisq = np.mean(chisq)
        lambda_gc = np.median(np.asarray(chisq)) / 0.4549
        return mean_chisq, lambda_gc

    def _ratio(self, intercept, intercept_se, mean_chisq):
        """Compute ratio (intercept - 1) / (mean chi^2 - 1)."""
        if mean_chisq > 1:
            ratio_se = intercept_se / (mean_chisq - 1)
            ratio = (intercept - 1) / (mean_chisq - 1)
        else:
            ratio = "NA"
            ratio_se = "NA"
        return ratio, ratio_se

    def _overlap_output(self, category_names, overlap_matrix, M_annot, M_tot,
                        print_coefficients):
        """LD Score regression summary table for overlapping categories."""
        overlap_matrix_prop = np.zeros([self.n_annot, self.n_annot])
        for i in range(self.n_annot):
            overlap_matrix_prop[i, :] = overlap_matrix[i, :] / M_annot

        prop_hsq_overlap = np.dot(
            overlap_matrix_prop, self.prop.T
        ).reshape((1, self.n_annot))
        prop_hsq_overlap_var = np.diag(
            np.dot(np.dot(overlap_matrix_prop, self.prop_cov),
                   overlap_matrix_prop.T)
        )
        prop_hsq_overlap_se = np.sqrt(
            np.maximum(0, prop_hsq_overlap_var)
        ).reshape((1, self.n_annot))
        one_d_convert = lambda x: np.array(x).reshape(np.prod(x.shape))
        prop_M_overlap = M_annot / M_tot
        enrichment = prop_hsq_overlap / prop_M_overlap
        enrichment_se = prop_hsq_overlap_se / prop_M_overlap
        overlap_matrix_diff = np.zeros([self.n_annot, self.n_annot])
        for i in range(self.n_annot):
            if not M_tot == M_annot[0, i]:
                overlap_matrix_diff[i, :] = overlap_matrix[i, :] / M_annot[
                    0, i
                ] - (M_annot - overlap_matrix[i, :]) / (M_tot - M_annot[0, i])

        diff_est = np.dot(overlap_matrix_diff, self.coef)
        diff_cov = np.dot(
            np.dot(overlap_matrix_diff, self.coef_cov), overlap_matrix_diff.T
        )
        diff_se = np.sqrt(np.diag(diff_cov))
        diff_p = [
            "NA"
            if diff_se[i] == 0
            else 2 * tdist.sf(abs(diff_est[i] / diff_se[i]), self.n_blocks)
            for i in range(self.n_annot)
        ]

        df = pd.DataFrame(
            {
                "Category": category_names,
                "Prop._SNPs": one_d_convert(prop_M_overlap),
                "Prop._h2": one_d_convert(prop_hsq_overlap),
                "Prop._h2_std_error": one_d_convert(prop_hsq_overlap_se),
                "Enrichment": one_d_convert(enrichment),
                "Enrichment_std_error": one_d_convert(enrichment_se),
                "Enrichment_p": diff_p,
                "Coefficient": one_d_convert(self.coef),
                "Coefficient_std_error": self.coef_se,
                "Coefficient_z-score": one_d_convert(self.coef)
                / one_d_convert(self.coef_se),
            }
        )
        if print_coefficients:
            df = df[
                ["Category", "Prop._SNPs", "Prop._h2", "Prop._h2_std_error",
                 "Enrichment", "Enrichment_std_error", "Enrichment_p",
                 "Coefficient", "Coefficient_std_error", "Coefficient_z-score"]
            ]
        else:
            df = df[
                ["Category", "Prop._SNPs", "Prop._h2", "Prop._h2_std_error",
                 "Enrichment", "Enrichment_std_error", "Enrichment_p"]
            ]
        return df

    def summary(self, ref_ld_colnames=None, P=None, K=None, overlap=False):
        """Return a human-readable text summary of the regression."""
        if P is not None and K is not None:
            T = "Liability"
            c = h2_obs_to_liab(1, P, K)
        else:
            T = "Observed"
            c = 1

        out = [
            "Total " + T + " scale h2: "
            + _s(c * self.tot) + " (" + _s(c * self.tot_se) + ")"
        ]
        if self.n_annot > 1:
            if ref_ld_colnames is None:
                ref_ld_colnames = ["CAT_" + str(i) for i in range(self.n_annot)]
            out.append("Categories: " + " ".join(ref_ld_colnames))
            if not overlap:
                out.append(T + " scale h2: " + _s(c * self.cat))
                out.append(T + " scale h2 SE: " + _s(c * self.cat_se))
                out.append("Proportion of SNPs: " + _s(self.M_prop))
                out.append("Proportion of h2g: " + _s(self.prop))
                out.append("Enrichment: " + _s(self.enrichment))
                out.append("Coefficients: " + _s(self.coef))
                out.append("Coefficient SE: " + _s(self.coef_se))

        out.append("Lambda GC: " + _s(self.lambda_gc))
        out.append("Mean Chi^2: " + _s(self.mean_chisq))
        if self.constrain_intercept:
            out.append(
                "Intercept: constrained to {C}".format(C=_s(self.intercept))
            )
        else:
            out.append(
                "Intercept: " + _s(self.intercept) + " ("
                + _s(self.intercept_se) + ")"
            )
            if self.mean_chisq > 1:
                if self.ratio < 0:
                    out.append("Ratio < 0 (usually indicates GC correction).")
                else:
                    out.append(
                        "Ratio: " + _s(self.ratio) + " (" + _s(self.ratio_se)
                        + ")"
                    )
            else:
                out.append("Ratio: NA (mean chi^2 < 1)")

        return remove_brackets("\n".join(out))

    def _update_weights(self, ld, w_ld, N, M, hsq, intercept, ii=None):
        if intercept is None:
            intercept = self.__null_intercept__
        return self.weights(ld, w_ld, N, M, hsq, intercept, ii)

    @classmethod
    def weights(cls, ld, w_ld, N, M, hsq, intercept=None, ii=None):
        """Heritability regression weights (approx. reciprocal CVF)."""
        M = float(M)
        if intercept is None:
            intercept = 1
        hsq = max(hsq, 0.0)
        hsq = min(hsq, 1.0)
        ld = np.fmax(ld, 1.0)
        w_ld = np.fmax(w_ld, 1.0)
        c = hsq * N / M
        het_w = 1.0 / (2 * np.square(intercept + np.multiply(c, ld)))
        oc_w = 1.0 / w_ld
        w = np.multiply(het_w, oc_w)
        return w


class Gencov(LD_Score_Regression):
    """Genetic covariance via cross-trait LD Score regression."""

    __null_intercept__ = 0

    def __init__(self, z1, z2, x, w, N1, N2, M, hsq1, hsq2, intercept_hsq1,
                 intercept_hsq2, n_blocks=200, intercept_gencov=None,
                 slow=False, twostep=None):
        self.intercept_hsq1 = intercept_hsq1
        self.intercept_hsq2 = intercept_hsq2
        self.hsq1 = hsq1
        self.hsq2 = hsq2
        self.N1 = N1
        self.N2 = N2
        y = z1 * z2
        step1_ii = None
        if twostep is not None:
            step1_ii = np.logical_and(z1 ** 2 < twostep, z2 ** 2 < twostep)

        LD_Score_Regression.__init__(
            self, y, x, w, np.sqrt(N1 * N2), M, n_blocks,
            intercept=intercept_gencov, slow=slow, step1_ii=step1_ii,
        )
        self.p, self.z = p_z_norm(self.tot, self.tot_se)
        self.mean_z1z2 = np.mean(np.multiply(z1, z2))

    def summary(self, ref_ld_colnames, P=None, K=None):
        """Return a human-readable text summary of the regression."""
        out = []
        if (
            P is not None
            and K is not None
            and all((i is not None for i in P))
            and all((i is not None for i in K))
        ):
            T = "Liability"
            c = gencov_obs_to_liab(1, P[0], P[1], K[0], K[1])
        else:
            T = "Observed"
            c = 1

        out.append(
            "Total " + T + " scale gencov: "
            + _s(c * self.tot) + " (" + _s(c * self.tot_se) + ")"
        )
        if self.n_annot > 1:
            out.append("Categories: " + str(" ".join(ref_ld_colnames)))
            out.append(T + " scale gencov: " + _s(c * self.cat))
            out.append(T + " scale gencov SE: " + _s(c * self.cat_se))
            out.append("Proportion of SNPs: " + _s(self.M_prop))
            out.append("Proportion of gencov: " + _s(self.prop))
            out.append("Enrichment: " + _s(self.enrichment))

        out.append("Mean z1*z2: " + _s(self.mean_z1z2))
        if self.constrain_intercept:
            out.append(
                "Intercept: constrained to {C}".format(C=_s(self.intercept))
            )
        else:
            out.append(
                "Intercept: " + _s(self.intercept) + " ("
                + _s(self.intercept_se) + ")"
            )
        return remove_brackets("\n".join(out))

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None,
                     ii=None):
        """Update function for IRWLS (genetic covariance)."""
        rho_g = M * x[0][0] / Nbar
        if intercept is None:  # if the regression includes an intercept
            intercept = x[0][1]
        ld = ref_ld_tot[:, 0].reshape(w_ld.shape)  # remove intercept
        if ii is not None:
            N1 = self.N1[ii].reshape((w_ld.shape))
            N2 = self.N2[ii].reshape((w_ld.shape))
        else:
            N1 = self.N1
            N2 = self.N2
        return self.weights(
            ld, w_ld, N1, N2, np.sum(M), self.hsq1, self.hsq2, rho_g,
            intercept, self.intercept_hsq1, self.intercept_hsq2, ii,
        )

    def _update_weights(self, ld, w_ld, sqrt_n1n2, M, rho_g, intercept,
                        ii=None):
        """Weight function with the same signature as ``Hsq._update_weights``."""
        w = self.weights(
            ld, w_ld, self.N1, self.N2, M, self.hsq1, self.hsq2, rho_g,
            intercept, self.intercept_hsq1, self.intercept_hsq2,
        )
        return w

    @classmethod
    def weights(cls, ld, w_ld, N1, N2, M, h1, h2, rho_g, intercept_gencov=None,
                intercept_hsq1=None, intercept_hsq2=None, ii=None):
        """Genetic-covariance regression weights (approx. reciprocal CVF)."""
        M = float(M)
        if intercept_gencov is None:
            intercept_gencov = 0
        if intercept_hsq1 is None:
            intercept_hsq1 = 1
        if intercept_hsq2 is None:
            intercept_hsq2 = 1

        h1, h2 = max(h1, 0.0), max(h2, 0.0)
        h1, h2 = min(h1, 1.0), min(h2, 1.0)
        rho_g = min(rho_g, 1.0)
        rho_g = max(rho_g, -1.0)
        ld = np.fmax(ld, 1.0)
        w_ld = np.fmax(w_ld, 1.0)
        a = np.multiply(N1, h1 * ld) / M + intercept_hsq1
        b = np.multiply(N2, h2 * ld) / M + intercept_hsq2
        sqrt_n1n2 = np.sqrt(np.multiply(N1, N2))
        c = np.multiply(sqrt_n1n2, rho_g * ld) / M + intercept_gencov
        try:
            het_w = 1.0 / (np.multiply(a, b) + np.square(c))
        except FloatingPointError:  # bizarre error; should never happen
            raise FloatingPointError("Why did you set hsq intercept <= 0?")
        oc_w = 1.0 / w_ld
        w = np.multiply(het_w, oc_w)
        return w


class RG(object):
    """Genetic correlation between trait 1 and trait 2."""

    def __init__(self, z1, z2, x, w, N1, N2, M, intercept_hsq1=None,
                 intercept_hsq2=None, intercept_gencov=None, n_blocks=200,
                 slow=False, twostep=None):
        self.intercept_gencov = intercept_gencov
        self._negative_hsq = None
        n_snp, n_annot = x.shape
        hsq1 = Hsq(
            np.square(z1), x, w, N1, M, n_blocks=n_blocks,
            intercept=intercept_hsq1, slow=slow, twostep=twostep,
        )
        hsq2 = Hsq(
            np.square(z2), x, w, N2, M, n_blocks=n_blocks,
            intercept=intercept_hsq2, slow=slow, twostep=twostep,
        )
        gencov = Gencov(
            z1, z2, x, w, N1, N2, M, hsq1.tot, hsq2.tot, hsq1.intercept,
            hsq2.intercept, n_blocks, intercept_gencov=intercept_gencov,
            slow=slow, twostep=twostep,
        )
        gencov.N1 = None  # save memory
        gencov.N2 = None
        self.hsq1, self.hsq2, self.gencov = hsq1, hsq2, gencov
        if hsq1.tot <= 0 or hsq2.tot <= 0:
            self._negative_hsq = True
            self.rg_ratio = self.rg = self.rg_se = "NA"
            self.p = self.z = "NA"
        else:
            rg_ratio = np.array(
                gencov.tot / np.sqrt(hsq1.tot * hsq2.tot)
            ).reshape((1, 1))
            denom_delete_values = np.sqrt(
                np.multiply(hsq1.tot_delete_values, hsq2.tot_delete_values)
            )
            rg = jk.RatioJackknife(
                rg_ratio, gencov.tot_delete_values, denom_delete_values
            )
            self.rg_jknife = float(rg.jknife_est)
            self.rg_se = float(rg.jknife_se)
            self.rg_ratio = float(rg_ratio)
            self.rg = self.rg_ratio
            self.p, self.z = p_z_norm(self.rg_ratio, self.rg_se)

    def summary(self, silly=False):
        """Return a human-readable text summary of the genetic correlation."""
        out = []
        if self._negative_hsq:
            out.append("Genetic Correlation: nan (nan) (h2  out of bounds) ")
            out.append("Z-score: nan (nan) (h2  out of bounds)")
            out.append("P: nan (nan) (h2  out of bounds)")
            out.append("WARNING: One of the h2's was out of bounds.")
            out.append(
                "This usually indicates a data-munging error "
                "or that h2 or N is low."
            )
        elif (self.rg_ratio > 1.2 or self.rg_ratio < -1.2) and not silly:
            out.append("Genetic Correlation: nan (nan) (rg out of bounds) ")
            out.append("Z-score: nan (nan) (rg out of bounds)")
            out.append("P: nan (nan) (rg out of bounds)")
            out.append("WARNING: rg was out of bounds.")
            if self.intercept_gencov is None:
                out.append(
                    "This often means that h2 is not significantly "
                    "different from zero."
                )
            else:
                out.append(
                    "This often means that you have constrained the "
                    "intercepts to the wrong values."
                )
        else:
            out.append(
                "Genetic Correlation: " + _s(self.rg_ratio) + " ("
                + _s(self.rg_se) + ")"
            )
            out.append("Z-score: " + _s(self.z))
            out.append("P: " + _s(self.p))
        return remove_brackets("\n".join(out))
