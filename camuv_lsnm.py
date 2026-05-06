"""
CAM-UV with LSNM residuals (LSNM-UV-Base).

This is Maeda & Shimizu (2021) CAM-UV exactly as published, with a single
change: _get_residual computes the two-step GAMLSS LSNM residual

    eta_hat_i = (x_i - f_hat(K_i)) / g_hat(K_i)          [paper Eq. 14]

instead of the additive residual  x_i - GAM(K_i).

Everything else -- fit, Algorithm 1 (parent search), Algorithm 2
(neighbourhood construction + UBP/UCP detection), independence tests,
prior knowledge handling -- is identical to camuv-original.py.
"""

import copy
import itertools

import numpy as np
from pygam import LinearGAM
from sklearn.utils import check_array

from lingam.hsic import hsic_test_gamma
from lingam.utils import f_correlation


class CAMUV_LSNM:
    """CAM-UV with LSNM residuals.  See camuv-original.py for the original."""

    def __init__(
        self,
        alpha=0.01,
        num_explanatory_vals=2,
        independence="hsic",
        ind_corr=0.5,
        prior_knowledge=None,
        verbose=False,
    ):
        # Check parameters
        if num_explanatory_vals <= 0:
            raise ValueError("num_explanatory_vals must be > 0.")

        if alpha < 0:
            raise ValueError("alpha must be >= 0.")

        if independence not in ("hsic", "fcorr"):
            raise ValueError("independence must be 'hsic' or 'fcorr'.")

        if ind_corr < 0.0:
            raise ValueError("ind_corr must be an float greater than 0.")

        self._num_explanatory_vals = num_explanatory_vals
        self._alpha = alpha
        self._independence = independence
        self._ind_corr = ind_corr
        self._pk_dict = self._make_pk_dict(prior_knowledge)
        self._verbose = verbose

    def fit(self, X):
        """Fit the model to X.  Identical to CAMUV.fit."""
        X = check_array(X)

        n = X.shape[0]
        d = X.shape[1]
        N = self._get_neighborhoods(X)
        P = self._find_parents(X, self._num_explanatory_vals, N)

        U = []

        if self._verbose:
            print(f"\n--- Algorithm 2: UBP/UCP detection ---")
            print(f"  P = {P}")
        for i in range(d):
            for j in range(d)[i + 1 :]:
                if (i in P[j]) or (j in P[i]):
                    if self._verbose:
                        print(f"  ({i},{j}): SKIP — gate (a) directed edge exists")
                    continue
                if (i not in N[j]) or (j not in N[i]):
                    if self._verbose:
                        print(f"  ({i},{j}): SKIP — gate (b) not in neighbourhood")
                    continue

                i_residual = self._get_residual(X, i, P[i])
                j_residual = self._get_residual(X, j, P[j])
                in_X = np.reshape(i_residual, [n, 1])
                in_Y = np.reshape(j_residual, [n, 1])
                if not self._is_independent(in_X, in_Y):
                    if not set([i, j]) in U:
                        U.append(set([i, j]))
                        if self._verbose:
                            print(f"  ({i},{j}): DEPENDENT → UBP detected!")
                elif self._verbose:
                    print(f"  ({i},{j}): independent → no UBP")

        self._U = U
        self._P = P

        return self._estimate_adjacency_matrix(X, P, U)

    def _make_pk_dict(self, prior_knowledge):
        if prior_knowledge is None:
            return None

        pk_dict = dict()
        for pair in prior_knowledge:
            if not pair[1] in pk_dict:
                pk_dict[pair[1]] = [pair[0]]
            else:
                pk_dict[pair[1]].append(pair[0])
        return pk_dict

    # ─────────────────────────────────────────────────────────────────────────
    # ONLY CHANGE vs camuv-original.py: LSNM two-step GAMLSS residual
    # ─────────────────────────────────────────────────────────────────────────

    def _get_residual(self, X, explained_i, explanatory_ids):
        """
        Compute the LSNM residual  eta_hat_i = (x_i - f_hat_i^1(K_i)) / g_hat_i^1(K_i).

        This implements paper Eq. (14) via a two-step procedure.

        The LSNM model for x_i is (paper Eq. 2):
            x_i = f_i^1(K_i) + g_i^1(K_i) * eta_i

        where K_i = explanatory_ids are the observed parents of x_i.
        Both f_i^1 (location) and g_i^1 (scale) are unknown nonlinear functions
        of K_i that must be estimated from data before eta_i can be recovered.

        A LinearGAM fits an additive spline model:
            y = beta_0 + sum_k s_k(x_k) + eps
        where each s_k is a smooth spline over one predictor.  LinearGAM().fit(X_K, y)
        learns the spline coefficients; .predict(X_K) returns the fitted values
            y_hat = beta_0_hat + sum_k s_hat_k(x_k)
        which approximates E[y | K_i] nonparametrically.

        Step 1 uses LinearGAM to estimate f_i^1(K_i) = E[x_i | K_i].
        Step 2 uses LinearGAM again on log(r_i^2) to estimate 2*log(g_i^1(K_i)).

        Falls back to the plain additive residual (= CAM-UV behaviour) if either
        GAM fit fails (e.g. constant feature, too few samples).
        """
        explanatory_ids = list(explanatory_ids)

        # K_i = empty: no observed parents to regress out.
        # eta_hat_i = x_i  (paper Eq. 14 with f_i^1 = 0, g_i^1 = 1)
        if len(explanatory_ids) == 0:
            return X[:, explained_i]

        X_expl = X[:, explanatory_ids]   # shape (n, |K_i|) -- values of observed parents K_i
        xi     = X[:, explained_i]       # shape (n,)       -- values of x_i

        # ---------------------------------------------------------------------
        # Step 1 -- estimate location f_i^1(K_i) and compute location residual r_i
        #
        # Model:  x_i = f_i^1(K_i) + g_i^1(K_i) * eta_i
        #
        # Fit LinearGAM:
        #   x_i = beta_0 + sum_{k in K_i} s_k(x_k) + eps
        #
        # Prediction:
        #   f_hat_i^1(K_i) = beta_0_hat + sum_k s_hat_k(x_k)
        #                  = E_hat[x_i | K_i]
        #
        # Location residual:
        #   r_i = x_i - f_hat_i^1(K_i)
        #       ~= g_i^1(K_i) * eta_i          (scale g_i^1 still present)
        #
        # This is the same step as Pham's _get_residual.  For homoscedastic
        # models (g_i^1 = const) r_i already equals eta_i up to a constant.
        # For LSNM, the scale g_i^1(K_i) must still be removed (Step 2).
        # ---------------------------------------------------------------------
        try:
            gam_loc   = LinearGAM().fit(X_expl, xi)
            loc_pred  = gam_loc.predict(X_expl)    # f_hat_i^1(K_i)
            loc_resid = xi - loc_pred               # r_i = x_i - f_hat_i^1(K_i)
        except Exception:
            # Fallback: demean only (no spline fit)
            loc_resid = xi - xi.mean()

        # ---------------------------------------------------------------------
        # Step 2 -- estimate scale g_i^1(K_i) from log(r_i^2) and divide
        #
        # From Step 1:  r_i ~= g_i^1(K_i) * eta_i
        #
        # Squaring and taking log:
        #   log(r_i^2) = 2 * log(g_i^1(K_i)) + log(eta_i^2)
        #                \_____________________/  \__________/
        #                  depends on K_i          noise term, indep of K_i
        #
        # Therefore:
        #   E[ log(r_i^2) | K_i ] = 2 * log(g_i^1(K_i))
        #
        # Fit LinearGAM on log(r_i^2) to estimate this conditional mean:
        #   log(r_i^2) = beta_0 + sum_{k in K_i} s_k(x_k) + eps
        #
        # Prediction:
        #   log_scale_hat = E_hat[ log(r_i^2) | K_i ]
        #                 ~= 2 * log(g_hat_i^1(K_i))
        #
        # Recover g_hat_i^1(K_i):
        #   g_hat_i^1(K_i) = exp( 0.5 * log_scale_hat )
        #
        # Clip to [1e-6, inf) to avoid division by near-zero scale.
        #
        # Final LSNM residual (paper Eq. 14):
        #   eta_hat_i = r_i / g_hat_i^1(K_i)
        #             ~= g_i^1(K_i) * eta_i / g_hat_i^1(K_i)
        #             ~= eta_i
        # ---------------------------------------------------------------------
        try:
            log_sq    = np.log(loc_resid ** 2 + 1e-8)   # log(r_i^2), +1e-8 avoids log(0)
            gam_scale = LinearGAM().fit(X_expl, log_sq)
            log_scale = gam_scale.predict(X_expl)        # E_hat[log(r_i^2) | K_i]
            scale     = np.clip(np.exp(0.5 * log_scale), 1e-6, None)  # g_hat_i^1(K_i)
            return loc_resid / scale                     # eta_hat_i  -- paper Eq. (14)
        except Exception:
            return loc_resid    # scale fit failed: return r_i (= CAM-UV / Pham fallback)

    # ─────────────────────────────────────────────────────────────────────────
    # Everything below is identical to camuv-original.py
    # ─────────────────────────────────────────────────────────────────────────

    def _is_independent(self, X, Y):
        if self._independence == "hsic":
            threshold = self._alpha
        elif self._independence == "fcorr":
            threshold = self._ind_corr
        is_independent, _ = self._is_independent_by(X, Y, threshold)
        return is_independent

    def _is_independent_by(self, X, Y, threshold):
        is_independent = False
        if self._independence == "hsic":
            _, value = hsic_test_gamma(X, Y)
            is_independent = value > threshold
        elif self._independence == "fcorr":
            value = f_correlation(X, Y)
            is_independent = value < threshold
        return is_independent, value

    def _get_neighborhoods(self, X):
        n = X.shape[0]
        d = X.shape[1]
        N = [set() for i in range(d)]
        if self._verbose:
            print(f"\n--- _get_neighborhoods ---")
        for i in range(d):
            for j in range(d)[i + 1 :]:
                in_X = np.reshape(X[:, i], [n, 1])
                in_Y = np.reshape(X[:, j], [n, 1])
                if not self._is_independent(in_X, in_Y):
                    N[i].add(j)
                    N[j].add(i)
                    if self._verbose:
                        print(f"  HSIC(x{i}, x{j}): dependent → added to N")
                elif self._verbose:
                    print(f"  HSIC(x{i}, x{j}): independent → NOT in N")
        if self._verbose:
            print(f"  N = {N}")
        return N

    def _find_parents(self, X, maxnum_vals, N):
        n = X.shape[0]
        d = X.shape[1]
        P = [set() for i in range(d)]  # Parents
        t = 2
        Y = copy.deepcopy(X)

        if self._verbose:
            print(f"\n--- _find_parents ---")

        while True:
            changed = False
            variables_set_list = list(itertools.combinations(set(range(d)), t))
            if self._verbose:
                print(f"\n  While-loop: t={t}, subsets={variables_set_list}")
            for variables_set in variables_set_list:
                variables_set = set(variables_set)

                if not self._check_identified_causality(variables_set, P):
                    if self._verbose:
                        print(f"  {variables_set}: skip (_check_identified_causality=False)")
                    continue

                child, is_independence_with_K = self._get_child(
                    X, variables_set, P, N, Y
                )
                if child is None:
                    if self._verbose:
                        print(f"  {variables_set}: _get_child → None")
                    continue
                if not is_independence_with_K:
                    if self._verbose:
                        print(f"  {variables_set}: _get_child → child={child}, but NOT independent of parents")
                    continue

                parents = variables_set - {child}
                withou_k = self._check_independence_withou_K(parents, child, P, N, Y)
                if self._verbose:
                    print(f"  {variables_set}: _check_independence_withou_K(parents={parents}, child={child}) = {withou_k}")
                if not withou_k:
                    continue

                for parent in parents:
                    P[child].add(parent)
                    changed = True
                    if self._verbose:
                        print(f"  *** PARENT ASSIGNED: x{parent} → x{child}  (P={P}) ***")
                    Y = self._get_residuals_matrix(X, Y, P, child)

            if changed:
                t = 2
            else:
                t += 1
                if t > maxnum_vals:
                    break

        # Prune non-parents
        if self._verbose:
            print(f"\n  Pruning: P before = {P}")
        for i in range(d):
            non_parents = set()
            for j in P[i]:
                residual_i = self._get_residual(X, i, P[i] - {j})
                residual_j = self._get_residual(X, j, P[j])
                in_X = np.reshape(residual_i, [n, 1])
                in_Y = np.reshape(residual_j, [n, 1])
                if self._is_independent(in_X, in_Y):
                    non_parents.add(j)
                    if self._verbose:
                        print(f"  Prune: x{j} removed from P[{i}] (independent)")
                elif self._verbose:
                    print(f"  Prune: x{j} kept in P[{i}] (dependent)")
            P[i] = P[i] - non_parents
        if self._verbose:
            print(f"  Pruning: P after = {P}")

        return P

    def _check_prior_knowledge(self, xj_list, xi):
        if self._pk_dict is not None:
            for xj in xj_list:
                if (xi in self._pk_dict) and (xj in self._pk_dict[xi]):
                    return True
        return False

    def _get_residuals_matrix(self, X, Y_old, P, child):
        Y = copy.deepcopy(Y_old)
        Y[:, child] = self._get_residual(X, child, P[child])
        return Y

    def _get_child(self, X, variables_set, P, N, Y):
        n = X.shape[0]

        prev_independence = 0.0 if self._independence == "hsic" else 1.0
        max_independence_child = None

        for child in variables_set:
            parents = variables_set - {child}

            if self._check_prior_knowledge(parents, child):
                continue

            if not self._check_correlation(child, parents, N):
                if self._verbose:
                    print(f"    _get_child: child={child}, parents={parents} → _check_correlation=False")
                continue

            residual = self._get_residual(X, child, parents | P[child])
            in_X = np.reshape(residual, [n, 1])
            in_Y = np.reshape(Y[:, list(parents)], [n, len(parents)])
            is_ind, value = self._is_independent_by(in_X, in_Y, prev_independence)
            if self._verbose:
                print(f"    _get_child: child={child}, parents={parents} | resid(x{child}|{parents | P[child]}) vs Y[{list(parents)}]: "
                      f"val={value:.6f}, prev={prev_independence:.6f}, ind={is_ind}")
            if is_ind:
                prev_independence = value
                max_independence_child = child

        if self._independence == "hsic":
            is_independent = prev_independence > self._alpha
        elif self._independence == "fcorr":
            is_independent = prev_independence < self._ind_corr

        if self._verbose:
            print(f"    _get_child result: child={max_independence_child}, is_independent={is_independent} "
                  f"(best={prev_independence:.6f} vs alpha={self._alpha})")

        return max_independence_child, is_independent

    def _check_independence_withou_K(self, parents, child, P, N, Y):
        n = Y.shape[0]
        for parent in parents:
            in_X = np.reshape(Y[:, child], [n, 1])
            in_Y = np.reshape(Y[:, parent], [n, 1])
            if self._is_independent(in_X, in_Y):
                return False
        return True

    def _check_identified_causality(self, variables_set, P):
        variables_list = list(variables_set)
        for i in variables_list:
            for j in variables_list[variables_list.index(i) + 1 :]:
                if (j in P[i]) or (i in P[j]):
                    return False
        return True

    def _check_correlation(self, child, parents, N):
        for parent in parents:
            if parent not in N[child]:
                return False
        return True

    def _estimate_adjacency_matrix(self, X, P, U):
        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i, parents in enumerate(P):
            for parent in parents:
                B[i, parent] = 1
        for confounded_pair in U:
            confounded_pair = list(confounded_pair)
            B[confounded_pair[0], confounded_pair[1]] = np.nan
            B[confounded_pair[1], confounded_pair[0]] = np.nan
        self._adjacency_matrix = B
        return self

    @property
    def adjacency_matrix_(self):
        return self._adjacency_matrix
