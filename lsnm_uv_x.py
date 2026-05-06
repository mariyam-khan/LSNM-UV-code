"""
LSNM-UV-X Algorithm.

Two-stage algorithm for causal discovery in Location-Scale Noise Models with
hidden variables (bow-free ADMGs).

  Stage 1 — CAMUV_LSNM (camuv-lsnm.py):
      CAM-UV (Maeda & Shimizu, 2021, Algorithms 1 & 2) exactly as published,
      with the additive residual replaced by the LSNM residual
          eta_i = (x_i - f_hat(K_i)) / g_hat(K_i)           [Eq. (14)]

  Stage 2 — checkVISIBLE:
      Re-examines invisible pairs (NaN entries) by searching over regression
      sets; follows Pham et al. (2025) / cam-uv-x_extended.py.

Requirements: lingam, pygam, numpy
"""

import itertools
import numpy as np
from camuv_lsnm import CAMUV_LSNM


# ─────────────────────────────────────────────────────────────────────────────
# LSNM-UV-X = CAMUV_LSNM (Stage 1) + checkVISIBLE (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

class LSNMUV_X(CAMUV_LSNM):
    """
    LSNM-UV-X: CAM-UV with LSNM residuals + checkVISIBLE.

    Stage 1 is CAMUV_LSNM -- Maeda & Shimizu (2021) Algorithms 1 & 2
    with LSNM residuals (the only change vs original CAM-UV).

    Stage 2 re-examines invisible pairs (NaN entries) to orient or
    remove them, following Pham et al. (2025).

    Parameters
    ----------
    alpha               : HSIC significance level (default 0.01)
    num_explanatory_vals: max |K| in parent search  (default 3, i.e. d=3)
    max_regress_size    : max regression-set size in checkVISIBLE (default 2)
    """

    def __init__(
        self,
        alpha:                float = 0.01,
        num_explanatory_vals: int   = 3,
        max_regress_size:     int   = 2,
    ):
        super().__init__(alpha=alpha, num_explanatory_vals=num_explanatory_vals)
        self._max_regress_size = max_regress_size

    # ── Public fit ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray):
        """
        Fit LSNM-UV-X to data matrix X  (n_samples x p_features).

        Returns self; adjacency matrix accessible via .adjacency_matrix_.
        Convention: A[i,j] = 1 => x_j -> x_i;  A[i,j] = NaN => invisible pair.
        """
        # Stage 1: CAMUV_LSNM (Maeda21 Alg 1 + Alg 2 with LSNM residuals)
        super().fit(X)
        mat = self._adjacency_matrix.copy()

        # Stage 2: checkVISIBLE
        mat = self._check_visible(X, mat)
        self._adjacency_matrix = mat
        return self

    # ── checkVISIBLE ──────────────────────────────────────────────────────────

    def _check_visible(self, X: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """
        Re-examine invisible pairs (NaN entries) using broader regression sets.

        For each NaN pair (x_i, x_j) with i < j, search over pairs of regression
        sets (S_1, S_2) drawn from:
            Q = {parents of x_i} u {parents of x_j}
              u {other NaN-neighbours of x_i} u {other NaN-neighbours of x_j}

        Tests  (using LSNM residuals h = _get_residual):

        (b) h(i, S_1 u {x_j}) _|_ h(j, S_2)   (x_j included in x_i's regression)
                -> x_i is NOT a parent of x_j  (iNotParent)

        (c) h(i, S_1) _|_ h(j, S_2 u {x_i})   (x_i included in x_j's regression)
                -> x_j is NOT a parent of x_i  (jNotParent)

        Resolution:
            iNotParent only   ->  edge x_j -> x_i  (A[i,j]=1, A[j,i]=0)
            jNotParent only   ->  edge x_i -> x_j  (A[j,i]=1, A[i,j]=0)
            both or neither   ->  pair remains NaN (invisible)

        Note: the original Pham2025 test (a) h(i,S1) _|_ h(j,S2) -> non-edge is
        intentionally omitted here.  NaN pairs arrive already confirmed as
        hidden by Algorithm 2 (CAMUV_LSNM pairwise residual test).  Test (a)
        uses different conditioning sets and can "absorb" the hidden-cause signal
        via overcontrol (conditioning on another child of the shared u_k), giving
        a spurious non-edge verdict.  Only directional evidence (b)/(c) is safe
        to apply to these confirmed-hidden pairs.

        However, when BOTH (b) and (c) succeed -- both directions are ruled out --
        this resolves the pair as a visible non-edge (set to 0), matching
        cam-uv-x_extended.py's logic.

        Follows Pham et al. (2025), Algorithm 3 (orientation step only).
        """
        n     = X.shape[0]
        p     = mat.shape[0]
        mat_new = mat.copy()

        # Only process upper-triangle pairs to avoid processing each pair twice
        nan_pairs = [
            (i, j) for i in range(p) for j in range(i + 1, p)
            if np.isnan(mat[i, j]) and np.isnan(mat[j, i])
        ]

        for (x_i, x_j) in nan_pairs:
            # ── Build candidate regression set Q ──────────────────────────────
            nan_xi = set(np.where(np.isnan(mat_new[x_i, :]))[0]) - {x_i, x_j}
            nan_xj = set(np.where(np.isnan(mat_new[x_j, :]))[0]) - {x_i, x_j}
            P_i    = set(np.where(mat_new[x_i, :] == 1)[0])
            P_j    = set(np.where(mat_new[x_j, :] == 1)[0])
            Q      = nan_xi | nan_xj | P_i | P_j

            if len(Q) == 0:
                continue   # no candidates, leave as NaN

            iNotParent = False
            jNotParent = False
            isNonEdge  = False

            max_sz = min(self._max_regress_size, len(Q))

            for sz_i in range(1, max_sz + 1):
                if isNonEdge:
                    break
                for s1 in itertools.combinations(Q, sz_i):
                    if isNonEdge:
                        break
                    for sz_j in range(1, max_sz + 1):
                        if isNonEdge:
                            break
                        for s2 in itertools.combinations(Q, sz_j):
                            if isNonEdge:
                                break
                            expl_i = set(s1)
                            expl_j = set(s2)

                            r_i = self._get_residual(X, x_i, expl_i).reshape(n, 1)
                            r_j = self._get_residual(X, x_j, expl_j).reshape(n, 1)

                            # (b) x_i not parent of x_j  (evidence for x_j -> x_i)
                            r_i2 = self._get_residual(
                                X, x_i, expl_i | {x_j}
                            ).reshape(n, 1)
                            if self._is_independent(r_i2, r_j):
                                iNotParent = True

                            # (c) x_j not parent of x_i  (evidence for x_i -> x_j)
                            r_j2 = self._get_residual(
                                X, x_j, expl_j | {x_i}
                            ).reshape(n, 1)
                            if self._is_independent(r_i, r_j2):
                                jNotParent = True

                            # Both ruled out -> visible non-edge; stop searching
                            if iNotParent and jNotParent:
                                mat_new[x_i, x_j] = 0
                                mat_new[x_j, x_i] = 0
                                isNonEdge = True
                                break

            # ── Resolve pair ──────────────────────────────────────────────────
            if not isNonEdge:
                if iNotParent and not jNotParent:
                    # x_j -> x_i
                    mat_new[x_i, x_j] = 1
                    mat_new[x_j, x_i] = 0
                elif jNotParent and not iNotParent:
                    # x_i -> x_j
                    mat_new[x_j, x_i] = 1
                    mat_new[x_i, x_j] = 0
                # neither True: leave as NaN (invisible)

        return mat_new
