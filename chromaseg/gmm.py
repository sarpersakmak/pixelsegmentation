"""
Title:       gmm.py
Author:      Sarper Sakmak
ID:          14008175400
Section:     2
Assignment:  CMPE 442 Programming Assignment-3
Description:
    - Implements the GMM class (EM algorithm) entirely from scratch using NumPy.
    - STEP 1: Initialization via K-Means means + identity-scaled covariances
              + uniform weights (_initialize).
    - STEP 2: E-step — compute per-pixel soft responsibilities using the
              multivariate Gaussian log-PDF with the log-sum-exp trick for
              numerical stability (_e_step).
    - STEP 3: M-step — update means, full covariances, and mixing weights
              from responsibilities; add 1e-6 * I regularization to each
              covariance to prevent singularity (_m_step).
    - STEP 4: Log-likelihood computation for convergence check
              (_compute_log_likelihood).
    - STEP 5: Multi-restart strategy (n_init runs); best result kept via
              highest log-likelihood (fit).
"""

import numpy as np  # type: ignore
from kmeans import KMeans  # type: ignore

SEED = 14008175400 % 200   # seed = 0


class GMM:
    """
    Gaussian Mixture Model (EM algorithm) implemented from scratch.

    Runs the algorithm n_init times and keeps the result with the
    highest final log-likelihood.

    Parameters
    ----------
    k        : int   – number of Gaussian components
    n_init   : int   – number of independent restarts (default 5)
    max_iter : int   – maximum EM iterations per run (default 100)
    tol      : float – convergence tolerance on log-likelihood (default 1e-4)
    seed     : int   – base random seed (default 0)
    reg      : float – covariance regularisation constant (default 1e-6)
    """

    def __init__(self, k: int, n_init: int = 5,
                 max_iter: int = 100, tol: float = 1e-4,
                 seed: int = SEED, reg: float = 1e-6) -> None:
        self.k        = k
        self.n_init   = n_init
        self.max_iter = max_iter
        self.tol      = tol
        self.seed     = seed
        self.reg      = reg

        # Results (populated after fit)
        self.labels_          = None   # type: ignore
        self.means_           = None   # type: ignore
        self.covariances_     = None   # type: ignore
        self.weights_         = None   # type: ignore
        self.log_likelihood_  = None   # type: ignore

    # ── STEP 1: Initialization ────────────────────────────────────────────
    def _initialize(self, X: np.ndarray, run_idx: int) -> tuple:
        """
        Initialize GMM parameters using K-Means centroids.

        Means       ← K-Means centroids (1 random restart, seeded per run).
        Covariances ← identity * overall pixel variance for each component.
        Weights     ← uniform 1/k.

        Parameters
        ----------
        X       : np.ndarray, shape (N, D) – pixel feature matrix
        run_idx : int – current restart index (varies K-Means seed)

        Returns
        -------
        means       : np.ndarray, shape (k, D)
        covariances : np.ndarray, shape (k, D, D)
        weights     : np.ndarray, shape (k,)
        """
        km = KMeans(k=self.k, n_init=1, max_iter=100, tol=1e-4,
                    seed=self.seed + run_idx)
        km.fit(X)

        D           = X.shape[1]
        var         = float(np.var(X))
        means       = km.centroids_.copy()                # (k, D)
        covariances = np.array([np.eye(D) * var
                                for _ in range(self.k)])  # (k, D, D)
        weights     = np.full(self.k, 1.0 / self.k)      # (k,)
        return means, covariances, weights

    # ── Helper: log multivariate Gaussian PDF ─────────────────────────────
    @staticmethod
    def _log_gaussian(X: np.ndarray, mean: np.ndarray,
                      cov: np.ndarray) -> np.ndarray:
        """
        Compute log N(x | mean, cov) for every row of X using NumPy only.

        Uses Cholesky decomposition for stable log-determinant computation
        and efficient solve of (x - mu)^T Σ^{-1} (x - mu).

        Parameters
        ----------
        X    : np.ndarray, shape (N, D) – data points
        mean : np.ndarray, shape (D,)   – component mean
        cov  : np.ndarray, shape (D, D) – component covariance

        Returns
        -------
        log_prob : np.ndarray, shape (N,) – log-PDF values
        """
        N, D   = X.shape
        diff   = X - mean                               # (N, D)

        try:
            L       = np.linalg.cholesky(cov)           # (D, D) lower triangular
            # log|cov| = 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            # Solve L z = diff^T  →  z = L^{-1} diff^T
            z       = np.linalg.solve(L, diff.T)        # (D, N)
            maha    = np.sum(z ** 2, axis=0)            # (N,)
        except np.linalg.LinAlgError:
            # Fallback: use pinv if Cholesky fails
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                log_det = -500.0
            cov_inv = np.linalg.pinv(cov)               # type: ignore
            maha    = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)

        log_prob = (-0.5 * (D * np.log(2.0 * np.pi) + log_det + maha))
        return log_prob                                 # (N,)

    # ── STEP 2: E-step ────────────────────────────────────────────────────
    def _e_step(self, X: np.ndarray, means: np.ndarray,
                covariances: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
        """
        Compute soft responsibilities γ_{ik} for every pixel i and
        component k using the log-sum-exp trick for numerical stability.

        Parameters
        ----------
        X           : np.ndarray, shape (N, D)
        means       : np.ndarray, shape (k, D)
        covariances : np.ndarray, shape (k, D, D)
        weights     : np.ndarray, shape (k,)

        Returns
        -------
        responsibilities : np.ndarray, shape (N, k) – soft assignments in [0,1]
        """
        N = X.shape[0]
        log_resp = np.zeros((N, self.k))  # (N, k)

        for j in range(self.k):
            log_resp[:, j] = (np.log(weights[j] + 1e-300) +
                              self._log_gaussian(X, means[j],
                                                 covariances[j]))

        # log-sum-exp normalisation for numerical stability
        log_resp_max = log_resp.max(axis=1, keepdims=True)   # (N, 1)
        log_sum      = (log_resp_max.squeeze() +
                        np.log(np.sum(
                            np.exp(log_resp - log_resp_max), axis=1
                        ) + 1e-300))                          # (N,)

        log_resp -= log_sum[:, np.newaxis]                    # normalise
        return np.exp(log_resp)                               # (N, k)

    # ── STEP 3: M-step ────────────────────────────────────────────────────
    def _m_step(self, X: np.ndarray,
                responsibilities: np.ndarray) -> tuple:
        """
        Update GMM parameters from soft responsibilities.

        Covariance regularisation: Σ_k += reg * I to prevent singularity.

        Parameters
        ----------
        X                : np.ndarray, shape (N, D)
        responsibilities : np.ndarray, shape (N, k)

        Returns
        -------
        means       : np.ndarray, shape (k, D)
        covariances : np.ndarray, shape (k, D, D)
        weights     : np.ndarray, shape (k,)
        """
        N, D = X.shape
        Nk   = responsibilities.sum(axis=0)          # (k,) effective counts

        weights = Nk / N                             # (k,)
        means   = (responsibilities.T @ X) / Nk[:, np.newaxis]  # (k, D)

        covariances = np.zeros((self.k, D, D))
        for j in range(self.k):
            diff          = X - means[j]             # (N, D)
            r             = responsibilities[:, j]   # (N,)
            # Weighted outer product: Σ_k = (1/N_k) Σ_i r_{ik} (x_i-μ_k)(x_i-μ_k)^T
            weighted_diff = diff * r[:, np.newaxis]  # (N, D)
            cov_k         = (weighted_diff.T @ diff) / (Nk[j] + 1e-300)
            # Regularisation
            cov_k        += self.reg * np.eye(D)
            covariances[j] = cov_k

        return means, covariances, weights

    # ── STEP 4: Log-likelihood ────────────────────────────────────────────
    def _compute_log_likelihood(self, X: np.ndarray, means: np.ndarray,
                                covariances: np.ndarray,
                                weights: np.ndarray) -> float:
        """
        Compute the total log-likelihood of the data under current parameters.

        L = Σ_i log Σ_k w_k N(x_i | μ_k, Σ_k)

        Uses log-sum-exp for numerical stability.

        Parameters
        ----------
        X           : np.ndarray, shape (N, D)
        means       : np.ndarray, shape (k, D)
        covariances : np.ndarray, shape (k, D, D)
        weights     : np.ndarray, shape (k,)

        Returns
        -------
        log_likelihood : float
        """
        N = X.shape[0]
        log_probs = np.zeros((N, self.k))

        for j in range(self.k):
            log_probs[:, j] = (np.log(weights[j] + 1e-300) +
                               self._log_gaussian(X, means[j],
                                                  covariances[j]))

        # log-sum-exp per pixel, then sum
        log_max    = log_probs.max(axis=1, keepdims=True)
        ll         = float(
            np.sum(log_max.squeeze() +
                   np.log(np.sum(np.exp(log_probs - log_max),
                                 axis=1) + 1e-300))
        )
        return ll

    # ── STEP 5: Fit (multi-restart) ───────────────────────────────────────
    def fit(self, X: np.ndarray) -> "GMM":
        """
        Run GMM EM algorithm n_init times; keep result with highest
        log-likelihood.

        Parameters
        ----------
        X : np.ndarray, shape (N, D) – pixel feature matrix (normalized)

        Returns
        -------
        self : GMM – fitted instance (self.labels_, self.means_,
                      self.covariances_, self.weights_,
                      self.log_likelihood_ are populated)
        """
        best_ll          = -np.inf
        best_means       = None   # type: ignore
        best_covariances = None   # type: ignore
        best_weights     = None   # type: ignore
        best_labels      = None   # type: ignore

        for run in range(self.n_init):
            print(f"[gmm]   Run {run + 1}/{self.n_init} | K={self.k}")
            means, covariances, weights = self._initialize(X, run_idx=run)
            prev_ll = -np.inf

            for it in range(self.max_iter):
                # E-step
                responsibilities = self._e_step(X, means,
                                                covariances, weights)
                # M-step
                means, covariances, weights = self._m_step(X,
                                                           responsibilities)
                # Log-likelihood
                ll = self._compute_log_likelihood(X, means,
                                                  covariances, weights)

                if (it + 1) % 20 == 0 or it == 0:
                    print(f"[gmm]     Iteration {it + 1:>3} | "
                          f"Log-Likelihood: {ll:.4f}")

                if abs(ll - prev_ll) < self.tol:
                    print(f"[gmm]     Converged at iteration {it + 1}.")
                    break
                prev_ll = ll

            print(f"[gmm]   Run {run + 1} final | Log-Likelihood: {ll:.4f}")

            if ll > best_ll:
                best_ll          = ll
                best_means       = means.copy()
                best_covariances = covariances.copy()
                best_weights     = weights.copy()
                best_labels      = np.argmax(responsibilities, axis=1).copy()

        self.means_          = best_means
        self.covariances_    = best_covariances
        self.weights_        = best_weights
        self.log_likelihood_ = best_ll
        self.labels_         = best_labels
        print(f"[gmm] Best log-likelihood across {self.n_init} runs: "
              f"{self.log_likelihood_:.4f}\n")
        return self