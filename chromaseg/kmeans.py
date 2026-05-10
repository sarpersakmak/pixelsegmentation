"""
Title:       kmeans.py
Author:      Sarper Sakmak
ID:          14008175400
Section:     2
Assignment:  CMPE 442 Programming Assignment-3
Description:
    - Implements the KMeans class entirely from scratch using only NumPy.
    - STEP 1: Centroid initialization via random sampling from the data
              (_init_centroids).
    - STEP 2: Cluster assignment using Euclidean distance and argmin
              (_assign_clusters).
    - STEP 3: Centroid update as mean of assigned points
              (_update_centroids).
    - STEP 4: Convergence check based on centroid shift tolerance (tol).
    - STEP 5: Multi-restart strategy (n_init runs); best result kept via
              lowest inertia (fit).
"""

import numpy as np  # type: ignore

SEED = 14008175400 % 200   # seed = 0


class KMeans:
    """
    K-Means clustering implemented from scratch with NumPy.

    Runs the algorithm n_init times and keeps the result with the
    lowest inertia (sum of squared distances to nearest centroid).

    Parameters
    ----------
    k        : int   – number of clusters
    n_init   : int   – number of independent restarts (default 10)
    max_iter : int   – maximum EM iterations per run (default 300)
    tol      : float – convergence tolerance on centroid shift (default 1e-4)
    seed     : int   – base random seed for reproducibility (default 0)
    """

    def __init__(self, k: int, n_init: int = 10,
                 max_iter: int = 300, tol: float = 1e-4,
                 seed: int = SEED) -> None:
        self.k         = k
        self.n_init    = n_init
        self.max_iter  = max_iter
        self.tol       = tol
        self.seed      = seed

        # Results (populated after fit)
        self.labels_    = None   # type: ignore
        self.centroids_ = None   # type: ignore
        self.inertia_   = None   # type: ignore

    # ── STEP 1: Centroid initialization ──────────────────────────────────
    def _init_centroids(self, X: np.ndarray, run_idx: int) -> np.ndarray:
        """
        Randomly sample k distinct data points as initial centroids.

        Parameters
        ----------
        X       : np.ndarray, shape (N, D) – pixel feature matrix
        run_idx : int – current restart index (used to vary seed per run)

        Returns
        -------
        centroids : np.ndarray, shape (k, D)
        """
        rng     = np.random.RandomState(self.seed + run_idx)
        indices = rng.choice(X.shape[0], size=self.k, replace=False)
        return X[indices].copy()

    # ── STEP 2: Cluster assignment ────────────────────────────────────────
    def _assign_clusters(self, X: np.ndarray,
                         centroids: np.ndarray) -> np.ndarray:
        """
        Assign each pixel to its nearest centroid via Euclidean distance.

        Uses broadcasting to compute the full (N, k) distance matrix
        without explicit Python loops.

        Parameters
        ----------
        X         : np.ndarray, shape (N, D) – pixel features
        centroids : np.ndarray, shape (k, D) – current centroids

        Returns
        -------
        labels : np.ndarray, shape (N,) – cluster index for each pixel
        """
        # (N, 1, D) - (1, k, D) → (N, k, D) → sum of squares → (N, k)
        diff   = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)          # (N, k)
        return np.argmin(sq_dist, axis=1)            # (N,)

    # ── STEP 3: Centroid update ───────────────────────────────────────────
    def _update_centroids(self, X: np.ndarray,
                          labels: np.ndarray,
                          centroids: np.ndarray) -> np.ndarray:
        """
        Recompute each centroid as the mean of its assigned pixels.

        If a cluster becomes empty (no assigned pixels), its centroid is
        re-initialised by randomly sampling a data point to avoid NaN.

        Parameters
        ----------
        X         : np.ndarray, shape (N, D) – pixel features
        labels    : np.ndarray, shape (N,)   – current cluster assignments
        centroids : np.ndarray, shape (k, D) – previous centroids

        Returns
        -------
        new_centroids : np.ndarray, shape (k, D)
        """
        new_centroids = np.zeros_like(centroids)
        rng = np.random.RandomState(self.seed)
        for j in range(self.k):
            mask = labels == j
            if mask.sum() == 0:
                # Re-initialize empty cluster with a random data point
                new_centroids[j] = X[rng.randint(0, X.shape[0])]
            else:
                new_centroids[j] = X[mask].mean(axis=0)
        return new_centroids

    # ── STEP 4 & 5: Fit (multi-restart) ──────────────────────────────────
    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Run K-Means n_init times; store the result with lowest inertia.

        Parameters
        ----------
        X : np.ndarray, shape (N, D) – pixel feature matrix (normalized)

        Returns
        -------
        self : KMeans – fitted instance (self.labels_, self.centroids_,
                         self.inertia_ are populated)
        """
        best_labels    = None   # type: ignore
        best_centroids = None   # type: ignore
        best_inertia   = np.inf

        for run in range(self.n_init):
            print(f"[kmeans]   Run {run + 1}/{self.n_init} | K={self.k}")
            centroids = self._init_centroids(X, run_idx=run)

            for it in range(self.max_iter):
                labels       = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels, centroids)

                # Convergence: max centroid shift < tol
                shift = np.max(np.linalg.norm(new_centroids - centroids,
                                              axis=1))
                centroids = new_centroids

                # Inertia = sum of squared distances to nearest centroid
                diff    = X - centroids[labels]
                inertia = float(np.sum(diff ** 2))

                if (it + 1) % 50 == 0 or it == 0:
                    print(f"[kmeans]     Iteration {it + 1:>3} | "
                          f"Inertia: {inertia:.4f} | Shift: {shift:.6f}")

                if shift < self.tol:
                    print(f"[kmeans]     Converged at iteration {it + 1}.")
                    break

            print(f"[kmeans]   Run {run + 1} final | Inertia: {inertia:.4f}")

            if inertia < best_inertia:
                best_inertia   = inertia
                best_labels    = labels.copy()
                best_centroids = centroids.copy()

        self.labels_    = best_labels
        self.centroids_ = best_centroids
        self.inertia_   = best_inertia
        print(f"[kmeans] Best inertia across {self.n_init} runs: "
              f"{self.inertia_:.4f}\n")
        return self

    # ── Predict ───────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new data points to the nearest fitted centroid.

        Parameters
        ----------
        X : np.ndarray, shape (N, D) – pixel features

        Returns
        -------
        labels : np.ndarray, shape (N,) – cluster assignments
        """
        return self._assign_clusters(X, self.centroids_)