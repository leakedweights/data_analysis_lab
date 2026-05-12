"""Per-class GMM cluster features for v2.

Fits one Gaussian Mixture Model per training class on the v2 feature
matrix. The number of components is selected per class by BIC over
``K ∈ {2, 3, 4, 5}``. At inference, every event gets a soft membership
probability under each per-class GMM; the concatenated probability
vector is appended to the v2 feature matrix as ``sum_c K_c`` extra
columns.

Why per-class GMMs
------------------
The Suspicious class has at least two sub-archetypes in training
(large-packet-Normal-like and small-packet-DDoS-like) that share one
class label. The single label is what made the model collapse on the
genericity split, where the Suspicious distribution shifted toward the
small-packet sub-archetype that was a minority in training. Per-class
soft membership lets the classifier learn "this looks like Suspicious
sub-type A but not B" instead of treating Suspicious as a single
homogeneous distribution.

Design choices
--------------
* **Diagonal covariance.** With ~21 input features and the smallest
  class (DDoS) holding ~2,400 training events, full covariance has too
  many parameters to fit stably (231 covariance params per component
  for 21D). Diagonal gives 21 variance params per component — robust
  even on the small classes — and is symmetric across classes.
* **BIC over K ∈ {2,3,4,5}.** Two is the floor (otherwise we are not
  sub-clustering at all); five is the ceiling so the total feature
  count stays bounded (≤ 15 cluster features for 3 classes). BIC
  trades fit against parameter count, so it is the right criterion
  when the goal is "how many genuine sub-types are there?".
* **Fit on StandardScaled features.** The v2 features mix log-rate
  (~10), ratio (~1), and binary (0/1) scales; a diagonal-covariance
  GMM assumes comparable variances per axis. Scaling first makes the
  cluster shape driven by feature relationships, not absolute
  magnitudes.
* **No labels at inference.** The class labels enter only through the
  fit (deciding which subset of training rows feeds each per-class
  GMM). At inference we ask every GMM to score every event regardless
  of true label, so this is purely a deterministic feature transform
  on the evaluation side — the same lifecycle as ``StandardScaler``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class PerClassGMMFeaturizer:
    """Append per-class GMM soft-membership probabilities to a feature matrix."""

    K_grid: tuple[int, ...] = (2, 3, 4, 5)
    cov_type: str = "diag"
    random_state: int = 42
    max_iter: int = 200
    # Filled by ``fit``.
    scaler_: StandardScaler | None = field(default=None, init=False)
    gmms_: list[GaussianMixture] = field(default_factory=list, init=False)
    k_chosen_: list[int] = field(default_factory=list, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    n_classes_: int = field(default=0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerClassGMMFeaturizer":
        self.n_classes_ = int(y.max()) + 1
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        n_feat = X.shape[1]

        self.gmms_ = []
        self.k_chosen_ = []
        self.feature_names_ = []

        for c in range(self.n_classes_):
            X_c = Xs[y == c]
            best_bic = np.inf
            best_gmm: GaussianMixture | None = None
            best_k = self.K_grid[0]
            for k in self.K_grid:
                # Skip K values that would leave fewer training rows
                # per component than parameters per component
                # (means + variances = 2 * n_feat). This is a soft
                # rule of thumb to avoid degenerate GMMs on small
                # classes; without it sklearn can produce variances
                # of effectively zero on outlier rows.
                if X_c.shape[0] < k * 2 * n_feat:
                    continue
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=self.cov_type,
                    random_state=self.random_state,
                    max_iter=self.max_iter,
                    reg_covar=1e-4,
                ).fit(X_c)
                bic = gmm.bic(X_c)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_k = k

            if best_gmm is None:
                # Class too small for even K=2 with 2*n_feat rows per
                # component — fall back to a single-component Gaussian
                # so the feature column still exists and is consistent
                # with other classes' membership shape.
                best_gmm = GaussianMixture(
                    n_components=1, covariance_type=self.cov_type,
                    random_state=self.random_state, max_iter=self.max_iter,
                    reg_covar=1e-4,
                ).fit(X_c)
                best_k = 1
            self.gmms_.append(best_gmm)
            self.k_chosen_.append(best_k)
            self.feature_names_.extend(
                [f"gmm_c{c}_k{k}" for k in range(best_k)]
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("PerClassGMMFeaturizer must be fit before transform")
        Xs = self.scaler_.transform(X)
        return np.concatenate(
            [gmm.predict_proba(Xs) for gmm in self.gmms_], axis=1,
        )

    def transform_summary(self, X: np.ndarray) -> np.ndarray:
        """Return two OOD-style summary columns: max-membership and
        normalized entropy across the per-class soft probabilities.

        * ``gmm_max_prob`` — max over all per-class sub-cluster
          probabilities. Low values indicate "this event doesn't
          resemble any training archetype well", which is exactly the
          out-of-distribution signal the genericity split needs and
          the v2 feature set otherwise lacks.
        * ``gmm_entropy``  — Shannon entropy of the per-class
          probability vector after concatenation and re-normalization
          (so it sums to 1). High values mean ambiguous between
          archetypes; near-zero means a confident match. Together with
          ``gmm_max_prob`` this gives the model both a confidence and
          a uncertainty signal, which are not redundant: a confidently
          wrong cluster (high max, low entropy) and a confidently
          out-of-distribution event (low max, low entropy because all
          probs are tiny) are different beasts, and the model can
          learn to treat them differently.
        """
        if self.scaler_ is None:
            raise RuntimeError("PerClassGMMFeaturizer must be fit before transform")
        full = self.transform(X)
        max_prob = full.max(axis=1)
        # Re-normalize across the concatenated vector to get a proper
        # distribution. Without this each per-class block sums to 1
        # and the "joint entropy" would be inflated by trivial
        # cross-block uncertainty.
        row_sum = full.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        norm = full / row_sum
        # Shannon entropy with base e; clip to avoid log(0).
        eps = 1e-12
        entropy = -np.sum(norm * np.log(norm + eps), axis=1)
        return np.column_stack([max_prob, entropy])

    @property
    def n_features_out(self) -> int:
        return sum(self.k_chosen_)

    @property
    def summary_feature_names(self) -> list[str]:
        return ["gmm_max_prob", "gmm_entropy"]
