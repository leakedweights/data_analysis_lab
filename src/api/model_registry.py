"""Train, serialize, and load the v2+cluster DDoS-detector models for
the interactive demo.

Single boundary with the streaming engine:

    X_aug = registry.featurize_window(events_df, components_df)
    y_pred = registry.predict(X_aug)

The featurizer side runs ``features_v2.featurize`` and concatenates the
per-class GMM soft-membership probabilities from
``cluster_features.PerClassGMMFeaturizer`` (same code path as
``bench_cluster.py``). The classifier side wraps each estimator with a
``_ScaledPredictor`` when needed so the engine never sees the
augmentation scaler directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.cluster_features import PerClassGMMFeaturizer
from src.features_v2 import featurize
from src.train_v2 import _model_specs
from src.utils.data_pipeline import load_components, load_events

logger = logging.getLogger(__name__)

MODEL_DIR = Path("/app/models")
FEATURE_VERSION = "v2+cluster"


class _ScaledPredictor:
    """Wraps an estimator with a pre-fit scaler. Used for the
    scale-sensitive models (LR, KNN) so the registry returns a uniform
    predictor object regardless of model type."""

    def __init__(self, model, scaler: StandardScaler):
        self._model = model
        self._scaler = scaler

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))


def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


class ModelRegistry:
    """Holds the v2+cluster pipeline (featurizer + augmented scaler +
    five classifiers) with a switchable current model."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self._model_dir = model_dir
        self._gmm: PerClassGMMFeaturizer | None = None
        self._aug_scaler: StandardScaler | None = None
        # name -> (predictor_obj, needs_scaling). For scaled models the
        # predictor is _ScaledPredictor; for tree models it is the raw
        # estimator. needs_scaling is kept for introspection only.
        self._models: dict[str, tuple[object, bool]] = {}
        self._current: str | None = None

    # -- introspection used by app.py --------------------------------------

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    @property
    def current_name(self) -> str | None:
        return self._current

    @property
    def current_model(self) -> object | None:
        """Returns a truthy sentinel when a model is selected, ``None``
        otherwise. The streaming engine routes prediction through
        :meth:`predict` rather than calling this object directly."""
        if self._current is None:
            return None
        entry = self._models.get(self._current)
        return entry[0] if entry else None

    def select(self, name: str) -> None:
        if name not in self._models:
            raise KeyError(f"Unknown model: {name}")
        self._current = name

    # -- inference surface for the streaming engine ------------------------

    def featurize_window(
        self, events: pd.DataFrame, components: pd.DataFrame,
    ) -> np.ndarray:
        """Produce the augmented (v2 + per-class GMM probs) feature
        matrix for one window. Caller passes the events and the
        components belonging to those events (joined on ``Attack ID``).
        """
        if self._gmm is None:
            raise RuntimeError("ModelRegistry not initialized — call load_or_train() first")

        fm = featurize(events, components, label_col=None)
        X = fm.X
        # ``featurize`` already fills NaN component aggregates with
        # sensible fallbacks. Belt-and-braces here: any residual NaN
        # (e.g., from an upstream schema drift) collapses to 0 so we
        # never feed NaN to predict.
        if not np.all(np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        gmm_probs = self._gmm.transform(X)
        return np.hstack([X, gmm_probs])

    def predict(self, X_aug: np.ndarray) -> np.ndarray:
        if self._current is None:
            raise RuntimeError("No model selected")
        predictor, _ = self._models[self._current]
        return predictor.predict(X_aug)

    # -- training / loading ------------------------------------------------

    def load_or_train(self) -> None:
        """Load serialized models from disk, or train from scratch.

        A cached manifest is only honored when its ``feature_version``
        matches the current pipeline (``v2+cluster``) so stale v1 caches
        are rejected automatically.
        """
        self._model_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._model_dir / "manifest.joblib"

        if manifest_path.exists():
            try:
                meta = joblib.load(manifest_path)
            except Exception:
                logger.warning("Failed to read manifest; retraining", exc_info=True)
                meta = None
            if meta and meta.get("feature_version") == FEATURE_VERSION:
                self._load_from_disk(meta)
                return
            logger.info(
                "Cached manifest is %s, current pipeline is %s — retraining.",
                meta.get("feature_version") if meta else "missing",
                FEATURE_VERSION,
            )

        self._train_and_persist()

    # -- internal: load -----------------------------------------------------

    def _load_from_disk(self, meta: dict) -> None:
        logger.info("Loading cached v2+cluster pipeline from %s", self._model_dir)
        self._gmm = joblib.load(self._model_dir / "gmm_featurizer.joblib")
        self._aug_scaler = joblib.load(self._model_dir / "aug_scaler.joblib")
        for name in meta["names"]:
            path = self._model_dir / f"{_safe_filename(name)}.joblib"
            estimator, needs_scaling = joblib.load(path)
            predictor = (
                _ScaledPredictor(estimator, self._aug_scaler) if needs_scaling
                else estimator
            )
            self._models[name] = (predictor, needs_scaling)
        self._current = meta.get("default", meta["names"][0])
        logger.info("Loaded %d models, active: %s", len(self._models), self._current)

    # -- internal: train ----------------------------------------------------

    def _train_and_persist(self) -> None:
        logger.info("No usable cache — training v2+cluster from data ...")
        train_ev = load_events("train")
        train_comp = load_components("train")
        assert train_ev["Attack ID"].is_unique, "events Attack ID must be unique"
        assert not train_comp.empty, "components table must be non-empty"

        # Match v1's startup-time budget: subsample events for faster
        # cold start, then keep only the components that belong to the
        # sampled events so the join is still valid.
        MAX_ROWS = 50_000
        if len(train_ev) > MAX_ROWS:
            logger.info("Subsampling %d → %d events", len(train_ev), MAX_ROWS)
            n_total = len(train_ev)
            sampled_parts = []
            for label, idx in train_ev.groupby("Type", observed=True).groups.items():
                n_class = len(idx)
                take = min(n_class, max(1, MAX_ROWS * n_class // n_total))
                sampled_parts.append(
                    train_ev.loc[idx].sample(n=take, random_state=42)
                )
            train_ev = pd.concat(sampled_parts, ignore_index=True)
            train_comp = train_comp[
                train_comp["Attack ID"].isin(train_ev["Attack ID"])
            ].reset_index(drop=True)

        logger.info("Featurizing %d events / %d components", len(train_ev), len(train_comp))
        fm = featurize(train_ev, train_comp, label_col="Type")
        X, y = fm.X, fm.y

        logger.info("Fitting per-class GMM featurizer (BIC over K∈{2..5}) ...")
        self._gmm = PerClassGMMFeaturizer().fit(X, y)
        logger.info("  K per class: %s (total cluster features: %d)",
                    self._gmm.k_chosen_, self._gmm.n_features_out)

        X_aug = np.hstack([X, self._gmm.transform(X)])
        self._aug_scaler = StandardScaler().fit(X_aug)
        X_aug_s = self._aug_scaler.transform(X_aug)

        for name, (factory, needs_scaling) in _model_specs().items():
            logger.info("Training %s ...", name)
            estimator = factory()
            estimator.fit(X_aug_s if needs_scaling else X_aug, y)
            predictor = (
                _ScaledPredictor(estimator, self._aug_scaler) if needs_scaling
                else estimator
            )
            self._models[name] = (predictor, needs_scaling)
            joblib.dump(
                (estimator, needs_scaling),
                self._model_dir / f"{_safe_filename(name)}.joblib",
            )

        joblib.dump(self._gmm, self._model_dir / "gmm_featurizer.joblib")
        joblib.dump(self._aug_scaler, self._model_dir / "aug_scaler.joblib")

        default_name = "Random Forest"
        joblib.dump(
            {
                "names": list(self._models.keys()),
                "default": default_name,
                "feature_version": FEATURE_VERSION,
            },
            self._model_dir / "manifest.joblib",
        )
        self._current = default_name
        logger.info("Trained and cached %d models. Active: %s",
                    len(self._models), self._current)
