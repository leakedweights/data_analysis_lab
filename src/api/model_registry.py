"""Train, serialize, and load ML models for the DDoS detection demo."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.simulator import TYPE_TO_INT
from src.utils.data_pipeline import load_events

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "Packet speed", "Data speed", "Avg packet len",
    "Avg source IP count", "Detect count", "Port number",
]

MODEL_DIR = Path("/app/models")


class _ScaledPredictor:
    """Wraps a model with a pre-fitted scaler."""

    def __init__(self, model, scaler: StandardScaler):
        self._model = model
        self._scaler = scaler

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df["Type"].map(TYPE_TO_INT).values
    return X, y


def _build_models(
    X_train: np.ndarray, y_train: np.ndarray, scaler: StandardScaler
) -> dict[str, object]:
    """Train all 6 models and return name→predictor mapping."""
    X_train_s = scaler.transform(X_train)

    specs: list[tuple[str, object, np.ndarray]] = [
        ("Baseline (majority)", DummyClassifier(strategy="most_frequent"), X_train),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            X_train_s,
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
            X_train,
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100, max_depth=15, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            X_train,
        ),
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5, n_jobs=-1), X_train_s),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.15, random_state=42,
            ),
            X_train,
        ),
    ]

    trained: dict[str, object] = {}
    for name, model, X in specs:
        logger.info("Training %s ...", name)
        model.fit(X, y_train)
        needs_scaling = name in ("Logistic Regression", "KNN (k=5)")
        trained[name] = _ScaledPredictor(model, scaler) if needs_scaling else model

    return trained


class ModelRegistry:
    """Holds all trained models in memory with a switchable current model."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self._model_dir = model_dir
        self._models: dict[str, object] = {}
        self._current: str | None = None

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    @property
    def current_name(self) -> str | None:
        return self._current

    @property
    def current_model(self) -> object | None:
        if self._current is None:
            return None
        return self._models.get(self._current)

    def select(self, name: str) -> None:
        if name not in self._models:
            raise KeyError(f"Unknown model: {name}")
        self._current = name

    def load_or_train(self) -> None:
        """Load serialized models from disk, or train from scratch."""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._model_dir / "manifest.joblib"

        if manifest.exists():
            logger.info("Loading cached models from %s", self._model_dir)
            meta = joblib.load(manifest)
            for name in meta["names"]:
                path = self._model_dir / f"{_safe_filename(name)}.joblib"
                self._models[name] = joblib.load(path)
            self._current = meta.get("default", meta["names"][0])
            logger.info("Loaded %d models, active: %s", len(self._models), self._current)
            return

        logger.info("No cached models found — training from data ...")
        train_ev = load_events("train")
        # Subsample for faster startup in demo; 50k stratified rows is sufficient
        MAX_ROWS = 50_000
        if len(train_ev) > MAX_ROWS:
            logger.info("Subsampling %d → %d rows for faster training", len(train_ev), MAX_ROWS)
            train_ev = (
                train_ev.groupby("Type", group_keys=False, observed=True)
                .apply(lambda g: g.sample(n=min(len(g), MAX_ROWS * len(g) // len(train_ev)), random_state=42))
                .reset_index(drop=True)
            )
        X_train, y_train = _prepare_features(train_ev)

        scaler = StandardScaler()
        scaler.fit(X_train)

        self._models = _build_models(X_train, y_train, scaler)

        # Serialize
        for name, model in self._models.items():
            path = self._model_dir / f"{_safe_filename(name)}.joblib"
            joblib.dump(model, path)

        default_name = "Random Forest"
        joblib.dump(
            {"names": list(self._models.keys()), "default": default_name},
            manifest,
        )
        self._current = default_name
        logger.info("Trained and cached %d models.", len(self._models))


def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")
