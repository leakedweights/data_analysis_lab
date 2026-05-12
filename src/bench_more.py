"""Benchmark additional derived features beyond v2 A+B.

Three configurations across all five v2 models with
``class_weight="balanced"`` (kept fixed so the feature additions are
isolated from the class-weight calibration):

* ``ab``           — 21 v2 A+B features (current featurizer).
* ``ab_extras``    — 21 v2 + 7 extras (component count + robust stats).
* ``ab_extras_ood`` — 21 v2 + 7 extras + 2 OOD summary columns
                     (``gmm_max_prob``, ``gmm_entropy``) from per-class
                     GMMs fit on the v2 features.

Output: ``results/more_per_class.csv`` with one row per
(model, config, dataset, class) plus a synthetic ``__macro__`` row.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.cluster_features import PerClassGMMFeaturizer
from src.features_v2 import featurize
from src.features_v2_extras import EXTRA_FEATURE_NAMES, extra_features
from src.simulator import TYPE_ORDER
from src.utils.data_pipeline import load_components, load_events

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _make_models():
    cw = "balanced"
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight=cw, random_state=42), True),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=10, class_weight=cw, random_state=42), False),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, max_depth=15, class_weight=cw,
                                   random_state=42, n_jobs=-1), False),
        "KNN (k=5)": (
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.1,
                                           class_weight=cw, random_state=42), False),
    }


def _train_and_eval(
    config: str, X_train, y_train, X_test, y_test, X_gen, y_gen,
) -> list[dict]:
    print(f"\n=== {config} ({X_train.shape[1]} features) ===")
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)
    Xgn_s = scaler.transform(X_gen)

    rows: list[dict] = []
    for name, (m, scale) in _make_models().items():
        t0 = time.perf_counter()
        m.fit(Xtr_s if scale else X_train, y_train)
        print(f"  fit {name:<22} in {time.perf_counter()-t0:5.1f}s")
        for split, (X, Xs, y) in [
            ("test",       (X_test, Xte_s, y_test)),
            ("genericity", (X_gen,  Xgn_s, y_gen)),
        ]:
            yp = m.predict(Xs if scale else X)
            p, r, f, s = precision_recall_fscore_support(
                y, yp, labels=list(range(len(TYPE_ORDER))), zero_division=0)
            for ci, cname in enumerate(TYPE_ORDER):
                rows.append({
                    "model": name, "config": config, "dataset": split,
                    "class": cname, "precision": float(p[ci]),
                    "recall": float(r[ci]), "f1": float(f[ci]),
                    "support": int(s[ci]),
                })
            rows.append({
                "model": name, "config": config, "dataset": split,
                "class": "__macro__",
                "precision": float(np.mean(p)),
                "recall": float(np.mean(r)),
                "f1": float(f1_score(y, yp, average="macro", zero_division=0)),
                "support": int(np.sum(s)),
            })
    return rows


def main() -> None:
    print("loading splits...")
    tr_e = load_events("train"); tr_c = load_components("train")
    te_e = load_events("test"); te_c = load_components("test")
    gn_e = load_events("genericity"); gn_c = load_components("genericity")

    print("featurizing v2 A+B...")
    tr = featurize(tr_e, tr_c, "Type")
    te = featurize(te_e, te_c, "Type")
    gn = featurize(gn_e, gn_c, "Type")

    print("computing extras (component count + robust stats)...")
    t0 = time.perf_counter()
    Etr = extra_features(tr_e, tr_c)
    Ete = extra_features(te_e, te_c)
    Egn = extra_features(gn_e, gn_c)
    print(f"  done in {time.perf_counter()-t0:.1f}s; shape={Etr.shape}")
    print(f"  extras: {EXTRA_FEATURE_NAMES}")

    print("fitting per-class GMMs for OOD signal...")
    t0 = time.perf_counter()
    clusterer = PerClassGMMFeaturizer().fit(tr.X, tr.y)
    print(f"  fit in {time.perf_counter()-t0:.1f}s; K per class={clusterer.k_chosen_}")
    Otr = clusterer.transform_summary(tr.X)
    Ote = clusterer.transform_summary(te.X)
    Ogn = clusterer.transform_summary(gn.X)
    print(f"  OOD shape: {Otr.shape}; cols={clusterer.summary_feature_names}")

    Xtr_e   = np.concatenate([tr.X, Etr], axis=1)
    Xte_e   = np.concatenate([te.X, Ete], axis=1)
    Xgn_e   = np.concatenate([gn.X, Egn], axis=1)
    Xtr_eo  = np.concatenate([tr.X, Etr, Otr], axis=1)
    Xte_eo  = np.concatenate([te.X, Ete, Ote], axis=1)
    Xgn_eo  = np.concatenate([gn.X, Egn, Ogn], axis=1)

    rows: list[dict] = []
    rows += _train_and_eval("ab",
                            tr.X,   tr.y, te.X,   te.y, gn.X,   gn.y)
    rows += _train_and_eval("ab_extras",
                            Xtr_e,  tr.y, Xte_e,  te.y, Xgn_e,  gn.y)
    rows += _train_and_eval("ab_extras_ood",
                            Xtr_eo, tr.y, Xte_eo, te.y, Xgn_eo, gn.y)

    df = pd.DataFrame(rows)
    out = RESULTS / "more_per_class.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(df[df["class"] == "__macro__"]
          .pivot_table(index=["model", "config"], columns="dataset",
                       values="f1", aggfunc="first")
          .round(4).to_string())


if __name__ == "__main__":
    main()
