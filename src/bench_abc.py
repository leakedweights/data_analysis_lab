"""Run all three configurations end-to-end and dump per-class metrics +
macro F1 to ``results/abc_per_class.csv`` so the plot script can build
the comparison figures from a single source.

Configurations
--------------
* ``v2``     — original v2 featurizer (``features_v2_baseline``) with
               sklearn ``class_weight="balanced"``.
* ``ab``     — Fix A+B featurizer (``features_v2``) with balanced.
* ``abc``    — Fix A+B featurizer + gen-prior class weights.

Output schema: model, config, dataset, class, precision, recall, f1,
support, plus a separate ``f1_macro`` row per (model, config, dataset).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features_v2 import featurize as featurize_ab
from src.features_v2_baseline import featurize_baseline
from src.simulator import TYPE_ORDER
from src.train_v2 import _gen_prior_weights
from src.utils.data_pipeline import load_components, load_events

RESULTS = Path(__file__).resolve().parents[1] / "results"

MODEL_NAMES = [
    "Logistic Regression", "Decision Tree", "Random Forest",
    "KNN (k=5)", "Gradient Boosting",
]


def _make_models(class_weight):
    cw = "balanced" if class_weight is None else class_weight
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


def _train_and_eval(config: str, train_fm, test_fm, gen_fm, class_weight) -> list[dict]:
    print(f"\n=== {config} ===")
    scaler = StandardScaler().fit(train_fm.X)
    Xtr_s = scaler.transform(train_fm.X)
    Xte_s = scaler.transform(test_fm.X)
    Xgn_s = scaler.transform(gen_fm.X)

    rows: list[dict] = []
    for name, (m, scale) in _make_models(class_weight).items():
        print(f"  fitting {name}...")
        m.fit(Xtr_s if scale else train_fm.X, train_fm.y)
        for split, fm, Xs in [("test", test_fm, Xte_s),
                              ("genericity", gen_fm, Xgn_s)]:
            yp = m.predict(Xs if scale else fm.X)
            p, r, f, s = precision_recall_fscore_support(
                fm.y, yp, labels=list(range(len(TYPE_ORDER))),
                zero_division=0)
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
                "f1": float(f1_score(fm.y, yp, average="macro", zero_division=0)),
                "support": int(np.sum(s)),
            })
    return rows


def main() -> None:
    print("loading splits...")
    tr_e = load_events("train"); tr_c = load_components("train")
    te_e = load_events("test"); te_c = load_components("test")
    gn_e = load_events("genericity"); gn_c = load_components("genericity")

    print("featurizing baseline...")
    tr_b = featurize_baseline(tr_e, tr_c, "Type")
    te_b = featurize_baseline(te_e, te_c, "Type")
    gn_b = featurize_baseline(gn_e, gn_c, "Type")
    print(f"  baseline X shape: {tr_b.X.shape}")

    print("featurizing A+B...")
    tr_a = featurize_ab(tr_e, tr_c, "Type")
    te_a = featurize_ab(te_e, te_c, "Type")
    gn_a = featurize_ab(gn_e, gn_c, "Type")
    print(f"  A+B X shape: {tr_a.X.shape}")

    rows: list[dict] = []
    rows += _train_and_eval("v2", tr_b, te_b, gn_b, class_weight=None)
    rows += _train_and_eval("ab", tr_a, te_a, gn_a, class_weight=None)
    rows += _train_and_eval("abc", tr_a, te_a, gn_a,
                            class_weight=_gen_prior_weights(tr_a.y))

    df = pd.DataFrame(rows)
    out = RESULTS / "abc_per_class.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(df[df["class"] == "__macro__"]
          .pivot_table(index=["model", "config"], columns="dataset",
                       values="f1", aggfunc="first")
          .round(4).to_string())


if __name__ == "__main__":
    main()
