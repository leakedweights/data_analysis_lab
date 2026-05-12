"""Side-by-side comparison: v2 baseline (model_comparison_v2_orig.csv) vs
v2 ABC (model_comparison_v2_abc.csv). Reports macro F1 deltas on test +
genericity and runs per-class metrics on the new ABC bundle so we can see
which class moved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features_v2 import featurize
from src.simulator import TYPE_ORDER
from src.train_v2 import _gen_prior_weights
from src.utils.data_pipeline import load_components, load_events

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _f1_table(df: pd.DataFrame, feats: str, resample: str) -> pd.DataFrame:
    # Bracket access — ``df.resample`` collides with the DataFrame method.
    sub = df[(df["features"] == feats) & (df["resample"] == resample)]
    return sub.pivot_table(
        index="model", columns="dataset", values="f1_macro", aggfunc="first"
    )


def main() -> None:
    orig = pd.read_csv(RESULTS / "model_comparison_v2_orig.csv")
    abc = pd.read_csv(RESULTS / "model_comparison_v2_abc.csv")

    orig_t = _f1_table(orig, "v2", "none")
    abc_t = _f1_table(abc, "v2_abc", "none")

    common = sorted(set(orig_t.index) & set(abc_t.index))
    cmp = pd.DataFrame(index=common)
    cmp["v2_test"] = orig_t.loc[common, "test"]
    cmp["abc_test"] = abc_t.loc[common, "test"]
    cmp["Δ test"] = cmp["abc_test"] - cmp["v2_test"]
    cmp["v2_gen"] = orig_t.loc[common, "genericity"]
    cmp["abc_gen"] = abc_t.loc[common, "genericity"]
    cmp["Δ gen"] = cmp["abc_gen"] - cmp["v2_gen"]
    print("=" * 78)
    print("F1 MACRO — v2 (orig) vs v2 ABC (Fix A+B+C, no resampling)")
    print("=" * 78)
    print(cmp.round(4).to_string())

    # Per-class metrics on the new bundle.
    tr_e = load_events("train"); tr_c = load_components("train")
    te_e = load_events("test"); te_c = load_components("test")
    gn_e = load_events("genericity"); gn_c = load_components("genericity")
    tr = featurize(tr_e, tr_c, "Type")
    te = featurize(te_e, te_c, "Type")
    gn = featurize(gn_e, gn_c, "Type")

    cw = _gen_prior_weights(tr.y)
    print(f"\nGen-prior class weights: {cw}")

    scaler = StandardScaler().fit(tr.X)
    Xtr_s = scaler.transform(tr.X)
    Xte_s = scaler.transform(te.X)
    Xgn_s = scaler.transform(gn.X)

    models = {
        "LR":  (LogisticRegression(max_iter=1000, class_weight=cw, random_state=42), True),
        "DT":  (DecisionTreeClassifier(max_depth=10, class_weight=cw, random_state=42), False),
        "RF":  (RandomForestClassifier(n_estimators=100, max_depth=15,
                  class_weight=cw, random_state=42, n_jobs=-1), False),
        "GBT": (HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                  learning_rate=0.1, class_weight=cw, random_state=42), False),
    }
    for _, (m, scale) in models.items():
        m.fit(Xtr_s if scale else tr.X, tr.y)

    for split, fm, Xs in [("TEST", te, Xte_s), ("GEN ", gn, Xgn_s)]:
        print(f"\n===== ABC — {split} =====")
        for name, (m, scale) in models.items():
            yp = m.predict(Xs if scale else fm.X)
            print(f"\n--- {name} [{split}] ---")
            print(classification_report(
                fm.y, yp, target_names=TYPE_ORDER,
                labels=list(range(len(TYPE_ORDER))), zero_division=0, digits=3))
            cm = confusion_matrix(
                fm.y, yp, labels=list(range(len(TYPE_ORDER))))
            cm_n = (cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100).round(1)
            print("  row% confusion (rows=true, cols=pred [Norm,Susp,DDoS]):")
            for i, t in enumerate(TYPE_ORDER):
                print(f"    {t:20s}: {cm_n[i].tolist()}   (n={cm.sum(axis=1)[i]})")


if __name__ == "__main__":
    main()
