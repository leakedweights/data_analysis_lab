"""v2 live evaluation — train, run hping3 scenarios, compare honestly.

Trains the v2 models from ``train_v2.train_v2_models`` (single source of
truth), runs each live scenario through ``LiveTrafficGeneratorV2``, and
generates a synthetic baseline matching the same scenario profile.

Crucial differences from ``eval_live.py`` (v1):

* Models train on the v2 feature set (component-aggregate stats, real
  source IP measurements when available).
* The live capture path is windowed and uses the same featurizer as
  training, so there is no train/serve skew.
* The synthetic baseline is also componentized (via
  ``synthetic.TrafficGenerator.generate_components``) so it goes through
  the *same* featurizer — the comparison is apples-to-apples.

Usage (inside docker eval target with hping3 + NET_RAW):
    python -m src.eval_live_v2
    python -m src.eval_live_v2 --scenarios live_syn_flood live_dns_amp
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.features_v2 import featurize
from src.live_capture_v2 import LIVE_SCENARIOS, LiveTrafficGeneratorV2, hping3_available
from src.simulator import TYPE_ORDER
from src.synthetic import TrafficGenerator
from src.train_v2 import V2Bundle, train_v2_models
from src.utils.data_pipeline import load_components, load_events

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "svg"


def _train_bundle() -> V2Bundle:
    print("Loading training data...")
    train_ev = load_events("train")
    train_co = load_components("train")
    print(f"  {len(train_ev):,} events / {len(train_co):,} components")
    print("Featurizing...")
    fm = featurize(train_ev, train_co, label_col="Type")
    print(f"  {fm.X.shape}")
    print("Training models...")
    return train_v2_models(fm)


def _evaluate(bundle: V2Bundle, name: str, X, y, source: str, scenario: str) -> dict:
    t0 = time.perf_counter()
    y_pred = bundle.predict(name, X)
    infer_s = time.perf_counter() - t0
    n = int(len(y))

    f1_mac = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=list(range(len(TYPE_ORDER))))
    rep = classification_report(y, y_pred, target_names=TYPE_ORDER,
                                labels=list(range(len(TYPE_ORDER))),
                                zero_division=0)

    print(f"\n  --- {name} [{source}] ---")
    print(f"    Events:      {n}")
    print(f"    F1 macro:    {f1_mac:.4f}")
    print(f"    F1 weighted: {f1_wt:.4f}")
    print(f"    Inference:   {infer_s:.4f}s")
    print(rep)
    print(f"    Confusion matrix:\n{cm}")

    return {
        "model": name,
        "source": source,
        "scenario": scenario,
        "n_events": n,
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_wt), 4),
        "inference_time_s": round(float(infer_s), 4),
        "throughput_eps": round(n / infer_s) if infer_s > 0 else 0,
    }


def _synthetic_for_scenario(name: str, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic events + components matching a live scenario's
    traffic-type mix."""
    from src.live_capture import HPING_ATTACKS
    scenario = LIVE_SCENARIOS[name]
    type_seconds = {"Normal traffic": 0.0, "Suspicious traffic": 0.0, "DDoS attack": 0.0}
    for phase in scenario.phases:
        spec = HPING_ATTACKS[phase.attack_name]
        type_seconds[spec.traffic_type] += phase.duration_s
    total = sum(type_seconds.values()) or 1.0
    ddos_ratio = type_seconds["DDoS attack"] / total
    susp_ratio = type_seconds["Suspicious traffic"] / total

    n_events = max(20, int(scenario.duration_seconds / 5))  # ~one window per 5s
    gen = TrafficGenerator(seed=seed)
    events = gen.generate_events(
        n=n_events, ddos_ratio=ddos_ratio, suspicious_ratio=susp_ratio,
    )
    components = gen.generate_components(events)
    return events, components


def _run_live(name: str, target: str, interface: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = LIVE_SCENARIOS[name]
    gen = LiveTrafficGeneratorV2(target=target, interface=interface, window_s=5.0,
                                 sample_s=0.1)
    print(f"\n  Running live scenario '{name}' ({scenario.duration_seconds}s)...")
    events, components = gen.run_scenario(scenario)
    print(f"  Captured {len(events)} windows / {len(components)} samples")
    return events, components


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 live evaluation")
    parser.add_argument("--scenarios", nargs="+",
                        default=list(LIVE_SCENARIOS.keys()),
                        choices=list(LIVE_SCENARIOS.keys()))
    parser.add_argument("--target", default="127.0.0.1")
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip hping3 (synthetic baseline only)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = _train_bundle()

    has_hping3 = hping3_available() and not args.skip_live
    if not has_hping3:
        print("\nhping3 not available (or --skip-live). Synthetic baseline only.")

    all_results: list[dict] = []

    for scenario_name in args.scenarios:
        print(f"\n{'=' * 72}\nSCENARIO: {scenario_name}\n{'=' * 72}")

        synth_ev, synth_co = _synthetic_for_scenario(scenario_name)
        synth_fm = featurize(synth_ev, synth_co, label_col="Type")
        print(f"\n  Synthetic baseline: {len(synth_ev)} events")

        for name in bundle.names():
            all_results.append(_evaluate(
                bundle, name, synth_fm.X, synth_fm.y,
                f"synthetic_{scenario_name}", scenario_name,
            ))

        if has_hping3:
            live_ev, live_co = _run_live(scenario_name, args.target, args.interface)
            if len(live_ev) == 0:
                print("  No live events captured")
                continue
            live_fm = featurize(live_ev, live_co, label_col="Type")
            for name in bundle.names():
                all_results.append(_evaluate(
                    bundle, name, live_fm.X, live_fm.y,
                    f"hping3_{scenario_name}", scenario_name,
                ))

    results_df = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "live_evaluation_v2.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    print("\n" + "=" * 72)
    print("SUMMARY (v2)")
    print("=" * 72)
    for scenario_name in args.scenarios:
        sc_df = results_df[results_df["scenario"] == scenario_name]
        synth = sc_df[sc_df["source"].str.startswith("synthetic")]
        live = sc_df[sc_df["source"].str.startswith("hping3")]
        print(f"\n--- {scenario_name} ---")
        if len(live) > 0:
            merged = (
                synth[["model", "f1_macro"]].rename(columns={"f1_macro": "synth"})
                .merge(live[["model", "f1_macro"]].rename(columns={"f1_macro": "live"}),
                       on="model")
            )
            merged["delta"] = merged["live"] - merged["synth"]
            print(merged.to_string(index=False))
        else:
            print(synth[["model", "f1_macro"]].to_string(index=False))

    _generate_plots(results_df)


def _generate_plots(df: pd.DataFrame) -> None:
    if df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = df["scenario"].unique()
    MODELS = list(df["model"].unique())
    SHORT = {
        "Logistic Regression": "LogReg", "Decision Tree": "DTree",
        "Random Forest": "RF", "KNN (k=5)": "KNN", "Gradient Boosting": "GBT",
    }

    n_sc = len(scenarios)
    fig, axes = plt.subplots(1, n_sc, figsize=(6 * n_sc, 5.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, scenario in zip(axes, scenarios):
        sc_df = df[df["scenario"] == scenario]
        synth = sc_df[sc_df["source"].str.startswith("synthetic")]
        live = sc_df[sc_df["source"].str.startswith("hping3")]
        models = [m for m in MODELS if m in synth["model"].values]
        x = np.arange(len(models))
        width = 0.35

        synth_f1 = [synth.loc[synth["model"] == m, "f1_macro"].iloc[0] for m in models]
        ax.bar(x - width / 2, synth_f1, width, label="Synthetic",
               color="#2563eb", edgecolor="white", linewidth=0.5)

        if not live.empty:
            live_f1 = [
                live.loc[live["model"] == m, "f1_macro"].iloc[0]
                if m in live["model"].values else 0.0
                for m in models
            ]
            ax.bar(x + width / 2, live_f1, width, label="hping3 (real)",
                   color="#ef4444", edgecolor="white", linewidth=0.5)

        sc_label = scenario.replace("live_", "").replace("_", " ").title()
        ax.set_title(sc_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(m, m) for m in models], fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("F1 macro", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("v2: Synthetic vs Real (hping3) — windowed, source-IP measured",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "live_vs_synthetic_v2.svg", format="svg",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PLOTS_DIR / 'live_vs_synthetic_v2.svg'}")


if __name__ == "__main__":
    main()
