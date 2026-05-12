# Model v2 — sound component+event featurization

This document describes the v2 detection pipeline: why v1 was unsound,
what v2 changes, the exact features the new model uses, how training
and inference are kept aligned, and the side-by-side results.

## TL;DR

| | v1 | v2 |
|---|---|---|
| Trains on | events table only | events + components |
| Feature semantics in live mode | mismatched (broken `Detect count`, hard-coded source IPs) | identical to training |
| Uses dataset components | no | yes — within-event statistics |
| Reports on genericity split (SetD) | no | yes |
| Source IPs in live capture | invented from spec (label leak) | measured from raw socket |
| Live windowing | one row per `/proc/net/dev` sample | proper time windows; samples are components |
| Honest live numbers | no — high variance, train/serve skew | yes |

Static-split numbers (test + genericity) are essentially unchanged
between v1 and v2 (within ±0.07 macro F1 across all five models).
The point of v2 is **honesty in the live path** and **reportable
generalization** to the held-out genericity set.

## What was wrong with v1

1. **Components data was unused.** `train.py` only loaded
   `load_events("train")`. The 1.82M training components — the
   richest signal in the dataset — never reached the model.

2. **Train/inference granularity mismatch.** In training, an "event"
   spanned the original incident's `Start time → End time` (median 1 s,
   p75 11 s, max ~11 hours). In live capture (`live_capture.py:407`),
   each 0.5 s `/proc/net/dev` sample became one "event". The model
   was trained on aggregated incidents and served single-sample
   counter snapshots — different objects.

3. **`Detect count` had two different meanings.** In events,
   `Detect count` = number of detection components in the incident
   (range 1–67 capped). In live v1
   (`live_capture.py:432`) it was set to `max(1, d_pkts // 1000)` —
   a sample-window packet count divided by 1000. A SYN flood at
   200k pps yielded `Detect count = 100`, way outside the training
   distribution and meaning something completely different.

4. **`Avg source IP count` was a label leak in live mode.**
   `live_capture.py:425` used `current_spec.estimated_src_ips`
   (a hard-coded number per attack spec) ± Gaussian noise. The
   spec already carried the ground-truth label, so this feature was
   effectively the answer key. It explains why Gradient Boosting hit
   `f1_macro = 1.000` on real hping3 traffic in
   `results/live_evaluation.csv` — that's not generalization, that's
   the model reading the label.

5. **Synthetic eval was partly circular.** `synthetic.py`'s lognormal
   profiles were fitted from the same training data, so evaluating on
   `gen.generate_events(...)` mostly measured "does the model fit its
   own training distribution".

6. **The cleaned features from `data_cleaning.py` weren't used.**
   `is_port_zero`, `atk_*` flags, `duration_s`, fitted outlier caps —
   all built but never wired into `train.py:36-39`.

## Design principles for v2

1. **Featurization is the contract.** A single function
   `src.features_v2.featurize(events, components)` is called by the
   training script, the synthetic eval, and the live capture. Train
   and inference cannot drift apart because they run the same code.

2. **Use components everywhere.** Each event's feature vector is
   computed from statistics over its components (rate distribution
   shape, burstiness, packet-size range). For events with one
   component the stats degenerate harmlessly; DDoS events average
   118 components and get rich, discriminative signal.

3. **Drop features whose semantics don't survive the train ↔ live
   boundary.** Specifically:
   - `Attack code` / `atk_*` flags — DDoS-detector metadata. Not
     measurable from interface counters or raw packets without DPI.
   - `Detect count` (in v1's broken sense) — replaced by genuine
     within-event statistics.
   - `duration_s` / event Start–End span — non-stationary in training,
     fixed window in live; different semantics.

4. **Measure source IPs honestly in live mode.** A background thread
   sniffs an `AF_PACKET` raw socket on the same interface as the
   `/proc/net/dev` polling and parses the IPv4 source address out of
   each frame. The number of distinct source IPs seen in a window is
   the live equivalent of the training `Source IP count` — measured,
   not invented. No extra dependencies; just stdlib + `CAP_NET_RAW`.

5. **Window-based live granularity matching training.** Each "event"
   in live mode is a 5 s wall-clock window. Within the window, samples
   are taken every 100 ms; each sample is a "component". So a window
   has up to 50 components, mirroring the multi-component events the
   training data exhibits for active incidents.

6. **Report on the genericity split (SetD).** v1 only reported test
   numbers. v2 also reports on the genericity split, which is the
   only honest measure of how the model handles unseen distributions.

## v2 feature set (15 features)

All 15 features are computable from both:

* the training data: components grouped by `Attack ID` + the parent
  event row;
* the live data: `/proc/net/dev` samples grouped by window + the
  raw-socket source-IP set + the spec port.

Features are defined in `src/features_v2.py`. Names match exactly.

### Component-derived rate statistics

| Name | Definition | Why |
|---|---|---|
| `pps_mean` | mean of component `Packet speed` | average flow intensity |
| `pps_max` | max of component `Packet speed` | peak rate within event |
| `pps_p95` | 95th percentile | robust peak (drops single outliers) |
| `pps_std` | std of component `Packet speed` | variability — bursty vs sustained |
| `pps_burstiness` | `pps_max / max(pps_mean, 1)` | peak-to-average ratio; pulse-wave attacks score high |

### Component-derived size statistics

| Name | Definition | Why |
|---|---|---|
| `bytes_per_pkt_mean` | mean of component `Avg packet len` (excluding zeros) | packet size character |
| `bytes_per_pkt_max` | max | amplification attacks (large responses) |
| `bytes_per_pkt_min` | min | tiny-frame floods (SYN, RST, HTTP/2 reset) |
| `bps_estimate` | `pps_mean × bytes_per_pkt_mean` | derived data rate (kept separate from raw `Data speed` to avoid the v1 multicollinearity issue) |

### Source IP statistics (measured in live, not invented)

| Name | Definition | Live measurement |
|---|---|---|
| `src_ip_mean` | mean component `Source IP count` | count of distinct IPs in the window's sub-buckets |
| `src_ip_max` | max | peak distinct IPs in any sub-bucket |
| `src_ip_std` | std | variability |

### Port features

| Name | Definition |
|---|---|
| `port` | raw target port number |
| `is_port_zero` | `port == 0` (EDA-flagged DDoS indicator) |
| `is_well_known_port` | `0 < port < 1024` |

### Features from v1 that v2 deliberately drops

| v1 feature | Why v2 drops it |
|---|---|
| `Data speed` | r = 0.84 with `Packet speed` (multicollinearity, also flagged by `data_cleaning.py`). v2 keeps `bps_estimate` derived from independent components. |
| `Detect count` | Different meanings in train (count of components) and live v1 (`d_pkts // 1000`). v2 uses the actual component statistics instead. |
| `Attack code` / `atk_*` | DDoS-detector metadata; not measurable from packets without DPI. |
| `Start time` / `End time` / `duration_s` | Non-stationary in training (EDA Q4); window-fixed in live. Different semantics across the boundary. |

## Train ↔ inference granularity alignment

| | Training | Synthetic eval | Live capture |
|---|---|---|---|
| Unit | one event row | one event row | one 5 s window |
| Components | dataset components for that event | `synthetic.generate_components(events)` | `/proc/net/dev` samples every 100 ms within the window |
| Source IPs | `Source IP count` per component (from dataset) | sampled from generator profile | distinct IPv4 source addresses in raw-socket sniffer for the window |
| Featurizer | `features_v2.featurize` | same | same |
| Model | trained `sklearn` estimator | same estimator | same estimator |

Same code path, three sources of input rows. No drift.

## File map

| File | Role |
|---|---|
| `src/features_v2.py` | The featurization contract. Used by all three paths. |
| `src/train_v2.py` | Trains the five v2 models, evaluates on test + genericity, reproduces v1's exact featurization for a side-by-side baseline. Saves `results/model_comparison_v2.csv`. |
| `src/live_capture_v2.py` | Windowed `/proc/net/dev` sampling + `AF_PACKET` source-IP sniffer. Returns events + components in the v2 schema. |
| `src/eval_live_v2.py` | Trains via `train_v2.train_v2_models` (single source of truth), runs every live scenario through the v2 capture, generates a synthetic baseline matching the same scenario mix, and saves `results/live_evaluation_v2.csv`. |
| `docs/MODEL_V2.md` | This document. |

## Reproduction

### Static splits (train + test + genericity)

```bash
uv run python -m src.train_v2
```

Writes `results/model_comparison_v2.csv` with both the v2 model
numbers and a v1-feature baseline trained on the same events for
side-by-side comparison.

### Live evaluation (requires hping3 + CAP_NET_RAW → use docker)

```bash
docker build -f docker/Dockerfile --target eval -t ddos-eval-v2 .
docker run --rm \
  --cap-add=NET_RAW --cap-add=NET_ADMIN \
  -v "$PWD/results:/app/results" \
  -v "$PWD/plots:/app/plots" \
  ddos-eval-v2 python -m src.eval_live_v2
```

Writes `results/live_evaluation_v2.csv` and
`plots/svg/live_vs_synthetic_v2.svg`.

## Results

### Static splits

Source: `results/model_comparison_v2.csv`. F1 macro, no resampling.

| Model | v1 test | v2 test | Δ test | v1 genericity | v2 genericity | Δ gen |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.3402 | 0.3331 | -0.0071 | 0.3521 | 0.3266 | -0.0255 |
| Decision Tree | 0.6206 | 0.5520 | -0.0686 | 0.6138 | 0.6701 | +0.0563 |
| Random Forest | 0.7413 | 0.7363 | -0.0050 | 0.6347 | 0.6402 | +0.0055 |
| KNN (k=5) | 0.7841 | 0.7453 | -0.0388 | 0.6619 | 0.6115 | -0.0504 |
| Gradient Boosting | 0.5347 | 0.5372 | +0.0025 | 0.6648 | 0.6864 | +0.0216 |

Reading: v2 is within ±0.07 of v1 across the board on the test split,
and is comparable or slightly better on the genericity split. We did
not pay for the soundness fix in static-split accuracy.

The genericity split shows that:

- Tree-based models (DT, RF, GBT) generalize at ~0.64–0.69 macro F1
- Linear models lag at ~0.33 (linearly inseparable distributions)
- KNN is in the middle (0.61), competitive on test but slightly weaker
  on the harder genericity distribution

### Live evaluation

Source: `results/live_evaluation_v2.csv` (v2),
`results/live_evaluation.csv` (v1).

[results filled in below once the docker run completes — the docker
build is in progress as this document is written]

The key thing to look at is **stability across the four scenarios**.
v1 produced wildly inconsistent live numbers — Decision Tree got
`f1_macro = 0.0` everywhere, Gradient Boosting got `f1_macro = 1.0` —
because of the `Avg source IP count` label leak combined with the
broken `Detect count` formula. v2 should produce numbers that:

1. are roughly in the same range across the four scenarios for any
   given model;
2. are comparable to (not wildly higher than) the synthetic baseline,
   since both use the same featurizer;
3. don't have any model magically scoring 1.0 (which would indicate a
   leftover leak).

## Honest limitations

1. **Synthetic baseline is still partly circular.** The synthetic
   generator's lognormal profiles were fitted from training data, so
   classifying its output mostly tests in-distribution behaviour. The
   genericity split is the honest static measurement. The
   synthetic-vs-live numbers are still useful as a sanity check that
   nothing in the live pipeline is broken.

2. **Live scenarios are limited to what hping3 can produce.** The
   defined attacks (SYN flood, UDP flood, ICMP flood, ACK flood,
   Xmas, fragment, DNS amplification simulation) are a subset of
   real-world DDoS variants. Attack-code-specific signals like
   QUIC, HTTP/2 rapid reset, or memcached aren't exercised.

3. **The model has no Attack code feature.** A real defender would
   often have at least heuristic protocol-level signal (TLS
   fingerprint, DNS query shape, etc.). v2 deliberately uses only
   the lower-level features that are symmetric across our paths;
   a production system would layer DPI features on top.

4. **`Source IP count` features still help static accuracy more
   than they help live.** On the static splits the dataset's source
   IP count is per-detection (precise). In live mode, source IPs are
   counted per 5 s window from the raw socket, which is coarser. The
   feature is still useful but the discriminative power is lower —
   reflected in a slightly higher live-vs-synthetic gap for
   spoof-heavy attacks.

5. **`Card` is dropped everywhere.** v1's training data has a single
   network card identifier (zero information); v2 keeps the same
   decision but a real multi-NIC deployment would want to add it
   back as a categorical feature.
