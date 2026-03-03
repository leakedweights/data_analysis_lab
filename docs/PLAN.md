# Project Plan — SCLDDoS2024 DDoS Detection

Roadmap for milestones M3–M6: feature engineering, modeling, evaluation, and application.

---

## 1. Project Overview

**Dataset:** SCLDDoS2024 — 264,770 events and 1,820,091 components across 3 classes (Normal 94.1%, Suspicious 5.1%, DDoS 0.9%).

**Completed work:**
- Data pipeline with CSV/Parquet loading (`src/utils/data_pipeline.py`) — see [DATA.md](DATA.md)
- Exploratory data analysis (`src/eda.py`, 10 plots) — see [EDA.md](EDA.md)
- Data cleaning pipeline (`src/utils/data_cleaning.py`) with train-fitted thresholds saved to `data/cleaning_config.json` — see [DATA.md § Data Cleaning](DATA.md#data-cleaning)

**Key EDA findings driving the plan:**
1. Extreme class imbalance (105:1 Normal-to-DDoS ratio)
2. `Avg packet len` is the strongest single discriminator (DDoS median 143 bytes vs Normal 1,278)
3. `Detect count` mean separates DDoS well (118 vs 6 for Normal), but medians are identical (1)
4. Temporal patterns are non-stationary — avoid time-dependent features
5. `Packet speed` and `Data speed` are redundant (r = 0.84); Data speed already dropped in cleaning
6. Port 0 is a DDoS indicator (already captured as `is_port_zero`)

---

## 2. Feature Engineering (M3)

**Module:** `src/feature_engineering.py`

### 2.1 Component aggregation

Join components to events via `Attack ID` and compute per-event statistics:

| Aggregation | Columns | Rationale |
|-------------|---------|-----------|
| std / max / min | Packet speed, Avg packet len | DDoS events have high intra-event variance from sustained multi-vector attacks |
| count distinct | Source IP count | Captures spoofed-IP diversity within a single event |
| time span | Component Time (max − min) | Duration of detection activity, complementing event-level `duration_s` |

### 2.2 Log transforms

Apply `log1p` to heavily right-skewed features:
- `Detect count` (median 1, max 67 after clipping, mean 6.9)
- `Avg source IP count` (median 1, max 61 after clipping, mean 6.6)

These features have long tails where the mean is several times the median. Log transforms reduce the influence of extreme values and improve linearity for models that benefit from it (e.g., Logistic Regression).

### 2.3 Interaction features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `pkt_speed_per_pkt_len` | `Packet speed / Avg packet len` | DDoS sends many small packets → high ratio |
| `detect_x_pkt_len` | `Detect count * Avg packet len` | Captures volume × size interaction |

### 2.4 Port grouping

Extend the existing `is_port_zero` flag with binary indicators for well-known ports:
- `is_port_80` (HTTP)
- `is_port_443` (HTTPS)
- `is_port_53` (DNS)

These ports appear in the top targeted ports from EDA (Q7). Avoids one-hot encoding 65K port values.

### 2.5 Feature selection

After initial model training:
1. Drop features with near-zero tree-based importance
2. Remove features with pairwise correlation > 0.95 (keep the one with higher target correlation)
3. Target final feature set of 15–25 features

---

## 3. Modeling (M4)

**Module:** `src/modeling.py`

### 3.1 Dual classification approach

| Stage | Task | Purpose |
|-------|------|---------|
| Primary | Binary: DDoS vs rest | High-recall DDoS detector — the critical security task |
| Secondary | 3-class: Normal / Suspicious / DDoS | Full classification for operational context |

The binary model is the priority — missing a DDoS attack (false negative) is the most costly error.

### 3.2 Model selection

| Model | Role | Why |
|-------|------|-----|
| Logistic Regression | Baseline | Simple, interpretable, establishes a floor |
| Decision Tree | Baseline | Non-linear baseline, reveals feature splits |
| Random Forest | Primary candidate | Strong tabular performance, handles imbalance well with class weights |
| XGBoost or LightGBM | Primary candidate | State-of-the-art for tabular data, native class weighting, fast training |

### 3.3 Handling class imbalance

Approaches to compare (all applied to training data only):
1. **`class_weight='balanced'`** — built-in to sklearn/XGBoost, inversely proportional to class frequencies
2. **SMOTE** — synthetic oversampling of minority classes on training folds only (never on validation/test)
3. **Random undersampling** — reduce Normal class to match Suspicious/DDoS counts

### 3.4 Hyperparameter tuning

- **Method:** `RandomizedSearchCV` (or Optuna if deeper search is needed)
- **CV:** Stratified 5-fold on training data to preserve class ratios
- **Scoring:** macro F1 (balances performance across all classes)
- **Budget:** 50–100 iterations for RandomizedSearch

### 3.5 Reproducible pipeline

Use `sklearn.Pipeline` to wrap preprocessing and model:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("features", FeatureEngineer()),       # custom transformer
    ("model", XGBClassifier(...))
])
```

This ensures feature engineering is applied consistently during training, cross-validation, and inference.

---

## 4. Evaluation (M5)

**Module:** `src/evaluation.py`

### 4.1 Data splits

| Split | Source | Purpose |
|-------|--------|---------|
| Train | SetA + SetB (264,770 events) | Model fitting |
| Test | SetC (130,000 events) | Primary evaluation |
| Genericity | SetD (437,657 events) | Generalization check — different time period/conditions |

### 4.2 Metrics

| Metric | Why |
|--------|-----|
| Per-class precision, recall, F1 | Understand per-class performance; accuracy is misleading at 94% Normal |
| Macro F1 | Single number balancing all classes equally |
| Confusion matrix | Visualize error patterns |
| PR-AUC | Better than ROC-AUC under extreme class imbalance |

**Not using accuracy** as the primary metric — a model predicting "Normal" for everything achieves 94.1% accuracy.

### 4.3 Error analysis

Focus on false negatives for DDoS (missed attacks):
- Examine misclassified DDoS events: which attack codes, port numbers, packet sizes?
- Compare feature distributions of correctly vs incorrectly classified DDoS events
- Identify if specific DDoS sub-types (e.g., DNS amplification vs SYN flood) are harder to detect

### 4.4 Feature importance

- **Tree-based importance:** built-in feature importances from Random Forest / XGBoost
- **SHAP values:** model-agnostic explanations, force plots for individual predictions
- Verify that the model's top features align with EDA findings (Avg packet len, Detect count, port 0)

### 4.5 Threshold tuning

The default 0.5 classification threshold may not be optimal:
- Sweep thresholds on a validation set (held-out from training or CV fold)
- Optimize for DDoS recall ≥ 0.95 while keeping precision acceptable
- Report performance at the chosen threshold vs default

---

## 5. Application (M6)

### 5.1 Inference pipeline

**Module:** `src/inference.py`

- Load trained model + cleaning config + feature engineering config
- Accept new event data (single event or batch)
- Apply cleaning → feature engineering → prediction
- Return predicted class + confidence score

```python
from src.inference import DDoSDetector

detector = DDoSDetector.load("models/best_model.pkl")
result = detector.predict(event_data)
# → {"prediction": "DDoS attack", "confidence": 0.97, "alert": True}
```

### 5.2 Alerting and reporting

When DDoS is detected:
- Log alert with event details (victim IP, port, confidence score)
- Generate summary report: detection count, top targeted IPs, attack types
- Output format: structured JSON logs + human-readable summary

### 5.3 Docker deployment

**Files:** `Dockerfile`, `scripts/simulate_traffic.py`

Containerized demo that replays test set events through the inference pipeline:

```
┌─────────────────────────────────────────────┐
│  Docker Container                           │
│                                             │
│  simulate_traffic.py                        │
│    ├── reads test events (SetC)             │
│    ├── feeds events to DDoSDetector         │
│    └── prints real-time predictions/alerts  │
│                                             │
│  src/inference.py                           │
│    ├── loads model + config                 │
│    └── returns predictions                  │
└─────────────────────────────────────────────┘
```

- **Dockerfile:** Python base image, install dependencies from `pyproject.toml`, copy source + model artifacts
- **Simulate script:** iterate through test events with a configurable delay, print predictions and alerts to stdout
- **Demo command:** `docker build -t ddos-detector . && docker run ddos-detector`

---

## 6. File Structure (planned)

```
src/
├── feature_engineering.py   # M3 — feature transforms + sklearn transformer
├── modeling.py              # M4 — model training, tuning, pipeline
├── evaluation.py            # M5 — metrics, plots, error analysis
├── inference.py             # M6 — production inference pipeline
├── eda.py                   # (existing) exploratory analysis
└── utils/
    ├── data_pipeline.py     # (existing) data loading
    └── data_cleaning.py     # (existing) cleaning pipeline

scripts/
└── simulate_traffic.py      # M6 — traffic replay for demo

models/                       # trained model artifacts (.pkl)
Dockerfile                    # M6 — containerized deployment
docs/
├── DATA.md                  # (existing) dataset documentation
├── EDA.md                   # (existing) EDA findings
└── PLAN.md                  # this document
```

---

## 7. Milestone Dependencies

```
M3 Feature Engineering
 └──→ M4 Modeling (needs engineered features)
       └──→ M5 Evaluation (needs trained models)
             └──→ M6 Application (needs best model + evaluation results)
```

Each milestone produces artifacts consumed by the next. Feature engineering and cleaning configs are serialized so downstream steps are reproducible without re-running upstream.
