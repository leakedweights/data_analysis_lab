# Dataset Documentation

## SCLDDoS2024

DDoS attack detection dataset. Source: [SharePoint](https://bmeedu-my.sharepoint.com/personal/skopko_tamas_vik_bme_hu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fskopko%5Ftamas%5Fvik%5Fbme%5Fhu%2FDocuments%2Fddos%2Ddata%2D2024&ga=1)

### Splits

| Split | Sets | Purpose |
|-------|------|---------|
| `train` | SetA + SetB | Model training |
| `test` | SetC | Model evaluation |
| `genericity` | SetD | Genericity check |

### File Structure

Each set has two CSV files:

- **Components** — individual detection records (one row per detection within an event)
- **Events** — aggregated event summaries with a classification label (`Type`)

### Components Columns

| Column | Type | Description |
|--------|------|-------------|
| Attack ID | int | Unique event identifier |
| Detect count | int | Component index within the event |
| Card | category | Network card identifier |
| Victim IP | category | Anonymized target IP |
| Port number | int | Target port |
| Attack code | category | Component type (e.g. "SYN Attack", "DNS", "High volume traffic") |
| Packet speed | int | Packet rate (pps) |
| Data speed | int | Data rate (bps) |
| Avg packet len | int | Average packet length (bytes) |
| Source IP count | int | Number of source IPs |
| Time | datetime | Component start time |

### Events Columns

| Column | Type | Description |
|--------|------|-------------|
| Attack ID | int | Unique event identifier |
| Card | category | Network card identifier |
| Victim IP | category | Anonymized target IP |
| Port number | int | Target port |
| Attack code | category | Component type |
| Detect count | int | Number of components in the event |
| Packet speed | int | Packet rate (pps) |
| Data speed | int | Data rate (bps) |
| Avg packet len | int | Average packet length (bytes) |
| Avg source IP count | int | Average number of source IPs |
| Start time | datetime | Event start time |
| End time | datetime | Event end time |
| Type | category | Classification: "Normal traffic", "Suspicious traffic", or "DDoS attack" |

### Row Counts

| File | Rows |
|------|------|
| SetA components | 586,642 |
| SetA events | 134,770 |
| SetB components | 1,233,449 |
| SetB events | 130,000 |
| SetC components | 1,247,266 |
| SetC events | 130,000 |
| SetD components | 2,452,610 |
| SetD events | 437,657 |

### Storage Formats

The pipeline supports two formats:

- **CSV** (`data/*.csv`) — raw extracted files, ~560 MB total, git-ignored
- **Parquet** (`data/parquet/*.parquet`) — zstd-compressed, ~88 MB total, committable to git

Parquet preserves all dtypes (categoricals, datetimes, integers) so loading is faster and requires no re-parsing. Files are pre-split by purpose:

| File | Size |
|------|------|
| `train_components.parquet` | 20 MB |
| `train_events.parquet` | 6.7 MB |
| `test_components.parquet` | 14 MB |
| `test_events.parquet` | 3.6 MB |
| `genericity_components.parquet` | 35 MB |
| `genericity_events.parquet` | 11 MB |

To generate Parquet from CSVs:

```bash
uv run python src/utils/data_pipeline.py
```

### Data Quality Analysis

A comprehensive audit of the training data (SetA + SetB) revealed the following issues:

**Missing values:**
- 4 `End time` values in SetA events are `"0"` — coerced to `NaT` during loading.
- No other missing values in either table.
- No duplicate rows in either table.

**Constant / low-information columns:**
- `Card` has a single unique value (`sga10gq0`) across all data — zero information.
- `Set` is metadata tracking the source file, not a feature.

**Nonsensical values:**
- **`Avg packet len` = 0**: 42,254 events (16.0%) and ~17% of components have zero-length average packets. Since every network packet has at least a header, these are measurement artifacts / missing data, not true zero-byte packets.
- No negative values found in any numeric column.
- All port numbers are within the valid range [0–65,535].

**Extreme outliers (training events):**

| Column | Mean | Median | 99th pct | Max | Issue |
|--------|------|--------|----------|-----|-------|
| Packet speed | 71,649 | 64,550 | 207,300 | 7,475,824 | Max is 36x the 99th percentile |
| Detect count | 6.9 | 1 | 67 | 12,534 | Max is 187x the 99th percentile |
| Avg source IP count | 6.6 | 1 | 61 | 18,602 | Max is 305x the 99th percentile |
| Data speed | 83.8 | 78 | 221 | 6,702 | Moderate outliers |

**Redundant features:**
- `Packet speed` and `Data speed` are highly correlated (r = 0.84). Keeping both adds multicollinearity without additional information.

**Structural issues:**
- `Attack code` contains compound comma-separated strings (e.g. `"DNS, High volume traffic"`), requiring multi-label encoding rather than simple one-hot.
- Port 0 appears in ~25% of events and is disproportionately associated with DDoS attacks — a useful indicator.
- 94% of component-level `Attack code` values are `"High volume traffic"` — very low entropy.

---

### Data Cleaning

Cleaning is implemented in `src/utils/data_cleaning.py`. All thresholds are fitted on training data only and applied to all splits to prevent data leakage. The fitted thresholds are saved to `data/cleaning_config.json`.

Run cleaning:

```bash
uv run python -m src.utils.data_cleaning
```

#### Cleaning decisions

| Issue | Approach | Rationale |
|-------|----------|-----------|
| `Card` column (constant) | Drop | Zero information content |
| `Set` column (metadata) | Drop | Not a feature |
| 4 missing `End time` | Fill with `Start time` | Yields duration = 0; only 4 rows, no data loss |
| 16% zero `Avg packet len` | Impute with train median + add `avg_pkt_len_was_zero` flag | 16% data loss unacceptable; zeros are measurement artifacts; binary flag lets model learn missingness |
| Extreme outliers | Clip at 99th percentile (train-fitted) | Consistent with EDA approach; preserves all rows; caps stored in config for reproducibility |
| `Data speed` (r=0.84 with `Packet speed`) | Drop column | Redundant; Packet speed retained as slightly more discriminative |
| Compound `Attack code` | Token-based binary flags (`atk_*` columns) | Fixed token list from 9 known attack types; consistent columns across splits |
| Port 0 as DDoS signal | Add `is_port_zero` binary flag | Strongest port-based signal from EDA; avoids 65K one-hot columns |
| `Start time`, `End time` | Derive `duration_s`, then drop timestamps | Temporal patterns are non-stationary (EDA Q4); duration captures useful info |
| `Time` (components) | Drop | Same non-stationarity concern |

#### Cleaned events schema (20 columns)

| Column | Type | Source |
|--------|------|--------|
| Attack ID | int32 | Original — row identifier |
| Victim IP | category | Original — kept for joins/analysis, not for ML directly |
| Port number | int32 | Original |
| Detect count | int32 | Capped at 99th pct (67) |
| Packet speed | int64 | Capped at 99th pct (207,300) |
| Avg packet len | int32 | Zeros imputed with median (1,285), then capped (1,506) |
| Avg source IP count | int32 | Capped at 99th pct (61) |
| Type | category | Target variable (unchanged) |
| duration_s | float64 | Derived: End time − Start time in seconds, clipped ≥ 0 |
| avg_pkt_len_was_zero | int8 | 1 if original Avg packet len was 0 |
| atk_syn_attack | int8 | Attack code contains "SYN Attack" |
| atk_dns | int8 | Attack code contains "DNS" |
| atk_ntp | int8 | Attack code contains "NTP" |
| atk_generic_udp | int8 | Attack code contains "Generic UDP" |
| atk_high_volume_traffic | int8 | Attack code contains "High volume traffic" |
| atk_suspicious_traffic | int8 | Attack code contains "Suspicious traffic" |
| atk_cldap | int8 | Attack code contains "CLDAP" |
| atk_ssdp | int8 | Attack code contains "SSDP" |
| atk_memcached | int8 | Attack code contains "memcached" |
| is_port_zero | int8 | 1 if Port number == 0 |

**Dropped:** Card, Set, Data speed, Start time, End time, Attack code (raw).

#### Cleaned components schema (18 columns)

Same structure minus events-specific columns (no `duration_s`, no `Type`, no `Avg source IP count`). Has `Source IP count` instead, and `Detect count` represents the component index within the event.

**Dropped:** Card, Set, Data speed, Time, Attack code (raw).

#### Fitted thresholds (training data)

Saved to `data/cleaning_config.json`:

| Parameter | Value |
|-----------|-------|
| Event Packet speed cap | 207,300 |
| Event Avg packet len cap | 1,506 |
| Event Avg source IP count cap | 61 |
| Event Detect count cap | 67 |
| Event Avg pkt len median (non-zero) | 1,285 |
| Component Packet speed cap | 468,000 |
| Component Avg packet len cap | 1,518 |
| Component Source IP count cap | 2,499 |
| Component Avg pkt len median (non-zero) | 1,465 |
| Cap quantile | 0.99 |

#### Usage

```python
from src.utils.data_cleaning import fit_and_clean_train, clean_split, CleaningConfig
from pathlib import Path

# Fit on train + clean all splits
clean_evt, clean_comp, cfg = fit_and_clean_train()

# Clean another split with the same thresholds
test_evt, test_comp = clean_split("test", cfg)

# Save / reload config
cfg.to_json(Path("data/cleaning_config.json"))
cfg = CleaningConfig.from_json(Path("data/cleaning_config.json"))
```

### Usage

```python
from src.utils.data_pipeline import load_components, load_events, load_all

# Load a specific split (prefers Parquet if available, falls back to CSV)
train_components = load_components("train")   # SetA + SetB, 1.8M rows
test_events = load_events("test")             # SetC, 130K rows

# Load everything at once
data = load_all()  # dict with keys like "train_components", "test_events", etc.
```

A `Set` column is added automatically to track the source set (e.g. "SetA", "SetB").
