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

### Data Quality Notes

- 4 rows in SetA events have `End time = "0"` — coerced to `NaT` during loading.

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
