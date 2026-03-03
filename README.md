# Data Analysis Lab

## Data Setup

The project uses the [SCLDDoS2024 dataset](https://bmeedu-my.sharepoint.com/personal/skopko_tamas_vik_bme_hu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fskopko%5Ftamas%5Fvik%5Fbme%5Fhu%2FDocuments%2Fddos%2Ddata%2D2024&ga=1) hosted on SharePoint.

1. Download the dataset from the link above (SharePoint will package it as a `OneDrive_*.zip`)
2. Place the zip file in `data/raw/`
3. Run the extraction script:

```bash
uv run python src/utils/extract_data.py
```

This will unzip the `.csv.gz` files from the OneDrive archive and decompress them into `data/` as CSVs.

4. Convert to Parquet (optional but recommended — ~88 MB total, committable to git):

```bash
uv run python src/utils/data_pipeline.py
```

## Loading Data

The data pipeline provides typed DataFrames with train/test/genericity splits:

```python
from src.utils.data_pipeline import load_components, load_events, load_all

train_components = load_components("train")       # SetA + SetB
test_events = load_events("test")                 # SetC
genericity_components = load_components("genericity")  # SetD

# Or load everything at once
data = load_all()  # dict: "train_components", "train_events", "test_components", etc.
```

The pipeline loads from Parquet if available (fast, types preserved), otherwise falls back to CSV.

See [docs/DATA.md](docs/DATA.md) for the full schema and row counts.
