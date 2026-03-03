"""Load SCLDDoS2024 dataset with proper types and train/test/genericity splits.

Supports both CSV (raw) and Parquet (fast, compact) formats.
Use save_parquet() to convert CSVs to Parquet, then load functions will
prefer Parquet automatically.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PARQUET_DIR = DATA_DIR / "parquet"

Split = Literal["train", "test", "genericity"]

_SPLIT_SETS: dict[Split, list[str]] = {
    "train": ["SetA", "SetB"],
    "test": ["SetC"],
    "genericity": ["SetD"],
}

_CATEGORICAL_COLS_COMPONENTS = ["Card", "Victim IP", "Attack code"]
_CATEGORICAL_COLS_EVENTS = ["Card", "Victim IP", "Attack code", "Type"]

_COMPONENT_DTYPES = {
    "Attack ID": "int32",
    "Detect count": "int32",
    "Port number": "int32",
    "Packet speed": "int64",
    "Data speed": "int64",
    "Avg packet len": "int32",
    "Source IP count": "int32",
}

_EVENT_DTYPES = {
    "Attack ID": "int32",
    "Port number": "int32",
    "Detect count": "int32",
    "Packet speed": "int64",
    "Data speed": "int64",
    "Avg packet len": "int32",
    "Avg source IP count": "int32",
}


def _load_components_from_csv(split: Split) -> pd.DataFrame:
    frames = []
    for set_name in _SPLIT_SETS[split]:
        path = DATA_DIR / f"SCLDDoS2024_{set_name}_components.csv"
        df = pd.read_csv(path, dtype=_COMPONENT_DTYPES, parse_dates=["Time"])
        df["Set"] = set_name
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    for col in _CATEGORICAL_COLS_COMPONENTS:
        result[col] = result[col].astype("category")
    return result


def _load_events_from_csv(split: Split) -> pd.DataFrame:
    frames = []
    for set_name in _SPLIT_SETS[split]:
        path = DATA_DIR / f"SCLDDoS2024_{set_name}_events.csv"
        df = pd.read_csv(path, dtype=_EVENT_DTYPES)
        # Some rows have "0" instead of a datetime — coerce to NaT
        df["Start time"] = pd.to_datetime(df["Start time"], errors="coerce")
        df["End time"] = pd.to_datetime(df["End time"], errors="coerce")
        df["Set"] = set_name
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    for col in _CATEGORICAL_COLS_EVENTS:
        result[col] = result[col].astype("category")
    return result


def load_components(split: Split) -> pd.DataFrame:
    """Load component data for the given split. Prefers Parquet if available."""
    pq = PARQUET_DIR / f"{split}_components.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return _load_components_from_csv(split)


def load_events(split: Split) -> pd.DataFrame:
    """Load event data for the given split. Prefers Parquet if available."""
    pq = PARQUET_DIR / f"{split}_events.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return _load_events_from_csv(split)


def load_all() -> dict[str, pd.DataFrame]:
    """Load all splits. Returns dict keyed by '{split}_components' / '{split}_events'."""
    result = {}
    for split in _SPLIT_SETS:
        result[f"{split}_components"] = load_components(split)
        result[f"{split}_events"] = load_events(split)
    return result


def save_parquet() -> None:
    """Convert all CSV splits to Parquet files in data/parquet/."""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for split in _SPLIT_SETS:
        for kind, loader in [("components", _load_components_from_csv), ("events", _load_events_from_csv)]:
            name = f"{split}_{kind}.parquet"
            path = PARQUET_DIR / name
            if path.exists():
                print(f"SKIP  {name} (already exists)")
                continue
            print(f"SAVE  {name}")
            df = loader(split)
            df.to_parquet(path, compression="zstd")
    print("Done.")


if __name__ == "__main__":
    save_parquet()
