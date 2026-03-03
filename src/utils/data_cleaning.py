"""Clean and prepare SCLDDoS2024 DataFrames for modeling.

Design principle: compute thresholds on train data only (fit),
then apply them to all splits (transform) to avoid data leakage.

Run with: uv run python -m src.utils.data_cleaning
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.data_pipeline import Split, load_components, load_events

# Columns to drop (zero-information or metadata-only)
_DROP_COLS_EVENTS = ["Card", "Set"]
_DROP_COLS_COMPONENTS = ["Card", "Set"]

# Numeric columns eligible for outlier capping
_CAP_COLS_EVENTS = [
    "Packet speed",
    "Data speed",
    "Avg packet len",
    "Avg source IP count",
    "Detect count",
]
_CAP_COLS_COMPONENTS = [
    "Packet speed",
    "Data speed",
    "Avg packet len",
    "Source IP count",
]

# Known attack tokens from EDA Q3 analysis
_ATTACK_TOKENS = [
    "SYN Attack",
    "DNS",
    "NTP",
    "Generic UDP",
    "High volume traffic",
    "Suspicious traffic",
    "CLDAP",
    "SSDP",
    "memcached",
]

DEFAULT_CAP_QUANTILE = 0.99


@dataclass
class CleaningConfig:
    """Thresholds and statistics computed from training data.

    Fitted once on training data, then applied to all splits
    to prevent data leakage.
    """

    event_caps: dict[str, float] = field(default_factory=dict)
    component_caps: dict[str, float] = field(default_factory=dict)
    event_avg_pkt_len_median: float = 0.0
    component_avg_pkt_len_median: float = 0.0
    cap_quantile: float = DEFAULT_CAP_QUANTILE

    def to_json(self, path: Path) -> None:
        """Serialize config to JSON for reproducibility."""
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> CleaningConfig:
        """Load a previously fitted config."""
        return cls(**json.loads(path.read_text()))


def fit_config(
    events: pd.DataFrame,
    components: pd.DataFrame,
    cap_quantile: float = DEFAULT_CAP_QUANTILE,
) -> CleaningConfig:
    """Compute cleaning thresholds from training data."""
    cfg = CleaningConfig(cap_quantile=cap_quantile)

    for col in _CAP_COLS_EVENTS:
        cfg.event_caps[col] = float(events[col].quantile(cap_quantile))

    for col in _CAP_COLS_COMPONENTS:
        cfg.component_caps[col] = float(components[col].quantile(cap_quantile))

    nonzero_evt = events.loc[events["Avg packet len"] > 0, "Avg packet len"]
    cfg.event_avg_pkt_len_median = float(nonzero_evt.median())

    nonzero_comp = components.loc[components["Avg packet len"] > 0, "Avg packet len"]
    cfg.component_avg_pkt_len_median = float(nonzero_comp.median())

    return cfg


def _encode_attack_code(df: pd.DataFrame, col: str = "Attack code") -> pd.DataFrame:
    """Expand compound Attack code into binary indicator columns.

    E.g. "DNS, High volume traffic" -> atk_dns=1, atk_high_volume_traffic=1.
    """
    code_str = df[col].astype(str).str.strip()
    for token in _ATTACK_TOKENS:
        safe_name = "atk_" + token.lower().replace(" ", "_")
        df[safe_name] = (
            code_str.str.contains(token, case=False, regex=False).astype("int8")
        )
    return df


def clean_events(events: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Apply cleaning pipeline to an events DataFrame.

    Steps:
        1. Drop useless columns (Card, Set)
        2. Handle missing End time (fill with Start time -> duration=0)
        3. Compute duration_s derived feature
        4. Treat zero Avg packet len (flag + impute with train median)
        5. Cap extreme outliers using train-fitted thresholds
        6. Drop Data speed (r=0.84 with Packet speed)
        7. Encode Attack code into binary atk_* columns
        8. Add is_port_zero indicator
        9. Drop raw columns replaced by encodings
    """
    df = events.copy()

    # 1. Drop zero-information columns
    df = df.drop(columns=[c for c in _DROP_COLS_EVENTS if c in df.columns])

    # 2. Handle missing End time (4 NaT values in training)
    mask_nat = df["End time"].isna()
    df.loc[mask_nat, "End time"] = df.loc[mask_nat, "Start time"]

    # 3. Compute duration
    df["duration_s"] = (
        (df["End time"] - df["Start time"]).dt.total_seconds().clip(lower=0)
    )

    # 4. Zero Avg packet len -> flag + impute
    df["avg_pkt_len_was_zero"] = (df["Avg packet len"] == 0).astype("int8")
    df.loc[df["Avg packet len"] == 0, "Avg packet len"] = int(
        cfg.event_avg_pkt_len_median
    )

    # 5. Cap outliers
    for col, cap in cfg.event_caps.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    # 6. Drop Data speed (redundant with Packet speed)
    df = df.drop(columns=["Data speed"], errors="ignore")

    # 7. Encode Attack code
    df = _encode_attack_code(df, col="Attack code")

    # 8. Port-zero indicator
    df["is_port_zero"] = (df["Port number"] == 0).astype("int8")

    # 9. Drop raw columns replaced by encodings
    df = df.drop(columns=["Start time", "End time", "Attack code"])

    return df


def clean_components(
    components: pd.DataFrame, cfg: CleaningConfig
) -> pd.DataFrame:
    """Apply cleaning pipeline to a components DataFrame.

    Steps:
        1. Drop useless columns (Card, Set)
        2. Treat zero Avg packet len (flag + impute)
        3. Cap extreme outliers
        4. Drop Data speed (redundant with Packet speed)
        5. Encode Attack code into binary atk_* columns
        6. Add is_port_zero indicator
        7. Drop raw columns replaced by encodings
    """
    df = components.copy()

    # 1. Drop zero-information columns
    df = df.drop(columns=[c for c in _DROP_COLS_COMPONENTS if c in df.columns])

    # 2. Zero Avg packet len -> flag + impute
    df["avg_pkt_len_was_zero"] = (df["Avg packet len"] == 0).astype("int8")
    df.loc[df["Avg packet len"] == 0, "Avg packet len"] = int(
        cfg.component_avg_pkt_len_median
    )

    # 3. Cap outliers
    for col, cap in cfg.component_caps.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    # 4. Drop Data speed (redundant with Packet speed)
    df = df.drop(columns=["Data speed"], errors="ignore")

    # 5. Encode Attack code
    df = _encode_attack_code(df, col="Attack code")

    # 6. Port-zero indicator
    df["is_port_zero"] = (df["Port number"] == 0).astype("int8")

    # 7. Drop raw columns replaced by encodings
    df = df.drop(columns=["Attack code", "Time"])

    return df


def fit_and_clean_train() -> tuple[pd.DataFrame, pd.DataFrame, CleaningConfig]:
    """Load training data, fit thresholds, and return cleaned DataFrames."""
    raw_events = load_events("train")
    raw_components = load_components("train")

    cfg = fit_config(raw_events, raw_components)
    return clean_events(raw_events, cfg), clean_components(raw_components, cfg), cfg


def clean_split(
    split: Split, cfg: CleaningConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean a split using a pre-fitted config."""
    raw_events = load_events(split)
    raw_components = load_components(split)
    return clean_events(raw_events, cfg), clean_components(raw_components, cfg)


def main() -> None:
    print("Fitting cleaning config on training data...")
    clean_evt, clean_comp, config = fit_and_clean_train()
    print(f"  Events:     {clean_evt.shape}")
    print(f"  Components: {clean_comp.shape}")
    print(f"  Event caps: {config.event_caps}")
    print(f"  Component caps: {config.component_caps}")
    print(f"  Avg pkt len median (events): {config.event_avg_pkt_len_median}")
    print(f"  Avg pkt len median (components): {config.component_avg_pkt_len_median}")

    cfg_path = Path(__file__).resolve().parents[2] / "data" / "cleaning_config.json"
    config.to_json(cfg_path)
    print(f"  Config saved to {cfg_path}")

    for split in ("test", "genericity"):
        evt, comp = clean_split(split, config)
        print(f"  {split} events: {evt.shape}, components: {comp.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
