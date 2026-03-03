"""Extract DDoS dataset from a OneDrive zip (or raw .csv.gz files) into data/."""

import gzip
import shutil
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "data"

FILES = [
    "SCLDDoS2024_SetA_components.csv.gz",
    "SCLDDoS2024_SetA_events.csv.gz",
    "SCLDDoS2024_SetB_components.csv.gz",
    "SCLDDoS2024_SetB_events.csv.gz",
    "SCLDDoS2024_SetC_components.csv.gz",
    "SCLDDoS2024_SetC_events.csv.gz",
    "SCLDDoS2024_SetD_components.csv.gz",
    "SCLDDoS2024_SetD_events.csv.gz",
]


def _unzip_onedrive(zip_path: Path) -> None:
    """Unpack a OneDrive zip into data/raw/, keeping only expected .csv.gz files."""
    print(f"UNZIP  {zip_path.name} -> {RAW_DIR}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            fname = Path(member).name
            if fname in FILES:
                dst = RAW_DIR / fname
                if not dst.exists():
                    with zf.open(member) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)


def _find_onedrive_zip() -> Path | None:
    """Look for a OneDrive zip in data/raw/ or data/."""
    for directory in (RAW_DIR, OUT_DIR):
        if not directory.exists():
            continue
        for p in directory.glob("OneDrive_*.zip"):
            return p
    return None


def extract_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: if a OneDrive zip is present, unpack the .csv.gz files first
    zip_path = _find_onedrive_zip()
    if zip_path:
        _unzip_onedrive(zip_path)

    # Step 2: decompress each .csv.gz -> .csv
    found_any = False
    for fname in FILES:
        src = RAW_DIR / fname
        dst = OUT_DIR / fname.removesuffix(".gz")

        if not src.exists():
            print(f"SKIP  {fname} (not found)")
            continue

        found_any = True
        if dst.exists():
            print(f"SKIP  {dst.name} (already extracted)")
            continue

        print(f"EXTRACT  {fname} -> {dst.name}")
        with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if not found_any:
        print(f"Error: no .csv.gz files found in {RAW_DIR}")
        print("Download the OneDrive zip from SharePoint and place it in data/raw/")
        raise SystemExit(1)

    print("Done.")


if __name__ == "__main__":
    extract_all()
