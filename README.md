# Data Analysis Lab — SCLDDoS2024 DDoS Detection

End-to-end pipeline for the BME *Haladó adatelemzési módszerek laboratórium* (Lab02-Intro)
DDoS-detection task: data wrangling, EDA, feature engineering, modeling,
evaluation on a held-out *genericity* split, and two live demos
(host-network monitor + FastAPI/dashboard simulator).

See [PROJECT_INFO.md](PROJECT_INFO.md) for course context and the
milestone schedule, and [docs/PLAN.md](docs/PLAN.md) for the original
M3–M6 roadmap.

## Data Setup

The project uses the [SCLDDoS2024 dataset](https://bmeedu-my.sharepoint.com/personal/skopko_tamas_vik_bme_hu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fskopko%5Ftamas%5Fvik%5Fbme%5Fhu%2FDocuments%2Fddos%2Ddata%2D2024&ga=1) hosted on SharePoint.

1. Download the dataset from the link above (SharePoint packages it as `OneDrive_*.zip`).
2. Place the zip file in `data/raw/`.
3. Extract the CSVs:

   ```bash
   uv run python src/utils/extract_data.py
   ```

4. Convert to Parquet (optional but recommended — ~88 MB total, committable to git):

   ```bash
   uv run python src/utils/data_pipeline.py
   ```

5. Run the cleaning step (fits outlier caps + median imputation on train,
   applies the same thresholds to test/genericity, saves
   `data/cleaning_config.json`):

   ```bash
   uv run python -m src.utils.data_cleaning
   ```

The data pipeline exposes typed DataFrames with train/test/genericity splits:

```python
from src.utils.data_pipeline import load_components, load_events, load_all

train_components = load_components("train")            # SetA + SetB
test_events      = load_events("test")                 # SetC
gen_components   = load_components("genericity")       # SetD

data = load_all()  # dict: "train_components", "train_events", "test_components", ...
```

Parquet is preferred when present; otherwise the loader falls back to CSV.
Full schema, row counts, and the cleaning decisions are documented in
[docs/DATA.md](docs/DATA.md).

## Project Layout

```
src/
├── utils/
│   ├── extract_data.py        # unzip + decompress raw OneDrive dump
│   ├── data_pipeline.py       # CSV ↔ Parquet, typed loaders
│   └── data_cleaning.py       # train-fitted caps, imputation, atk_*/is_port_zero flags
├── eda.py, eda_figures.py     # 10 EDA plots → plots/svg/01_*..10_*.svg
├── features_v2*.py            # v2 feature builders (event + component aggregates)
├── cluster_features.py        # per-class GMM soft-membership featurizer
├── train.py, train_v2.py      # v1 (events only) and v2 (events+components) training
├── bench_abc.py, bench_cluster.py, bench_more.py  # ablations
├── plot_*.py                  # figure generation for each experiment
├── synthetic.py, simulator.py # offline scenario generation
├── live_capture*.py, host_sampler.py  # NIC sampling for live mode
├── eval_live*.py              # hping3 vs synthetic comparison
├── monitor.py                 # rich-TUI live host-network monitor
├── api/                       # FastAPI demo (model registry, stream engine, static UI)
└── dashboard/                 # Streamlit dashboard
docs/                          # DATA, EDA, PLAN, MODEL_V2*, ...
docker/                        # multi-stage Dockerfile + docker-compose
results/, plots/svg/           # CSV metrics + figures
```

## Reproducing the Modeling Pipeline

After the data setup above:

```bash
# v1 baseline (events only) — establishes the floor reported in docs/MODEL_V2.md
uv run python -m src.train

# v2 — events + within-event component aggregates, balanced class weights
uv run python -m src.train_v2

# v2 + per-class GMM cluster features (BIC-selected K, see docs/MODEL_V2_CLUSTER.md)
uv run python -m src.bench_cluster

# Optional resampling ablations
uv run python -m src.train_v2 --resample smote
uv run python -m src.train_v2 --resample all
```

Each script writes metrics to `results/` and figures to `plots/svg/`.
The narrative for each experiment lives in `docs/`:

- [MODEL_V2.md](docs/MODEL_V2.md) — why v1 was unsound, what v2 fixes,
  and side-by-side numbers
- [MODEL_V2_ABC.md](docs/MODEL_V2_ABC.md) — A+B+C ablation
- [MODEL_V2_CLUSTER.md](docs/MODEL_V2_CLUSTER.md) — per-class GMM features
- [MODEL_V2_MORE.md](docs/MODEL_V2_MORE.md) — additional feature studies
- [EDA.md](docs/EDA.md) — exploratory findings driving the design

## Live Host-Network Monitor

`src/monitor.py` runs the v2+cluster pipeline against a real NIC and
renders a rich TUI with per-class probabilities, sparklines, and alerts.

```bash
# Auto-detect interface; first run trains + caches the pipeline under ./models
uv run python -m src.monitor

# Pin an interface / model / alert threshold
uv run python -m src.monitor --interface lo --model "Gradient Boosting" --alert-threshold 2
```

Source-IP counting needs `CAP_NET_RAW` — run with `sudo` or grant the
capability. Without it the sniffer reports zero distinct IPs and the
`src_ip_*` features fall back to zero; the monitor still runs.

## FastAPI Demo + Dashboard (Docker)

Multi-stage build under `docker/` ships three targets: `api`, `dashboard`,
and `eval` (live evaluation). The compose file wires API + Streamlit
dashboard + Redis for the streaming channel.

```bash
cd docker
docker compose up --build
```

- API:        http://localhost:8000 (static UI under `/`, JSON endpoints under `/api/*`)
- Dashboard:  http://localhost:8501

On first start the API trains and caches the model registry; subsequent
runs are warm. The `api` service requests `NET_ADMIN`/`NET_RAW` so the
optional `hping3` live-traffic scenarios work from inside the container.

## Evaluating Against Live / Synthetic Traffic

```bash
# v2 hping3 vs synthetic scenarios — writes results/live_evaluation_v2.csv
uv run python -m src.eval_live_v2 --scenarios live_syn_flood live_dns_amp

# Skip hping3 (synthetic only)
uv run python -m src.eval_live_v2 --skip-live
```

## Dependencies

Managed with `uv` (see `pyproject.toml`). Core deps: `pandas`, `numpy`,
`scikit-learn`, `imbalanced-learn`, `pyarrow`, `matplotlib`, `seaborn`.
The `demo` optional group adds `fastapi`, `uvicorn`, `redis`, `joblib`,
`pydantic`, and `rich` for the API/monitor stack.

```bash
uv sync                      # core
uv sync --extra demo         # + API/monitor extras
```
