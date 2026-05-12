# Multi-stage Dockerfile for DDoS Detection Demo
# Build context: project root (docker-compose sets context: ..)

# ---- base ----
FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for sklearn/numpy + hping3 for live traffic mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ hping3 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Install all deps including demo extras
RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.34" \
    "redis>=5.0" \
    "streamlit>=1.45" \
    "joblib>=1.4" \
    "plotly>=6.0" \
    "requests>=2.32" \
    "scikit-learn>=1.8" \
    "numpy>=2.0" \
    "pandas>=1.5,<3" \
    "pyarrow>=15.0" \
    "pydantic>=2.0"

COPY src/ ./src/
COPY data/ ./data/

RUN pip install --no-cache-dir "imbalanced-learn>=0.12" "matplotlib>=3.8" "seaborn>=0.13"

# ---- eval target (live traffic evaluation) ----
FROM base AS eval

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.eval_live"]

# ---- api target ----
FROM base AS api

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- dashboard target ----
FROM base AS dashboard

# Streamlit theme/config
COPY src/dashboard/.streamlit /root/.streamlit

EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
