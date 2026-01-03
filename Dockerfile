FROM python:3.11-slim

# ---- Environment hardening ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- System deps (minimal, safe) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python deps first (cache-friendly) ----
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip check

# ---- Copy application code ----
COPY app ./app

# ---- Security: non-root user ----
RUN useradd -m appuser
USER appuser

# ---- Networking ----
EXPOSE 8000

# ---- Health check (K8s & Docker compatible) ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -f http://localhost:8000/health || exit 1

# ---- Run ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
