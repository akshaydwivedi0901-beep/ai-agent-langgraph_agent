# ---- Base Image ----
FROM python:3.11-slim

# ---- Set Working Directory ----
WORKDIR /app

# ---- System deps (recommended) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy Code ----
COPY . /app

# ---- Install Dependencies ----
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn redis

# ---- Expose Port ----
EXPOSE 8080

# ---- Run FastAPI ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
