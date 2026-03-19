# backend AlpineRoute -- python:3.11-slim + pip wheels (ref A07)
# les wheels rasterio PyPI embarquent leur propre GDAL, pas besoin de conda

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libsqlite3-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

# volume pour le cache DEM + base SQLite
VOLUME /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /app/backend

CMD ["uvicorn", "alpineroute.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info"]
