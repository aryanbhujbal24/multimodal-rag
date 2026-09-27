FROM python:3.12-slim

# libgomp is required by faiss; the rest keeps the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies first so a code change does not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY eval ./eval

ENV PYTHONPATH=/srv PYTHONUNBUFFERED=1
RUN useradd -m appuser && mkdir -p /srv/data && chown -R appuser /srv
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fs http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
