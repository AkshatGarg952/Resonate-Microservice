#!/bin/bash
# =============================================================================
# Resonate Microservice — Production Start Script
#
# Uses gunicorn with multiple UvicornWorker processes instead of a single
# uvicorn instance. This allows concurrent AI request handling without
# blocking the event loop.
#
# Workers = 4 (tune based on Railway/Render vCPU count):
#   - 2 vCPUs  → 4 workers
#   - 4 vCPUs  → 8 workers
#
# Deploy: Set your Railway/Render start command to: sh start.sh
# =============================================================================

WORKERS=${GUNICORN_WORKERS:-4}
PORT=${PORT:-10000}

echo "Starting Resonate Microservice: ${WORKERS} workers on port ${PORT}"

exec gunicorn app.main:app \
  --workers "${WORKERS}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:${PORT}" \
  --timeout 120 \
  --keepalive 5 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --access-logfile - \
  --error-logfile -
