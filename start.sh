#!/bin/bash
set -e
PORT=${PORT:-8080}
echo "Starting server on port $PORT"
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1