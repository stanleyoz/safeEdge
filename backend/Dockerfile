# SafeEdge cloud backend — Alibaba Cloud Function Compute (custom container)
FROM python:3.10-slim

WORKDIR /app

# System libs for opencv-python-headless (libGL not needed in headless, but
# glib is). Keep minimal for fast cold starts on Function Compute.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Backend package + reused cloud skills + dashboard UI
COPY backend/   /app/backend/
COPY cloud/     /app/cloud/
COPY dashboard/ /app/dashboard/
COPY config/    /app/config/

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# Function Compute sends HTTP to the container's listen port.
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
