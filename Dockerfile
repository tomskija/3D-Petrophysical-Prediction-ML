FROM python:3.11-slim

# System dependencies (cmake in case segyio has to build from source on arm64)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY rockPropCalculator/ ./rockPropCalculator/

ENV PYTHONPATH=/workspace/rockPropCalculator
ENV PYTHONUNBUFFERED=1

# Keep container alive for devcontainer use — run script manually
CMD ["sleep", "infinity"]
