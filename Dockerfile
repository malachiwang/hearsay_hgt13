# syntax=docker/dockerfile:1

FROM python:3.11-slim

ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        python -m pip install --no-cache-dir \
            --index-url "${PYTORCH_CPU_INDEX_URL}" torch torchaudio; \
    else \
        python -m pip install --no-cache-dir torch torchaudio; \
    fi \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Download only the allowlisted Eliya files into the exact directory expected
# by Deepfake.py, and cache the transitive WavLM-large backbone for offline use.
RUN python scripts/download_eliya.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Fail the build if either model snapshot cannot be resolved without network.
RUN python scripts/download_eliya.py --verify-only

ENTRYPOINT ["python", "run_hearsay.py"]
