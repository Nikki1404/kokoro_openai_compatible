FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    CONFIG_PATH=app/config.yaml \
    HF_HOME=/models/hf \
    HF_HUB_CACHE=/models/hf/hub \
    TRANSFORMERS_CACHE=/models/hf/transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir git+https://github.com/nvidia/kokoro.git


RUN pip install --no-cache-dir "huggingface_hub>=0.24.0"

RUN python - << 'PY'
from huggingface_hub import snapshot_download
repo_id = "hexgrad/Kokoro-82M"
snapshot_download(repo_id=repo_id, local_dir=None, allow_patterns=["*.json","*.safetensors","*.bin","*.pt","*.txt","*.model","*.vocab","*.tiktoken","*"],)
print("Downloaded:", repo_id)
PY

COPY app /app/app

EXPOSE 8080

ENV HF_HUB_OFFLINE=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
