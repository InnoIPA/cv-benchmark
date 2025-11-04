# 基底映像：PyTorch + CUDA（與多數驅動相容）
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei

# 安裝系統套件（ffmpeg 供 OpenCV 讀取影片）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 複製需求並安裝
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 複製主程式
COPY run-cv-benchmark.py ./

# 預設進入點（參數由 docker run 傳入）
ENTRYPOINT ["python", "run-cv-benchmark.py"]


