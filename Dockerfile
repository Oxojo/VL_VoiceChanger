# NeoShowcase 用（Runtime / Dockerfile）。GPU を前提にしない CPU 版。
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 音声ファイルの読み書きに必要な共有ライブラリ。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY engine/ ./engine/
COPY vlvc/ ./vlvc/
COPY config/ ./config/

# CUDA 関連の巨大な依存を入れず、CPU 版を明示的に選ぶ。
# バイナリ配布がない依存はソースビルドせず、ビルド時点で失敗させる。
RUN python -m pip install --upgrade pip \
    && python -m pip install --only-binary=:all: 'torch>=2.6' --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --only-binary=:all: '.[engine,web]'

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/models /app/modules /app/logs \
    && chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4)"

# 複数 worker はジョブ履歴・モデルキャッシュを共有しないため使わない。
CMD ["python", "-m", "uvicorn", "engine.web.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
