ARG PYTHON_IMAGE=python:3.11.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS test
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    RUFF_CACHE_DIR=/tmp/ruff-cache
WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
      --constraint constraints/ci.txt setuptools \
    && python -m pip install --no-cache-dir --no-build-isolation \
      --constraint constraints/ci.txt --editable ".[dev]"
RUN addgroup --system --gid 10001 marine-ptz \
    && adduser --system --uid 10001 --ingroup marine-ptz --home /nonexistent \
      --no-create-home marine-ptz
USER 10001:10001
CMD ["/bin/sh", "-c", "ruff format --check src tests tools && ruff check src tests tools && python -m pytest -p no:cacheprovider && python -m compileall -q src tests tools && python tools/smoke_check.py"]

FROM ${PYTHON_IMAGE} AS runtime-headless
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app
COPY constraints/container-runtime-cu128.txt constraints/vision-cu128.txt ./constraints/
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.11.0 torchvision==0.26.0
RUN python -m pip install --no-cache-dir \
      --constraint constraints/container-runtime-cu128.txt \
      setuptools PyYAML pyserial filelock numpy matplotlib pillow requests psutil polars \
      nvidia-ml-py ultralytics-thop opencv-python-headless \
    && python -m pip install --no-cache-dir --no-deps ultralytics==8.4.104
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps .
COPY --chmod=0555 scripts/container_entrypoint.sh /usr/local/bin/marine-ptz-entrypoint
COPY --chown=10001:10001 configs ./configs
COPY --chown=10001:10001 tools/runtime_preflight.py ./tools/runtime_preflight.py
RUN addgroup --system --gid 10001 marine-ptz \
    && adduser --system --uid 10001 --ingroup marine-ptz --home /nonexistent \
      --no-create-home marine-ptz
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/marine-ptz-entrypoint"]
CMD ["python", "-m", "marine_ptz.vision_cli", "--help"]
