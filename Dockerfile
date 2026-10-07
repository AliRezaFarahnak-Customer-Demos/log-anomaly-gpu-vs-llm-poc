# Full image, not -slim: Triton compiles a small CUDA helper at the first training step and needs gcc.
FROM python:3.12

ARG PIP_INDEX_URL=""
ARG GIT_SHA=nogit
ARG GIT_DIRTY=false
ARG IMAGE_TAG=local

WORKDIR /app

# Dependencies first so code changes do not invalidate this layer. No model weights in the image:
# they are downloaded to the shared ML_ROOT by the download job.
COPY docker/requirements.lock docker/requirements.lock
RUN pip install --no-cache-dir ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} -r docker/requirements.lock

COPY pyproject.toml README.md ./
COPY src src
COPY configs configs
COPY data/synthetic data/synthetic
RUN pip install --no-cache-dir --no-deps .

ENV ML_ROOT=/mnt/ml \
    HF_HOME=/mnt/ml/hf-cache \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    GIT_SHA=${GIT_SHA} \
    GIT_DIRTY=${GIT_DIRTY} \
    IMAGE_TAG=${IMAGE_TAG}

ENTRYPOINT ["python", "-m", "logpoc"]
