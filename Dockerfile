FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HAP_MULTI=1 \
    HAP_MULTI_SET=provided4 \
    HAP_MULTI_THREADS=3 \
    HAP_MULTI_WORKER_TIMEOUT=3200 \
    OMP_NUM_THREADS=3 \
    MKL_NUM_THREADS=3 \
    NUMBA_NUM_THREADS=3

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
