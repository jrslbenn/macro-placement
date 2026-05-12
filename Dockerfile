FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir numba scipy matplotlib tqdm absl-py

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .