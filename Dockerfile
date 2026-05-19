FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HAP_MULTI=1 \
    HAP_MULTI_SET=provided4 \
    HAP_MULTI_THREADS=3 \
    HAP_MULTI_WORKER_TIMEOUT=3500 \
    HAP_TOTAL_TIME_BUDGET=2850 \
    HAP_HARD_TIME_BUDGET=2850 \
    HAP_ENABLE_PLOTS=0 \
    OMP_NUM_THREADS=3 \
    MKL_NUM_THREADS=3 \
    NUMBA_NUM_THREADS=3

WORKDIR /app

# Explicit submodule copies — build fails loudly if `git submodule update --init
# external/MacroPlacement` wasn't run before `docker build`.
COPY external/MacroPlacement/Testcases/ICCAD04/ /app/external/MacroPlacement/Testcases/ICCAD04/
COPY external/MacroPlacement/CodeElements/Plc_client/ /app/external/MacroPlacement/CodeElements/Plc_client/

COPY . .
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "macro_place.evaluate", "/app/placer.py"]
CMD ["--all"]
