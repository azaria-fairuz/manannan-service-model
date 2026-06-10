#-- stage 1
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ccache \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install Nuitka

COPY trained_model.onnx trained_risks.npy converter.py main.py ./
RUN python converter.py && \
    rm trained_model.onnx trained_risks.npy converter.py

RUN --mount=type=cache,target=/root/.cache/Nuitka \
    python -m nuitka --standalone --remove-output --jobs=2 main.py

#-- stage 2
FROM python:3.11-slim AS runner

WORKDIR /app
COPY --from=builder /build/main.dist ./main.dist

EXPOSE 5106
CMD ["./main.dist/main.bin"]