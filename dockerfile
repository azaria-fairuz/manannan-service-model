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

COPY main.py assets.py ./
RUN python -m nuitka --standalone --remove-output main.py

#-- stage 2
FROM python:3.11-slim AS runner

WORKDIR /app
COPY --from=builder /build/main.dist ./main.dist

EXPOSE 5106
CMD ["./main.dist/main.bin"]