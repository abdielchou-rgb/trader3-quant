# trader3-quant 多阶段构建
# ── 构建层：装依赖 + 质量门（ruff/mypy 只在构建层跑，运行层零工具链）──
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --prefix=/install --no-deps .

# ── 运行层：最小镜像 ──
FROM python:3.11-slim

LABEL org.opencontainers.image.title="trader3-quant"
LABEL org.opencontainers.image.description="Quantitative trading engine for China A-shares"
LABEL org.opencontainers.image.source="https://github.com/abdielchou-rgb/trader3-quant"

# 安全：非 root 运行
RUN useradd -m -u 1000 trader
WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/trader3 ./trader3
COPY --from=builder /build/evolve ./evolve
COPY --from=builder /build/pyproject.toml ./
COPY --from=builder /build/requirements.txt ./requirements.txt
COPY --from=builder /build/LICENSE ./LICENSE

USER trader

# 默认暴露 API（uvicorn）；/metrics 同端口
EXPOSE 8000

# 健康检查（仅探进程内 HTTP 可达；未配 key 时 /metrics 无鉴权）
HEALTHCHECK --interval=60s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "trader3.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
