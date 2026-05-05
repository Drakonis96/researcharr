FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=5001

WORKDIR /app

FROM base AS deps

COPY webapp/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /tmp/requirements.txt

FROM base

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY public /app/public
COPY webapp /app/webapp

RUN chmod +x /app/webapp/start.sh

WORKDIR /app/webapp

EXPOSE 5000 5001

CMD ["./start.sh"]