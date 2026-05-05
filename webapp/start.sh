#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_VENV_PYTHON="$BASE_DIR/../.venv/bin/python"
LOCAL_VENV_PYTHON="$BASE_DIR/venv/bin/python"
SYSTEM_PYTHON_BIN="$(command -v python3 || command -v python || true)"

cd "$BASE_DIR"

if [ -x "$ROOT_VENV_PYTHON" ]; then
	PYTHON_BIN="$ROOT_VENV_PYTHON"
elif [ -x "$LOCAL_VENV_PYTHON" ]; then
	PYTHON_BIN="$LOCAL_VENV_PYTHON"
elif [ -n "$SYSTEM_PYTHON_BIN" ]; then
	PYTHON_BIN="$SYSTEM_PYTHON_BIN"
else
	echo "No Python interpreter found. Create .venv at the repo root or webapp/venv, or install python3." >&2
	exit 1
fi

MCP_HOST="${MCP_HOST:-127.0.0.1}"
MCP_PORT="${MCP_PORT:-5001}"

cleanup() {
	if [ -n "$MCP_PID" ] && kill -0 "$MCP_PID" 2>/dev/null; then
		kill "$MCP_PID" 2>/dev/null || true
	fi
	if [ -n "$FLASK_PID" ] && kill -0 "$FLASK_PID" 2>/dev/null; then
		kill "$FLASK_PID" 2>/dev/null || true
	fi
	wait "$MCP_PID" "$FLASK_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

"$PYTHON_BIN" mcp_server.py &
MCP_PID=$!
echo "MCP server starting on ${MCP_HOST}:${MCP_PORT}" >&2

"$PYTHON_BIN" app.py &
FLASK_PID=$!

wait -n "$MCP_PID" "$FLASK_PID" 2>/dev/null || true