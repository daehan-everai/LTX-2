#!/usr/bin/env bash
set -euo pipefail

# ── LTX-2: install dependencies and launch the training UI ─────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration (single source of truth) ─────────────────────────────────
export PORT="${PORT:-8675}"
# TensorBoard process. Bound to loopback and reverse-proxied through Next.js
# so users only need to forward one port.
export TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
export TENSORBOARD_HOST="${TENSORBOARD_HOST:-127.0.0.1}"
# Sub-path under which TensorBoard is proxied. Must match the rewrite in
# ui/next.config.ts and the --path_prefix flag in api/tensorboard/route.ts.
export TENSORBOARD_PATH_PREFIX="${TENSORBOARD_PATH_PREFIX:-/tensorboard}"

# ── Helpers ────────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "→ $*"; }

# ── 1. Python environment (uv) ─────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

info "Syncing Python dependencies (uv sync)..."
uv sync

export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# ── 2. Node.js / npm ───────────────────────────────────────────────────────
command -v node &>/dev/null || die "Node.js not found. Install Node 20+ and retry."
command -v npm  &>/dev/null || die "npm not found."

# ── 3. Launch UI (Next.js + worker via concurrently) ───────────────────────
info "Launching UI on http://localhost:${PORT}  (TensorBoard at ${TENSORBOARD_PATH_PREFIX}/)"
cd "$SCRIPT_DIR/ui"

if [[ "${1:-}" == "--dev" ]]; then
    info "Running in development mode (hot-reload)..."
    [[ -d node_modules ]] || npm install
    exec npm run dev
else
    info "Running production build..."
    exec npm run build_and_start
fi
