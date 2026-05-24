#!/usr/bin/env bash
# =============================================================================
# BigV — Wildlife Spotter Start Script
# Usage: bash start.sh [--install] [--port 8000]
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PORT="${PORT:-8000}"
VENV_DIR="$(pwd)/.venv"
INSTALL=false

# Parse args
for arg in "$@"; do
  case $arg in
    --install) INSTALL=true ;;
    --port=*) PORT="${arg#*=}" ;;
  esac
done

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║   🦁  BigV — Wildlife Spotter v1.0   ║"
echo "  ║   Agentic AI · Real-Time Detection   ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ─── Virtual environment ──────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]] || [[ "$INSTALL" == "true" ]]; then
  echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip -q
  echo -e "${YELLOW}Installing dependencies...${NC}"
  pip install -r requirements.txt
  echo -e "${GREEN}✅ Dependencies installed.${NC}"
else
  source "$VENV_DIR/bin/activate"
fi

# ─── Create logs dir ──────────────────────────────────────────────────────────
mkdir -p logs

# ─── Load .env if present ─────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
  echo -e "${CYAN}Loading .env...${NC}"
  set -a; source .env; set +a
fi

# ─── Start server ─────────────────────────────────────────────────────────────
echo -e "${GREEN}"
echo "  Starting server on http://localhost:${PORT}"
echo "  Open your browser at: http://localhost:${PORT}"
echo "  Press Ctrl+C to stop"
echo -e "${NC}"

cd backend
exec "$VENV_DIR/bin/python" -m uvicorn main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --reload \
  --log-level info

# Made with Bob
