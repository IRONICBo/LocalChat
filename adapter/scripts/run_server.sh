#!/bin/bash
# =============================================================================
# Run Script for ChatAdapter Server
# =============================================================================
# This script starts the ChatAdapter proxy server with OpenAI-compatible backend
#
# Usage:
#   chmod +x scripts/run_server.sh
#   ./scripts/run_server.sh
#
# Environment variables can be overridden:
#   OPENAI_API_KEY=your-key ./scripts/run_server.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "ChatAdapter Proxy Server"
echo "=============================================="

# ==================== Environment Configuration ====================

# LLM Backend Configuration
export LLM_BACKEND="${LLM_BACKEND:-openai}"

# OpenAI-compatible API settings
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-123456}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://0.0.0.0:23333/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-internlm/internlm2-chat-1_8b}"

# PII Detection settings
export PII_DETECTION_METHOD="${PII_DETECTION_METHOD:-E2E}"
export DETECTION_STRATEGY="${DETECTION_STRATEGY:-balanced}"

# Database settings
export DATABASE_URL="${DATABASE_URL:-sqlite:///${PROJECT_DIR}/localchat.db}"

# Server settings
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Logging
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export ENABLE_DETAILED_LOGGING="${ENABLE_DETAILED_LOGGING:-false}"

# ==================== Display Configuration ====================

echo ""
echo "Configuration:"
echo "  LLM Backend:     $LLM_BACKEND"
echo "  API Base:        $OPENAI_API_BASE"
echo "  Model:           $OPENAI_MODEL"
echo "  Detection:       $PII_DETECTION_METHOD"
echo "  Strategy:        $DETECTION_STRATEGY"
echo "  Database:        $DATABASE_URL"
echo "  Server:          http://$HOST:$PORT"
echo ""

# ==================== Check Dependencies ====================

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Warning: No conda environment detected."
    echo "Please activate the chatadapter environment:"
    echo "  conda activate chatadapter"
    echo ""

    # Try to activate conda
    if command -v conda &> /dev/null; then
        echo "Attempting to activate chatadapter environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate chatadapter 2>/dev/null || {
            echo "Failed to activate chatadapter environment."
            echo "Please run: ./scripts/setup_env.sh first"
            exit 1
        }
        echo "Activated conda environment: $CONDA_DEFAULT_ENV"
    fi
elif [[ "$CONDA_DEFAULT_ENV" != "chatadapter" ]]; then
    echo "Warning: Current environment is '$CONDA_DEFAULT_ENV', not 'chatadapter'"
    echo "Consider running: conda activate chatadapter"
fi

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null; then
    echo "ERROR: uvicorn is not installed"
    echo "Please run: pip install uvicorn[standard]"
    exit 1
fi

# ==================== Start Server ====================

cd "$PROJECT_DIR"

echo "Starting ChatAdapter server..."
echo "Press Ctrl+C to stop"
echo ""
echo "=============================================="
echo ""

# Start the FastAPI server
uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --log-level "${LOG_LEVEL,,}"
