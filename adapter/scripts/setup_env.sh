#!/bin/bash
# =============================================================================
# Setup Script for ChatAdapter Environment
# =============================================================================
# This script creates the conda environment and installs all dependencies
#
# Usage:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "ChatAdapter Environment Setup"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Environment name
ENV_NAME="chatadapter"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Keeping existing environment. Activating..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $ENV_NAME
        echo "Environment activated. Installing/updating packages..."
    fi
fi

# Create new environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install --upgrade pip

# FastAPI and web server
pip install fastapi uvicorn[standard] python-multipart

# Database
pip install sqlalchemy

# HTTP client
pip install requests httpx

# Data processing
pip install pydantic

# PII Detection - Presidio
echo ""
echo "Installing Presidio (PII detection)..."
pip install presidio-analyzer presidio-anonymizer

# Download spaCy model for Presidio
echo ""
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_lg

# Faker for synthetic data generation
pip install faker

# OpenAI compatible client
pip install openai

# Optional: pandas for data analysis
pip install pandas openpyxl

# Optional: for testing
pip install pytest pytest-asyncio

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To start the server, run:"
echo "  ./scripts/run_server.sh"
echo ""
echo "To test the API, run:"
echo "  ./scripts/test_api.sh"
echo ""
