#!/usr/bin/env bash

# Check for python3 first, then python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Neither python3 nor python found in PATH"
    echo "Please install Python 3.13+ via Homebrew:"
    echo "brew install python"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Create virtual environment
$PYTHON_CMD -m venv .venv && \
source .venv/bin/activate && \
pip install -r requirements.txt && \
echo "✅ Virtual environment ready!" && \
echo "📝 Next steps:" && \
echo "1. Copy your Alpaca paper trading keys into config/base_config.yaml" && \
echo "2. Run: source .venv/bin/activate" && \
echo "3. Test with: python run_trading_bot.py --mode child" && \
mkdir -p logs