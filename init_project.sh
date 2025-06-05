#!/usr/bin/env bash
python -m venv .venv && \
source .venv/bin/activate && \
pip install -r requirements.txt && \
echo "Virtualenv ready. Copy your Alpaca keys into config/base_config.yaml before first run." && \
mkdir -p logs
