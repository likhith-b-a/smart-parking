#!/usr/bin/env bash
set -e

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. Run with:"
echo "  source .venv/bin/activate && python app.py"
