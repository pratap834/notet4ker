#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "=========================================="
echo "Building Physician Notetaker Application"
echo "=========================================="

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install spaCy models (if needed)
# python -m spacy download en_core_web_sm

echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
