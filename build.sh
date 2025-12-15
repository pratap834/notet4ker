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
# Install problematic packages first with pre-built wheels only
pip install --only-binary=:all: blis==0.7.11 cymem==2.0.8 murmurhash==1.0.10 preshed==3.0.9 thinc==8.2.2

# Install remaining dependencies
pip install -r requirements.txt --prefer-binary

# Install spaCy models (if needed)
# python -m spacy download en_core_web_sm

echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
