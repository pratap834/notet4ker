#!/bin/bash
# Physician Notetaker Web Application Startup Script
# This script activates the virtual environment and starts the Flask server

echo "================================================"
echo "  Physician Notetaker Web Application"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    echo "Then run: source .venv/bin/activate"
    echo "Then run: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "[1/2] Activating virtual environment..."
source .venv/bin/activate

# Start Flask application
echo "[2/2] Starting Flask server..."
echo ""
echo "The web application will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "================================================"
echo ""

python app.py
