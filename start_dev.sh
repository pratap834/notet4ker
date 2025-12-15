#!/usr/bin/env bash
# Local development startup script

echo "=========================================="
echo "Starting Physician Notetaker (Development)"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠ No .env file found. Creating from template..."
    cp .env.example .env
    echo "✓ Created .env file. Please update with your settings."
fi

# Load environment variables
export FLASK_ENV=development
export FLASK_APP=app.py

# Start the application
echo "Starting Flask development server..."
python app.py
