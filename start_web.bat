@echo off
REM Physician Notetaker Web Application Startup Script
REM This script activates the virtual environment and starts the Flask server

echo ================================================
echo   Physician Notetaker Web Application
echo ================================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then run: .venv\Scripts\activate
    echo Then run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Start Flask application
echo [2/2] Starting Flask server...
echo.
echo The web application will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================
echo.

python app.py

pause
