@echo off
echo Setting up ISRO GNSS Monitoring Dashboard...
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if virtual environment exists, if not create one
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing/Updating required packages...
pip install --upgrade pip
pip install -r dashboard_requirements_updated.txt
if %ERRORLEVEL% neq 0 (
    echo Warning: Some packages failed to install. Trying with the original requirements file...
    pip install -r dashboard_requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to install required packages.
        echo Please check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo Starting the dashboard...
echo ========================================
echo The dashboard will open in your default web browser.
echo If it doesn't open automatically, go to http://localhost:8501
echo.
echo Login Credentials:
echo Admin:     admin / admin123
echo Operator:  operator / operator123
echo Viewer:    viewer / viewer123
echo.
echo ========================================

streamlit run app.py

pause
