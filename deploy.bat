@echo off

REM Simple deployment script for the Sleep Disorder Prediction API (Windows)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the application
echo Starting the application...
python app.py