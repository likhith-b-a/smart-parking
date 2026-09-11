@echo off
setlocal

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Done. Run with:
echo   .venv\Scripts\activate ^&^& python app.py
