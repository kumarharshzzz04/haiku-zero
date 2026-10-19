@echo off
setlocal
chcp 65001 >nul
title HAIKU ZERO - desk sheet (Streamlit)
rem Basic front end that ships with this repo: seed box, model picker, knobs, and every
rem poem with its measured numbers. No Node.js, no build step. The React desk is private.
cd /d "%~dp0"

set "PY=pytorch_with_gpu\Scripts\python.exe"

if not exist "%PY%" (
    echo  No environment yet - running setup.bat first.
    call setup.bat
)
if not exist "%PY%" exit /b 1

if not exist checkpoints\best.pt (
    echo.
    echo  [X] No model at checkpoints\best.pt - train one with train.bat first.
    pause
    exit /b 1
)

"%PY%" -c "import streamlit" 2>nul
if errorlevel 1 (
    echo  Installing Streamlit for the desk sheet - one time only...
    "%PY%" -m pip install -r requirements.txt
)

echo.
echo  Desk sheet on http://localhost:8601  -  Ctrl+C in this window stops it.
echo.
"%PY%" -m streamlit run streamlit_app.py --server.port 8601 ^
    --browser.gatherUsageStats false
