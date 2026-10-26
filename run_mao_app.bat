@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3.12 -m venv .venv
        if errorlevel 1 py -3 -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python was not found or virtual environment creation failed.
    echo Install Python 3.11 or 3.12, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo Installing MAO application dependencies. An internet connection is required on first launch.
    .venv\Scripts\python.exe -m pip install --upgrade pip
    if errorlevel 1 goto :failed
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

.venv\Scripts\python.exe -m streamlit run app.py
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo ERROR: Setup or application launch failed. Review the messages above.
pause
exit /b 1
