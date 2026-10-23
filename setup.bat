
@echo off
title Stellar Classification - Setup and Launch
cd /d "%~dp0"

echo ==========================================
echo     STELLAR CLASSIFICATION AI PROJECT
echo ==========================================
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed.
    echo Install Git from https://git-scm.com/downloads
    pause
    exit /b 1
)

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed.
    echo Install Python 3.11 from:
    echo https://www.python.org/downloads/release/python-3119/
    echo Make sure Python is added to PATH.
    pause
    exit /b 1
)

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 was not found.
    echo Install Python 3.11 and run this file again.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: Project files are missing.
    echo Clone the GitHub repository first.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 goto error
)

echo.
echo [2/3] Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [3/3] Launching application...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto error

goto end

:error
echo.
echo Setup or launch failed. Check the error messages above.
pause
exit /b 1

:end
pause
