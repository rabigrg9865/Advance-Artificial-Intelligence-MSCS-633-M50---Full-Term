@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "WHEELS_DIR=%~dp0wheels"
set "ARGS=%*"

REM One-click behavior:
REM - If local .venv is missing/broken, rebuild it.
REM - If dependencies are missing, install them.
REM - Then run chatbot.
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Existing .venv is broken. Rebuilding...
        rmdir /s /q "%VENV_DIR%"
    )
)

if not exist "%VENV_PY%" (
    echo Creating local virtual environment...
    py -3 -m venv "%VENV_DIR%" >nul 2>&1 || python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        echo Install Python 3 and try again.
        pause
        exit /b 1
    )
)

REM Install dependencies only when missing.
"%VENV_PY%" -c "import django, chatterbot" >nul 2>&1
if errorlevel 1 (
    if exist "%WHEELS_DIR%" (
        echo First run setup: installing dependencies from local wheels...
        "%VENV_PY%" -m pip install --no-index --find-links "%WHEELS_DIR%" -r requirements.txt
    ) else (
        echo First run setup: installing dependencies from internet...
        "%VENV_PY%" -m pip install -r requirements.txt
    )

    if errorlevel 1 (
        if exist "%WHEELS_DIR%" (
            echo Local wheel install incomplete. Retrying from internet...
            "%VENV_PY%" -m pip install -r requirements.txt
        )
    )

    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

:run_local
echo Launching chatbot...
"%VENV_PY%" chatbot.py %ARGS%
if errorlevel 1 (
    echo Chatbot exited with an error.
    pause
    exit /b 1
)
exit /b 0
