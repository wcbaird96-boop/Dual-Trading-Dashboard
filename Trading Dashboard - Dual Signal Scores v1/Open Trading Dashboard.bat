@echo off
setlocal

cd /d "%~dp0"

set "APP_FILE=app.py"
set "PYTHON_EXE="
set "LAUNCH_MODE="

if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    set "LAUNCH_MODE=venv"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "LAUNCH_MODE=py"
    )
)

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Create a virtual environment or install Python, then try again.
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo Could not find "%APP_FILE%" in:
    echo %cd%
    pause
    exit /b 1
)

if /i "%~1"=="--dry-run" (
    echo Launcher check passed.
    echo Working directory: %cd%
    echo Python source: %LAUNCH_MODE%
    echo App file: %APP_FILE%
    exit /b 0
)

title Trading Dashboard
echo Launching Trading Dashboard...
echo.

if /i "%LAUNCH_MODE%"=="venv" (
    call "%PYTHON_EXE%" -m streamlit run "%APP_FILE%"
) else (
    call py -m streamlit run "%APP_FILE%"
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The dashboard closed with exit code %EXIT_CODE%.
    pause
)

endlocal & exit /b %EXIT_CODE%
