@echo off
:: ========================================
:: Universal Python Script Launcher
:: ========================================
:: Configuration Section - Modify variables below to change the script to run
:: ========================================
set SCRIPT_NAME=main.py
set SERVICE_NAME=Qwen-TTS Tool Service
:: ========================================

echo ========================================
echo      Start %SERVICE_NAME%
echo ========================================
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found, please run setup.bat first
    echo.
    echo Command: setup.bat
    pause
    exit /b 1
)

:: Check if script exists
if not exist "%SCRIPT_NAME%" (
    echo [ERROR] %SCRIPT_NAME% file not found
    pause
    exit /b 1
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Start program
echo [INFO] Starting %SERVICE_NAME%...
echo [TIP] Press Ctrl+C to stop the program
echo.
echo ========================================
python %SCRIPT_NAME%

:: Handle program exit
echo.
echo ========================================
echo      %SERVICE_NAME% Stopped
echo ========================================
pause