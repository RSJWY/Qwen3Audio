@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    Python Virtual Environment Setup
echo ========================================
echo.

:: Get project name from current directory
for %%I in (.) do set "PROJECT_NAME=%%~nxI"
echo [INFO] Project: %PROJECT_NAME%
echo [INFO] Current directory: %cd%
echo.

:: Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found, please install Python 3.7+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Python version detected:
python --version
echo.

:: Check for virtual environment name parameter
set "VENV_NAME=.venv"
if not "%1"=="" (
    set "VENV_NAME=%1"
    echo [INFO] Using custom virtual environment name: %VENV_NAME%
) else (
    echo [INFO] Using default virtual environment name: %VENV_NAME%
)
echo.

:: Remove existing virtual environment if exists
if exist "%VENV_NAME%" (
    echo [INFO] Removing existing virtual environment '%VENV_NAME%'...
    rmdir /s /q "%VENV_NAME%"
    if %errorlevel% equ 0 (
        echo [DONE] Old virtual environment removed
    ) else (
        echo [WARN] Failed to remove old virtual environment, continuing...
    )
    echo.
)

:: Create virtual environment
echo [INFO] Creating virtual environment '%VENV_NAME%'...
python -m venv "%VENV_NAME%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [DONE] Virtual environment created successfully
echo.

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call "%VENV_NAME%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [DONE] Virtual environment activated
echo.

::Configure pip to use Tsinghua mirror by default
echo [INFO] Configuring pip to use USTC mirror...
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
pip config set global.trusted-host pypi.mirrors.ustc.edu.cn
echo [DONE] USTC mirror configured
echo.

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARN] pip upgrade failed, continuing with dependencies...
) else (
    echo [DONE] pip upgraded successfully
)
echo.

:: Install dependencies
echo [INFO] Installing project dependencies...

:: Check for different dependency files in order of preference
set "DEP_FILE="
if exist "requirements.txt" (
    set "DEP_FILE=requirements.txt"
) else if exist "pyproject.toml" (
    set "DEP_FILE=pyproject.toml"
) else if exist "Pipfile" (
    set "DEP_FILE=Pipfile"
) else if exist "setup.py" (
    set "DEP_FILE=setup.py"
)

if not "!DEP_FILE!"=="" (
    echo [INFO] Found dependency file: !DEP_FILE!
    
    if "!DEP_FILE!"=="requirements.txt" (
        pip install -r requirements.txt
    ) else if "!DEP_FILE!"=="pyproject.toml" (
        pip install .
    ) else if "!DEP_FILE!"=="Pipfile" (
        echo [INFO] Pipfile detected, please use 'pipenv install' instead
        echo [WARN] Continuing without installing dependencies
    ) else if "!DEP_FILE!"=="setup.py" (
        pip install -e .
    )
    
    if %errorlevel% neq 0 (
        echo [ERROR] Dependencies installation failed
        pause
        exit /b 1
    )
    echo [DONE] Dependencies installed successfully
) else (
    echo [WARN] No dependency file found (requirements.txt, pyproject.toml, setup.py, Pipfile)
    echo [INFO] Virtual environment created without dependencies
)
echo.

:: Show installed packages
echo [INFO] Installed packages:
pip list --format=columns
echo.

:: Success message
echo ========================================
echo     Environment Setup Complete!
echo ========================================
echo.
echo Project: %PROJECT_NAME%
echo Virtual Environment: %VENV_NAME%
echo.
echo Usage Instructions:
echo 1. Activate virtual environment:
echo    %VENV_NAME%\Scripts\activate.bat
echo.
echo 2. Run your Python scripts:
echo    python your_script.py
echo.
echo 3. Deactivate virtual environment:
echo    deactivate
echo.
echo 4. To recreate environment:
echo    setup.bat [custom_venv_name]
echo.
echo ========================================

pause