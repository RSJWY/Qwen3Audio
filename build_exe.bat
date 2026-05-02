@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo Qwen3-TTS EXE Build Tool
echo ============================================================
echo.

REM Activate venv if exists
if exist "%~dp0.venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%~dp0.venv\Scripts\activate.bat"
)

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

set MODELS_DIR=%~dp0models
if not exist "%MODELS_DIR%" (
    echo NOTE: models/ not found. Run: python download_models.py --for-exe
    choice /C YN /M "Continue without models?"
    if errorlevel 2 exit /b 0
)

if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo Building...
pyinstaller qwen3_tts.spec --clean

if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Build completed!
echo Output: %~dp0dist\Qwen3-TTS\
echo ============================================================

if exist "%MODELS_DIR%" (
    echo Copying models...
    xcopy "%MODELS_DIR%" "%~dp0dist\Qwen3-TTS\models\" /E /I /Y >nul
    echo Offline package ready!
) else (
    echo Online EXE created. Models will download on first run.
)

pause
