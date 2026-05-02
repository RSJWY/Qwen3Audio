@echo off
REM Qwen3-TTS EXE 打包脚本 (Windows)
REM 使用 PyInstaller 将应用打包为独立可执行文件

setlocal EnableDelayedExpansion

echo ============================================================
echo Qwen3-TTS EXE 打包工具
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查 PyInstaller
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo 正在安装 PyInstaller...
    pip install pyinstaller
)

REM 检查是否下载了模型
set MODELS_DIR=%~dp0models
if not exist "%MODELS_DIR%" (
    echo.
    echo 注意: 未找到 models/ 目录
    echo 如果需要离线部署，请先运行:
    echo   python download_models.py --for-exe
    echo.
    echo 继续打包将创建在线版EXE（需要网络下载模型）
    echo.
    choice /C YN /M "是否继续打包"
    if errorlevel 2 exit /b 0
)

REM 清理旧的构建文件
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo.
echo 开始打包...
echo.

REM 运行 PyInstaller
pyinstaller qwen3_tts.spec --clean

if errorlevel 1 (
    echo.
    echo 错误: 打包失败
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 打包完成！
echo ============================================================
echo.
echo 输出目录: %~dp0dist\Qwen3-TTS\
echo 可执行文件: %~dp0dist\Qwen3-TTS\Qwen3-TTS.exe
echo.

REM 如果有模型目录，复制到输出目录
if exist "%MODELS_DIR%" (
    echo 正在复制模型文件到输出目录...
    xcopy "%MODELS_DIR%" "%~dp0dist\Qwen3-TTS\models\" /E /I /Y
    echo 模型文件已复制
    echo.
    echo 离线部署包已准备完成！
    echo 将 dist\Qwen3-TTS\ 目录复制到目标机器即可使用
) else (
    echo 在线版EXE已创建
    echo 首次运行时会自动下载模型到用户缓存目录
)

echo.
pause
