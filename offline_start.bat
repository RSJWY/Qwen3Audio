@echo off
REM Qwen3-TTS 离线模式启动脚本 (Windows)
REM 设置环境变量以启用离线模式并使用本地模型

setlocal

REM 设置离线模式环境变量
set QWEN3_TTS_OFFLINE=1

REM 设置模型目录（如果 models/ 存在于当前目录）
if exist "%~dp0models" (
    set QWEN3_TTS_MODELS_DIR=%~dp0models
    echo 使用本地模型目录: %~dp0models
)

REM 设置 HuggingFace 离线模式
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo ============================================================
echo Qwen3-TTS 离线模式
echo ============================================================
echo.
echo 离线模式已启用
echo 模型目录: %QWEN3_TTS_MODELS_DIR%
echo.

REM 启动应用
python main.py --mode all %*

endlocal
