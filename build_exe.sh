#!/bin/bash
# Qwen3-TTS EXE 打包脚本 (Linux/Mac)
# 使用 PyInstaller 将应用打包为独立可执行文件

set -e

echo "============================================================"
echo "Qwen3-TTS EXE 打包工具"
echo "============================================================"
echo

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi

# 检查 PyInstaller
if ! python3 -c "import PyInstaller" &> /dev/null; then
    echo "正在安装 PyInstaller..."
    pip3 install pyinstaller
fi

# 检查是否下载了模型
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

if [ ! -d "$MODELS_DIR" ]; then
    echo
    echo "注意: 未找到 models/ 目录"
    echo "如果需要离线部署，请先运行:"
    echo "  python3 download_models.py --for-exe"
    echo
    echo "继续打包将创建在线版EXE（需要网络下载模型）"
    echo
    read -p "是否继续打包? [y/N] " choice
    case "$choice" in
        y|Y ) ;;
        * ) exit 0 ;;
    esac
fi

# 清理旧的构建文件
rm -rf build dist

echo
echo "开始打包..."
echo

# 运行 PyInstaller
pyinstaller qwen3_tts.spec --clean

echo
echo "============================================================"
echo "打包完成！"
echo "============================================================"
echo
echo "输出目录: $SCRIPT_DIR/dist/Qwen3-TTS/"
echo "可执行文件: $SCRIPT_DIR/dist/Qwen3-TTS/Qwen3-TTS"
echo

# 如果有模型目录，复制到输出目录
if [ -d "$MODELS_DIR" ]; then
    echo "正在复制模型文件到输出目录..."
    cp -r "$MODELS_DIR" "$SCRIPT_DIR/dist/Qwen3-TTS/models/"
    echo "模型文件已复制"
    echo
    echo "离线部署包已准备完成！"
    echo "将 dist/Qwen3-TTS/ 目录复制到目标机器即可使用"
else
    echo "在线版EXE已创建"
    echo "首次运行时会自动下载模型到用户缓存目录"
fi

echo
