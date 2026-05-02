#!/bin/bash
# Qwen3-TTS 离线模式启动脚本 (Linux/Mac)
# 设置环境变量以启用离线模式并使用本地模型

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置离线模式环境变量
export QWEN3_TTS_OFFLINE=1

# 设置模型目录（如果 models/ 存在于当前目录）
if [ -d "$SCRIPT_DIR/models" ]; then
    export QWEN3_TTS_MODELS_DIR="$SCRIPT_DIR/models"
    echo "使用本地模型目录: $SCRIPT_DIR/models"
fi

# 设置 HuggingFace 离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "Qwen3-TTS 离线模式"
echo "============================================================"
echo
echo "离线模式已启用"
echo "模型目录: $QWEN3_TTS_MODELS_DIR"
echo

# 启动应用
python3 main.py --mode all "$@"
