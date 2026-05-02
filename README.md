# Qwen3-TTS UI

现代化的 Qwen3-TTS Gradio 界面，支持三种语音合成模式：预设音色、语音设计、声音克隆。支持 0.6B 和 1.7B 两种模型大小。

## 功能特性

### 支持两种模型大小

| 模型大小 | 显存需求 | 模型体积 | 特点 |
|---------|---------|---------|------|
| **0.6B** | ~2GB | ~1.5GB/模型 | 轻量级，适合低显存设备 |
| **1.7B** | ~4GB | ~3.5GB/模型 | 高质量，推荐使用 |

可在 Web UI 中一键切换模型大小，无需重启服务。

### 三大核心功能

| 功能 | 0.6B 模型 | 1.7B 模型 | 说明 |
|------|-----------|-----------|------|
| **Custom Voice** | Qwen3-TTS-12Hz-0.6B-CustomVoice | Qwen3-TTS-12Hz-1.7B-CustomVoice | 9种预设音色 |
| **指令控制** | ❌ 不支持 | ✅ 支持 | 情感/风格指令控制（仅 1.7B） |
| **Voice Design** | ❌ 不存在 | Qwen3-TTS-12Hz-1.7B-VoiceDesign | 自然语言描述设计任意音色（仅 1.7B） |
| **Voice Clone** | Qwen3-TTS-12Hz-0.6B-Base | Qwen3-TTS-12Hz-1.7B-Base | 3秒参考音频即可克隆声音 |

### 预设音色（Custom Voice）

| 音色 | 中文描述 | English Description | 母语 |
|------|----------|---------------------|------|
| Vivian | 明亮、略带锋芒的年轻女声 | Bright, slightly edgy young female voice | 中文 |
| Serena | 温暖、柔和的年轻女声 | Warm, gentle young female voice | 中文 |
| Uncle_Fu | 低沉醇厚的成熟男声 | Seasoned male voice with a low, mellow timbre | 中文 |
| Dylan | 清亮自然的京味年轻男声 | Youthful Beijing male voice | 京味方言 |
| Eric | 略带沙哑亮度的成都男声 | Lively Chengdu male voice | 川味方言 |
| Ryan | 节奏感强的动感男声 | Dynamic male voice with strong rhythmic drive | English |
| Aiden | 阳光清澈的美式男中音 | Sunny American male voice with a clear midrange | English |
| Ono_Anna | 轻快俏皮的日系女声 | Playful Japanese female voice | 日语 |
| Sohee | 温暖富有情感的韩语女声 | Warm Korean female voice with rich emotion | 韩语 |

### 支持语言

中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语（共10种）+ 自动检测

## 安装

### 环境要求

- Python 3.10+
- CUDA GPU（推荐，显存 ≥ 8GB）
- 或 CPU（推理较慢）

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/RSJY/Qwen3Audio.git
cd Qwen3Audio

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 【重要】先安装 CUDA 版本的 PyTorch
# 默认 pip install torch 是 CPU 版本，必须手动安装 CUDA 版本！

# 卸载 CPU 版本（如果已安装）
pip uninstall torch torchvision torchaudio

# 安装 CUDA 12.1 版本（推荐，适配 CUDA 12.x 显卡驱动）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 11.8 版本（适配旧显卡）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 可选：安装 Flash Attention 2 加速推理（需要 CUDA GPU）
pip install flash-attn --no-build-isolation
```

### 验证 CUDA 安装

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# 应输出: CUDA available: True
```

如果显示 `False`，说明安装的是 CPU 版本 PyTorch，请重新安装 CUDA 版本。

## 使用方法

### 启动 Web UI

```bash
# 默认启动（不预加载模型，首次使用时自动下载）
python main.py

# 预加载特定模型
python main.py --mode custom_voice   # 预加载预设音色模型
python main.py --mode voice_design   # 预加载语音设计模型
python main.py --mode base           # 预加载声音克隆模型
python main.py --mode all            # 预加载所有模型

# 完整参数
python main.py \
  --mode all \
  --port 7860 \
  --ip 0.0.0.0 \
  --share \
  --dtype bfloat16 \
  --device cuda:0
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | none | 预加载模型：custom_voice, voice_design, base, all, none |
| `--model-size` | 1.7B | 模型大小：0.6B（轻量）或 1.7B（高质量） |
| `--port` | 7860 | 服务端口 |
| `--ip` | 0.0.0.0 | 绑定地址 |
| `--share` | False | 创建公网 Gradio 链接 |
| `--model-dir` | None | 自定义模型目录（跳过自动下载） |
| `--dtype` | bfloat16 | 模型精度：float16, bfloat16 |
| `--device` | cuda:0 | 设备：cuda:0, cuda:1, cpu |
| `--offline` | False | 离线模式（模型必须预先下载） |

### 一键启动脚本

```bash
# Windows
start.bat

# Linux/Mac
chmod +x start.sh
./start.sh
```

## 离线部署

### 方案一：离线启动模式

在有网络的环境下预先下载模型，然后在离线环境启动：

```bash
# 步骤1: 在有网络环境下下载模型
python download_models.py

# 步骤2: 将模型目录复制到离线机器
# 模型位置: ~/.cache/qwen3-tts/

# 步骤3: 在离线机器上启动
python main.py --offline --mode all

# 或使用离线启动脚本
offline_start.bat  # Windows
./offline_start.sh  # Linux/Mac
```

### 方案二：打包为 EXE（完全离线部署）

将应用打包为独立可执行文件，可在无 Python 环境的机器上运行：

```bash
# 步骤1: 下载模型用于打包
python download_models.py --for-exe

# 步骤2: 打包 EXE
build_exe.bat  # Windows
./build_exe.sh  # Linux/Mac

# 步骤3: 分发
# 输出目录: dist/Qwen3-TTS/
# 将整个目录复制到目标机器即可使用
```

**打包后的目录结构：**
```
dist/Qwen3-TTS/
├── Qwen3-TTS.exe      # 主程序
├── models/            # 预下载的模型（离线可用）
│   ├── tokenizer/
│   ├── custom_voice/
│   ├── voice_design/
│   └── base/
├── _internal/         # 依赖库
└── ...
```

### 环境变量配置

| 环境变量 | 说明 |
|---------|------|
| `QWEN3_TTS_OFFLINE` | 设为 `1` 启用离线模式 |
| `QWEN3_TTS_MODELS_DIR` | 自定义模型目录路径 |
| `HF_HUB_OFFLINE` | 设为 `1` 禁用 HuggingFace 网络请求 |

```bash
# 示例：使用自定义模型目录
export QWEN3_TTS_MODELS_DIR=/path/to/models
python main.py
```

## 模型自动下载

首次运行时，模型会自动从 HuggingFace 下载到 `~/.cache/qwen3-tts/` 目录：

**Tokenizer（两种大小共用）：**
- `Qwen/Qwen3-TTS-Tokenizer-12Hz`

**0.6B 模型（轻量级）：**
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` - 预设音色模型（不支持指令控制）
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` - 声音克隆模型

**1.7B 模型（高质量）：**
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` - 预设音色模型（支持指令控制）
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` - 语音设计模型
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` - 声音克隆模型

### 国内用户加速

如果 HuggingFace 下载缓慢，系统会自动尝试 ModelScope 镜像：

```bash
pip install modelscope  # 安装后自动使用国内镜像
```

或手动下载：

```bash
# 使用 ModelScope 下载（1.7B 模型）
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ~/.cache/qwen3-tts/1.7B/custom_voice

# 下载 0.6B 模型
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local_dir ~/.cache/qwen3-tts/0.6B/custom_voice
```

## API 使用

```python
from app import TTSEngine

# 初始化引擎（默认使用 1.7B 模型）
engine = TTSEngine(device="cuda:0", dtype="bfloat16")

# 初始化引擎（使用 0.6B 轻量模型）
engine = TTSEngine(device="cuda:0", dtype="bfloat16", model_size="0.6B")

# 运行时切换模型大小
engine.set_model_size("0.6B")  # 切换到 0.6B
engine.set_model_size("1.7B")  # 切换到 1.7B

# 1. 预设音色生成（1.7B 支持指令控制，0.6B 会自动忽略 instruct）
audio, sr = engine.generate_custom_voice(
    text="你好，欢迎使用Qwen3-TTS！",
    language="Chinese",
    speaker="Vivian",
    instruct="用温柔的语气说"  # 仅 1.7B 有效
)

# 2. 语音设计生成（仅 1.7B 支持）
audio, sr = engine.generate_voice_design(
    text="Hello, this is a voice design test.",
    language="English",
    instruct="A deep, authoritative male voice with a British accent"
)

# 3. 声音克隆生成
audio, sr = engine.generate_voice_clone(
    text="这是克隆后的声音。",
    language="Chinese",
    ref_audio="reference.wav",  # 或 (audio_array, sample_rate) 元组
    ref_text="这是参考音频的转录文本",  # 可选但推荐
    x_vector_only_mode=False
)

# 保存音频
import soundfile as sf
sf.write("output.wav", audio, sr)
```

## 项目结构

```
qwen3-tts-ui/
├── main.py              # 入口：CLI参数解析 + 启动UI
├── pyproject.toml       # 项目配置
├── requirements.txt     # 依赖列表
├── start.bat            # Windows一键启动
├── start.sh             # Linux/Mac一键启动
├── offline_start.bat    # Windows离线启动
├── offline_start.sh     # Linux/Mac离线启动
├── download_models.py   # 模型预下载脚本
├── build_exe.bat        # Windows EXE打包脚本
├── build_exe.sh         # Linux/Mac EXE打包脚本
├── qwen3_tts.spec       # PyInstaller配置
├── README.md            # 说明文档
└── app/
    ├── __init__.py      # 包初始化
    ├── config.py        # 配置：音色、语言、模型ID
    ├── model_manager.py # 模型管理：自动下载、GPU内存管理
    ├── tts_engine.py    # TTS引擎：三种生成模式统一API
    ├── ui.py            # Gradio UI：三标签页界面
    └── style.css        # 暗色主题样式
```

## 注意事项

### GPU 显存
1. **0.6B 模型**：需要约 2GB 显存（bfloat16），建议 4GB+
2. **1.7B 模型**：需要约 4GB 显存（bfloat16），建议 8GB+
3. **首次启动**：模型下载约 1.5GB（0.6B）或 3.5GB（1.7B）/模型，请确保网络畅通
3. **麦克风录制**：远程部署需 HTTPS 才能使用浏览器麦克风
4. **模型切换**：0.6B 仅支持预设音色（无指令控制）和声音克隆，1.7B 支持全部功能
5. **EXE 打包**：打包后体积较大（约 10-15GB，含模型），请确保足够磁盘空间

## 致谢

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - 阿里云通义千问语音合成模型
- [Gradio](https://gradio.app/) - 机器学习 Web 界面框架

## License

Apache 2.0
