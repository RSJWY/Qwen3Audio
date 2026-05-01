# Qwen3-TTS UI

现代化的 Qwen3-TTS Gradio 界面，支持三种语音合成模式：预设音色、语音设计、声音克隆。

## 功能特性

### 三大核心功能（基于 1.7B 模型）

| 功能 | 模型 | 说明 |
|------|------|------|
| **Custom Voice** | Qwen3-TTS-12Hz-1.7B-CustomVoice | 9种预设音色 + 情感/风格指令控制 |
| **Voice Design** | Qwen3-TTS-12Hz-1.7B-VoiceDesign | 自然语言描述设计任意音色 |
| **Voice Clone** | Qwen3-TTS-12Hz-1.7B-Base | 3秒参考音频即可克隆声音 |

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
git clone https://github.com/your-username/qwen3-tts-ui.git
cd qwen3-tts-ui

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 可选：安装 Flash Attention 2 加速推理
pip install flash-attn --no-build-isolation
```

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
| `--port` | 7860 | 服务端口 |
| `--ip` | 0.0.0.0 | 绑定地址 |
| `--share` | False | 创建公网 Gradio 链接 |
| `--model-dir` | None | 自定义模型目录（跳过自动下载） |
| `--dtype` | bfloat16 | 模型精度：float16, bfloat16 |
| `--device` | cuda:0 | 设备：cuda:0, cuda:1, cpu |

### 一键启动脚本

```bash
# Windows
start.bat

# Linux/Mac
chmod +x start.sh
./start.sh
```

## 模型自动下载

首次运行时，模型会自动从 HuggingFace 下载到 `~/.cache/qwen3-tts/` 目录：

- `Qwen/Qwen3-TTS-Tokenizer-12Hz` - 分词器
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` - 预设音色模型
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` - 语音设计模型
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` - 声音克隆模型

### 国内用户加速

如果 HuggingFace 下载缓慢，系统会自动尝试 ModelScope 镜像：

```bash
pip install modelscope  # 安装后自动使用国内镜像
```

或手动下载：

```bash
# 使用 ModelScope 下载
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ~/.cache/qwen3-tts/custom_voice
```

## API 使用

```python
from app import TTSEngine

# 初始化引擎
engine = TTSEngine(device="cuda:0", dtype="bfloat16")

# 1. 预设音色生成
audio, sr = engine.generate_custom_voice(
    text="你好，欢迎使用Qwen3-TTS！",
    language="Chinese",
    speaker="Vivian",
    instruct="用温柔的语气说"  # 可选
)

# 2. 语音设计生成
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

1. **GPU 显存**：1.7B 模型需要约 4GB 显存（bfloat16），建议 8GB+
2. **首次启动**：模型下载约 3-4GB，请确保网络畅通
3. **麦克风录制**：远程部署需 HTTPS 才能使用浏览器麦克风
4. **模型切换**：三种模式共用基础模型，切换时仅需加载差异部分

## 致谢

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - 阿里云通义千问语音合成模型
- [Gradio](https://gradio.app/) - 机器学习 Web 界面框架

## License

Apache 2.0
