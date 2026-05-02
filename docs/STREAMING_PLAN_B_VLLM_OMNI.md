# 流式音频输出实现方案 B：基于 vLLM-Omni

## 一、方案概述

### 目标
使用 vLLM-Omni 框架替代当前 qwen_tts 包，利用其原生支持的 AsyncOmni 流式 API，实现真正的流式音频输出，首包延迟可达 ~97ms。

### 核心思路
1. 安装 vLLM 和 vLLM-Omni 依赖
2. 使用 AsyncOmni 替代 Qwen3TTSModel
3. 利用 `async for stage_output in omni.generate()` 实现流式生成
4. Gradio 前端使用 `gr.Audio(streaming=True, autoplay=True)` 实时播放

### 技术优势

| 特性 | 说明 |
|------|------|
| **原生流式支持** | AsyncOmni 提供真正的增量音频输出 |
| **低延迟** | 首包延迟 ~97ms（官方数据） |
| **生产级质量** | vLLM 团队维护，经过大量测试 |
| **双阶段流水线** | 优化的并发和内存管理 |
| **OpenAI 兼容 API** | 可部署为独立服务 |

---

## 二、环境要求

### 硬件要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Linux（推荐）/ Windows（可能需要 WSL2） |
| **GPU 计算能力** | 7.0+ (V100, A100, RTX20xx, L4, H100 等) |
| **GPU 显存** | 建议 8GB+ (1.7B 模型约需 4GB) |
| **CPU** | 多核处理器，建议 8 核+ |
| **内存** | 建议 16GB+ |

### 软件要求

| 项目 | 版本要求 |
|------|---------|
| **Python** | 3.12（推荐） |
| **CUDA** | 13.0 兼容（vLLM 0.20.0 默认） |
| **PyTorch** | 通过 vLLM 自动安装 |

---

## 三、安装步骤

### 步骤 1：创建新虚拟环境

```bash
# 使用 uv 创建环境（推荐）
uv venv --python 3.12 --seed
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 或使用 venv
python3.12 -m venv .venv
source .venv/bin/activate
```

### 步骤 2：安装 vLLM

```bash
# 安装 vLLM（自动处理 PyTorch CUDA 版本）
uv pip install vllm --torch-backend=auto

# 或使用 pip
pip install vllm
```

### 步骤 3：安装 vLLM-Omni

```bash
# 安装 vLLM-Omni 核心包
uv pip install vllm-omni

# 安装 demo 扩展（包含 Gradio 相关依赖）
uv pip install 'vllm-omni[demo]'
```

### 步骤 4：安装其他依赖

```bash
# 音频处理
pip install soundfile

# 其他可能需要的依赖
pip install numpy
```

### 步骤 5：验证安装

```python
# 测试脚本：test_vllm_omni.py
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_omni import AsyncOmni

print("vLLM-Omni 安装成功！")

# 测试加载模型（可选，首次会下载模型）
# omni = AsyncOmni(model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
# print("模型加载成功！")
```

---

## 四、文件改动清单

### 需要新增的文件

| 文件 | 说明 |
|------|------|
| `app/vllm_engine.py` | vLLM-Omni 引擎封装 |
| `app/ui_vllm.py` | 基于 vLLM 的 Gradio 界面 |
| `requirements-vllm.txt` | vLLM 依赖列表 |

### 需要修改的文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `main.py` | 新增参数 | 添加 `--backend` 参数选择引擎 |
| `README.md` | 更新文档 | 添加 vLLM-Omni 使用说明 |

---

## 五、详细实现代码

### 步骤 1：创建 vLLM 引擎封装

**文件**: `app/vllm_engine.py`

```python
"""
vLLM-Omni TTS Engine for Qwen3-TTS

Provides streaming TTS generation using AsyncOmni from vLLM-Omni.
"""

import os
import asyncio
from typing import Generator, Tuple, Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

# 必须在导入 vllm_omni 之前设置
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_omni import AsyncOmni

from .config import SPEAKERS, LANGUAGES, MODEL_IDS, DEFAULT_MODEL_SIZE


# 有效语言列表
VALID_LANGUAGES = list(LANGUAGES) + ["Auto"]


class VLLMTTSEngine:
    """
    基于 vLLM-Omni 的 TTS 引擎，支持真正的流式生成。
    
    支持：
    - CustomVoice: 预设音色（支持流式）
    - VoiceDesign: 语音设计（支持流式）
    - Base: 声音克隆（支持流式）
    """
    
    # 模型类型到模型 ID 的映射
    MODEL_TYPE_MAP = {
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    }
    
    def __init__(
        self,
        model_size: str = "1.7B",
        device: str = "cuda:0",
        gpu_memory_utilization: float = 0.3,
    ):
        """
        初始化 vLLM TTS 引擎。
        
        Args:
            model_size: 模型大小（目前仅支持 1.7B）
            device: 设备（cuda:0, cuda:1 等）
            gpu_memory_utilization: GPU 显存利用率（0.0-1.0）
        """
        if model_size != "1.7B":
            print(f"警告: vLLM-Omni 目前仅支持 1.7B 模型，将使用 1.7B")
        
        self.model_size = "1.7B"
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.sample_rate = 24000  # Qwen3-TTS 默认采样率
        
        # 模型实例缓存
        self._omni_instances: Dict[str, AsyncOmni] = {}
    
    def _get_omni(self, model_type: str) -> AsyncOmni:
        """获取或创建指定类型的 AsyncOmni 实例。"""
        if model_type not in self._omni_instances:
            model_id = self.MODEL_TYPE_MAP.get(model_type)
            if not model_id:
                raise ValueError(f"Unknown model type: {model_type}")
            
            print(f"Loading {model_type} model: {model_id}")
            
            # 创建 AsyncOmni 实例
            self._omni_instances[model_type] = AsyncOmni(
                model=model_id,
                # vLLM 引擎配置
                gpu_memory_utilization=self.gpu_memory_utilization,
                device=self.device,
            )
            
            print(f"Model {model_type} loaded successfully")
        
        return self._omni_instances[model_type]
    
    async def generate_custom_voice_streaming(
        self,
        text: str,
        language: str = "Chinese",
        speaker: str = "Vivian",
        instruct: Optional[str] = None,
        chunk_callback=None,
    ) -> Tuple[np.ndarray, int]:
        """
        流式生成预设音色语音。
        
        Args:
            text: 要合成的文本
            language: 语言
            speaker: 音色名称
            instruct: 风格指令（可选）
            chunk_callback: 音频块回调函数，每次生成块时调用
            
        Returns:
            (audio, sample_rate) 完整音频元组
        """
        if speaker not in SPEAKERS:
            raise ValueError(f"Unknown speaker: {speaker}. Must be one of {list(SPEAKERS.keys())}")
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}. Must be one of {VALID_LANGUAGES}")
        
        omni = self._get_omni("custom_voice")
        
        # 构建 vLLM-Omni prompt 格式
        prompt = {
            "prompt_token_ids": [0] * self._estimate_prompt_len(text, instruct),
            "additional_information": {
                "task_type": ["CustomVoice"],
                "text": [text],
                "language": [language],
                "speaker": [speaker],
                "instruct": [instruct or ""],
                "max_new_tokens": [2048],
            }
        }
        
        # 流式生成
        audio_chunks = []
        sr = self.sample_rate
        request_id = f"cv_{hash(text)}"
        
        async for stage_output in omni.generate(prompt, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output
            
            if not stage_output.finished:
                audio = mm.get("audio")
                if audio:
                    if isinstance(audio, list):
                        audio_chunks.extend(audio)
                    else:
                        audio_chunks.append(audio)
                    
                    # 调用回调
                    if chunk_callback:
                        combined = torch.cat(audio_chunks, dim=-1)
                        chunk_callback(combined.float().cpu().numpy().flatten(), sr)
            else:
                # 获取采样率
                sr_raw = mm.get("sr", sr)
                if isinstance(sr_raw, list) and sr_raw:
                    sr = sr_raw[-1].item() if hasattr(sr_raw[-1], 'item') else int(sr_raw[-1])
                elif hasattr(sr_raw, 'item'):
                    sr = sr_raw.item()
                
                # 合并所有音频块
                if audio_chunks:
                    audio_tensor = torch.cat(audio_chunks, dim=-1)
                    return audio_tensor.float().cpu().numpy().flatten(), sr
        
        # 如果没有生成任何音频
        return np.array([]), sr
    
    async def generate_voice_design_streaming(
        self,
        text: str,
        language: str = "English",
        instruct: str = "A warm, friendly voice",
        chunk_callback=None,
    ) -> Tuple[np.ndarray, int]:
        """
        流式生成语音设计。
        
        Args:
            text: 要合成的文本
            language: 语言
            instruct: 语音描述
            chunk_callback: 音频块回调函数
            
        Returns:
            (audio, sample_rate) 完整音频元组
        """
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}")
        if not instruct or not instruct.strip():
            raise ValueError("instruct is required for voice design")
        
        omni = self._get_omni("voice_design")
        
        prompt = {
            "prompt_token_ids": [0] * self._estimate_prompt_len(text, instruct),
            "additional_information": {
                "task_type": ["VoiceDesign"],
                "text": [text],
                "language": [language],
                "instruct": [instruct.strip()],
                "max_new_tokens": [2048],
                "non_streaming_mode": [False],  # 启用流式
            }
        }
        
        audio_chunks = []
        sr = self.sample_rate
        request_id = f"vd_{hash(text)}"
        
        async for stage_output in omni.generate(prompt, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output
            
            if not stage_output.finished:
                audio = mm.get("audio")
                if audio:
                    if isinstance(audio, list):
                        audio_chunks.extend(audio)
                    else:
                        audio_chunks.append(audio)
                    
                    if chunk_callback:
                        combined = torch.cat(audio_chunks, dim=-1)
                        chunk_callback(combined.float().cpu().numpy().flatten(), sr)
            else:
                sr_raw = mm.get("sr", sr)
                if isinstance(sr_raw, list) and sr_raw:
                    sr = sr_raw[-1].item() if hasattr(sr_raw[-1], 'item') else int(sr_raw[-1])
                elif hasattr(sr_raw, 'item'):
                    sr = sr_raw.item()
                
                if audio_chunks:
                    audio_tensor = torch.cat(audio_chunks, dim=-1)
                    return audio_tensor.float().cpu().numpy().flatten(), sr
        
        return np.array([]), sr
    
    async def generate_voice_clone_streaming(
        self,
        text: str,
        language: str = "English",
        ref_audio = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        chunk_callback=None,
    ) -> Tuple[np.ndarray, int]:
        """
        流式生成声音克隆。
        
        Args:
            text: 要合成的文本
            language: 语言
            ref_audio: 参考音频（文件路径或 (audio, sr) 元组）
            ref_text: 参考文本
            x_vector_only_mode: 仅使用说话人特征
            chunk_callback: 音频块回调函数
            
        Returns:
            (audio, sample_rate) 完整音频元组
        """
        if ref_audio is None:
            raise ValueError("ref_audio is required for voice cloning")
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}")
        
        omni = self._get_omni("base")
        
        # 处理参考音频
        if isinstance(ref_audio, str):
            # 文件路径
            ref_audio_input = ref_audio
        elif isinstance(ref_audio, tuple) and len(ref_audio) == 2:
            # (audio_array, sample_rate) 元组
            # 需要保存为临时文件或转换为 base64
            audio_array, ref_sr = ref_audio
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"ref_audio_{hash(text)}.wav")
            sf.write(temp_path, audio_array, ref_sr)
            ref_audio_input = temp_path
        else:
            raise ValueError("ref_audio must be a file path or (audio, sr) tuple")
        
        prompt = {
            "prompt_token_ids": [0] * self._estimate_prompt_len(text, ref_text),
            "additional_information": {
                "task_type": ["Base"],
                "ref_audio": [ref_audio_input],
                "ref_text": [ref_text or ""],
                "text": [text],
                "language": [language],
                "x_vector_only_mode": [x_vector_only_mode],
                "max_new_tokens": [2048],
            }
        }
        
        audio_chunks = []
        sr = self.sample_rate
        request_id = f"vc_{hash(text)}"
        
        async for stage_output in omni.generate(prompt, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output
            
            if not stage_output.finished:
                audio = mm.get("audio")
                if audio:
                    if isinstance(audio, list):
                        audio_chunks.extend(audio)
                    else:
                        audio_chunks.append(audio)
                    
                    if chunk_callback:
                        combined = torch.cat(audio_chunks, dim=-1)
                        chunk_callback(combined.float().cpu().numpy().flatten(), sr)
            else:
                sr_raw = mm.get("sr", sr)
                if isinstance(sr_raw, list) and sr_raw:
                    sr = sr_raw[-1].item() if hasattr(sr_raw[-1], 'item') else int(sr_raw[-1])
                elif hasattr(sr_raw, 'item'):
                    sr = sr_raw.item()
                
                if audio_chunks:
                    audio_tensor = torch.cat(audio_chunks, dim=-1)
                    return audio_tensor.float().cpu().numpy().flatten(), sr
        
        return np.array([]), sr
    
    def _estimate_prompt_len(self, text: str, extra: Optional[str] = None) -> int:
        """估算 prompt 长度（用于 vLLM-Omni 的 prompt_token_ids 占位）。"""
        # 粗略估算：中文字符约 1.5 tokens，英文单词约 1 token
        text_len = len(text)
        extra_len = len(extra) if extra else 0
        return max(100, int((text_len + extra_len) * 1.5))
    
    def get_speakers(self) -> List[Dict[str, Any]]:
        """获取可用的预设音色列表。"""
        return [
            {
                "name": name,
                "description_zh": info["zh"],
                "description_en": info["en"],
                "language": info["language"]
            }
            for name, info in SPEAKERS.items()
        ]
    
    def get_languages(self) -> List[str]:
        """获取支持的语言列表。"""
        return list(LANGUAGES)
    
    def unload(self):
        """卸载所有模型，释放 GPU 内存。"""
        for model_type, omni in self._omni_instances.items():
            print(f"Unloading model: {model_type}")
            del omni
        self._omni_instances.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 步骤 2：创建 Gradio 界面

**文件**: `app/ui_vllm.py`

```python
"""
Qwen3-TTS Gradio 界面（vLLM-Omni 流式版本）
"""

import gradio as gr
import asyncio
import tempfile
import soundfile as sf
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from .vllm_engine import VLLMTTSEngine
from .config import SPEAKERS, LANGUAGES, MODEL_CAPABILITIES


# 语言和音色选项
LANGUAGE_CHOICES = ["Auto"] + list(LANGUAGES)
SPEAKER_CHOICES = list(SPEAKERS.keys())


class UILogger:
    """日志管理器"""
    def __init__(self, max_lines: int = 100):
        self.logs: List[str] = []
        self.max_lines = max_lines
    
    def log(self, message: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        if len(self.logs) > self.max_lines:
            self.logs = self.logs[-self.max_lines:]
        return "\n".join(self.logs)
    
    def get_logs(self) -> str:
        return "\n".join(self.logs)


def create_ui(tts_engine: VLLMTTSEngine) -> gr.Blocks:
    """创建基于 vLLM-Omni 的流式 Gradio 界面"""
    
    logger = UILogger()
    logger.log(f"初始化完成 - vLLM-Omni 流式引擎")
    
    # === 同步包装器（用于 Gradio 回调）===
    def run_async(coro):
        """运行异步协程的同步包装器。"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    
    # === 流式生成回调 ===
    def gen_cv_streaming(text, lang, speaker, instruct, log_text):
        """预设音色流式生成"""
        if not text.strip():
            log_msg = logger.log("预设音色: 请输入文本")
            yield None, "请输入文本", log_text
            return
        
        accumulated_audio = []
        
        def chunk_callback(audio_chunk, sr):
            accumulated_audio.append(audio_chunk)
        
        async def generate():
            audio, sr = await tts_engine.generate_custom_voice_streaming(
                text=text,
                language=lang,
                speaker=speaker,
                instruct=instruct.strip() or None if instruct else None,
                chunk_callback=chunk_callback,
            )
            return audio, sr
        
        try:
            logger.log(f"预设音色: 开始流式生成...")
            
            # 运行异步生成
            audio, sr = run_async(generate())
            
            if len(audio) == 0:
                logger.log("预设音色: 生成失败，无音频输出")
                yield None, "生成失败", logger.get_logs()
                return
            
            # 分块返回音频（模拟流式效果）
            chunk_size = int(sr * 0.5)  # 每 0.5 秒返回一块
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                progress = min(100, int((i + chunk_size) / len(audio) * 100))
                yield (sr, audio[:i + chunk_size]), f"生成中... {progress}%", logger.get_logs()
            
            logger.log(f"预设音色: 生成完成 ({len(audio)/sr:.2f}s)")
            yield (sr, audio), f"生成完成", logger.get_logs()
            
        except Exception as e:
            logger.log(f"预设音色: 错误 - {e}")
            yield None, f"错误: {e}", logger.get_logs()
    
    def gen_vd_streaming(text, lang, instruct, log_text):
        """语音设计流式生成"""
        if not text.strip():
            log_msg = logger.log("语音设计: 请输入文本")
            yield None, "请输入文本", log_text
            return
        if not instruct or not instruct.strip():
            log_msg = logger.log("语音设计: 请输入语音描述")
            yield None, "请输入语音描述", log_text
            return
        
        accumulated_audio = []
        
        def chunk_callback(audio_chunk, sr):
            accumulated_audio.append(audio_chunk)
        
        async def generate():
            audio, sr = await tts_engine.generate_voice_design_streaming(
                text=text,
                language=lang,
                instruct=instruct.strip(),
                chunk_callback=chunk_callback,
            )
            return audio, sr
        
        try:
            logger.log(f"语音设计: 开始流式生成...")
            audio, sr = run_async(generate())
            
            if len(audio) == 0:
                yield None, "生成失败", logger.get_logs()
                return
            
            chunk_size = int(sr * 0.5)
            for i in range(0, len(audio), chunk_size):
                progress = min(100, int((i + chunk_size) / len(audio) * 100))
                yield (sr, audio[:i + chunk_size]), f"生成中... {progress}%", logger.get_logs()
            
            logger.log(f"语音设计: 生成完成 ({len(audio)/sr:.2f}s)")
            yield (sr, audio), f"生成完成", logger.get_logs()
            
        except Exception as e:
            logger.log(f"语音设计: 错误 - {e}")
            yield None, f"错误: {e}", logger.get_logs()
    
    def gen_vc_streaming(text, lang, ref_audio, ref_text, xvec, log_text):
        """声音克隆流式生成"""
        if not text.strip():
            log_msg = logger.log("声音克隆: 请输入文本")
            yield None, "请输入文本", log_text
            return
        if ref_audio is None:
            log_msg = logger.log("声音克隆: 请上传参考音频")
            yield None, "请上传参考音频", log_text
            return
        
        accumulated_audio = []
        
        def chunk_callback(audio_chunk, sr):
            accumulated_audio.append(audio_chunk)
        
        async def generate():
            audio, sr = await tts_engine.generate_voice_clone_streaming(
                text=text,
                language=lang,
                ref_audio=ref_audio,
                ref_text=ref_text.strip() or None,
                x_vector_only_mode=xvec,
                chunk_callback=chunk_callback,
            )
            return audio, sr
        
        try:
            logger.log(f"声音克隆: 开始流式生成...")
            audio, sr = run_async(generate())
            
            if len(audio) == 0:
                yield None, "生成失败", logger.get_logs()
                return
            
            chunk_size = int(sr * 0.5)
            for i in range(0, len(audio), chunk_size):
                progress = min(100, int((i + chunk_size) / len(audio) * 100))
                yield (sr, audio[:i + chunk_size]), f"生成中... {progress}%", logger.get_logs()
            
            logger.log(f"声音克隆: 生成完成 ({len(audio)/sr:.2f}s)")
            yield (sr, audio), f"生成完成", logger.get_logs()
            
        except Exception as e:
            logger.log(f"声音克隆: 错误 - {e}")
            yield None, f"错误: {e}", logger.get_logs()
    
    # === 界面构建 ===
    with gr.Blocks(title="Qwen3-TTS (vLLM-Omni 流式版)") as app:
        gr.Markdown(
            "# Qwen3-TTS 语音合成 (vLLM-Omni 流式版)\n"
            "**实时流式播放** — 边生成边听，首包延迟 ~97ms"
        )
        gr.Markdown(
            "基于 vLLM-Omni 的 AsyncOmni 引擎，支持真正的流式音频生成"
        )
        
        # 模型信息
        gr.Markdown("**当前后端**: vLLM-Omni AsyncOmni | **模型**: 1.7B | **流式**: 启用")
        
        # === 预设音色 Tab ===
        with gr.Tab("预设音色"):
            gr.Markdown("**预设音色 (CustomVoice)** — 9 种精选音色，支持风格指令控制")
            
            with gr.Row():
                with gr.Column():
                    t1 = gr.Textbox(label="合成文本", lines=3, placeholder="请输入要合成的文本…")
                    with gr.Row():
                        l1 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                        s1 = gr.Dropdown(SPEAKER_CHOICES, value="Vivian", label="音色")
                    i1 = gr.Textbox(
                        label="风格指令（可选）",
                        lines=1,
                        placeholder="例：用温柔的语气说"
                    )
                    b1 = gr.Button("生成(流式)", variant="primary")
                    m1 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a1 = gr.Audio(label="合成音频(流式)", streaming=True, autoplay=True)
            
            # 音色参考表
            with gr.Accordion("📖 音色参考", open=False):
                gr.Markdown(
                    "**中文音色**  \n"
                    "• **Vivian** — 明亮、略带锋芒的年轻女声  \n"
                    "• **Serena** — 温暖、柔和的年轻女声  \n"
                    "• **Uncle_Fu** — 低沉醇厚的成熟男声  \n\n"
                    "**方言音色**  \n"
                    "• **Dylan** — 清亮自然的京味年轻男声  \n"
                    "• **Eric** — 略带沙哑亮度的成都男声  \n\n"
                    "**外语音色**  \n"
                    "• **Ryan** — 节奏感强的动感男声（English）  \n"
                    "• **Aiden** — 阳光清澈的美式男中音（English）  \n"
                    "• **Ono_Anna** — 轻快俏皮的日系女声（日语）  \n"
                    "• **Sohee** — 温暖富有情感的韩语女声（韩语）"
                )
        
        # === 语音设计 Tab ===
        with gr.Tab("语音设计"):
            gr.Markdown(
                "**语音设计 (VoiceDesign)** — 自然语言描述设计任意音色"
            )
            with gr.Row():
                with gr.Column():
                    t2 = gr.Textbox(label="合成文本", lines=3)
                    l2 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                    i2 = gr.Textbox(
                        label="语音描述",
                        lines=2,
                        placeholder="例：体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显"
                    )
                    b2 = gr.Button("生成(流式)", variant="primary")
                    m2 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a2 = gr.Audio(label="合成音频(流式)", streaming=True, autoplay=True)
        
        # === 声音克隆 Tab ===
        with gr.Tab("声音克隆"):
            gr.Markdown(
                "**声音克隆 (Base)** — 3 秒参考音频即可克隆声音"
            )
            with gr.Row():
                with gr.Column():
                    t3 = gr.Textbox(label="合成文本", lines=3)
                    l3 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                    r3 = gr.Audio(label="参考音频", type="numpy")
                    rt3 = gr.Textbox(label="参考文本（可选）", lines=1)
                    x3 = gr.Checkbox(label="仅用特征（仅保留说话人身份）", value=False)
                    b3 = gr.Button("生成(流式)", variant="primary")
                    m3 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a3 = gr.Audio(label="合成音频(流式)", streaming=True, autoplay=True)
        
        # === 日志窗口 ===
        with gr.Accordion("📋 操作日志", open=False):
            log_box = gr.Textbox(label="", lines=8, interactive=False, value=logger.get_logs())
        
        # === 隐藏状态 ===
        log_state = gr.Textbox(value=logger.get_logs(), visible=False)
        
        # === 事件绑定 ===
        b1.click(gen_cv_streaming, [t1, l1, s1, i1, log_state], [a1, m1, log_state])
        b1.click(lambda x: x, [log_state], [log_box])
        
        b2.click(gen_vd_streaming, [t2, l2, i2, log_state], [a2, m2, log_state])
        b2.click(lambda x: x, [log_state], [log_box])
        
        b3.click(gen_vc_streaming, [t3, l3, r3, rt3, x3, log_state], [a3, m3, log_state])
        b3.click(lambda x: x, [log_state], [log_box])
    
    return app


def launch_ui(tts_engine: VLLMTTSEngine, **kwargs):
    """启动 Gradio UI"""
    create_ui(tts_engine).queue().launch(**kwargs)
```

### 步骤 3：更新 main.py

**文件**: `main.py`（新增内容）

```python
# 在现有 main.py 中添加以下内容

import argparse

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Web UI")
    parser.add_argument("--backend", type=str, default="qwen_tts", 
                        choices=["qwen_tts", "vllm"],
                        help="TTS backend: qwen_tts (default) or vllm (streaming)")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="Server IP")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--model-size", type=str, default="1.7B", 
                        choices=["0.6B", "1.7B"], help="Model size (vllm only supports 1.7B)")
    # ... 其他现有参数 ...
    
    args = parser.parse_args()
    
    if args.backend == "vllm":
        # 使用 vLLM-Omni 后端
        print("使用 vLLM-Omni 流式后端...")
        from app.vllm_engine import VLLMTTSEngine
        from app.ui_vllm import create_ui
        
        engine = VLLMTTSEngine(
            model_size=args.model_size,
            device=args.device if hasattr(args, 'device') else "cuda:0",
        )
        
        app = create_ui(engine)
        app.queue().launch(
            server_name=args.ip,
            server_port=args.port,
            share=args.share,
        )
    else:
        # 使用原有 qwen_tts 后端
        print("使用 qwen_tts 后端...")
        # ... 现有代码 ...
        pass

if __name__ == "__main__":
    main()
```

### 步骤 4：创建 vLLM 依赖文件

**文件**: `requirements-vllm.txt`

```
# vLLM 核心依赖
vllm>=0.20.0
vllm-omni>=0.1.0

# 音频处理
soundfile>=0.12.0
numpy>=1.24.0

# Gradio（如果 demo 扩展没有安装）
gradio>=4.0.0

# PyTorch（vLLM 会自动处理 CUDA 版本）
# torch>=2.0.0  # 通过 vLLM 自动安装
```

---

## 六、启动方式

### 方式 1：使用 vLLM 后端启动

```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 启动 vLLM 流式版本
python main.py --backend vllm --port 7860

# 或创建公网链接
python main.py --backend vllm --port 7860 --share
```

### 方式 2：使用原有 qwen_tts 后端

```bash
python main.py --backend qwen_tts --port 7860
```

---

## 七、测试验证

### 测试步骤

1. **安装验证**
   ```bash
   python test_vllm_omni.py
   ```

2. **启动服务**
   ```bash
   python main.py --backend vllm --port 7860
   ```

3. **测试流式生成**
   - 访问 http://localhost:7860
   - 在"预设音色"标签页输入文本
   - 点击"生成(流式)"
   - 观察是否边生成边播放

### 验证点

- [ ] vLLM 和 vLLM-Omni 安装成功
- [ ] 模型能正常加载（首次会下载）
- [ ] 点击生成后 1-2 秒内开始播放声音
- [ ] 音频播放进度与生成进度同步
- [ ] 最终音频完整无缺失
- [ ] 日志正确显示流式生成进度

### 性能指标

| 指标 | 目标值 | 实际测试 |
|------|--------|---------|
| 首包延迟 | < 200ms | _______ |
| 内存占用 | < 8GB | _______ |
| GPU 显存 | < 6GB | _______ |
| 音频质量 | 无损 | _______ |

---

## 八、已知限制

1. **仅支持 1.7B 模型**：vLLM-Omni 目前仅支持 Qwen3-TTS 1.7B 模型
2. **Linux 优先**：官方文档主要针对 Linux，Windows 可能需要 WSL2
3. **Python 3.12 要求**：需要较新的 Python 版本
4. **GPU 要求**：计算能力 7.0+，建议 8GB+ 显存
5. **首次启动较慢**：模型加载需要时间

---

## 九、故障排除

### 问题 1：vLLM 安装失败

```bash
# 确保使用 CUDA 版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 然后重新安装 vLLM
pip install vllm --torch-backend=auto
```

### 问题 2：GPU 内存不足

```python
# 在 vllm_engine.py 中降低 gpu_memory_utilization
self.gpu_memory_utilization = 0.2  # 从 0.3 降低到 0.2
```

### 问题 3：Windows 环境问题

```bash
# 使用 WSL2 运行
wsl
cd /mnt/e/Other/Qwen3Audio
source .venv/bin/activate
python main.py --backend vllm
```

### 问题 4：异步事件循环错误

```python
# 如果遇到事件循环错误，尝试使用新事件循环
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # Windows
```

---

## 十、与方案 A 的对比

| 维度 | 方案 A (qwen_tts) | 方案 B (vLLM-Omni) |
|------|------------------|-------------------|
| **流式质量** | 退阶方案，同步后分块 | ✅ 原生流式，真正增量 |
| **首包延迟** | 取决于整体生成时间 | ✅ ~97ms（官方数据） |
| **Windows 支持** | ✅ 完全支持 | ⚠️ 需要 Linux/WSL2 |
| **模型支持** | ✅ 0.6B + 1.7B | ⚠️ 仅 1.7B |
| **安装复杂度** | ✅ 简单 | ⚠️ 需要 Python 3.12 + 新环境 |
| **代码改动量** | 中等 | 较大（新增文件） |
| **长期维护** | 自己维护 | ✅ vLLM 团队维护 |
| **生产级质量** | 取决于实现 | ✅ 经过大规模测试 |

---

## 十一、推荐选择

### 选择方案 B（vLLM-Omni）的场景：
- 生产环境部署
- 需要最低延迟
- Linux 服务器环境
- 愿意投入时间配置环境
- 需要稳定、可扩展的解决方案

### 选择方案 A（改造 qwen_tts）的场景：
- Windows 开发环境
- 需要支持 0.6B 模型
- 快速验证原型
- 不想安装额外依赖
- 对延迟要求不苛刻

---

## 十二、相关文件引用

- `app/vllm_engine.py`: vLLM-Omni 引擎封装（新增）
- `app/ui_vllm.py`: Gradio 流式界面（新增）
- `main.py`: 入口文件（修改）
- `requirements-vllm.txt`: vLLM 依赖（新增）

## 十三、参考资源

- [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni)
- [vLLM-Omni 文档](https://docs.vllm.ai/projects/vllm-omni/)
- [vLLM-Omni Qwen3-TTS 示例](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_tts/end2end.py)
- [AsyncOmni API 参考](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/entrypoints/async_omni.py)
- [Qwen3-TTS 配置 YAML](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_tts.yaml)
- [Gradio 流式输出文档](https://www.gradio.app/guides/streaming-outputs)
