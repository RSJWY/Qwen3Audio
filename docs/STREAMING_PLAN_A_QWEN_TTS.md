# 流式音频输出实现方案 A：改造 qwen_tts

## 一、方案概述

### 目标
在现有 qwen_tts 包基础上，实现真正的流式音频输出，使用户在点击生成后约 1-2 秒开始听到声音，边生成边播放，无需等待完整音频生成完毕。

### 核心思路
1. 利用 Qwen3-TTS 底层模型的增量生成能力（KV cache + 自回归生成）
2. 使用 tokenizer 的 `chunked_decode()` 方法做分块音频解码
3. 将同步 API 改造为 Python generator，逐块 yield 音频数据
4. Gradio 前端使用 `gr.Audio(streaming=True, autoplay=True)` 实现实时播放

### 技术可行性依据

| 层级 | 支持情况 | 证据来源 |
|------|---------|---------|
| 模型架构 | ✅ 支持 | Dual-Track hybrid streaming generation，首包延迟 97ms |
| 底层 generate | ✅ 支持 | 有 `past_key_values` 缓存机制，可增量生成 codec codes |
| tokenizer 解码 | ✅ 支持 | 有 `chunked_decode()` 方法，可按 chunk 解码音频 |
| qwen_tts API | ❌ 需改造 | `generate_*()` 是同步阻塞，需包装成 generator |

---

## 二、文件改动清单

### 需要修改的文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `app/tts_engine.py` | 新增方法 | 添加 `generate_*_streaming()` 流式生成方法 |
| `app/ui.py` | 修改 | 添加流式 UI 组件和回调函数 |
| `requirements.txt` | 可能修改 | 无需新增依赖 |

### 需要新增的文件

无

---

## 三、详细实现步骤

### 步骤 1：分析 qwen_tts 底层代码结构

**位置**: `qwen_tts/inference/qwen3_tts_model.py`

当前 `generate_custom_voice()` 的实现逻辑：
```python
# 当前实现（同步）
result = model.generate_custom_voice(
    text=text,
    language=language,
    speaker=speaker,
    instruct=instruct
)
# 返回 (wavs, sample_rate) - 完整音频
```

底层调用链：
1. `Qwen3TTSModel.generate_custom_voice()` 
2. → `self.model.generate(...)` 生成 codec codes
3. → `self.model.speech_tokenizer.decode()` 解码为音频

**关键发现**：
- `model.generate()` 内部支持增量生成（有 `past_key_values` 参数）
- `speech_tokenizer` 有 `chunked_decode(codes, chunk_size=300, left_context_size=25)` 方法

### 步骤 2：在 tts_engine.py 中添加流式生成方法

**文件**: `app/tts_engine.py`

**新增代码**:

```python
import numpy as np
from typing import Generator, Tuple, Optional
import torch

class TTSEngine:
    # ... 现有代码 ...
    
    def generate_custom_voice_streaming(
        self,
        text: str,
        language: str = "Chinese",
        speaker: str = "Vivian",
        instruct: Optional[str] = None,
        chunk_size: int = 300,
        yield_every_n_chunks: int = 1
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        流式生成预设音色语音。
        
        Args:
            text: 要合成的文本
            language: 语言
            speaker: 音色名称
            instruct: 风格指令（可选，仅 1.7B 支持）
            chunk_size: 每个音频 chunk 的帧数（默认 300）
            yield_every_n_chunks: 每生成多少 chunk 返回一次（默认 1）
            
        Yields:
            (audio_chunk, sample_rate) 元组
        """
        if speaker not in SPEAKERS:
            raise ValueError(f"Unknown speaker: {speaker}")
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}")
        
        # 0.6B 不支持指令控制
        caps = MODEL_CAPABILITIES.get(self.model_size, {})
        if not caps.get('instruct_control', False):
            instruct = None
        
        model = self.model_manager.load_model("custom_voice")
        
        # 调用底层模型的流式生成方法
        # 注意：这里需要访问 model.model (底层的 Qwen3TTSForConditionalGeneration)
        # 和 model.speech_tokenizer (12Hz tokenizer)
        
        # 方案 A1: 如果 qwen_tts 暴露了流式接口
        try:
            # 尝试使用非流式模式获取中间结果
            # 这是一个概念实现，实际需要根据 qwen_tts API 调整
            result = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                non_streaming_mode=False  # 尝试启用流式模式
            )
            
            # 如果返回的是完整音频，我们需要另一种方法
            if isinstance(result, tuple) and len(result) == 2:
                wavs, sr = result
                # 将完整音频分块返回（退阶方案）
                audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
                chunk_samples = int(chunk_size * sr / 12)  # 12Hz -> 24000Hz
                
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i:i + chunk_samples]
                    if len(chunk) > 0:
                        yield chunk, sr
                        
        except Exception as e:
            # 退阶到同步生成后分块
            audio, sr = self.generate_custom_voice(text, language, speaker, instruct)
            chunk_samples = int(chunk_size * sr / 12)
            
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                if len(chunk) > 0:
                    yield chunk, sr
    
    def generate_voice_design_streaming(
        self,
        text: str,
        language: str = "English",
        instruct: str = "A warm, friendly voice",
        chunk_size: int = 300
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        流式生成语音设计。
        
        Args:
            text: 要合成的文本
            language: 语言
            instruct: 语音描述
            chunk_size: 每个音频 chunk 的帧数
            
        Yields:
            (audio_chunk, sample_rate) 元组
        """
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}")
        
        caps = MODEL_CAPABILITIES.get(self.model_size, {})
        if not caps.get('voice_design', False):
            raise NotImplementedError(
                f"Voice Design is not supported for {self.model_size} model. "
                "Please use 1.7B model size."
            )
        
        # 目前退阶到同步生成后分块
        audio, sr = self.generate_voice_design(text, language, instruct)
        chunk_samples = int(chunk_size * sr / 12)
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                yield chunk, sr
    
    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str = "English",
        ref_audio = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        chunk_size: int = 300
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        流式生成声音克隆。
        
        Args:
            text: 要合成的文本
            language: 语言
            ref_audio: 参考音频
            ref_text: 参考文本
            x_vector_only_mode: 仅使用说话人特征
            chunk_size: 每个音频 chunk 的帧数
            
        Yields:
            (audio_chunk, sample_rate) 元组
        """
        if ref_audio is None:
            raise ValueError("ref_audio is required")
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}")
        
        # 目前退阶到同步生成后分块
        audio, sr = self.generate_voice_clone(
            text, language, ref_audio, ref_text, x_vector_only_mode
        )
        chunk_samples = int(chunk_size * sr / 12)
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                yield chunk, sr
```

### 步骤 3：修改 ui.py 添加流式 UI

**文件**: `app/ui.py`

**修改内容**:

```python
import gradio as gr
import tempfile
import soundfile as sf
import numpy as np
from typing import Generator

# ... 现有导入 ...

def create_ui(tts_engine: TTSEngine) -> gr.Blocks:
    """创建 Gradio 界面"""
    
    logger = UILogger()
    logger.log(f"初始化完成，当前模型大小: {tts_engine.model_size}")
    
    # ... 现有辅助函数 ...
    
    # === 流式生成回调函数 ===
    def gen_cv_streaming(text, lang, speaker, instruct, log_text, progress=gr.Progress()):
        """流式生成预设音色"""
        if not text.strip():
            log_msg = logger.log("预设音色(流式): 请输入文本")
            yield None, "请输入文本", log_text
            return
        
        caps = MODEL_CAPABILITIES.get(tts_engine.model_size, {})
        if not caps.get('instruct_control', False):
            instruct = None
        
        try:
            model_name = get_current_model_name('custom_voice')
            log_msg = logger.log(f"预设音色(流式): 开始生成 - 模型 {model_name}")
            
            # 创建临时文件用于累积音频
            accumulated_audio = []
            sr = None
            
            for audio_chunk, sample_rate in tts_engine.generate_custom_voice_streaming(
                text=text, language=lang, speaker=speaker,
                instruct=instruct.strip() or None if instruct else None
            ):
                accumulated_audio.append(audio_chunk)
                sr = sample_rate
                
                # 合并当前所有 chunks
                combined = np.concatenate(accumulated_audio)
                
                # 写入临时文件供 Gradio 播放
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, combined, sr)
                    log_msg = logger.log(f"预设音色(流式): 已生成 {len(combined)/sr:.2f}s 音频")
                    yield (sr, combined), f"生成中... ({len(combined)/sr:.2f}s)", log_msg
            
            log_msg = logger.log(f"预设音色(流式): 生成完成 ({model_name})")
            final_audio = np.concatenate(accumulated_audio)
            yield (sr, final_audio), f"生成完成 ({model_name})", log_msg
            
        except Exception as e:
            log_msg = logger.log(f"预设音色(流式): 错误 - {e}")
            yield None, f"错误: {e}", log_msg
    
    def gen_vd_streaming(text, lang, instruct, log_text, progress=gr.Progress()):
        """流式生成语音设计"""
        caps = MODEL_CAPABILITIES.get(tts_engine.model_size, {})
        if not caps.get('voice_design', False):
            log_msg = logger.log("语音设计(流式): 当前模型不支持")
            yield None, "当前模型大小不支持语音设计", log_text
            return
        
        if not text.strip() or not instruct.strip():
            log_msg = logger.log("语音设计(流式): 请输入文本和描述")
            yield None, "请输入文本和语音描述", log_text
            return
        
        try:
            model_name = get_current_model_name('voice_design')
            log_msg = logger.log(f"语音设计(流式): 开始生成 - 模型 {model_name}")
            
            accumulated_audio = []
            sr = None
            
            for audio_chunk, sample_rate in tts_engine.generate_voice_design_streaming(
                text=text, language=lang, instruct=instruct.strip()
            ):
                accumulated_audio.append(audio_chunk)
                sr = sample_rate
                combined = np.concatenate(accumulated_audio)
                
                log_msg = logger.log(f"语音设计(流式): 已生成 {len(combined)/sr:.2f}s")
                yield (sr, combined), f"生成中... ({len(combined)/sr:.2f}s)", log_msg
            
            log_msg = logger.log(f"语音设计(流式): 生成完成 ({model_name})")
            final_audio = np.concatenate(accumulated_audio)
            yield (sr, final_audio), f"生成完成 ({model_name})", log_msg
            
        except Exception as e:
            log_msg = logger.log(f"语音设计(流式): 错误 - {e}")
            yield None, f"错误: {e}", log_msg
    
    def gen_vc_streaming(text, lang, ref_audio, ref_text, xvec, log_text, progress=gr.Progress()):
        """流式生成声音克隆"""
        if not text.strip():
            log_msg = logger.log("声音克隆(流式): 请输入文本")
            yield None, "请输入文本", log_text
            return
        if ref_audio is None:
            log_msg = logger.log("声音克隆(流式): 请上传参考音频")
            yield None, "请上传参考音频", log_text
            return
        
        try:
            model_name = get_current_model_name('base')
            log_msg = logger.log(f"声音克隆(流式): 开始生成 - 模型 {model_name}")
            
            accumulated_audio = []
            sr = None
            
            for audio_chunk, sample_rate in tts_engine.generate_voice_clone_streaming(
                text=text, language=lang, ref_audio=ref_audio,
                ref_text=ref_text.strip() or None,
                x_vector_only_mode=xvec
            ):
                accumulated_audio.append(audio_chunk)
                sr = sample_rate
                combined = np.concatenate(accumulated_audio)
                
                log_msg = logger.log(f"声音克隆(流式): 已生成 {len(combined)/sr:.2f}s")
                yield (sr, combined), f"生成中... ({len(combined)/sr:.2f}s)", log_msg
            
            log_msg = logger.log(f"声音克隆(流式): 生成完成 ({model_name})")
            final_audio = np.concatenate(accumulated_audio)
            yield (sr, final_audio), f"生成完成 ({model_name})", log_msg
            
        except Exception as e:
            log_msg = logger.log(f"声音克隆(流式): 错误 - {e}")
            yield None, f"错误: {e}", log_msg
    
    # === 界面构建 ===
    initial_caps = MODEL_CAPABILITIES.get(tts_engine.model_size, {})
    
    with gr.Blocks(title="Qwen3-TTS") as app:
        # ... 现有头部和模型切换代码 ...
        
        # === 预设音色 Tab（流式版本）===
        with gr.Tab("预设音色(流式)") as tab_cv_stream:
            gr.Markdown("**预设音色 (流式)** — 实时播放，边生成边听")
            
            with gr.Row():
                with gr.Column():
                    t1s = gr.Textbox(label="合成文本", lines=3, placeholder="请输入要合成的文本…")
                    with gr.Row():
                        l1s = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                        s1s = gr.Dropdown(SPEAKER_CHOICES, value="Vivian", label="音色")
                    i1s = gr.Textbox(
                        label="风格指令（可选）",
                        lines=1,
                        placeholder="例：用温柔的语气说",
                        visible=initial_caps.get('instruct_control', False)
                    )
                    b1s = gr.Button("生成(流式)", variant="primary")
                    m1s = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    # 关键：streaming=True, autoplay=True
                    a1s = gr.Audio(label="合成音频(流式)", streaming=True, autoplay=True)
        
        # === 语音设计 Tab（流式版本）===
        with gr.Tab("语音设计(流式)", visible=initial_caps.get('voice_design', False)) as tab_vd_stream:
            gr.Markdown("**语音设计 (流式)** — 实时播放")
            
            with gr.Row():
                with gr.Column():
                    t2s = gr.Textbox(label="合成文本", lines=3)
                    l2s = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                    i2s = gr.Textbox(label="语音描述", lines=2)
                    b2s = gr.Button("生成(流式)", variant="primary")
                    m2s = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a2s = gr.Audio(label="合成音频(流式)", streaming=True, autoplay=True)
        
        # === 声音克隆 Tab（流式版本）===
        with gr.Tab("声音克隆(流式)"):
            gr.Markdown("**声音克隆 (流式)** — 实时播放")
            
            with gr.Row():
                with gr.Column():
                    t3s = gr.Textbox(label="合成文本", lines=3)
                    l3s = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                    r3s = gr.Audio(label="参考音频", type="numpy")
                    rt3s = gr.Textbox(label="参考文本（可选）", lines=1)
                    x3s = gr.Checkbox(label="仅用特征", value=False)
                    b3s = gr.Button("生成(流式)", variant="primary")
                    m3s = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a3s = gr.Audio(label="合成音频(流式)", streaming=True, autoplay=True)
        
        # === 事件绑定（流式版本）===
        # 注意：Gradio 流式输出需要使用 generator 函数
        
        b1s.click(
            gen_cv_streaming,
            [t1s, l1s, s1s, i1s, log_state],
            [a1s, m1s, log_state]
        )
        b1s.click(lambda x: x, [log_state], [log_box])
        
        b2s.click(
            gen_vd_streaming,
            [t2s, l2s, i2s, log_state],
            [a2s, m2s, log_state]
        )
        b2s.click(lambda x: x, [log_state], [log_box])
        
        b3s.click(
            gen_vc_streaming,
            [t3s, l3s, r3s, rt3s, x3s, log_state],
            [a3s, m3s, log_state]
        )
        b3s.click(lambda x: x, [log_state], [log_box])
        
        # ... 保留原有的非流式 Tab 和其他代码 ...
    
    return app
```

### 步骤 4：进阶优化（可选）

如果需要真正的底层流式生成（而非同步生成后分块），需要深入改造：

**方案 A2：直接访问底层模型**

```python
# 在 tts_engine.py 中添加

def _stream_generate_codes(self, model, text, language, speaker, instruct=None):
    """
    直接访问底层模型的增量生成能力。
    这是一个概念实现，需要根据 qwen_tts 实际 API 调整。
    """
    import torch
    
    # 1. 准备输入
    # 这里需要访问 model.model (Qwen3TTSForConditionalGeneration)
    # 和 model.speech_tokenizer
    
    # 2. 使用 model.generate() 的增量模式
    # 关键参数：
    # - use_cache=True
    # - past_key_values 用于缓存
    
    # 3. 使用 tokenizer.chunked_decode() 解码
    # chunked_decode(codes, chunk_size=300, left_context_size=25)
    
    # 伪代码：
    # with torch.no_grad():
    #     past_key_values = None
    #     generated_codes = []
    #     
    #     for step in range(max_steps):
    #         # 增量生成 codec codes
    #         outputs = model.model.generate(
    #             input_ids,
    #             past_key_values=past_key_values,
    #             use_cache=True,
    #             ...
    #         )
    #         past_key_values = outputs.past_key_values
    #         new_codes = outputs.codes  # 假设有这个字段
    #         generated_codes.append(new_codes)
    #         
    #         # 每隔一定步数解码并返回
    #         if len(generated_codes) % chunk_size == 0:
    #             audio_chunk = model.speech_tokenizer.chunked_decode(
    #                 torch.cat(generated_codes),
    #                 chunk_size=chunk_size
    #             )
    #             yield audio_chunk, self.sample_rate
    #     
    #     # 返回最后一批
    #     final_audio = model.speech_tokenizer.decode(generated_codes)
    #     yield final_audio, self.sample_rate
    pass
```

**注意事项**：
- 这需要深入了解 `qwen_tts` 内部实现
- 可能需要修改或继承 `Qwen3TTSModel` 类
- 建议先实现退阶方案（同步生成后分块），验证 UI 层流式播放可行后再深入底层

---

## 四、测试验证

### 测试步骤

1. **启动应用**
   ```bash
   python main.py
   ```

2. **测试流式生成**
   - 打开浏览器访问 http://localhost:7860
   - 切换到"预设音色(流式)"标签页
   - 输入文本，点击"生成(流式)"
   - 观察是否边生成边播放

3. **验证点**
   - [ ] 点击生成后 1-2 秒内开始播放声音
   - [ ] 音频播放进度与生成进度同步
   - [ ] 最终音频完整无缺失
   - [ ] 日志正确显示生成进度

### 性能指标

| 指标 | 目标值 | 测试方法 |
|------|--------|---------|
| 首包延迟 | < 2秒 | 从点击到首次听到声音 |
| 音频质量 | 无损 | 对比非流式生成结果 |
| CPU/GPU 占用 | 合理 | 监控资源使用 |

---

## 五、已知限制

1. **当前实现为退阶方案**：同步生成后分块返回，并非真正的底层流式生成
2. **首包延迟**：取决于整体生成时间，可能无法达到 97ms 的理论最小值
3. **内存占用**：需要缓存完整音频用于分块，内存占用与当前方案相同
4. **模型支持**：需要验证 0.6B 和 1.7B 模型是否都能正常工作

---

## 六、后续优化方向

1. **真正的底层流式**：研究 `qwen_tts` 源码，实现 codec codes 的增量生成
2. **更小的 chunk size**：调整 chunk 参数以降低首包延迟
3. **音频格式优化**：考虑使用 MP3 或 Opus 格式减少数据传输量
4. **并发处理**：支持多个请求并发流式生成

---

## 七、相关文件引用

- `app/tts_engine.py`: TTS 引擎主文件
- `app/ui.py`: Gradio 界面定义
- `app/model_manager.py`: 模型加载管理
- `app/config.py`: 配置常量

## 八、参考资源

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Gradio Streaming Outputs](https://www.gradio.app/guides/streaming-outputs)
- [Qwen3-TTS chunked_decode 源码](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py#L881-L889)
