"""
Qwen3-TTS Gradio 界面 - 极简版
"""

import gradio as gr
from typing import Tuple, Optional

from .config import SPEAKERS, LANGUAGES
from .tts_engine import TTSEngine


# 语言选项
LANGUAGE_CHOICES = ["Auto"] + LANGUAGES

# 音色选项
SPEAKER_CHOICES = list(SPEAKERS.keys())


def create_ui(tts_engine: TTSEngine) -> gr.Blocks:
    """创建极简 Gradio 界面"""

    # === 回调函数 ===
    def gen_cv(text, lang, speaker, instruct, progress=gr.Progress()):
        if not text.strip():
            return None, "请输入文本"
        try:
            progress(0.3, desc="加载模型中...")
            audio, sr = tts_engine.generate_custom_voice(
                text=text, language=lang, speaker=speaker,
                instruct=instruct.strip() or None
            )
            return (sr, audio), "生成成功"
        except Exception as e:
            return None, f"错误: {e}"

    def gen_vd(text, lang, instruct, progress=gr.Progress()):
        if not text.strip():
            return None, "请输入文本"
        if not instruct.strip():
            return None, "请输入语音描述"
        try:
            progress(0.3, desc="加载模型中...")
            audio, sr = tts_engine.generate_voice_design(
                text=text, language=lang, instruct=instruct.strip()
            )
            return (sr, audio), "生成成功"
        except Exception as e:
            return None, f"错误: {e}"

    def gen_vc(text, lang, ref_audio, ref_text, xvec, progress=gr.Progress()):
        if not text.strip():
            return None, "请输入文本"
        if ref_audio is None:
            return None, "请上传参考音频"
        try:
            progress(0.3, desc="加载模型中...")
            audio, sr = tts_engine.generate_voice_clone(
                text=text, language=lang, ref_audio=ref_audio,
                ref_text=ref_text.strip() or None,
                x_vector_only_mode=xvec
            )
            return (sr, audio), "生成成功"
        except Exception as e:
            return None, f"错误: {e}"

    # === 界面 ===
    with gr.Blocks(title="Qwen3-TTS") as app:
        gr.Markdown("# Qwen3-TTS 语音合成")

        with gr.Tab("预设音色"):
            with gr.Row():
                with gr.Column():
                    t1 = gr.Textbox(label="文本", lines=3)
                    with gr.Row():
                        l1 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                        s1 = gr.Dropdown(SPEAKER_CHOICES, value="Vivian", label="音色")
                    i1 = gr.Textbox(label="风格指令(可选)", lines=1)
                    b1 = gr.Button("生成", variant="primary")
                    m1 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a1 = gr.Audio(label="音频", type="numpy")
            b1.click(gen_cv, [t1, l1, s1, i1], [a1, m1])

        with gr.Tab("语音设计"):
            with gr.Row():
                with gr.Column():
                    t2 = gr.Textbox(label="文本", lines=3)
                    l2 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                    i2 = gr.Textbox(label="语音描述", lines=2)
                    b2 = gr.Button("生成", variant="primary")
                    m2 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a2 = gr.Audio(label="音频", type="numpy")
            b2.click(gen_vd, [t2, l2, i2], [a2, m2])

        with gr.Tab("声音克隆"):
            with gr.Row():
                with gr.Column():
                    t3 = gr.Textbox(label="文本", lines=3)
                    l3 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言")
                    r3 = gr.Audio(label="参考音频", type="numpy")
                    rt3 = gr.Textbox(label="参考文本(可选)", lines=1)
                    x3 = gr.Checkbox(label="仅用特征", value=False)
                    b3 = gr.Button("生成", variant="primary")
                    m3 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a3 = gr.Audio(label="音频", type="numpy")
            b3.click(gen_vc, [t3, l3, r3, rt3, x3], [a3, m3])

    return app


def launch_ui(tts_engine: TTSEngine, **kwargs):
    create_ui(tts_engine).launch(**kwargs)
