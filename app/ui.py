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
        # 页面头部
        gr.Markdown(
            "# Qwen3-TTS 语音合成\n"
            "通义千问语音合成模型 — 支持声音克隆、语音设计、超高质量拟人语音生成及自然语言语音控制"
        )
        gr.Markdown(
            "涵盖中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语等 10 种语言及多种方言音色 | "
            "基于离散多码本 LM 架构，端到端合成延迟低至 97ms"
        )

        with gr.Tab("预设音色"):
            gr.Markdown(
                "**预设音色 (Custom Voice)** — 提供 9 种精选音色，覆盖多种性别、年龄、语言与方言组合；"
                "支持通过指令控制情感、风格与韵律，实现「所想即所闻」"
            )
            gr.Markdown(
                "| 音色 | 描述 | 母语 |\n"
                "|------|------|------|\n"
                "| **Vivian** | 明亮、略带锋芒的年轻女声 | 中文 |\n"
                "| **Serena** | 温暖、柔和的年轻女声 | 中文 |\n"
                "| **Uncle_Fu** | 低沉醇厚的成熟男声 | 中文 |\n"
                "| **Dylan** | 清亮自然的京味年轻男声 | 京味方言 |\n"
                "| **Eric** | 略带沙哑亮度的成都男声 | 川味方言 |\n"
                "| **Ryan** | 节奏感强的动感男声 | English |\n"
                "| **Aiden** | 阳光清澈的美式男中音 | English |\n"
                "| **Ono_Anna** | 轻快俏皮的日系女声 | 日语 |\n"
                "| **Sohee** | 温暖富有情感的韩语女声 | 韩语 |"
            )
            with gr.Row():
                with gr.Column():
                    t1 = gr.Textbox(label="合成文本", lines=3, placeholder="请输入要合成的文本…")
                    with gr.Row():
                        l1 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言", info="支持 10 种语言及自动检测")
                        s1 = gr.Dropdown(SPEAKER_CHOICES, value="Vivian", label="音色", info="详见上方表格")
                    i1 = gr.Textbox(label="风格指令（可选）", lines=1, placeholder="例：用温柔的语气说 / Speak slowly and gently")
                    b1 = gr.Button("生成", variant="primary")
                    m1 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a1 = gr.Audio(label="合成音频", type="numpy")
            b1.click(gen_cv, [t1, l1, s1, i1], [a1, m1])

        with gr.Tab("语音设计"):
            gr.Markdown(
                "**语音设计** — 通过自然语言描述来设计任意音色，"
                "灵活控制音色、情感、韵律等多维声学属性"
            )
            with gr.Row():
                with gr.Column():
                    t2 = gr.Textbox(label="合成文本", lines=3, placeholder="请输入要合成的文本…")
                    l2 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言", info="支持 10 种语言及自动检测")
                    i2 = gr.Textbox(
                        label="语音描述", lines=2,
                        placeholder="例：体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显 / A deep, authoritative male voice with a British accent"
                    )
                    b2 = gr.Button("生成", variant="primary")
                    m2 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a2 = gr.Audio(label="合成音频", type="numpy")
            b2.click(gen_vd, [t2, l2, i2], [a2, m2])

        with gr.Tab("声音克隆"):
            gr.Markdown(
                "**声音克隆** — 仅需 3 秒参考音频即可快速克隆声音；"
                "提供参考文本可提升克隆质量"
            )
            with gr.Row():
                with gr.Column():
                    t3 = gr.Textbox(label="合成文本", lines=3, placeholder="请输入要合成的文本…")
                    l3 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言", info="支持 10 种语言及自动检测")
                    r3 = gr.Audio(label="参考音频", type="numpy")
                    rt3 = gr.Textbox(label="参考文本（可选，提供可提升克隆质量）", lines=1, placeholder="参考音频的转录文本…")
                    x3 = gr.Checkbox(label="仅用特征（仅保留说话人身份，不做韵律匹配）", value=False)
                    b3 = gr.Button("生成", variant="primary")
                    m3 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a3 = gr.Audio(label="合成音频", type="numpy")
            b3.click(gen_vc, [t3, l3, r3, rt3, x3], [a3, m3])

    return app


def launch_ui(tts_engine: TTSEngine, **kwargs):
    create_ui(tts_engine).launch(**kwargs)
