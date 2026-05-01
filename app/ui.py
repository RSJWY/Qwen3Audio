"""
Qwen3-TTS Gradio 界面

现代化三标签页界面，支持预设音色、语音设计、声音克隆。
"""

import os
import gradio as gr
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

from .config import SPEAKERS, LANGUAGES
from .tts_engine import TTSEngine


# 加载自定义 CSS
CSS_PATH = Path(__file__).parent / "style.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def get_speaker_choices() -> List[str]:
    """获取格式化的音色选项列表。"""
    return [
        f"{name} - {info['zh']}"
        for name, info in SPEAKERS.items()
    ]


def parse_speaker_choice(choice: str) -> str:
    """从格式化选项中提取音色名称。"""
    return choice.split(" - ")[0] if " - " in choice else choice


def create_ui(tts_engine: TTSEngine) -> gr.Blocks:
    """
    创建 Qwen3-TTS 的 Gradio 界面。

    Args:
        tts_engine: TTSEngine 实例

    Returns:
        Gradio Blocks 应用
    """

    # 音色卡片展示
    speaker_info_html = "<div class='speaker-grid'>"
    for name, info in SPEAKERS.items():
        speaker_info_html += f"""
        <div class='speaker-card'>
            <div class='name'>{name}</div>
            <div class='desc'>{info['zh']}</div>
            <div class='desc'>{info['en']}</div>
            <div class='lang'>{info['language']}</div>
        </div>
        """
    speaker_info_html += "</div>"

    # 语音设计示例
    voice_design_examples = [
        ["温柔的年轻女声，语速较慢，带有安抚感", "Chinese"],
        ["低沉磁性的男声，带有新闻播报的正式感", "Chinese"],
        ["活泼俏皮的女声，语速较快，充满活力", "Chinese"],
        ["沉稳厚重的老年男声，缓慢而庄重", "Chinese"],
    ]

    # 语言选项（含自动检测）
    language_choices = ["自动检测"] + LANGUAGES

    # 语言映射：界面显示 -> 引擎内部值
    LANG_DISPLAY_TO_INTERNAL = {"自动检测": "Auto"}
    for lang in LANGUAGES:
        LANG_DISPLAY_TO_INTERNAL[lang] = lang

    LANG_INTERNAL_TO_DISPLAY = {v: k for k, v in LANG_DISPLAY_TO_INTERNAL.items()}

    def get_status_html() -> str:
        """获取当前模型状态 HTML。"""
        try:
            status = tts_engine.get_status()
            current = status.get("current_model", "None")
            if current and current != "None":
                model_names = {
                    "custom_voice": "预设音色",
                    "voice_design": "语音设计",
                    "base": "声音克隆",
                }
                display_name = model_names.get(current, current)
                return f"<span class='status-badge loaded'>已加载模型：{display_name}</span>"
            return "<span class='status-badge unloaded'>尚未加载模型</span>"
        except Exception:
            return "<span class='status-badge unloaded'>状态不可用</span>"

    # === 标签页1：预设音色 ===
    def generate_custom_voice(
        text: str,
        language: str,
        speaker_choice: str,
        instruct: str,
        progress=gr.Progress()
    ) -> Tuple[Optional[Tuple[int, Any]], str]:
        """使用预设音色生成语音。"""
        if not text.strip():
            return None, "请输入要合成的文本。"

        speaker = parse_speaker_choice(speaker_choice)
        lang = LANG_DISPLAY_TO_INTERNAL.get(language, language)

        try:
            progress(0.3, desc="正在加载模型...")
            audio, sr = tts_engine.generate_custom_voice(
                text=text,
                language=lang,
                speaker=speaker,
                instruct=instruct.strip() if instruct.strip() else None
            )
            progress(1.0, desc="生成完成！")
            return (sr, audio), f"已使用 {speaker} 音色成功生成语音！"
        except Exception as e:
            return None, f"生成失败：{str(e)}"

    # === 标签页2：语音设计 ===
    def generate_voice_design(
        text: str,
        language: str,
        instruct: str,
        progress=gr.Progress()
    ) -> Tuple[Optional[Tuple[int, Any]], str]:
        """使用语音设计生成语音。"""
        if not text.strip():
            return None, "请输入要合成的文本。"
        if not instruct.strip():
            return None, "请输入语音描述指令。"

        lang = LANG_DISPLAY_TO_INTERNAL.get(language, language)

        try:
            progress(0.3, desc="正在加载模型...")
            audio, sr = tts_engine.generate_voice_design(
                text=text,
                language=lang,
                instruct=instruct.strip()
            )
            progress(1.0, desc="生成完成！")
            return (sr, audio), "语音设计生成成功！"
        except Exception as e:
            return None, f"生成失败：{str(e)}"

    # === 标签页3：声音克隆 ===
    def generate_voice_clone(
        text: str,
        language: str,
        ref_audio: Optional[Tuple[int, Any]],
        ref_text: str,
        x_vector_only: bool,
        progress=gr.Progress()
    ) -> Tuple[Optional[Tuple[int, Any]], str]:
        """使用声音克隆生成语音。"""
        if not text.strip():
            return None, "请输入要合成的文本。"
        if ref_audio is None:
            return None, "请上传或录制参考音频。"

        lang = LANG_DISPLAY_TO_INTERNAL.get(language, language)

        try:
            progress(0.3, desc="正在加载模型...")
            audio, sr = tts_engine.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=ref_audio,
                ref_text=ref_text.strip() if ref_text.strip() else None,
                x_vector_only_mode=x_vector_only
            )
            progress(1.0, desc="生成完成！")
            return (sr, audio), "声音克隆生成成功！"
        except Exception as e:
            return None, f"生成失败：{str(e)}"

    # === 构建界面 ===
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
        title="Qwen3-TTS 语音合成",
    ) as app:

        # 标题栏
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div id='header-title'>
                    <h1>Qwen3-TTS 语音合成</h1>
                    <p>支持预设音色 · 语音设计 · 声音克隆 · 10种语言</p>
                </div>
                """)

        # 模型状态
        status_display = gr.HTML(get_status_html())

        # 标签页
        with gr.Tabs() as tabs:

            # === 标签页1：预设音色 ===
            with gr.TabItem("🎙️ 预设音色"):
                gr.HTML("<p style='color: var(--text-secondary);'>选择预设音色，可选添加情感/风格指令控制语气。</p>")

                with gr.Row():
                    with gr.Column(scale=2):
                        text_input_cv = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要转换为语音的文本...",
                            lines=5,
                        )

                        with gr.Row():
                            language_cv = gr.Dropdown(
                                choices=language_choices,
                                value="自动检测",
                                label="语言",
                            )
                            speaker_cv = gr.Dropdown(
                                choices=get_speaker_choices(),
                                value=get_speaker_choices()[0],
                                label="音色",
                            )

                        instruct_cv = gr.Textbox(
                            label="风格指令（可选）",
                            placeholder="例如：用特别愤怒的语气说 / 温柔缓慢地说话",
                            lines=2,
                        )

                        gen_btn_cv = gr.Button("🎵 生成语音", variant="primary")
                        result_msg_cv = gr.Textbox(label="状态", interactive=False)

                    with gr.Column(scale=1):
                        audio_output_cv = gr.Audio(
                            label="生成音频",
                            type="numpy",
                            interactive=False,
                        )

                # 音色信息展示
                gr.HTML("<h3 style='color: var(--text-primary); margin-top: 1rem;'>可用音色一览</h3>")
                gr.HTML(speaker_info_html)

                gen_btn_cv.click(
                    generate_custom_voice,
                    inputs=[text_input_cv, language_cv, speaker_cv, instruct_cv],
                    outputs=[audio_output_cv, result_msg_cv],
                )

            # === 标签页2：语音设计 ===
            with gr.TabItem("🎨 语音设计"):
                gr.HTML("<p style='color: var(--text-secondary);'>用自然语言描述你想要的音色，AI 将为你设计专属声音。</p>")

                with gr.Row():
                    with gr.Column(scale=2):
                        text_input_vd = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要转换为语音的文本...",
                            lines=5,
                        )

                        with gr.Row():
                            language_vd = gr.Dropdown(
                                choices=language_choices,
                                value="自动检测",
                                label="语言",
                            )

                        instruct_vd = gr.Textbox(
                            label="语音描述（必填）",
                            placeholder="例如：体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显 / 低沉磁性的男声，带有新闻播报的正式感",
                            lines=3,
                        )

                        gen_btn_vd = gr.Button("🎵 生成语音", variant="primary")
                        result_msg_vd = gr.Textbox(label="状态", interactive=False)

                    with gr.Column(scale=1):
                        audio_output_vd = gr.Audio(
                            label="生成音频",
                            type="numpy",
                            interactive=False,
                        )

                # 示例
                gr.HTML("<h3 style='color: var(--text-primary); margin-top: 1rem;'>语音描述示例</h3>")
                gr.Examples(
                    examples=voice_design_examples,
                    inputs=[instruct_vd, language_vd],
                    label="点击使用",
                )

                gen_btn_vd.click(
                    generate_voice_design,
                    inputs=[text_input_vd, language_vd, instruct_vd],
                    outputs=[audio_output_vd, result_msg_vd],
                )

            # === 标签页3：声音克隆 ===
            with gr.TabItem("🎤 声音克隆"):
                gr.HTML("""
                <p style='color: var(--text-secondary);'>
                    上传或录制一段参考音频（建议3秒以上），即可克隆该声音并合成新内容。
                </p>
                <p style='color: var(--text-muted); font-size: 0.8rem;'>
                    提示：远程部署时需使用 HTTPS 才能使用浏览器麦克风功能。
                </p>
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        text_input_vc = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要让克隆声音朗读的文本...",
                            lines=5,
                        )

                        with gr.Row():
                            language_vc = gr.Dropdown(
                                choices=language_choices,
                                value="自动检测",
                                label="语言",
                            )

                        ref_audio_vc = gr.Audio(
                            label="参考音频",
                            sources=["upload", "microphone"],
                            type="numpy",
                        )

                        ref_text_vc = gr.Textbox(
                            label="参考音频转录文本（可选，但推荐填写以提升质量）",
                            placeholder="请输入参考音频对应的文字内容...",
                            lines=2,
                        )

                        x_vector_only_vc = gr.Checkbox(
                            label="仅使用说话人特征嵌入（更快但质量略低）",
                            value=False,
                        )

                        gen_btn_vc = gr.Button("🎵 生成语音", variant="primary")
                        result_msg_vc = gr.Textbox(label="状态", interactive=False)

                    with gr.Column(scale=1):
                        audio_output_vc = gr.Audio(
                            label="生成音频",
                            type="numpy",
                            interactive=False,
                        )

                gen_btn_vc.click(
                    generate_voice_clone,
                    inputs=[text_input_vc, language_vc, ref_audio_vc, ref_text_vc, x_vector_only_vc],
                    outputs=[audio_output_vc, result_msg_vc],
                )

        # 页面加载时刷新状态
        app.load(lambda: get_status_html(), outputs=[status_display])

    return app


def launch_ui(tts_engine: TTSEngine, **kwargs) -> None:
    """
    创建并启动 Gradio 界面。

    Args:
        tts_engine: TTSEngine 实例
        **kwargs: 传递给 gr.Blocks.launch() 的参数
    """
    app = create_ui(tts_engine)
    app.launch(**kwargs)
