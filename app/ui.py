"""
Qwen3-TTS Gradio 界面
"""

import gradio as gr
from typing import Tuple, Optional, List
from datetime import datetime

from .config import SPEAKERS, LANGUAGES, MODEL_SIZES, DEFAULT_MODEL_SIZE, MODEL_IDS, MODEL_CAPABILITIES
from .tts_engine import TTSEngine


# 语言选项
LANGUAGE_CHOICES = ["Auto"] + LANGUAGES

# 音色选项
SPEAKER_CHOICES = list(SPEAKERS.keys())

# 模型大小选项
MODEL_SIZE_CHOICES = MODEL_SIZES  # ["0.6B", "1.7B"]


class UILogger:
    """日志管理器"""
    def __init__(self, max_lines: int = 100):
        self.logs: List[str] = []
        self.max_lines = max_lines
    
    def log(self, message: str) -> str:
        """添加日志并返回完整日志文本"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        # 保持最大行数限制
        if len(self.logs) > self.max_lines:
            self.logs = self.logs[-self.max_lines:]
        return "\n".join(self.logs)
    
    def get_logs(self) -> str:
        return "\n".join(self.logs)


def create_ui(tts_engine: TTSEngine) -> gr.Blocks:
    """创建 Gradio 界面"""

    logger = UILogger()
    
    # 记录初始状态
    logger.log(f"初始化完成，当前模型大小: {tts_engine.model_size}")

    def get_current_model_name(model_type: str) -> str:
        """获取当前模型的确切名称"""
        model_ids = MODEL_IDS.get(tts_engine.model_size, MODEL_IDS[DEFAULT_MODEL_SIZE])
        return model_ids.get(model_type, "未知")

    def get_model_info_text() -> str:
        """获取模型信息文本"""
        size = tts_engine.model_size
        caps = MODEL_CAPABILITIES.get(size, {})
        model_ids = MODEL_IDS.get(size, {})
        
        lines = [
            f"**模型大小**: {size}",
            f"**预设音色**: {model_ids.get('custom_voice', '不支持')}",
            f"**语音设计**: {'不支持' if not caps.get('voice_design') else model_ids.get('voice_design', '不支持')}",
            f"**声音克隆**: {model_ids.get('base', '不支持')}",
            f"**指令控制**: {'支持' if caps.get('instruct_control') else '不支持'}",
        ]
        return "\n".join(lines)

    # === 回调函数 ===
    def switch_model_size(model_size, log_text, progress=gr.Progress()):
        """切换模型大小"""
        try:
            progress(0.2, desc=f"切换到 {model_size} 模型...")
            tts_engine.set_model_size(model_size)
            log_msg = logger.log(f"切换模型大小: {model_size}")
            
            # 获取新模型信息
            caps = MODEL_CAPABILITIES.get(model_size, {})
            model_ids = MODEL_IDS.get(model_size, {})
            
            # 构建状态文本
            status = f"当前: {model_size}"
            
            # 返回更新：日志、状态、模型信息、instruct可见性、voice_design可见性
            return (
                log_msg,           # 日志
                status,            # 状态文本
                model_size,        # dropdown值
                get_model_info_text(),  # 模型信息
                gr.update(visible=caps.get('instruct_control', False)),  # instruct输入框
                gr.update(visible=caps.get('voice_design', False)),       # 语音设计tab
            )
        except Exception as e:
            log_msg = logger.log(f"切换失败: {e}")
            return (
                log_msg, 
                f"切换失败: {e}", 
                tts_engine.model_size,
                get_model_info_text(),
                gr.update(),
                gr.update(),
            )

    def gen_cv(text, lang, speaker, instruct, log_text, progress=gr.Progress()):
        if not text.strip():
            log_msg = logger.log("预设音色: 请输入文本")
            return None, "请输入文本", log_text
        
        # 检查是否支持指令控制
        caps = MODEL_CAPABILITIES.get(tts_engine.model_size, {})
        if not caps.get('instruct_control', False):
            instruct = None  # 0.6B 不支持指令，强制忽略
        
        try:
            model_name = get_current_model_name('custom_voice')
            log_msg = logger.log(f"预设音色: 加载模型 {model_name}")
            progress(0.3, desc="加载模型中...")
            audio, sr = tts_engine.generate_custom_voice(
                text=text, language=lang, speaker=speaker,
                instruct=instruct.strip() or None if instruct else None
            )
            log_msg = logger.log(f"预设音色: 生成成功 (模型: {model_name})")
            return (sr, audio), f"生成成功 (模型: {model_name})", log_msg
        except Exception as e:
            log_msg = logger.log(f"预设音色: 错误 - {e}")
            return None, f"错误: {e}", log_msg

    def gen_vd(text, lang, instruct, log_text, progress=gr.Progress()):
        # 先检查是否支持语音设计
        caps = MODEL_CAPABILITIES.get(tts_engine.model_size, {})
        if not caps.get('voice_design', False):
            log_msg = logger.log("语音设计: 当前模型大小不支持此功能")
            return None, "当前模型大小不支持语音设计，请切换到 1.7B", log_text
        
        if not text.strip():
            log_msg = logger.log("语音设计: 请输入文本")
            return None, "请输入文本", log_text
        if not instruct.strip():
            log_msg = logger.log("语音设计: 请输入语音描述")
            return None, "请输入语音描述", log_text
        try:
            model_name = get_current_model_name('voice_design')
            log_msg = logger.log(f"语音设计: 加载模型 {model_name}")
            progress(0.3, desc="加载模型中...")
            audio, sr = tts_engine.generate_voice_design(
                text=text, language=lang, instruct=instruct.strip()
            )
            log_msg = logger.log(f"语音设计: 生成成功 (模型: {model_name})")
            return (sr, audio), f"生成成功 (模型: {model_name})", log_msg
        except Exception as e:
            log_msg = logger.log(f"语音设计: 错误 - {e}")
            return None, f"错误: {e}", log_msg

    def gen_vc(text, lang, ref_audio, ref_text, xvec, log_text, progress=gr.Progress()):
        if not text.strip():
            log_msg = logger.log("声音克隆: 请输入文本")
            return None, "请输入文本", log_text
        if ref_audio is None:
            log_msg = logger.log("声音克隆: 请上传参考音频")
            return None, "请上传参考音频", log_text
        try:
            model_name = get_current_model_name('base')
            log_msg = logger.log(f"声音克隆: 加载模型 {model_name}")
            progress(0.3, desc="加载模型中...")
            audio, sr = tts_engine.generate_voice_clone(
                text=text, language=lang, ref_audio=ref_audio,
                ref_text=ref_text.strip() or None,
                x_vector_only_mode=xvec
            )
            log_msg = logger.log(f"声音克隆: 生成成功 (模型: {model_name})")
            return (sr, audio), f"生成成功 (模型: {model_name})", log_msg
        except Exception as e:
            log_msg = logger.log(f"声音克隆: 错误 - {e}")
            return None, f"错误: {e}", log_msg

    # 获取初始能力
    initial_caps = MODEL_CAPABILITIES.get(tts_engine.model_size, {})
    
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

        # 模型大小切换
        with gr.Row():
            model_size_dd = gr.Dropdown(
                MODEL_SIZE_CHOICES,
                value=tts_engine.model_size,
                label="模型大小",
                info="0.6B 轻量（~2GB显存），1.7B 高质量（~4GB显存）"
            )
            model_size_btn = gr.Button("切换模型", variant="secondary")
            model_size_status = gr.Textbox(label="模型状态", value=f"当前: {tts_engine.model_size}", interactive=False, scale=1)
        
        # 当前模型信息面板
        model_info_box = gr.Markdown(get_model_info_text(), label="当前模型信息")

        # 预设音色 Tab
        with gr.Tab("预设音色") as tab_cv:
            cv_desc = "**预设音色 (Custom Voice)** — 提供 9 种精选音色，覆盖多种性别、年龄、语言与方言组合"
            if initial_caps.get('instruct_control'):
                cv_desc += "；支持通过指令控制情感、风格与韵律"
            else:
                cv_desc += "（当前模型不支持指令控制）"
            gr.Markdown(cv_desc)
            
            with gr.Row():
                with gr.Column():
                    t1 = gr.Textbox(label="合成文本", lines=3, placeholder="请输入要合成的文本…")
                    with gr.Row():
                        l1 = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="语言", info="支持 10 种语言及自动检测")
                        s1 = gr.Dropdown(SPEAKER_CHOICES, value="Vivian", label="音色", info="选择预设音色")
                    i1 = gr.Textbox(
                        label="风格指令（可选）", 
                        lines=1, 
                        placeholder="例：用温柔的语气说 / Speak slowly and gently",
                        visible=initial_caps.get('instruct_control', False)
                    )
                    b1 = gr.Button("生成", variant="primary")
                    m1 = gr.Textbox(label="状态", interactive=False)
                with gr.Column():
                    a1 = gr.Audio(label="合成音频", type="numpy")
            # 音色参考表
            with gr.Accordion("📖 音色参考", open=False):
                gr.Markdown(
                    "**中文音色**  \n"
                    "• **Vivian** — 明亮、略带锋芒的年轻女声  \n"
                    "• **Serena** — 温暖、柔和的年轻女声  \n"
                    "• **Uncle_Fu** — 低沉醇厚的成熟男声  \n\n"
                    "**方言音色**  \n"
                    "• **Dylan** — 清亮自然的京味年轻男声（京味方言）  \n"
                    "• **Eric** — 略带沙哑亮度的成都男声（川味方言）  \n\n"
                    "**外语音色**  \n"
                    "• **Ryan** — 节奏感强的动感男声（English）  \n"
                    "• **Aiden** — 阳光清澈的美式男中音（English）  \n"
                    "• **Ono_Anna** — 轻快俏皮的日系女声（日语）  \n"
                    "• **Sohee** — 温暖富有情感的韩语女声（韩语）"
                )

        # 语音设计 Tab（仅 1.7B 可见）
        with gr.Tab("语音设计", visible=initial_caps.get('voice_design', False)) as tab_vd:
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

        # 声音克隆 Tab
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

        # 日志窗口（页面底部）
        with gr.Accordion("📋 操作日志", open=False):
            log_box = gr.Textbox(
                label="", 
                lines=8, 
                interactive=False,
                value=logger.get_logs(),
                show_copy_button=True
            )

        # === 事件绑定 ===
        # 隐藏的日志状态（用于传递日志）
        log_state = gr.Textbox(value=logger.get_logs(), visible=False)
        
        model_size_btn.click(
            switch_model_size, 
            [model_size_dd, log_state], 
            [log_state, model_size_status, model_size_dd, model_info_box, i1, tab_vd]
        )
        
        b1.click(gen_cv, [t1, l1, s1, i1, log_state], [a1, m1, log_state])
        b1.click(lambda x: x, [log_state], [log_box])
        
        b2.click(gen_vd, [t2, l2, i2, log_state], [a2, m2, log_state])
        b2.click(lambda x: x, [log_state], [log_box])
        
        b3.click(gen_vc, [t3, l3, r3, rt3, x3, log_state], [a3, m3, log_state])
        b3.click(lambda x: x, [log_state], [log_box])

    return app


def launch_ui(tts_engine: TTSEngine, **kwargs):
    create_ui(tts_engine).launch(**kwargs)
