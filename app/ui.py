"""
Gradio UI for Qwen3-TTS Application

Modern, beautiful interface with three TTS modes.
"""

import os
import gradio as gr
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

from .config import SPEAKERS, LANGUAGES
from .tts_engine import TTSEngine


# Load custom CSS
CSS_PATH = Path(__file__).parent / "style.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def get_speaker_choices() -> List[str]:
    """Get formatted speaker choices for dropdown."""
    return [
        f"{name} - {info['zh']} ({info['en']})"
        for name, info in SPEAKERS.items()
    ]


def parse_speaker_choice(choice: str) -> str:
    """Extract speaker name from formatted choice."""
    return choice.split(" - ")[0] if " - " in choice else choice


def create_ui(tts_engine: TTSEngine) -> gr.Blocks:
    """
    Create the Gradio UI for Qwen3-TTS.
    
    Args:
        tts_engine: TTSEngine instance for TTS generation
        
    Returns:
        Gradio Blocks application
    """
    
    # Speaker info for display
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
    
    # Voice design examples
    voice_design_examples = [
        ["温柔的年轻女声，语速较慢，带有安抚感", "Chinese"],
        ["低沉磁性的男声，带有新闻播报的正式感", "Chinese"],
        ["A cheerful, energetic female voice with a slight southern drawl", "English"],
        ["An authoritative male voice, deep and resonant, like a documentary narrator", "English"],
    ]
    
    # Language choices with Auto option
    language_choices = ["Auto"] + LANGUAGES
    
    def get_status_html() -> str:
        """Get current model status as HTML."""
        try:
            status = tts_engine.get_status()
            current = status.get("current_model", "None")
            if current and current != "None":
                return f"<span class='status-badge loaded'>Model: {current}</span>"
            return "<span class='status-badge unloaded'>No model loaded</span>"
        except Exception:
            return "<span class='status-badge unloaded'>Status unavailable</span>"
    
    # === Tab 1: Custom Voice ===
    def generate_custom_voice(
        text: str,
        language: str,
        speaker_choice: str,
        instruct: str,
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[Optional[Tuple[int, Any]], str]:
        """Generate speech using custom voice."""
        if not text.strip():
            return None, "Please enter text to synthesize."
        
        speaker = parse_speaker_choice(speaker_choice)
        
        try:
            progress(0.3, desc="Loading model...")
            audio, sr = tts_engine.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct.strip() if instruct.strip() else None
            )
            progress(1.0, desc="Done!")
            # Gradio expects (sample_rate, audio_array)
            return (sr, audio), f"Generated successfully with {speaker}!"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # === Tab 2: Voice Design ===
    def generate_voice_design(
        text: str,
        language: str,
        instruct: str,
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[Optional[Tuple[int, Any]], str]:
        """Generate speech using voice design."""
        if not text.strip():
            return None, "Please enter text to synthesize."
        if not instruct.strip():
            return None, "Please provide a voice description instruction."
        
        try:
            progress(0.3, desc="Loading model...")
            audio, sr = tts_engine.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct.strip()
            )
            progress(1.0, desc="Done!")
            return (sr, audio), "Voice design generated successfully!"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # === Tab 3: Voice Clone ===
    def generate_voice_clone(
        text: str,
        language: str,
        ref_audio: Optional[Tuple[int, Any]],
        ref_text: str,
        x_vector_only: bool,
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[Optional[Tuple[int, Any]], str]:
        """Generate speech using voice clone."""
        if not text.strip():
            return None, "Please enter text to synthesize."
        if ref_audio is None:
            return None, "Please upload or record reference audio."
        
        try:
            progress(0.3, desc="Loading model...")
            # ref_audio from Gradio is (sample_rate, audio_array)
            audio, sr = tts_engine.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,  # Pass tuple directly
                ref_text=ref_text.strip() if ref_text.strip() else None,
                x_vector_only_mode=x_vector_only
            )
            progress(1.0, desc="Done!")
            return (sr, audio), "Voice clone generated successfully!"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # === Build UI ===
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
        title="Qwen3-TTS",
    ) as app:
        
        # Header
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div id='header-title'>
                    <h1>Qwen3-TTS</h1>
                    <p>Premium Text-to-Speech with Custom Voice, Voice Design, and Voice Clone</p>
                </div>
                """)
        
        # Model status
        status_display = gr.HTML(get_status_html())
        
        # Tabs
        with gr.Tabs() as tabs:
            
            # === Tab 1: Custom Voice ===
            with gr.TabItem("Custom Voice"):
                gr.HTML("<p style='color: var(--text-secondary);'>Choose a preset speaker and optionally add emotion/style instructions.</p>")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input_cv = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=5,
                        )
                        
                        with gr.Row():
                            language_cv = gr.Dropdown(
                                choices=language_choices,
                                value="Auto",
                                label="Language",
                            )
                            speaker_cv = gr.Dropdown(
                                choices=get_speaker_choices(),
                                value=get_speaker_choices()[0],
                                label="Speaker",
                            )
                        
                        instruct_cv = gr.Textbox(
                            label="Style Instruction (Optional)",
                            placeholder="e.g., Speak slowly and gently / 用特别愤怒的语气说",
                            lines=2,
                        )
                        
                        gen_btn_cv = gr.Button("Generate Speech", variant="primary")
                        result_msg_cv = gr.Textbox(label="Status", interactive=False)
                        
                    with gr.Column(scale=1):
                        audio_output_cv = gr.Audio(
                            label="Generated Audio",
                            type="numpy",
                            interactive=False,
                        )
                
                # Speaker info section
                gr.HTML("<h3 style='color: var(--text-primary); margin-top: 1rem;'>Available Speakers</h3>")
                gr.HTML(speaker_info_html)
                
                # Wire up events
                gen_btn_cv.click(
                    generate_custom_voice,
                    inputs=[text_input_cv, language_cv, speaker_cv, instruct_cv],
                    outputs=[audio_output_cv, result_msg_cv],
                )
            
            # === Tab 2: Voice Design ===
            with gr.TabItem("Voice Design"):
                gr.HTML("<p style='color: var(--text-secondary);'>Describe the voice you want using natural language.</p>")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input_vd = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=5,
                        )
                        
                        with gr.Row():
                            language_vd = gr.Dropdown(
                                choices=language_choices,
                                value="Auto",
                                label="Language",
                            )
                        
                        instruct_vd = gr.Textbox(
                            label="Voice Description (Required)",
                            placeholder="e.g., A warm, gentle female voice speaking slowly / 体现撒娇稚嫩的萝莉女声",
                            lines=3,
                        )
                        
                        gen_btn_vd = gr.Button("Generate Speech", variant="primary")
                        result_msg_vd = gr.Textbox(label="Status", interactive=False)
                        
                    with gr.Column(scale=1):
                        audio_output_vd = gr.Audio(
                            label="Generated Audio",
                            type="numpy",
                            interactive=False,
                        )
                
                # Examples section
                gr.HTML("<h3 style='color: var(--text-primary); margin-top: 1rem;'>Example Voice Descriptions</h3>")
                gr.Examples(
                    examples=voice_design_examples,
                    inputs=[instruct_vd, language_vd],
                    label="Click to use",
                )
                
                gen_btn_vd.click(
                    generate_voice_design,
                    inputs=[text_input_vd, language_vd, instruct_vd],
                    outputs=[audio_output_vd, result_msg_vd],
                )
            
            # === Tab 3: Voice Clone ===
            with gr.TabItem("Voice Clone"):
                gr.HTML("""
                <p style='color: var(--text-secondary);'>
                    Clone a voice from reference audio. Upload or record a short audio sample (3+ seconds recommended).
                </p>
                <p style='color: var(--text-muted); font-size: 0.8rem;'>
                    Note: For microphone access when deployed remotely, the server must use HTTPS.
                </p>
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input_vc = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter the text you want the cloned voice to speak...",
                            lines=5,
                        )
                        
                        with gr.Row():
                            language_vc = gr.Dropdown(
                                choices=language_choices,
                                value="Auto",
                                label="Language",
                            )
                        
                        ref_audio_vc = gr.Audio(
                            label="Reference Audio",
                            sources=["upload", "microphone"],
                            type="numpy",
                        )
                        
                        ref_text_vc = gr.Textbox(
                            label="Reference Transcript (Optional but recommended)",
                            placeholder="Enter the transcript of the reference audio for better quality...",
                            lines=2,
                        )
                        
                        x_vector_only_vc = gr.Checkbox(
                            label="Use speaker embedding only (faster, less accurate)",
                            value=False,
                        )
                        
                        gen_btn_vc = gr.Button("Generate Speech", variant="primary")
                        result_msg_vc = gr.Textbox(label="Status", interactive=False)
                        
                    with gr.Column(scale=1):
                        audio_output_vc = gr.Audio(
                            label="Generated Audio",
                            type="numpy",
                            interactive=False,
                        )
                
                gen_btn_vc.click(
                    generate_voice_clone,
                    inputs=[text_input_vc, language_vc, ref_audio_vc, ref_text_vc, x_vector_only_vc],
                    outputs=[audio_output_vc, result_msg_vc],
                )
        
        # Refresh status on load
        app.load(lambda: get_status_html(), outputs=[status_display])
    
    return app


def launch_ui(tts_engine: TTSEngine, **kwargs) -> None:
    """
    Create and launch the Gradio UI.
    
    Args:
        tts_engine: TTSEngine instance
        **kwargs: Additional arguments for gr.Blocks.launch()
    """
    app = create_ui(tts_engine)
    app.launch(**kwargs)