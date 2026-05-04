"""
Configuration constants for Qwen3-TTS application
"""

import os
import sys
from pathlib import Path

# Determine if running as PyInstaller bundle
def _is_frozen() -> bool:
    """Check if running as packaged executable (PyInstaller, cx_Freeze, etc.)."""
    # PyInstaller sets sys.frozen=True and sys._MEIPASS
    # cx_Freeze and others set sys.frozen=True without _MEIPASS
    return getattr(sys, 'frozen', False)

def _get_exe_dir() -> Path:
    """Get the directory where the EXE actually resides (not the temp extraction dir).
    
    PyInstaller unpacks _internal/ to sys._MEIPASS (a temp dir), but the EXE
    and its sibling folders (like models/) live next to the actual executable.
    We must look for models next to the EXE, not in the temp extraction dir.
    """
    if getattr(sys, 'frozen', False):
        # sys.executable points to the actual .exe file
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.parent.resolve()

def _get_bundle_dir() -> Path:
    """Get the internal PyInstaller bundle directory (sys._MEIPASS) or source root.
    
    This is where code/resources are unpacked at runtime. 
    For locating bundled code/data files (like style.css), use this.
    For locating files placed next to the EXE (like models/), use _get_exe_dir().
    """
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent.resolve()

def _get_offline_models_dir() -> Path:
    """Get the offline models directory.
    
    Priority: environment variable > EXE-adjacent models/ (with actual models) > user home cache
    
    In frozen (EXE) mode, models/ is placed next to the EXE file, NOT inside
    the PyInstaller _MEIPASS temp extraction directory.
    
    Note: This is used as a hint for the primary search location. The actual
    model search in ModelManager._get_model_path() checks multiple locations
    including this one, so even if this returns the user home cache, models
    found next to the EXE will still be discovered.
    """
    # Highest priority: explicit environment variable
    env_dir = os.environ.get('QWEN3_TTS_MODELS_DIR', '')
    if env_dir:
        return Path(env_dir)
    
    # If running as packaged exe, check for models/ next to the EXE
    if _is_frozen():
        exe_adjacent_models = _get_exe_dir() / "models"
        # Only return this if it actually contains model subdirectories
        # (not just an empty dir created by mkdir)
        if exe_adjacent_models.exists():
            has_models = any(
                (exe_adjacent_models / subdir).exists()
                for subdir in ("custom_voice", "voice_design", "base", "tokenizer",
                               "0.6B", "1.7B")
            )
            if has_models:
                return exe_adjacent_models
    
    # Default: user home cache (works for both online and offline)
    return Path.home() / ".cache" / "qwen3-tts"

# Cache directory for downloaded models (online mode)
CACHE_DIR = Path.home() / ".cache" / "qwen3-tts"

# Offline models directory (for frozen/standalone deployment)
OFFLINE_MODELS_DIR = _get_offline_models_dir()

# Flag for offline mode (no network access)
OFFLINE_MODE = os.environ.get('QWEN3_TTS_OFFLINE', '').lower() in ('1', 'true', 'yes')

# Supported model sizes
MODEL_SIZES = ["0.6B", "1.7B"]
DEFAULT_MODEL_SIZE = "1.7B"

# Model IDs from HuggingFace (organized by size)
# NOTE: 0.6B does NOT have VoiceDesign model, and CustomVoice does NOT support instruct control
MODEL_IDS = {
    "0.6B": {
        "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    },
    "1.7B": {
        "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    },
}

# Model capabilities (what each model size supports)
MODEL_CAPABILITIES = {
    "0.6B": {
        "custom_voice": True,       # Has CustomVoice model
        "voice_design": False,      # NO VoiceDesign model
        "base": True,               # Has Base model (voice clone)
        "instruct_control": False,  # CustomVoice does NOT support instruct
    },
    "1.7B": {
        "custom_voice": True,
        "voice_design": True,
        "base": True,
        "instruct_control": True,   # CustomVoice supports instruct control
    },
}

# Supported speakers with descriptions
SPEAKERS = {
    "Vivian": {
        "zh": "明亮、略带锋芒的年轻女声",
        "en": "Bright, slightly edgy young female voice",
        "language": "Chinese"
    },
    "Serena": {
        "zh": "温暖、柔和的年轻女声",
        "en": "Warm, gentle young female voice",
        "language": "Chinese"
    },
    "Uncle_Fu": {
        "zh": "低沉醇厚的成熟男声",
        "en": "Seasoned male voice with a low, mellow timbre",
        "language": "Chinese"
    },
    "Dylan": {
        "zh": "清亮自然的京味年轻男声",
        "en": "Youthful Beijing male voice with a clear, natural timbre",
        "language": "Chinese/Beijing Dialect"
    },
    "Eric": {
        "zh": "略带沙哑亮度的成都男声",
        "en": "Lively Chengdu male voice with a slightly husky brightness",
        "language": "Chinese/Sichuan Dialect"
    },
    "Ryan": {
        "zh": "节奏感强的动感男声",
        "en": "Dynamic male voice with strong rhythmic drive",
        "language": "English"
    },
    "Aiden": {
        "zh": "阳光清澈的美式男中音",
        "en": "Sunny American male voice with a clear midrange",
        "language": "English"
    },
    "Ono_Anna": {
        "zh": "轻快俏皮的日系女声",
        "en": "Playful Japanese female voice with a light, nimble timbre",
        "language": "Japanese"
    },
    "Sohee": {
        "zh": "温暖富有情感的韩语女声",
        "en": "Warm Korean female voice with rich emotion",
        "language": "Korean"
    }
}

# Supported languages
LANGUAGES = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian"
]

# Default generation parameters
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
