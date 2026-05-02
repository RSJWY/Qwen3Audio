"""
Configuration constants for Qwen3-TTS application
"""

import os
import sys
from pathlib import Path

# Determine if running as PyInstaller bundle
def _is_frozen() -> bool:
    """Check if running as PyInstaller bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def _get_bundle_dir() -> Path:
    """Get the directory of the PyInstaller bundle or source."""
    if _is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent

def _get_offline_models_dir() -> Path:
    """Get the offline models directory."""
    # Priority: environment variable > bundle models/ > user home cache
    env_dir = os.environ.get('QWEN3_TTS_MODELS_DIR', '')
    if env_dir:
        return Path(env_dir)
    
    # If running as frozen exe, check for bundled models
    if _is_frozen():
        bundled_models = _get_bundle_dir() / "models"
        if bundled_models.exists():
            return bundled_models
    
    # Default: user home cache (for online mode)
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
