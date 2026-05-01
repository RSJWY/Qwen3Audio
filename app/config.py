"""
Configuration constants for Qwen3-TTS application
"""

import os
from pathlib import Path

# Cache directory for downloaded models
CACHE_DIR = Path.home() / ".cache" / "qwen3-tts"

# Model IDs from HuggingFace
MODEL_IDS = {
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
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
