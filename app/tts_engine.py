"""
TTS Engine for Qwen3-TTS

Main interface for text-to-speech generation using Qwen3-TTS models.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any

from .model_manager import ModelManager
from .config import SPEAKERS, LANGUAGES, DEFAULT_SAMPLE_RATE, MODEL_SIZES, DEFAULT_MODEL_SIZE, MODEL_CAPABILITIES, MODEL_IDS

# Valid language values including Auto for auto-detection
VALID_LANGUAGES = list(LANGUAGES) + ["Auto"]


class TTSEngine:
    """
    Main TTS engine providing a clean API for Qwen3-TTS generation.
    
    Supports three generation modes:
    1. Custom Voice: Pre-defined speakers with optional emotion/style control
    2. Voice Design: Natural language voice description
    3. Voice Clone: Clone voice from reference audio
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        model_size: str = DEFAULT_MODEL_SIZE
    ):
        """
        Initialize TTSEngine.
        
        Args:
            model_manager: Optional ModelManager instance (if None, creates one)
            cache_dir: Directory for model cache
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            dtype: Model dtype ('bfloat16' or 'float16')
            model_size: Model size to use ('0.6B' or '1.7B', default: 1.7B)
        """
        if model_manager is not None:
            self.model_manager = model_manager
        else:
            self.model_manager = ModelManager(
                cache_dir=cache_dir,
                device=device,
                dtype=dtype,
                model_size=model_size
            )
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.model_size = model_size
    
    def generate_custom_voice(
        self,
        text: str,
        language: str = "Chinese",
        speaker: str = "Vivian",
        instruct: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using a pre-defined speaker voice.
        
        Args:
            text: Text to synthesize
            language: Language of the text (e.g., 'Chinese', 'English')
            speaker: Speaker name, one of: Vivian, Serena, Uncle_Fu, Dylan, Eric,
                     Ryan, Aiden, Ono_Anna, Sohee
            instruct: Optional emotion/style instruction (e.g., 'speak slowly and gently')
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if speaker not in SPEAKERS:
            raise ValueError(f"Unknown speaker: {speaker}. Must be one of {list(SPEAKERS.keys())}")
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}. Must be one of {VALID_LANGUAGES}")

        # 0.6B CustomVoice does NOT support instruct control — force ignore
        caps = MODEL_CAPABILITIES.get(self.model_size, {})
        if not caps.get('instruct_control', False):
            instruct = None

        model = self.model_manager.load_model("custom_voice")
        
        # Qwen3TTSModel.generate_custom_voice returns (wavs, sample_rate)
        result = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct if instruct else None
        )
        
        # Handle return value: (wavs_list, sample_rate)
        if isinstance(result, tuple) and len(result) == 2:
            wavs, sr = result
            # wavs is a list of numpy arrays
            if isinstance(wavs, (list, tuple)) and len(wavs) > 0:
                audio = wavs[0]
            else:
                audio = wavs
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            return audio, sr
        else:
            # Fallback
            return np.array(result), self.sample_rate
    
    def generate_voice_design(
        self,
        text: str,
        language: str = "English",
        instruct: str = "A warm, friendly voice"
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using a natural language voice description.
        
        Args:
            text: Text to synthesize
            language: Language of the text
            instruct: Natural language voice description
                      (e.g., 'A deep, authoritative male voice with slight echo')
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}. Must be one of {VALID_LANGUAGES}")

        # Check if voice_design is supported for current model size
        caps = MODEL_CAPABILITIES.get(self.model_size, {})
        if not caps.get('voice_design', False):
            raise NotImplementedError(
                f"Voice Design is not supported for {self.model_size} model. "
                "Please use 1.7B model size for this feature."
            )

        model = self.model_manager.load_model("voice_design")
        
        # Qwen3TTSModel.generate_voice_design returns (wavs, sample_rate)
        result = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct
        )
        
        # Handle return value: (wavs_list, sample_rate)
        if isinstance(result, tuple) and len(result) == 2:
            wavs, sr = result
            if isinstance(wavs, (list, tuple)) and len(wavs) > 0:
                audio = wavs[0]
            else:
                audio = wavs
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            return audio, sr
        else:
            return np.array(result), self.sample_rate
    
    def generate_voice_clone(
        self,
        text: str,
        language: str = "English",
        ref_audio: Union[str, tuple] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Clone a voice from reference audio and synthesize text.
        
        Args:
            text: Text to synthesize
            language: Language of the text
            ref_audio: Reference audio as file path (str), URL (str), or
                       tuple of (audio_array, sample_rate)
            ref_text: Transcript of the reference audio (improves quality)
            x_vector_only_mode: If True, only use speaker identity without
                                 prosody matching (faster but less accurate)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if ref_audio is None:
            raise ValueError("ref_audio is required for voice cloning")
        if language not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language: {language}. Must be one of {VALID_LANGUAGES}")

        model = self.model_manager.load_model("base")
        
        # Generate voice clone
        # Qwen3TTSModel.generate_voice_clone returns (wavs, sample_rate)
        result = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode
        )
        
        # Handle return value: (wavs_list, sample_rate)
        if isinstance(result, tuple) and len(result) == 2:
            wavs, sr = result
            if isinstance(wavs, (list, tuple)) and len(wavs) > 0:
                audio = wavs[0]
            else:
                audio = wavs
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            return audio, sr
        else:
            return np.array(result), self.sample_rate
    
    def get_speakers(self) -> List[Dict[str, Any]]:
        """
        Get list of available speakers with descriptions.
        
        Returns:
            List of dicts with keys: name, description_zh, description_en, language
        """
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
        """
        Get list of supported languages.
        
        Returns:
            List of language names
        """
        return list(LANGUAGES)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current engine status.
        
        Returns:
            Dictionary with status info
        """
        return self.model_manager.get_model_info()
    
    def ensure_model_loaded(self, model_type: str) -> None:
        """
        Ensure a specific model is loaded.
        
        Args:
            model_type: One of 'custom_voice', 'voice_design', 'base'
        """
        self.model_manager.load_model(model_type)
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model loading status.
        
        Returns:
            Dictionary with model status info
        """
        return self.get_status()
    
    def unload(self):
        """Unload current model and free GPU memory."""
        self.model_manager.unload_model()

    def set_model_size(self, model_size: str) -> None:
        """
        Switch to a different model size (0.6B or 1.7B).
        Unloads current model and updates model_size for subsequent loads.
        
        Args:
            model_size: Model size ('0.6B' or '1.7B')
        """
        if model_size not in MODEL_SIZES:
            raise ValueError(f"Unknown model size: {model_size}. Must be one of {MODEL_SIZES}")
        if model_size == self.model_size:
            return
        # Unload current model first (size change requires different model files)
        self.model_manager.unload_model()
        self.model_manager.model_size = model_size
        self.model_size = model_size
