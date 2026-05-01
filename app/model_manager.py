"""
Model Manager for Qwen3-TTS

Handles model downloading, caching, and loading with GPU memory management.
"""

import os
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from qwen_tts import Qwen3TTSModel

from .config import CACHE_DIR, MODEL_IDS, DEFAULT_DTYPE, DEFAULT_ATTN_IMPLEMENTATION


class ModelManager:
    """
    Manages Qwen3-TTS model downloading, caching, and loading.
    
    Features:
    - Auto-download from HuggingFace with caching
    - GPU memory management (load/unload on demand)
    - Thread-safe model loading
    - Progress callbacks for downloads
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        dtype: str = DEFAULT_DTYPE,
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    ):
        """
        Initialize ModelManager.
        
        Args:
            cache_dir: Directory for model cache (default: ~/.cache/qwen3-tts/)
            device: Device to load models on (default: auto-detect)
            dtype: Data type for model weights (default: bfloat16)
            attn_implementation: Attention implementation (default: flash_attention_2)
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        
        # Currently loaded model
        self.current_model: Optional[Qwen3TTSModel] = None
        self.current_model_type: Optional[str] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Model paths cache
        self._model_paths: Dict[str, str] = {}
    
    def _get_model_path(self, model_type: str, local_dir: Optional[str] = None) -> str:
        """
        Get or download model path.
        
        Args:
            model_type: One of 'custom_voice', 'voice_design', 'base', 'tokenizer'
            local_dir: Optional local directory path for the model
            
        Returns:
            Path to the model directory
        """
        if local_dir and os.path.exists(local_dir):
            return local_dir
        
        # Check cache
        if model_type in self._model_paths:
            cached_path = self._model_paths[model_type]
            if os.path.exists(cached_path):
                return cached_path
        
        # Get model ID
        if model_type not in MODEL_IDS:
            raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(MODEL_IDS.keys())}")
        
        model_id = MODEL_IDS[model_type]
        
        # Check if already downloaded
        model_cache_path = self.cache_dir / model_type
        if model_cache_path.exists() and any(model_cache_path.iterdir()):
            self._model_paths[model_type] = str(model_cache_path)
            return str(model_cache_path)
        
        # Download from HuggingFace
        print(f"Downloading {model_type} model from {model_id}...")
        try:
            downloaded_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(self.cache_dir),
                local_dir=str(model_cache_path),
                local_dir_use_symlinks=False
            )
            self._model_paths[model_type] = downloaded_path
            print(f"Model {model_type} downloaded successfully to {downloaded_path}")
            return downloaded_path
        except Exception as e:
            print(f"Error downloading model {model_type}: {e}")
            # Try ModelScope as fallback for China users
            try:
                from modelscope import snapshot_download as ms_snapshot_download
                print(f"Trying ModelScope mirror for {model_type}...")
                downloaded_path = ms_snapshot_download(
                    model_id=model_id,
                    cache_dir=str(self.cache_dir / model_type)
                )
                self._model_paths[model_type] = downloaded_path
                print(f"Model {model_type} downloaded from ModelScope to {downloaded_path}")
                return downloaded_path
            except ImportError:
                raise Exception(
                    f"Failed to download {model_type} from HuggingFace. "
                    "Install modelscope for China mirror: pip install modelscope"
                ) from e
            except Exception as ms_error:
                raise Exception(
                    f"Failed to download {model_type} from both HuggingFace and ModelScope"
                ) from ms_error
    
    def load_model(
        self,
        model_type: str,
        local_dir: Optional[str] = None,
        force_reload: bool = False
    ) -> Qwen3TTSModel:
        """
        Load a Qwen3-TTS model.
        
        Args:
            model_type: One of 'custom_voice', 'voice_design', 'base'
            local_dir: Optional local directory path for the model
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded Qwen3TTSModel instance
        """
        with self._lock:
            # Check if already loaded
            if not force_reload and self.current_model is not None and self.current_model_type == model_type:
                return self.current_model
            
            # Unload current model to free GPU memory
            if self.current_model is not None:
                print(f"Unloading current model: {self.current_model_type}")
                del self.current_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.current_model = None
                self.current_model_type = None
            
            # Get model path (download if needed)
            model_path = self._get_model_path(model_type, local_dir)
            
            # Load model
            print(f"Loading {model_type} model from {model_path}...")
            try:
                # Try with flash_attention_2 first
                model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    torch_dtype=getattr(torch, self.dtype),
                    attn_implementation=self.attn_implementation,
                    device_map=self.device
                )
                print(f"Model {model_type} loaded successfully with {self.attn_implementation}")
            except Exception as e:
                # Fallback without flash_attention_2
                print(f"Failed to load with {self.attn_implementation}, falling back to default attention: {e}")
                model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    torch_dtype=getattr(torch, self.dtype),
                    device_map=self.device
                )
                print(f"Model {model_type} loaded successfully with default attention")
            
            self.current_model = model
            self.current_model_type = model_type
            
            return model
    
    def unload_model(self):
        """Unload the current model and free GPU memory."""
        with self._lock:
            if self.current_model is not None:
                print(f"Unloading model: {self.current_model_type}")
                del self.current_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.current_model = None
                self.current_model_type = None
    
    def is_model_downloaded(self, model_type: str) -> bool:
        """
        Check if a model is already downloaded.
        
        Args:
            model_type: One of 'custom_voice', 'voice_design', 'base', 'tokenizer'
            
        Returns:
            True if model exists locally
        """
        model_cache_path = self.cache_dir / model_type
        return model_cache_path.exists() and any(model_cache_path.iterdir())
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model state.
        
        Returns:
            Dictionary with model info
        """
        return {
            "current_model": self.current_model_type,
            "device": self.device,
            "dtype": self.dtype,
            "cache_dir": str(self.cache_dir),
            "downloaded_models": [
                model_type for model_type in MODEL_IDS.keys()
                if self.is_model_downloaded(model_type)
            ]
        }
