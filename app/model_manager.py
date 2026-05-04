"""
Model Manager for Qwen3-TTS

Handles model downloading, caching, and loading with GPU memory management.
Supports both online and offline modes.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from qwen_tts import Qwen3TTSModel

from .config import (
    CACHE_DIR,
    OFFLINE_MODELS_DIR,
    OFFLINE_MODE,
    MODEL_IDS,
    MODEL_SIZES,
    DEFAULT_MODEL_SIZE,
    DEFAULT_DTYPE,
    DEFAULT_ATTN_IMPLEMENTATION
)


def _is_frozen() -> bool:
    """Check if running as packaged executable (PyInstaller, cx_Freeze, etc.)."""
    return getattr(sys, 'frozen', False)


def _get_exe_dir() -> Path:
    """Get the directory where the EXE actually resides.
    
    Models are placed next to the EXE (e.g. dist/Qwen3-TTS/models/),
    NOT inside the PyInstaller temp extraction dir (sys._MEIPASS).
    """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.parent.resolve()


def _get_bundle_dir() -> Path:
    """Get the internal PyInstaller bundle directory or source root.
    
    For locating code/data resources (like style.css inside the bundle).
    For locating files next to the EXE (like models/), use _get_exe_dir().
    """
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent.resolve()


def _is_valid_model_dir(path: Path) -> bool:
    """Check if a directory contains actual model files (not just empty/metadata dirs)."""
    if not path.exists():
        return False
    try:
        # A valid model dir should contain at least config.json or model files
        # Exclude dirs that only have HF metadata (.cache, .lock, ._____temp)
        meaningful_files = [
            f for f in path.iterdir()
            if f.name not in ('.cache', '.lock', '._____temp', '.gitattributes')
            and not f.name.startswith('models--')  # HF hub cache format
        ]
        # Check for config.json as a reliable indicator of a real model dir
        has_config = (path / "config.json").exists()
        has_safetensors = any(
            f.name.endswith('.safetensors') or f.name.endswith('.bin')
            for f in path.iterdir() if f.is_file()
        )
        return has_config or has_safetensors or len(meaningful_files) > 0
    except (OSError, PermissionError):
        return False


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
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        offline_mode: Optional[bool] = None,
        model_size: str = DEFAULT_MODEL_SIZE
    ):
        """
        Initialize ModelManager.

        Args:
            cache_dir: Directory for model cache (default: ~/.cache/qwen3-tts/)
            device: Device to load models on (default: auto-detect)
            dtype: Data type for model weights (default: bfloat16)
            attn_implementation: Attention implementation (default: flash_attention_2)
            offline_mode: Force offline mode (default: auto-detect from environment)
            model_size: Model size to use ('0.6B' or '1.7B', default: 1.7B)
        """
        # Validate model size
        if model_size not in MODEL_SIZES:
            raise ValueError(f"Unknown model size: {model_size}. Must be one of {MODEL_SIZES}")
        self.model_size = model_size

        # Determine offline mode
        if offline_mode is None:
            self.offline_mode = OFFLINE_MODE or _is_frozen()
        else:
            self.offline_mode = offline_mode

        # Determine cache directory: always use user home cache as base
        # The _get_model_path method searches multiple locations, so even in
        # offline mode, it will find models in ~/.cache/qwen3-tts/ if they exist.
        if cache_dir is not None:
            self.cache_dir = cache_dir
        else:
            # Always use user home cache as primary cache_dir.
            # This ensures models downloaded via `python main.py` (online mode)
            # are found even when the EXE is run in offline/frozen mode.
            # _get_model_path will also check EXE-adjacent models/ and OFFLINE_MODELS_DIR.
            self.cache_dir = CACHE_DIR
        
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
        
        Search order:
        1. Explicit local_dir (if provided)
        2. Previously cached path for this model_type
        3. EXE-adjacent models/ with model_size subdirectory (frozen mode)
        4. EXE-adjacent models/ without model_size subdirectory (frozen mode)
        5. Primary cache_dir with model_size subdirectory
        6. Primary cache_dir without model_size subdirectory
        7. User home cache (~/.cache/qwen3-tts/) with model_size subdirectory
        8. User home cache without model_size subdirectory
        9. Download from HuggingFace (online mode only)
        
        Args:
            model_type: One of 'custom_voice', 'voice_design', 'base', 'tokenizer'
            local_dir: Optional local directory path for the model
            
        Returns:
            Path to the model directory
        """
        if local_dir and os.path.exists(local_dir):
            return local_dir
        
        # Check in-memory path cache
        if model_type in self._model_paths:
            cached_path = self._model_paths[model_type]
            if os.path.exists(cached_path):
                return cached_path
        
        # Get model ID for validation
        model_ids = MODEL_IDS.get(self.model_size, MODEL_IDS[DEFAULT_MODEL_SIZE])
        if model_type not in model_ids:
            raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(model_ids.keys())}")

        model_id = model_ids[model_type]
        
        # Build comprehensive search paths
        # The model files can be organized in two layouts:
        #   Layout A (with model_size): base_dir/1.7B/custom_voice/
        #   Layout B (flat):            base_dir/custom_voice/
        # Both layouts exist in practice, so we search both.
        search_paths = []
        
        # 1. If frozen (EXE), check models/ next to the EXE first
        if _is_frozen():
            exe_models = _get_exe_dir() / "models"
            search_paths.append(exe_models / self.model_size / model_type)  # Layout A
            search_paths.append(exe_models / model_type)                    # Layout B
        
        # 2. Check primary cache_dir (may be offline dir or user-specified dir)
        search_paths.append(self.cache_dir / self.model_size / model_type)  # Layout A
        search_paths.append(self.cache_dir / model_type)                    # Layout B
        
        # 3. Always check user home cache as fallback (even in offline mode!)
        #    Users may have downloaded models via download_models.py or normal online use
        if CACHE_DIR != self.cache_dir:
            search_paths.append(CACHE_DIR / self.model_size / model_type)   # Layout A
            search_paths.append(CACHE_DIR / model_type)                     # Layout B
        
        # 4. Also check the OFFLINE_MODELS_DIR if different from cache_dir
        if OFFLINE_MODELS_DIR != self.cache_dir and OFFLINE_MODELS_DIR != CACHE_DIR:
            search_paths.append(OFFLINE_MODELS_DIR / self.model_size / model_type)  # Layout A
            search_paths.append(OFFLINE_MODELS_DIR / model_type)                     # Layout B
        
        # Search all paths
        for model_path in search_paths:
            if _is_valid_model_dir(model_path):
                self._model_paths[model_type] = str(model_path)
                print(f"Found local model {model_type} at: {model_path}")
                return str(model_path)
        
        # Offline mode: fail if model not found anywhere
        if self.offline_mode:
            searched = "\n  ".join(str(p) for p in search_paths)
            raise FileNotFoundError(
                f"模型 {model_type} ({self.model_size}) 未找到。离线模式下需要预先下载模型。\n"
                f"请运行: python download_models.py\n"
                f"或将模型放置到以下任一位置:\n  {searched}"
            )
        
        # Online mode: download from HuggingFace
        print(f"Downloading {model_type} model from {model_id}...")
        model_cache_path = self.cache_dir / model_type
        
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
        Check if a model is already downloaded in any searchable location.
        
        Args:
            model_type: One of 'custom_voice', 'voice_design', 'base', 'tokenizer'
            
        Returns:
            True if model exists locally in any searchable location
        """
        # Build the same search paths as _get_model_path
        search_paths = []
        
        if _is_frozen():
            exe_models = _get_exe_dir() / "models"
            search_paths.append(exe_models / self.model_size / model_type)
            search_paths.append(exe_models / model_type)
        
        search_paths.append(self.cache_dir / self.model_size / model_type)
        search_paths.append(self.cache_dir / model_type)
        
        if CACHE_DIR != self.cache_dir:
            search_paths.append(CACHE_DIR / self.model_size / model_type)
            search_paths.append(CACHE_DIR / model_type)
        
        if OFFLINE_MODELS_DIR != self.cache_dir and OFFLINE_MODELS_DIR != CACHE_DIR:
            search_paths.append(OFFLINE_MODELS_DIR / self.model_size / model_type)
            search_paths.append(OFFLINE_MODELS_DIR / model_type)
        
        return any(_is_valid_model_dir(p) for p in search_paths)
    
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
            "offline_mode": self.offline_mode,
            "frozen": _is_frozen(),
            "downloaded_models": [
                model_type for model_type in MODEL_IDS.keys()
                if self.is_model_downloaded(model_type)
            ]
        }
    
    def set_offline_mode(self, offline: bool) -> None:
        """
        Set offline mode dynamically.
        
        Note: cache_dir always stays as CACHE_DIR (user home cache).
        The _get_model_path method searches multiple locations including
        EXE-adjacent models/ and OFFLINE_MODELS_DIR as fallbacks.
        
        Args:
            offline: Whether to enable offline mode
        """
        self.offline_mode = offline
        # Keep cache_dir as CACHE_DIR so models downloaded in online mode
        # are still found when switching to offline mode.
        # _get_model_path searches multiple locations regardless of cache_dir.
        self._model_paths.clear()
