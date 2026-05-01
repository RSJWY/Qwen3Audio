"""Main entry point for the Qwen3-TTS Gradio application."""

from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Iterable, Optional

import torch

from app.tts_engine import TTSEngine
from app.model_manager import ModelManager
from app.ui import create_ui


PRELOAD_MODES = ("custom_voice", "voice_design", "base", "all", "none")
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(slots=True)
class AppConfig:
    mode: str
    port: int
    ip: str
    share: bool
    model_dir: Optional[str]
    dtype: str
    device: str

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_MAP[self.dtype]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Qwen3-TTS Gradio UI.")
    parser.add_argument(
        "--mode",
        choices=PRELOAD_MODES,
        default="none",
        help="Model preset to preload before launching the UI.",
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--ip", default="0.0.0.0", help="Server bind address.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional model directory to use instead of auto-download.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP),
        default="bfloat16",
        help="Torch precision for loaded models.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device identifier, for example cuda:0 or cuda:1.",
    )
    return parser


def parse_args() -> AppConfig:
    args = build_parser().parse_args()
    return AppConfig(
        mode=args.mode,
        port=args.port,
        ip=args.ip,
        share=args.share,
        model_dir=args.model_dir,
        dtype=args.dtype,
        device=args.device,
    )


def ensure_device_available(config: AppConfig) -> AppConfig:
    """Check device availability and auto-fallback if needed."""
    if not config.device.lower().startswith("cuda"):
        return config

    if torch.cuda.is_available():
        return config

    # Auto-fallback to CPU with warning
    print("\n" + "=" * 60)
    print("WARNING: CUDA is not available!")
    print("=" * 60)
    print("Possible causes:")
    print("  1. You have CPU-only PyTorch (most common)")
    print("  2. NVIDIA GPU drivers not installed")
    print("  3. CUDA toolkit version mismatch")
    print()
    print("To fix #1 - Install CUDA-enabled PyTorch:")
    print("  pip uninstall torch torchvision torchaudio")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("Falling back to CPU mode (slower inference)...")
    print("=" * 60 + "\n")
    
    # Return modified config with CPU device
    return AppConfig(
        mode=config.mode,
        port=config.port,
        ip=config.ip,
        share=config.share,
        model_dir=config.model_dir,
        dtype=config.dtype,
        device="cpu",
    )


def create_tts_engine(config: AppConfig) -> TTSEngine:
    """Create TTSEngine with proper configuration."""
    cache_dir = Path(config.model_dir) if config.model_dir else None
    
    model_manager = ModelManager(
        cache_dir=cache_dir,
        device=config.device,
        dtype=config.dtype,
    )
    
    return TTSEngine(model_manager=model_manager)


def preload_targets(mode: str) -> list[str]:
    """Get list of models to preload based on mode."""
    if mode == "all":
        return ["custom_voice", "voice_design", "base"]
    if mode == "none":
        return []
    return [mode]


def preload_models(tts_engine: TTSEngine, targets: Iterable[str]) -> None:
    """Preload specified models."""
    targets = list(targets)
    if not targets:
        return

    print("Pre-loading models...")
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] Loading {target}...")
        tts_engine.ensure_model_loaded(target)

    print("Model pre-load complete.")


def install_signal_handlers(tts_engine: TTSEngine) -> None:
    """Install signal handlers for graceful shutdown."""
    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        tts_engine.unload()
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)


def main() -> int:
    config = parse_args()
    tts_engine = None

    try:
        config = ensure_device_available(config)  # May fallback to CPU
        tts_engine = create_tts_engine(config)
        install_signal_handlers(tts_engine)
        preload_models(tts_engine, preload_targets(config.mode))

        print(f"\nStarting Qwen3-TTS UI on http://{config.ip}:{config.port}")
        print(f"Device: {config.device}")
        if config.share:
            print("Creating public Gradio share link...")
        
        ui = create_ui(tts_engine)
        ui.launch(server_name=config.ip, server_port=config.port, share=config.share)
        return 0
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
        return 0
    except RuntimeError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to launch Qwen3-TTS UI: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if tts_engine is not None:
            tts_engine.unload()


if __name__ == "__main__":
    raise SystemExit(main())
