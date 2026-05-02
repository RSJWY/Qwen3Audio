#!/usr/bin/env python
"""
模型预下载脚本 - 用于离线部署前下载所有模型

使用方法:
    python download_models.py                    # 下载所有模型到默认缓存目录
    python download_models.py --output ./models  # 下载到指定目录
    python download_models.py --models custom_voice base  # 只下载指定模型
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

# 模型配置
MODEL_IDS = {
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

MODEL_SIZES = {
    "tokenizer": "~500MB",
    "custom_voice": "~3.5GB",
    "voice_design": "~3.5GB",
    "base": "~3.5GB",
}


def download_model(model_type: str, output_dir: Path) -> bool:
    """
    下载单个模型。
    
    Args:
        model_type: 模型类型 (tokenizer, custom_voice, voice_design, base)
        output_dir: 输出目录
        
    Returns:
        是否成功
    """
    if model_type not in MODEL_IDS:
        print(f"❌ 未知模型类型: {model_type}")
        return False
    
    model_id = MODEL_IDS[model_type]
    model_dir = output_dir / model_type
    
    # 检查是否已下载
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"✓ 模型已存在: {model_type} ({model_dir})")
        return True
    
    print(f"\n{'='*60}")
    print(f"下载模型: {model_type}")
    print(f"HuggingFace ID: {model_id}")
    print(f"预计大小: {MODEL_SIZES[model_type]}")
    print(f"目标目录: {model_dir}")
    print('='*60)
    
    try:
        from huggingface_hub import snapshot_download
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"✅ 成功下载: {model_type}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败 ({model_type}): {e}")
        
        # 尝试 ModelScope 镜像
        try:
            print(f"\n尝试使用 ModelScope 镜像...")
            from modelscope import snapshot_download as ms_snapshot_download
            
            ms_snapshot_download(
                model_id=model_id,
                cache_dir=str(model_dir),
            )
            print(f"✅ 从 ModelScope 成功下载: {model_type}")
            return True
            
        except ImportError:
            print("❌ ModelScope 未安装，请运行: pip install modelscope")
            return False
        except Exception as ms_error:
            print(f"❌ ModelScope 下载也失败: {ms_error}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="下载 Qwen3-TTS 模型用于离线部署",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python download_models.py                       # 下载所有模型到默认缓存目录
    python download_models.py --output ./models     # 下载到指定目录
    python download_models.py --models tokenizer custom_voice base  # 只下载指定模型
    python download_models.py --for-exe             # 为 EXE 打包准备模型
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="模型下载目标目录（默认: ~/.cache/qwen3-tts/）"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=list(MODEL_IDS.keys()),
        default=list(MODEL_IDS.keys()),
        help="要下载的模型（默认: 全部）"
    )
    parser.add_argument(
        "--for-exe",
        action="store_true",
        help="为 EXE 打包准备模型（下载到 ./models/ 目录）"
    )
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.for_exe:
        output_dir = Path(__file__).parent / "models"
    elif args.output:
        output_dir = args.output
    else:
        output_dir = Path.home() / ".cache" / "qwen3-tts"
    
    print("\n" + "="*60)
    print("Qwen3-TTS 模型下载工具")
    print("="*60)
    print(f"目标目录: {output_dir}")
    print(f"下载模型: {', '.join(args.models)}")
    print()
    
    # 计算总大小
    def parse_size(size_str: str) -> float:
        """Parse size string like '~3.5GB' or '~500MB' to float in GB."""
        s = size_str.replace('~', '')
        if 'GB' in s:
            return float(s.replace('GB', '').strip())
        elif 'MB' in s:
            return float(s.replace('MB', '').strip()) / 1000
        return float(s)
    
    total_size = sum(parse_size(MODEL_SIZES[m]) for m in args.models)
    print(f"预计总下载量: ~{total_size:.1f}GB")
    print("请确保网络畅通，下载时间取决于网络速度...")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载模型
    results = {}
    for model_type in args.models:
        results[model_type] = download_model(model_type, output_dir)
    
    # 总结
    print("\n" + "="*60)
    print("下载完成总结")
    print("="*60)
    
    success = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for model_type, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {model_type}")
    
    print()
    print(f"成功: {success}/{len(results)}")
    if failed > 0:
        print(f"失败: {failed}/{len(results)}")
        return 1
    
    print("\n✅ 所有模型下载完成！")
    
    if args.for_exe:
        print(f"\n模型已准备用于 EXE 打包，位于: {output_dir}")
        print("运行 build_exe.bat 进行打包。")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
