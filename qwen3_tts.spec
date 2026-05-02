# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Qwen3-TTS

Usage:
    pyinstaller qwen3_tts.spec

Or use the build script:
    build_exe.bat (Windows)
    ./build_exe.sh (Linux/Mac)
"""

import sys
from pathlib import Path

# 项目根目录
project_root = Path(SPECPATH)

# 分析入口文件
a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # 包含样式文件
        ('app/style.css', 'app'),
        # 包含模型目录（如果存在）
        # 注意：模型文件很大，建议单独分发或使用 --add-data 动态指定
    ],
    hiddenimports=[
        # PyTorch 相关
        'torch',
        'torch.nn',
        'torch.cuda',
        'torch.utils',
        'torch.utils.data',
        
        # HuggingFace 相关
        'huggingface_hub',
        'transformers',
        'safetensors',
        
        # Qwen-TTS 核心
        'qwen_tts',
        
        # Gradio 相关
        'gradio',
        'gradio.blocks',
        'gradio.components',
        'gradio.layouts',
        'gradio.themes',
        
        # 音频处理
        'soundfile',
        'sox',
        'scipy',
        'numpy',
        
        # 网络相关
        'requests',
        'urllib3',
        
        # 其他可能需要的模块
        'PIL',
        'pillow',
        'tqdm',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
        
        #编码相关
        'encodings',
        'encodings.utf_8',
        'encodings.ascii',
        
        # asyncio 相关（Gradio需要）
        'asyncio',
        'asyncio.base_events',
        'asyncio.coroutines',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的模块以减小体积
        'tkinter',
        'matplotlib',
        'pandas',
        'jupyter',
        'IPython',
        'notebook',
    ],
    noarchive=False,
    optimize=0,
)

# PYZ 压缩包
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# 可执行文件
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Qwen3-TTS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 显示控制台窗口以便查看日志
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标: icon='icon.ico'
)

# 收集所有文件
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    a.zipfiles,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Qwen3-TTS',
)

# 如果需要打包模型文件，取消下面的注释并确保 models/ 目录存在
# models = Tree('models', prefix='models')
# coll += models
