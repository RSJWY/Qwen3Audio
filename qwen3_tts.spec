# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Qwen3-TTS"""

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

project_root = Path(SPECPATH)

# Collect data files
datas = [('app/style.css', 'app')]
datas += collect_data_files('gradio', include_py_files=False)
datas += collect_data_files('gradio_client', include_py_files=False)

# Critical packages that need version.txt or metadata
critical_packages = [
    'safehttpx', 'groovy', 'gradio', 'gradio_client', 
    'httpx', 'httpcore', 'h11', 'anyio', 'sniffio',
    'pydantic', 'pydantic_core', 'fastapi', 'starlette',
    'python_multipart', 'sse_starlette', 'uvicorn'
]

# Use copy_metadata for all critical packages
for pkg in critical_packages:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Additionally, scan all site-packages for version.txt files
# This is the most thorough approach
import site
for site_path in site.getsitepackages() + [site.getusersitepackages()]:
    site_dir = Path(site_path)
    if not site_dir.exists():
        continue
    
    # Scan all package directories for version.txt
    for item in site_dir.iterdir():
        if item.is_dir() and not item.name.startswith('_'):
            version_txt = item / 'version.txt'
            if version_txt.exists():
                # Add (source, destination) tuple
                datas.append((str(version_txt), item.name))

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'torch', 'torch.nn', 'torch.cuda', 'torch.utils', 'torch.utils.data',
        'huggingface_hub', 'transformers', 'safetensors',
        'qwen_tts',
        'gradio', 'gradio.blocks', 'gradio.components', 'gradio.layouts', 'gradio.themes',
        'gradio.routes', 'gradio.utils', 'gradio_client', 'gradio_client.documentation',
        'gradio._simple_templates', 'gradio._simple_templates.simpledropdown',
        'safehttpx', 'groovy', 'httpx', 'httpcore', 'httpcore._backends', 'httpcore._backends.sync',
        'h11', 'h11._abnf', 'anyio', 'anyio._backends', 'anyio._core', 'sniffio',
        'soundfile', 'sox', 'scipy', 'scipy.io', 'scipy.io.wavfile', 'numpy',
        'requests', 'urllib3',
        'PIL', 'tqdm', 'packaging', 'packaging.version', 'packaging.specifiers',
        'encodings', 'encodings.utf_8', 'encodings.ascii', 'encodings.cp1252',
        'asyncio', 'asyncio.base_events', 'asyncio.coroutines',
        'json', 'json.decoder', 'json.encoder',
        'typing_extensions', 'pydantic', 'pydantic_core', 'annotated_types',
        'starlette', 'fastapi', 'uvicorn',
        'pandas', 'pandas.core', 'pandas.core.frame',
        'python_multipart', 'sse_starlette',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib.pyplot', 'jupyter', 'IPython', 'notebook'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name='Qwen3-TTS',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
    console=True, disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None, icon=None,
)

coll = COLLECT(exe, a.binaries, a.datas, a.zipfiles, strip=False, upx=True, upx_exclude=[], name='Qwen3-TTS')