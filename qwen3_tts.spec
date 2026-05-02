# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Qwen3-TTS"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

project_root = Path(SPECPATH)

# Collect data files from gradio and dependencies
datas = [('app/style.css', 'app')]
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')

# Collect metadata for packages
for pkg in ['safehttpx', 'httpx', 'httpcore', 'h11', 'anyio', 'sniffio', 'pydantic', 'pydantic_core']:
    try:
        datas += copy_metadata(pkg)
    except:
        pass

# Explicitly add safehttpx version.txt (critical for gradio)
import site
site_packages = site.getsitepackages()[0]
for sp in site.getsitepackages():
    version_txt = Path(sp) / 'safehttpx' / 'version.txt'
    if version_txt.exists():
        datas.append((str(version_txt), 'safehttpx'))
        break

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
        'safehttpx', 'httpx', 'httpcore', 'httpcore._backends', 'httpcore._backends.sync',
        'h11', 'h11._abnf', 'anyio', 'anyio._backends', 'anyio._core', 'sniffio',
        'soundfile', 'sox', 'scipy', 'scipy.io', 'scipy.io.wavfile', 'numpy',
        'requests', 'urllib3',
        'PIL', 'tqdm', 'packaging', 'packaging.version', 'packaging.specifiers',
        'encodings', 'encodings.utf_8', 'encodings.ascii', 'encodings.cp1252',
        'asyncio', 'asyncio.base_events', 'asyncio.coroutines',
        'json', 'json.decoder', 'json.encoder',
        'typing_extensions', 'pydantic', 'pydantic_core', 'annotated_types',
        'starlette', 'fastapi', 'uvicorn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pandas', 'jupyter', 'IPython', 'notebook'],
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
