# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for standalone Adonis binary
# Build: pip install pyinstaller && pyinstaller adonis.spec
# Output: dist/adonis (or adonis.exe on Windows)
# Note: ML models (bge-small, gliner, spacy) are NOT bundled — they download on first run (~280MB) to keep binary lean.
# We bundle templates, prompts, and report assets.

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect fastapi/uvicorn/jinja2 data
datas = [
    ("src/adonis/llm/prompts", "adonis/llm/prompts"),
    ("src/adonis/report/template.html.j2", "adonis/report"),
    ("src/adonis/web/static", "adonis/web/static"),
    ("README.md", "."),
    ("PLAN.md", "."),
    (".env.example", "."),
]

hiddenimports = [
    "adonis.web.app",
    "adonis.cli.doctor",
    "adonis.config",
    "adonis.db",
    "adonis.pipeline.core",
    "adonis.pair.embed",
    "adonis.judge.demo",
    "adonis.verify.demo",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "jinja2",
    "faiss",
    "sentence_transformers",
    "gliner",
    "spacy",
    "rapidfuzz",
]

a = Analysis(
    ["src/adonis/web/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="adonis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="adonis-dist",
)
