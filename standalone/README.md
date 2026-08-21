# Standalone builds

## PyInstaller (recommended for non-Python users)

```bash
pip install pyinstaller
pyinstaller adonis.spec   # → dist/adonis / dist/adonis-dist/
./dist/adonis --version   # adonis 0.1.0
./dist/adonis             # serve on :8000
./dist/adonis 8080        # custom port
```

Models (~280MB: `bge-small-en-v1.5`, `gliner_medium-v2.1`, `en_core_web_sm`) download on first run, not bundled — keeps binary ~90MB.

Distribute per-OS:

```
adonis-0.1.0-linux
adonis-0.1.0-macos
adonis-0.1.0-win.exe
```

## Docker (fallback)

```bash
docker build -t adonis:0.1.0 .
docker run --rm -p 8000:8000 -v $PWD/data:/app/data -v $PWD/reports:/app/reports adonis:0.1.0
# or
docker compose up
```

## Verification

```bash
python -m adonis.cli.doctor
adonis --version
curl http://127.0.0.1:8000/api/status | jq .
```
