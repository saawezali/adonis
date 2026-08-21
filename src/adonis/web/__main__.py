"""Run the Adonis web console: python -m adonis.web [PORT]"""

from __future__ import annotations

import sys

import uvicorn

from adonis.web.app import app

HOST = "127.0.0.1"


def main() -> None:
    if "--version" in sys.argv or "-V" in sys.argv:
        from adonis import __version__

        print(f"adonis {__version__}")
        return
    if "--help" in sys.argv or "-h" in sys.argv:
        print("usage: adonis [PORT]  |  python -m adonis.web [PORT]")
        print("       adonis --version")
        print(f"  starts console at http://{HOST}:8000/ (override port: adonis 8080)")
        return
    port = 8000
    # allow bare port or --port
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        try:
            port = int(arg)
            break
        except ValueError:
            print(f"usage: adonis [PORT]  (default {HOST}:{port})")
            sys.exit(2)
    print(f"Adonis console: http://{HOST}:{port}/  (report at /report)")
    uvicorn.run(app, host=HOST, port=port)


if __name__ == "__main__":
    main()