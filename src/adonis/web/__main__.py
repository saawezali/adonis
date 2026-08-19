"""Run the Adonis web console: python -m adonis.web [PORT]"""

from __future__ import annotations

import sys

import uvicorn

from adonis.web.app import app

HOST = "127.0.0.1"


def main() -> None:
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"usage: python -m adonis.web [PORT]  (default {HOST}:{port})")
            sys.exit(2)
    print(f"Adonis console: http://{HOST}:{port}/  (report at /report)")
    uvicorn.run(app, host=HOST, port=port)


if __name__ == "__main__":
    main()