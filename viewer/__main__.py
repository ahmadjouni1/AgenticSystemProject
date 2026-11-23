"""Open a run in the browser:

    uv run python -m viewer runs/four_bar            # a run folder (with result/)
    uv run python -m viewer runs/four_bar/result/v0000   # or one version folder

Serves on 127.0.0.1 only and opens the default browser. Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from viewer.server import ViewerServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m viewer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run folder, its result folder, or one version folder")
    parser.add_argument("--port", type=int, default=0, help="port (default: any free port)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)
    try:
        server = ViewerServer(args.run, port=args.port)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"cannot open {args.run}: {exc}", file=sys.stderr)
        return 2
    print(f"viewer: {server.url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
