"""The viewer's HTTP server (standard library only), bound to 127.0.0.1.

| Route | What it returns |
| --- | --- |
| ``GET /`` and ``GET /static/...`` | the page, the app, and vendored three.js |
| ``GET /api/versions`` | the run's versions, newest first; ``last_failed`` if the latest compile failed its checks |
| ``GET /api/version/<id>`` | manifest, compile signals, sliders, and mode ("tree" or "loop") |
| ``GET /api/version/<id>/model.glb`` | the geometry, one node per body |
| ``POST /api/version/<id>/pose`` | body ``{joint: value}``; returns body transforms (200), or the reason a loop cannot close (409) |

Tree mechanisms are posed in the browser; only loop mechanisms call ``/pose``.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from sdk.errors import SDKError
from viewer.loader import GLB_FILE, MANIFEST_FILE, LoadedVersion, load_version

STATIC = Path(__file__).parent / "static"
LAST_FAILED = "last_failed"
_VERSION = re.compile(r"v\d{4,}")
# Explicit types: on Windows the registry may map .js to text/plain, and browsers refuse
# to run ES modules served with the wrong type.
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".json": "application/json", ".glb": "model/gltf-binary", ".css": "text/css",
        ".md": "text/plain; charset=utf-8", ".svg": "image/svg+xml"}


def find_versions(root: Path) -> tuple[str, dict[str, Path]]:
    """(title, {version id: folder}) for a run folder, its result folder, or one version."""
    root = Path(root).resolve()
    if (root / MANIFEST_FILE).is_file():
        return root.name, {root.name: root}
    result = root / "result" if (root / "result").is_dir() else root
    title = root.name if result != root else root.parent.name
    found = {p.name: p for p in sorted(result.iterdir(), reverse=True)
             if p.is_dir() and _VERSION.fullmatch(p.name) and (p / MANIFEST_FILE).is_file()}
    failed = result / LAST_FAILED
    if (failed / MANIFEST_FILE).is_file():
        found = {LAST_FAILED: failed, **found}
    return title, found


class Viewer:
    """Holds the run and the loaded models; one per server."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._loaded: dict[str, LoadedVersion] = {}
        self._lock = threading.Lock()

    def versions(self) -> dict[str, Any]:
        title, found = find_versions(self.root)
        entries = []
        for name, folder in found.items():
            signals = folder / "signals.json"
            count = len(json.loads(signals.read_text(encoding="utf-8"))) if signals.is_file() \
                else 0
            entries.append({"id": name, "failed": name == LAST_FAILED, "signals": count,
                            "label": "latest compile (failed checks)" if name == LAST_FAILED
                            else name})
        return {"title": title, "versions": entries}

    def folder(self, version: str) -> Path:
        _title, found = find_versions(self.root)
        if version not in found:
            raise KeyError(version)
        return found[version]

    def load(self, version: str) -> LoadedVersion:
        folder = self.folder(version)
        with self._lock:
            loaded = self._loaded.get(version)
            # last_failed is replaced by every compile: reload when it changed on disk.
            stamp = (folder / MANIFEST_FILE).stat().st_mtime_ns
            if loaded is None or getattr(loaded, "_stamp", None) != stamp:
                loaded = load_version(folder)
                loaded._stamp = stamp  # type: ignore[attr-defined]
                self._loaded[version] = loaded
            return loaded

    def describe(self, version: str) -> dict[str, Any]:
        loaded = self.load(version)
        return {"id": version, "manifest": loaded.manifest, "signals": loaded.signals,
                "drivers": loaded.drivers, "mode": loaded.mode}


def make_handler(viewer: Viewer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # keep the console quiet

        # -- responses --

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), MIME[".json"])

        def _not_found(self) -> None:
            self._json(404, {"error": "not found"})

        # -- routes --

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlsplit(self.path).path)
            try:
                if path in ("/", "/index.html"):
                    return self._static("index.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/api/versions":
                    return self._json(200, viewer.versions())
                match = re.fullmatch(r"/api/version/([\w]+)(/model\.glb)?", path)
                if match:
                    version, glb = match.groups()
                    if glb:
                        data = (viewer.folder(version) / GLB_FILE).read_bytes()
                        return self._send(200, data, MIME[".glb"])
                    return self._json(200, viewer.describe(version))
            except KeyError:
                return self._not_found()
            return self._not_found()

        def do_POST(self) -> None:  # noqa: N802
            match = re.fullmatch(r"/api/version/([\w]+)/pose", urlsplit(self.path).path)
            if not match:
                return self._not_found()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                values = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(values, dict):
                    raise ValueError("send a JSON object of joint values")
                values = {str(k): float(v) for k, v in values.items()}
                loaded = viewer.load(match.group(1))
            except KeyError:
                return self._not_found()
            except (ValueError, TypeError) as exc:
                return self._json(400, {"ok": False, "error": str(exc)})
            try:
                return self._json(200, {"ok": True, **loaded.solve(values)})
            except SDKError as exc:  # LoopClosureError, PoseError: an answer, not a crash
                return self._json(409, {"ok": False, "error": str(exc)})

        def _static(self, relative: str) -> None:
            target = (STATIC / relative).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                return self._not_found()
            self._send(200, target.read_bytes(),
                       MIME.get(target.suffix.lower(), "application/octet-stream"))

    return Handler


class ViewerServer:
    """``ViewerServer(run_dir).start()`` serves in a background thread (tests use this);
    ``serve_forever()`` blocks (the command line uses this)."""

    def __init__(self, root: str | Path, port: int = 0) -> None:
        self.viewer = Viewer(root)
        find_versions(self.viewer.root)  # fail early on a wrong folder
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self.viewer))
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> ViewerServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
