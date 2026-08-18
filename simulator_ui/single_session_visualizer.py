#!/usr/bin/env python3
"""Standalone one-session P1B visualizer with no sweep/ranking dependency."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ranked_session_visualizer import LazySessionRuntime, SessionEntry


ROOT = Path(__file__).resolve().parent
HTML_FILE = ROOT / "web_ui/single_session.html"
DEFAULT_DATA_ROOT = Path("/home/user/data/BEV")
DEFAULT_BACKBONE_SOURCE = Path(
    "/home/user/VGGT/method1_train_p1b_nll/vendor/backbone"
)
DEFAULT_BACKBONE_CHECKPOINT = Path(
    "/home/user/VGGT/method1_train_p1b_nll/checkpoints/model.pt"
)


@dataclass
class SingleSessionIndex:
    """Minimal index contract required by the shared inference runtime."""

    entry: SessionEntry
    checkpoint: str

    def __post_init__(self) -> None:
        self.entries = (self.entry,)
        self.by_key = {self.entry.session_key: self.entry}


def _safe_session_path(data_root: Path, session_key: str) -> Path:
    relative = Path(session_key)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe session key")
    path = (data_root / relative).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError("session escapes dataset root")
    if not (path / "COMPLETE").is_file():
        raise FileNotFoundError(f"session is not complete: {path}")
    return path


def _entry(data_root: Path, session_key: str, source_split: str, checkpoint: Path) -> SessionEntry:
    path = _safe_session_path(data_root, session_key)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    history = min(int(metadata["frame_count"]), 10)
    if history < 1:
        raise ValueError("session has no RGB frames")
    scene_id = str(metadata.get("scene_id", path.name))
    return SessionEntry(
        session_key=session_key,
        scene_key=f"hm3d:{scene_id}",
        source_split=source_split,
        selected_for_retraining=False,
        sample_id=f"standalone:{session_key}",
        checkpoint=str(checkpoint),
        reference_frame_id=history - 1,
        history_frame_count=history,
        metrics={},
    )


def _standalone_health(runtime: LazySessionRuntime, session_key: str) -> dict[str, Any]:
    health = runtime.health()
    health.pop("sweep_checkpoint_compatible", None)
    return {
        **health,
        "mode": "standalone_single_session",
        "session_key": session_key,
        "ranking_loaded": False,
        "sweep_metrics_loaded": False,
    }


def make_handler(
    runtime: LazySessionRuntime,
    session_key: str,
    html: bytes,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "P1BSingleSession/1.0"

        def _send_json(
            self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(html)
                    return
                if path == "/health":
                    self._send_json(_standalone_health(runtime, session_key))
                    return
                if path == "/api/session":
                    payload = runtime.session(session_key)
                    payload.pop("sweep_metrics", None)
                    payload.pop("selected_for_retraining", None)
                    self._send_json(payload)
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format_string: str, *args: object) -> None:
            if "/health" not in str(args[0]):
                super().log_message(format_string, *args)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--session-key", required=True)
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--backbone-source", type=Path, default=DEFAULT_BACKBONE_SOURCE
    )
    parser.add_argument(
        "--backbone-checkpoint", type=Path, default=DEFAULT_BACKBONE_CHECKPOINT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8893)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    entry = _entry(data_root, args.session_key, args.source_split, checkpoint)
    index = SingleSessionIndex(entry=entry, checkpoint=str(checkpoint))
    runtime = LazySessionRuntime(
        index=index,
        data_root=data_root,
        checkpoint=checkpoint,
        backbone_source=args.backbone_source,
        backbone_checkpoint=args.backbone_checkpoint,
        device=args.device,
        cache_sessions=1,
        expected_checkpoint_sha256=None,
        enforce_sweep_checkpoint=False,
    )
    html = HTML_FILE.read_bytes()
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(runtime, args.session_key, html)
    )
    server.daemon_threads = True
    print(
        json.dumps(
            {
                **_standalone_health(runtime, args.session_key),
                "url": f"http://{args.host}:{args.port}",
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
