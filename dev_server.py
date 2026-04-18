#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from extract_pose import HAND_MODEL_FILE, POSE_MODEL_FILE, extract_pose_from_video

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
DEFAULT_PORT = 4173

DEFAULT_EXTRACT_CONFIG = {
    "pose_detection_confidence": 0.45,
    "pose_presence_confidence": 0.45,
    "hand_detection_confidence": 0.35,
    "hand_presence_confidence": 0.35,
    "tracking_confidence": 0.35,
    "min_pose_visibility": 0.2,
    "pose_smoothing_window": 5,
    "hand_smoothing_window": 5,
    "interpolate_gap_frames": 4,
    "detection_stride": 1,
    "upsample_min_short_side": 720,
    "upsample_max_long_side": 1600,
    "hand_roi_expansion_factor": 1.35,
    "max_frames": None,
}


def sanitize_filename(raw_name: str | None) -> str:
    decoded = urllib.parse.unquote(raw_name or "")
    basename = os.path.basename(decoded).strip()
    if not basename:
        basename = "upload.mp4"
    safe = "".join(char if char.isalnum() or char in "._- " else "_" for char in basename).strip(" .")
    return safe or "upload.mp4"


class PoseDevServerHandler(SimpleHTTPRequestHandler):
    server_version = "PoseDevServer/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.respond_json({"ok": True, "service": "pose-dev-server"})
            return
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/extract-pose":
            self.respond_json({"ok": False, "error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            content_length = 0

        if content_length <= 0:
            self.respond_json({"ok": False, "error": "上传内容为空。"}, status=HTTPStatus.BAD_REQUEST)
            return

        if content_length > MAX_UPLOAD_BYTES:
            self.respond_json(
                {"ok": False, "error": f"视频过大，请控制在 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 以内。"},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        file_name = sanitize_filename(self.headers.get("X-Filename"))
        suffix = Path(file_name).suffix or ".mp4"

        try:
            with tempfile.TemporaryDirectory(prefix="pose-upload-", dir=str(ROOT / ".tmp")) as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                video_path = tmp_dir_path / f"input{suffix}"
                output_json_path = tmp_dir_path / f"{Path(file_name).stem}_pose.json"

                with video_path.open("wb") as video_file:
                    remaining = content_length
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        video_file.write(chunk)
                        remaining -= len(chunk)

                if video_path.stat().st_size == 0:
                    self.respond_json({"ok": False, "error": "上传内容为空。"}, status=HTTPStatus.BAD_REQUEST)
                    return

                extract_pose_from_video(
                    str(video_path),
                    str(output_json_path),
                    str(ROOT / POSE_MODEL_FILE),
                    str(ROOT / HAND_MODEL_FILE),
                    **DEFAULT_EXTRACT_CONFIG,
                )

                pose_json = json.loads(output_json_path.read_text(encoding="utf-8"))
                pose_json["audio_source"] = file_name
                pose_json["source_video"] = file_name
                self.respond_json(
                    {
                        "ok": True,
                        "json": pose_json,
                        "json_name": output_json_path.name,
                        "source_video": file_name,
                    }
                )
        except Exception as exc:
            self.respond_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def respond_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args():
    parser = argparse.ArgumentParser(description="本地动作跟练开发服务，支持静态页面和短视频转 JSON。")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口，默认 {DEFAULT_PORT}")
    return parser.parse_args()


def main():
    args = parse_args()
    (ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), PoseDevServerHandler)
    print(f"Serving pose viewer on http://{args.host}:{args.port}/pose_viewer.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
