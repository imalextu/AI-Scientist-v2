#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from paper_agent.config import load_config_from_text
from paper_agent.workflow import ThesisWorkflow

SCRIPT_DIR = Path(__file__).resolve().parent
UI_DIR = SCRIPT_DIR / "web_ui"

app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="/ui")

DEFAULTS: Dict[str, str] = {
    "config_path": "",
    "topic_path": "",
    "config_text": "",
    "topic_text": "",
}
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def resolve_input_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (SCRIPT_DIR / candidate).resolve()


def read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def set_defaults(config_path: Path, topic_path: Path) -> None:
    DEFAULTS["config_path"] = str(config_path)
    DEFAULTS["topic_path"] = str(topic_path)
    DEFAULTS["config_text"] = read_optional_text(config_path)
    DEFAULTS["topic_text"] = read_optional_text(topic_path)


def format_sse_event(event: str, payload: Dict[str, Any]) -> str:
    content = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {content}\n\n"


@app.get("/")
def index() -> Response:
    return send_from_directory(UI_DIR, "index.html")


@app.get("/api/initial")
def initial() -> Response:
    return jsonify(DEFAULTS)


@app.post("/api/jobs")
def create_job() -> Response:
    payload = request.get_json(silent=True) or {}
    config_text = str(payload.get("config_text", "")).strip()
    topic_text = str(payload.get("topic_text", "")).strip()
    forced_title_raw = str(payload.get("title", "")).strip()
    forced_title = forced_title_raw or None

    if not config_text:
        return jsonify({"error": "config_text 不能为空"}), 400
    if not topic_text:
        return jsonify({"error": "topic_text 不能为空"}), 400

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "config_text": config_text,
            "topic_text": topic_text,
            "forced_title": forced_title,
        }
    return jsonify({"job_id": job_id})


@app.get("/api/jobs/<job_id>/events")
def stream_job(job_id: str) -> Response:
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
    if not job:
        return jsonify({"error": "任务不存在或已开始执行"}), 404

    def event_stream() -> Any:
        updates: queue.Queue[tuple[str, Dict[str, Any]] | None] = queue.Queue(
            maxsize=512
        )
        stream_closed = threading.Event()

        def emit(event: str, payload: Dict[str, Any]) -> None:
            if stream_closed.is_set():
                return
            try:
                updates.put_nowait((event, payload))
            except queue.Full:
                if event == "stage_delta":
                    return
                try:
                    updates.get_nowait()
                except queue.Empty:
                    return
                try:
                    updates.put_nowait((event, payload))
                except queue.Full:
                    return

        def worker() -> None:
            try:
                config = load_config_from_text(
                    job["config_text"], project_root=SCRIPT_DIR
                )
                workflow = ThesisWorkflow(config, event_handler=emit)
                run_dir = workflow.run(
                    topic_text=job["topic_text"],
                    forced_title=job["forced_title"],
                )
                emit("done", {"run_dir": str(run_dir)})
            except Exception as exc:
                emit("job_error", {"message": str(exc)})
            finally:
                try:
                    updates.put_nowait(None)
                except queue.Full:
                    return

        threading.Thread(target=worker, daemon=True).start()
        yield format_sse_event("status", {"message": "任务已启动"})

        try:
            while True:
                try:
                    item = updates.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue

                if item is None:
                    break
                event, payload = item
                yield format_sse_event(event, payload)
        finally:
            stream_closed.set()

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers=headers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="管理学论文生成器本地 Web 界面")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="默认加载的配置文件路径",
    )
    parser.add_argument(
        "--topic-file",
        type=str,
        default="examples/topic_example.md",
        help="默认加载的题目描述文件路径",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_input_path(args.config)
    topic_path = resolve_input_path(args.topic_file)
    set_defaults(config_path, topic_path)

    if not UI_DIR.exists():
        raise FileNotFoundError(f"未找到前端目录: {UI_DIR}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
