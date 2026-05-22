from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BASE_URL = "https://api.stepfun.ai/v1"
DEFAULT_ENDPOINT = f"{DEFAULT_BASE_URL}/chat/completions"
DEFAULT_MODEL = "step-3.6"
DEFAULT_BANK_PATH = Path(__file__).with_name("stepfun_dance_prompt_bank.json")


@dataclass(frozen=True)
class FrameImage:
    index: int
    timestamp_sec: float
    path: Path
    data_url: str


def load_prompt_bank(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dance_motion_prompt(prompt_bank: dict, extra_guidance: str | None = None) -> str:
    lines = [
        "你是一个舞蹈动作分析教练，正在分析由 ffmpeg 从视频抽出的连续图片帧。",
        "请逐帧观察人物姿态，并把连续帧合并成可教学的动作解析。",
        "",
        "核心要求：",
        "1. 只输出 JSON，不要 Markdown，不要解释性前缀。",
        "2. 同时给出 frame_actions（逐帧动作）和 segments（连续动作段）。",
        "3. frame_actions 必须描述当前帧相对前后帧的动作状态，不要只描述衣服、背景或镜头。",
        "4. segments 要把语义一致的连续帧合并，避免每帧都切成独立动作段。",
        "5. 优先判断手腕、手肘、肩、胸、腰胯、膝盖和重心变化。",
        "6. 使用舞蹈教学口吻，给出短句 teaching_cue，适合前端直接展示。",
        "7. 时间字段用秒，保留 2 位小数；confidence 用 0 到 1 的小数。",
        "8. 如果画面无法判断动作，保留该帧但把 confidence 降低，并说明“画面遮挡/姿态不清”。",
        "",
        "常用舞蹈说法：",
    ]

    for item in prompt_bank.get("dance_terms", []):
        term = str(item.get("term", "")).strip()
        meaning = str(item.get("meaning", "")).strip()
        if term and meaning:
            lines.append(f"- {term}: {meaning}")

    lines.extend(["", "动作标签参考："])
    for item in prompt_bank.get("action_labels", []):
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        lines.append(f"- {label}")
        for hint in item.get("guidance", []):
            lines.append(f"  - {hint}")

    if extra_guidance:
        lines.extend(["", "额外要求：", extra_guidance])

    schema_hint = prompt_bank.get("output_schema_hint")
    if schema_hint:
        lines.extend(["", "输出 JSON 结构示例：", json.dumps(schema_hint, ensure_ascii=False, indent=2)])

    return "\n".join(lines)


def ffprobe_video(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "ffprobe 执行失败")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    return streams[0] if streams else {}


def parse_fraction(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    return float(value)


def get_video_fps(video_path: Path) -> float:
    metadata = ffprobe_video(video_path)
    return parse_fraction(metadata.get("avg_frame_rate")) or parse_fraction(metadata.get("r_frame_rate")) or 30.0


def build_ffmpeg_filter(sample_fps: float | None, max_width: int | None) -> str | None:
    filters = []
    if sample_fps:
        filters.append(f"fps={sample_fps}")
    if max_width:
        filters.append(f"scale='min({max_width},iw)':-2")
    return ",".join(filters) if filters else None


def extract_frames_with_ffmpeg(
    video_path: Path,
    frames_dir: Path,
    *,
    sample_fps: float | None = None,
    max_width: int | None = 960,
    image_quality: int = 3,
) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，请先安装 ffmpeg 后再抽帧。")

    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "frame_%06d.jpg"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    vf = build_ffmpeg_filter(sample_fps, max_width)
    if vf:
        cmd.extend(["-vf", vf])
    cmd.extend(["-q:v", str(image_quality), str(pattern)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "ffmpeg 抽帧失败")

    return sorted(frames_dir.glob("frame_*.jpg"))


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_frame_images(frame_paths: Iterable[Path], fps: float) -> list[FrameImage]:
    images = []
    for index, path in enumerate(frame_paths, start=1):
        timestamp = (index - 1) / fps if fps > 0 else 0.0
        images.append(
            FrameImage(
                index=index,
                timestamp_sec=round(timestamp, 3),
                path=path,
                data_url=image_to_data_url(path),
            )
        )
    return images


def chunks(items: list[FrameImage], size: int) -> Iterable[list[FrameImage]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_chat_payload(
    *,
    model: str,
    prompt_text: str,
    images: list[FrameImage],
    detail: str = "high",
    temperature: float = 0.1,
) -> dict:
    image_blocks = [
        {
            "type": "image_url",
            "image_url": {
                "url": image.data_url,
                "detail": detail,
            },
        }
        for image in images
    ]
    frame_index_text = "\n".join(
        f"- frame_index={image.index}, timestamp_sec={image.timestamp_sec:.2f}, file={image.path.name}"
        for image in images
    )
    text_blocks = []
    if frame_index_text:
        text_blocks.append(
            {
                "type": "text",
                "text": f"以下是本批次图片帧的索引和时间：\n{frame_index_text}",
            }
        )
    text_blocks.append({"type": "text", "text": prompt_text})

    return {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "你只输出可解析 JSON。不要输出 Markdown 代码块或额外说明。",
            },
            {
                "role": "user",
                "content": image_blocks + text_blocks,
            },
        ],
    }


def run_curl(payload: dict, api_key: str, endpoint: str) -> dict:
    payload_file = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
            json.dump(payload, f, ensure_ascii=False)
            payload_file = f.name

        cmd = [
            "curl",
            "--silent",
            "--show-error",
            "--fail-with-body",
            endpoint,
            "-H",
            "Content-Type: application/json",
            "-H",
            f"Authorization: Bearer {api_key}",
            "--data-binary",
            f"@{payload_file}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError((result.stdout or result.stderr).strip() or "StepFun 请求失败")
        return json.loads(result.stdout)
    finally:
        if payload_file:
            Path(payload_file).unlink(missing_ok=True)


def extract_output_text(response_json: dict) -> str:
    choices = response_json.get("choices", [])
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            if parts:
                return "\n".join(parts).strip()
    raise RuntimeError("StepFun 返回中未找到 message.content")


def extract_json_from_text(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            raw = "\n".join(lines[1:-1]).strip()
            if raw.lower().startswith("json\n"):
                raw = raw[5:].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("模型返回中未找到 JSON 对象")
        return json.loads(raw[start : end + 1])


def clamp_confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, round(number, 3)))


def normalize_analysis(data: dict) -> dict:
    frame_actions = []
    for item in data.get("frame_actions", []):
        if not isinstance(item, dict):
            continue
        frame_actions.append(
            {
                "frame_index": int(float(item.get("frame_index", 0) or 0)),
                "timestamp_sec": round(float(item.get("timestamp_sec", 0.0) or 0.0), 2),
                "action": str(item.get("action", "")),
                "body_parts": [str(x) for x in item.get("body_parts", []) if str(x).strip()],
                "dance_term": str(item.get("dance_term", "")),
                "direction": str(item.get("direction", "")),
                "teaching_cue": str(item.get("teaching_cue", "")),
                "confidence": clamp_confidence(item.get("confidence", 0.0)),
            }
        )
    frame_actions.sort(key=lambda item: (item["frame_index"], item["timestamp_sec"]))

    segments = []
    for item in data.get("segments", []):
        if not isinstance(item, dict):
            continue
        start = round(float(item.get("start_sec", 0.0) or 0.0), 2)
        end = round(float(item.get("end_sec", start) or start), 2)
        segments.append(
            {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(max(0.0, end - start), 2),
                "label": str(item.get("label", "")),
                "summary": str(item.get("summary", "")),
                "primary_actions": [str(x) for x in item.get("primary_actions", []) if str(x).strip()],
                "dance_terms": [str(x) for x in item.get("dance_terms", []) if str(x).strip()],
                "difficulty": int(float(item.get("difficulty", 1) or 1)),
                "teaching_cues": [str(x) for x in item.get("teaching_cues", []) if str(x).strip()],
                "confidence": clamp_confidence(item.get("confidence", 0.0)),
            }
        )
    segments.sort(key=lambda item: (item["start_sec"], item["end_sec"]))

    return {
        "video_summary": str(data.get("video_summary", "")),
        "frame_actions": frame_actions,
        "segments": segments,
    }


def merge_batch_analyses(batch_results: list[dict]) -> dict:
    merged = {"video_summary": "", "frame_actions": [], "segments": []}
    summaries = []
    for result in batch_results:
        analysis = normalize_analysis(result)
        if analysis["video_summary"]:
            summaries.append(analysis["video_summary"])
        merged["frame_actions"].extend(analysis["frame_actions"])
        merged["segments"].extend(analysis["segments"])
    merged["video_summary"] = "；".join(dict.fromkeys(summaries))
    merged["frame_actions"].sort(key=lambda item: (item["frame_index"], item["timestamp_sec"]))
    merged["segments"].sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    return merged


def analyze_batches(
    *,
    frame_images: list[FrameImage],
    prompt_text: str,
    api_key: str,
    endpoint: str,
    model: str,
    batch_size: int,
    detail: str,
    temperature: float,
) -> tuple[list[dict], dict]:
    raw_batches = []
    parsed_batches = []
    total = len(frame_images)
    for batch_index, batch in enumerate(chunks(frame_images, batch_size), start=1):
        payload = build_chat_payload(
            model=model,
            prompt_text=f"{prompt_text}\n\n当前是第 {batch_index} 批，共 {total} 帧中的 {len(batch)} 帧。请只分析这一批图片。",
            images=batch,
            detail=detail,
            temperature=temperature,
        )
        response_json = run_curl(payload, api_key, endpoint)
        output_text = extract_output_text(response_json)
        parsed = extract_json_from_text(output_text)
        raw_batches.append(
            {
                "batch_index": batch_index,
                "frame_indices": [image.index for image in batch],
                "payload_without_images": {
                    **payload,
                    "messages": [
                        payload["messages"][0],
                        {
                            "role": "user",
                            "content": [
                                item if item.get("type") != "image_url" else {"type": "image_url", "image_url": {"url": "<omitted>", "detail": detail}}
                                for item in payload["messages"][1]["content"]
                            ],
                        },
                    ],
                },
                "raw_response": response_json,
                "raw_output_text": output_text,
            }
        )
        parsed_batches.append(parsed)
    return raw_batches, merge_batch_analyses(parsed_batches)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 ffmpeg 抽帧，并调用 StepFun step-3.6 对舞蹈图片帧做动作解析。"
    )
    parser.add_argument("video", help="本地视频路径")
    parser.add_argument("-o", "--output", default="stepfun_dance_frame_analysis.json", help="输出 JSON 路径")
    parser.add_argument("--api-key", default=os.environ.get("STEPFUN_API_KEY"), help="StepFun API Key；默认读取 STEPFUN_API_KEY")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"请求地址，默认 {DEFAULT_ENDPOINT}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名，默认 {DEFAULT_MODEL}")
    parser.add_argument("--prompt-bank", default=str(DEFAULT_BANK_PATH), help="prompt bank JSON 路径")
    parser.add_argument("--extra-guidance", help="附加提示词")
    parser.add_argument("--sample-fps", type=float, default=None, help="抽帧采样 fps；不传则逐帧抽取")
    parser.add_argument("--max-frames", type=int, default=None, help="最多发送多少帧，默认不限制")
    parser.add_argument("--max-width", type=int, default=960, help="抽帧图片最大宽度，默认 960")
    parser.add_argument("--batch-size", type=int, default=12, help="每次请求发送的图片数量，默认 12")
    parser.add_argument("--detail", default="high", choices=["low", "high", "auto"], help="图片理解 detail")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--keep-frames", action="store_true", help="保留抽帧目录")
    parser.add_argument("--frames-dir", help="指定抽帧目录；不传则使用临时目录")
    parser.add_argument("--dry-run", action="store_true", help="只抽帧并输出 prompt/payload 预览，不请求接口")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"视频不存在: {video_path}", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("--batch-size 必须大于 0", file=sys.stderr)
        return 2

    prompt_bank_path = Path(args.prompt_bank)
    prompt_bank = load_prompt_bank(prompt_bank_path)
    prompt_text = build_dance_motion_prompt(prompt_bank, args.extra_guidance)

    temp_dir = None
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="stepfun_dance_frames_")
        frames_dir = Path(temp_dir.name)

    try:
        source_fps = get_video_fps(video_path)
        frame_fps = args.sample_fps or source_fps
        frame_paths = extract_frames_with_ffmpeg(
            video_path,
            frames_dir,
            sample_fps=args.sample_fps,
            max_width=args.max_width,
        )
        if args.max_frames is not None:
            frame_paths = frame_paths[: args.max_frames]
        frame_images = build_frame_images(frame_paths, frame_fps)

        if not frame_images:
            print("未抽取到任何图片帧", file=sys.stderr)
            return 1

        preview_payload = build_chat_payload(
            model=args.model,
            prompt_text=prompt_text,
            images=frame_images[: min(len(frame_images), args.batch_size)],
            detail=args.detail,
            temperature=args.temperature,
        )
        preview_payload["messages"][1]["content"] = [
            item
            if item.get("type") != "image_url"
            else {"type": "image_url", "image_url": {"url": "<omitted>", "detail": args.detail}}
            for item in preview_payload["messages"][1]["content"]
        ]

        result = {
            "video": str(video_path),
            "endpoint": args.endpoint,
            "model": args.model,
            "prompt_bank_path": str(prompt_bank_path),
            "prompt_text": prompt_text,
            "frame_count": len(frame_images),
            "source_fps": round(source_fps, 3),
            "analysis_fps": round(frame_fps, 3),
            "frames_dir": str(frames_dir),
            "preview_payload": preview_payload,
        }

        if not args.dry_run:
            if not args.api_key:
                print("缺少 StepFun API Key。请设置 STEPFUN_API_KEY 或通过 --api-key 传入。", file=sys.stderr)
                return 2
            raw_batches, analysis = analyze_batches(
                frame_images=frame_images,
                prompt_text=prompt_text,
                api_key=args.api_key,
                endpoint=args.endpoint,
                model=args.model,
                batch_size=args.batch_size,
                detail=args.detail,
                temperature=args.temperature,
            )
            result["raw_batches"] = raw_batches
            result["analysis"] = analysis

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"已输出: {args.output}")
        print(f"抽帧数量: {len(frame_images)}")
        if not args.dry_run and "analysis" in result:
            print(f"动作段数: {len(result['analysis']['segments'])}")
        return 0
    finally:
        if temp_dir and not args.keep_frames:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
