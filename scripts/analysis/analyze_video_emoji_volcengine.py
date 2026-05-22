import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "doubao-1-5-thinking-vision-pro"
DEFAULT_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/responses"
DEFAULT_BANK_PATH = Path(__file__).with_name("volcengine_emoji_prompt_bank.json")


def load_prompt_bank(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_analysis_prompt(prompt_bank: dict, extra_guidance: str | None) -> str:
    candidates = prompt_bank.get("emoji_candidates", [])
    lines = [
        "你是一个视频动作分析器。请分析输入视频中的舞蹈或肢体动作，并把动作转换成 emoji 时间段。",
        "要求：",
        "1. 输出必须是 JSON，不要输出 Markdown，不要加解释性前缀。",
        "2. 以时间段切分视频，识别动作变化点，输出 segments 数组。",
        "3. 每个 segment 必须包含: start_sec, end_sec, emoji, label, reason, confidence。",
        "4. confidence 用 0 到 1 的小数。",
        "5. 可以使用具象或意象化 emoji，不限于人物动作字面含义。",
        "6. 如果手部或动作明显向下压、向下落、向下洒，可以用 🌧️ 表示，即使画面里没有真的下雨。",
        "7. 相邻时间段如果语义一致，尽量合并，不要切得过碎。",
        "8. 优先从候选 emoji 里选；如果实在不匹配，再用 🕺 作为兜底。",
        "9. start_sec 和 end_sec 用秒，保留 2 位小数。",
        "10. 最终 JSON 顶层结构必须包含 video_summary 和 segments。",
        "",
        "候选 emoji 与语义：",
    ]

    for item in candidates:
        emoji = item["emoji"]
        label = item["label"]
        guidance = item.get("guidance", [])
        lines.append(f"- {emoji} {label}")
        for hint in guidance:
            lines.append(f"  - {hint}")

    if extra_guidance:
        lines.extend(["", "额外要求：", extra_guidance])

    example = prompt_bank.get("output_schema_hint")
    if example:
        lines.extend(
            [
                "",
                "输出示例结构：",
                json.dumps(example, ensure_ascii=False, indent=2),
            ]
        )

    return "\n".join(lines)


def build_payload(
    video_url: str,
    model: str,
    sampling_interval: int | None,
    max_sampling_frames: int | None,
    prompt_text: str,
) -> dict:
    video_item = {
        "type": "input_video",
        "video_url": video_url,
    }
    if sampling_interval is not None or max_sampling_frames is not None:
        video_config = {}
        if sampling_interval is not None:
            video_config["sampling_interval"] = sampling_interval
        if max_sampling_frames is not None:
            video_config["max_sampling_frames"] = max_sampling_frames
        video_item["video_analysis_config"] = video_config

    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    video_item,
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    },
                ],
            }
        ],
    }


def run_curl(payload: dict, api_key: str, endpoint: str, insecure: bool) -> dict:
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--http1.1",
        endpoint,
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Authorization: Bearer {api_key}",
        "-d",
        json.dumps(payload, ensure_ascii=False),
    ]
    if insecure:
        cmd.insert(1, "--insecure")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stdout or result.stderr).strip() or "curl 调用失败")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"接口返回不是合法 JSON: {exc}") from exc


def extract_output_text(response_json: dict) -> str:
    output = response_json.get("output", [])
    text_parts = []

    for item in output:
        for content in item.get("content", []):
            content_type = content.get("type")
            if content_type in {"output_text", "text"} and content.get("text"):
                text_parts.append(content["text"])

    if text_parts:
        return "\n".join(text_parts).strip()

    if "output_text" in response_json and response_json["output_text"]:
        return str(response_json["output_text"]).strip()

    raise RuntimeError("未在 responses 返回中找到文本内容")


def extract_json_from_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json\n"):
                text = text[5:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("模型返回中未找到 JSON 对象")

    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"模型返回 JSON 解析失败: {exc}") from exc


def normalize_segments(parsed: dict) -> dict:
    segments = parsed.get("segments")
    if not isinstance(segments, list):
        parsed["segments"] = []
        return parsed

    normalized = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        normalized.append(
            {
                "start_sec": round(float(segment.get("start_sec", 0.0)), 2),
                "end_sec": round(float(segment.get("end_sec", 0.0)), 2),
                "emoji": str(segment.get("emoji", "🕺")),
                "label": str(segment.get("label", "基础舞动")),
                "reason": str(segment.get("reason", "")),
                "confidence": max(0.0, min(1.0, float(segment.get("confidence", 0.0)))),
            }
        )

    parsed["segments"] = normalized
    return parsed


def build_result(
    *,
    video_url: str,
    model: str,
    prompt_bank_path: Path,
    prompt_text: str,
    response_json: dict | None,
    output_text: str | None,
    parsed_json: dict | None,
) -> dict:
    result = {
        "video_url": video_url,
        "model": model,
        "prompt_bank_path": str(prompt_bank_path),
        "prompt_text": prompt_text,
    }
    if response_json is not None:
        result["raw_response"] = response_json
    if output_text is not None:
        result["raw_output_text"] = output_text
    if parsed_json is not None:
        result["analysis"] = normalize_segments(parsed_json)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用火山引擎 doubao-1-5-thinking-vision-pro 分析视频动作并输出 emoji 时间段。"
    )
    parser.add_argument("--video-url", required=True, help="可公网访问的视频 URL")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ARK_API_KEY"),
        help="火山引擎 ARK API Key；默认读取环境变量 ARK_API_KEY",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名，默认 {DEFAULT_MODEL}")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Responses API endpoint")
    parser.add_argument(
        "--sampling-interval",
        type=int,
        default=None,
        help="视频采样间隔（秒）；不传则不发送 video_analysis_config",
    )
    parser.add_argument(
        "--max-sampling-frames",
        type=int,
        default=None,
        help="最大采样帧数；不传则不发送 video_analysis_config",
    )
    parser.add_argument(
        "--prompt-bank",
        default=str(DEFAULT_BANK_PATH),
        help="emoji prompt 库 JSON 文件路径",
    )
    parser.add_argument(
        "--extra-guidance",
        help="附加提示词，会附加到基础 prompt 后面",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="volcengine_video_emoji_analysis.json",
        help="输出结果 JSON 路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成 payload 和 prompt，不实际请求接口",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="给 curl 加 --insecure，遇到本机 TLS 问题时可尝试",
    )
    args = parser.parse_args()

    prompt_bank_path = Path(args.prompt_bank)
    prompt_bank = load_prompt_bank(prompt_bank_path)
    prompt_text = build_analysis_prompt(prompt_bank, args.extra_guidance)
    payload = build_payload(
        video_url=args.video_url,
        model=args.model,
        sampling_interval=args.sampling_interval,
        max_sampling_frames=args.max_sampling_frames,
        prompt_text=prompt_text,
    )

    if args.dry_run:
        result = {
            "prompt_bank_path": str(prompt_bank_path),
            "prompt_text": prompt_text,
            "payload": payload,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"dry-run 已输出到 {args.output}")
        return 0

    if not args.api_key:
        print("缺少 API Key。请设置 ARK_API_KEY 或通过 --api-key 传入。", file=sys.stderr)
        return 2

    response_json = run_curl(payload, args.api_key, args.endpoint, args.insecure)
    output_text = extract_output_text(response_json)
    parsed_json = extract_json_from_text(output_text)
    result = build_result(
        video_url=args.video_url,
        model=args.model,
        prompt_bank_path=prompt_bank_path,
        prompt_text=prompt_text,
        response_json=response_json,
        output_text=output_text,
        parsed_json=parsed_json,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    analysis = result.get("analysis", {})
    segments = analysis.get("segments", [])
    print(f"分析完成: {args.output}")
    print(f"视频摘要: {analysis.get('video_summary', '')}")
    print(f"动作段数: {len(segments)}")
    if segments:
        timeline = " ".join(
            f"[{seg['start_sec']:.2f}-{seg['end_sec']:.2f}s]{seg['emoji']}"
            for seg in segments
        )
        print(f"时间线: {timeline[:500]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
