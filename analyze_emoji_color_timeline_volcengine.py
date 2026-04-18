import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/responses"
DEFAULT_MODEL = "doubao-seed-1-6-vision-250815"


def build_prompt(extra_guidance: str | None = None) -> str:
    lines = [
        "你是一个视频分镜和时间轴分析器。",
        "请分析视频中的 emoji 画面变化和音频结构，并输出一个“emoji 变色持续时间时间轴”。",
        "",
        "任务要求：",
        "1. 识别视频里出现的 emoji 或主要图形符号。",
        "2. 识别每个 emoji 在什么时间段变成什么颜色，持续多久。",
        "3. 同时分析音频结构：节奏段落、能量变化、明显转场、打点、停顿、高潮、收尾。",
        "4. 说明每段颜色变化是否和音频节奏、段落、打点有关。",
        "5. 如果同一个 emoji 在一段时间内保持同一种颜色，就合并成一个连续时间段。",
        "",
        "输出要求：",
        "1. 只输出 JSON，不要 Markdown，不要解释。",
        "2. start_sec 和 end_sec 保留 2 位小数。",
        "3. duration_sec = end_sec - start_sec。",
        "4. confidence 为 0 到 1。",
        "",
        "JSON 顶层结构必须是：",
        "{",
        '  "video_summary": "一句话总结画面变化",',
        '  "audio_summary": "一句话总结音频结构",',
        '  "emoji_color_timeline": [',
        "    {",
        '      "start_sec": 0.00,',
        '      "end_sec": 2.50,',
        '      "duration_sec": 2.50,',
        '      "emoji": "😀",',
        '      "label": "笑脸 emoji",',
        '      "color": "yellow",',
        '      "hex": "#F7D447",',
        '      "visual_change": "从开场持续保持黄色",',
        '      "audio_relation": "对应前奏平稳段",',
        '      "confidence": 0.92',
        "    }",
        "  ],",
        '  "audio_structure": [',
        "    {",
        '      "start_sec": 0.00,',
        '      "end_sec": 2.50,',
        '      "section": "intro",',
        '      "energy": "low",',
        '      "beat_density": "sparse",',
        '      "cue": "前奏铺垫"',
        "    }",
        "  ],",
        '  "sync_observations": [',
        '    "某个 emoji 颜色变化与鼓点同步",',
        '    "某段颜色持续时间覆盖副歌段落"',
        "  ]",
        "}",
        "",
        "颜色命名要求：优先使用基础颜色词，例如 red, orange, yellow, green, cyan, blue, purple, pink, white, black, gray。",
        "如果无法确定精确 hex，可以给近似值，但不要留空。",
        "如果视频里没有清晰 emoji，而是类似 emoji 风格的图形，也按主要符号分析。",
    ]
    if extra_guidance:
        lines.extend(["", "额外要求：", extra_guidance])
    return "\n".join(lines)


def build_payload(video_url: str, model: str, prompt: str) -> dict:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_video",
                        "video_url": video_url,
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }


def run_curl(payload: dict, api_key: str, endpoint: str) -> dict:
    result = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail-with-body",
            endpoint,
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload, ensure_ascii=False),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stdout or result.stderr).strip() or "curl 调用失败")
    return json.loads(result.stdout)


def extract_output_text(response_json: dict) -> str:
    output = response_json.get("output", [])
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"].strip()
    raise RuntimeError("未找到模型文本输出")


def extract_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.splitlines()
        if len(parts) >= 3:
            raw = "\n".join(parts[1:-1]).strip()
            if raw.lower().startswith("json\n"):
                raw = raw[5:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def normalize_analysis(data: dict) -> dict:
    timeline = []
    for item in data.get("emoji_color_timeline", []):
        if not isinstance(item, dict):
            continue
        start = round(float(item.get("start_sec", 0.0)), 2)
        end = round(float(item.get("end_sec", start)), 2)
        timeline.append(
            {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(max(0.0, end - start), 2),
                "emoji": str(item.get("emoji", "")),
                "label": str(item.get("label", "")),
                "color": str(item.get("color", "")),
                "hex": str(item.get("hex", "")),
                "visual_change": str(item.get("visual_change", "")),
                "audio_relation": str(item.get("audio_relation", "")),
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.0)))),
            }
        )
    audio = []
    for item in data.get("audio_structure", []):
        if not isinstance(item, dict):
            continue
        start = round(float(item.get("start_sec", 0.0)), 2)
        end = round(float(item.get("end_sec", start)), 2)
        audio.append(
            {
                "start_sec": start,
                "end_sec": end,
                "section": str(item.get("section", "")),
                "energy": str(item.get("energy", "")),
                "beat_density": str(item.get("beat_density", "")),
                "cue": str(item.get("cue", "")),
            }
        )
    return {
        "video_summary": str(data.get("video_summary", "")),
        "audio_summary": str(data.get("audio_summary", "")),
        "emoji_color_timeline": timeline,
        "audio_structure": audio,
        "sync_observations": [str(x) for x in data.get("sync_observations", []) if str(x).strip()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="使用火山视频模型分析 emoji 颜色时间轴和音频结构。")
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("ARK_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--extra-guidance")
    parser.add_argument("-o", "--output", default="emoji_color_timeline_analysis.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt = build_prompt(args.extra_guidance)
    payload = build_payload(args.video_url, args.model, prompt)

    if args.dry_run:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "payload": payload}, f, ensure_ascii=False, indent=2)
        print(f"dry-run 已输出到 {args.output}")
        return 0

    if not args.api_key:
        print("缺少 ARK API Key", file=sys.stderr)
        return 2

    response_json = run_curl(payload, args.api_key, args.endpoint)
    output_text = extract_output_text(response_json)
    parsed = extract_json(output_text)
    normalized = normalize_analysis(parsed)
    result = {
        "video_url": args.video_url,
        "model": args.model,
        "prompt": prompt,
        "analysis": normalized,
        "raw_response": response_json,
        "raw_output_text": output_text,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"分析完成: {args.output}")
    print(f"视频摘要: {normalized['video_summary']}")
    print(f"音频摘要: {normalized['audio_summary']}")
    print(f"时间段数: {len(normalized['emoji_color_timeline'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
