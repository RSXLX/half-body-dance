import argparse
import json
import math
from collections import Counter
from pathlib import Path


NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_landmark(points: list, index: int):
    if index >= len(points):
        return None
    point = points[index]
    if not isinstance(point, dict):
        return None
    if point.get("v", 1.0) < 0.2:
        return None
    return point


def center(p1: dict, p2: dict) -> dict:
    return {
        "x": (p1["x"] + p2["x"]) / 2.0,
        "y": (p1["y"] + p2["y"]) / 2.0,
    }


def dist(p1: dict, p2: dict) -> float:
    return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])


def detect_actions(frame: dict) -> list[tuple[str, str]]:
    points = frame.get("pose_landmarks") or []
    if not points:
        return [("❓", "未识别")]

    ls = get_landmark(points, LEFT_SHOULDER)
    rs = get_landmark(points, RIGHT_SHOULDER)
    lw = get_landmark(points, LEFT_WRIST)
    rw = get_landmark(points, RIGHT_WRIST)
    lh = get_landmark(points, LEFT_HIP)
    rh = get_landmark(points, RIGHT_HIP)
    lk = get_landmark(points, LEFT_KNEE)
    rk = get_landmark(points, RIGHT_KNEE)
    la = get_landmark(points, LEFT_ANKLE)
    ra = get_landmark(points, RIGHT_ANKLE)

    core_points = [ls, rs, lh, rh]
    if any(point is None for point in core_points):
        return [("❓", "未识别")]

    shoulder_width = max(dist(ls, rs), 0.01)
    hip_center = center(lh, rh)
    shoulder_center = center(ls, rs)
    actions: list[tuple[str, str]] = []

    if lw and rw and lw["y"] < shoulder_center["y"] and rw["y"] < shoulder_center["y"]:
        actions.append(("🙌", "双手举起"))
    else:
        if lw and lw["y"] < ls["y"]:
            actions.append(("🙋", "左手举起"))
        if rw and rw["y"] < rs["y"]:
            actions.append(("🙋", "右手举起"))

    if lw and rw:
        hand_span = abs(lw["x"] - rw["x"])
        if hand_span > shoulder_width * 1.8:
            actions.append(("👐", "张开双臂"))

    if lk and rk and la and ra:
        hip_to_knee = ((abs(lh["y"] - lk["y"]) + abs(rh["y"] - rk["y"])) / 2.0)
        knee_to_ankle = ((abs(lk["y"] - la["y"]) + abs(rk["y"] - ra["y"])) / 2.0)
        ankle_span = abs(la["x"] - ra["x"])

        if hip_to_knee < knee_to_ankle * 0.75:
            actions.append(("🪑", "下蹲"))

        if ankle_span > shoulder_width * 1.7:
            actions.append(("🦵", "开腿"))

    body_offset = shoulder_center["x"] - hip_center["x"]
    if body_offset > shoulder_width * 0.18:
        actions.append(("↘️", "身体右倾"))
    elif body_offset < -shoulder_width * 0.18:
        actions.append(("↙️", "身体左倾"))

    if not actions:
        actions.append(("🕺", "基础舞动"))

    return actions


def build_segments(frames: list, min_duration: float) -> tuple[list[dict], Counter]:
    segments: list[dict] = []
    emoji_counter: Counter = Counter()

    current = None
    for frame in frames:
        actions = detect_actions(frame)
        emoji_list = [emoji for emoji, _ in actions]
        emojis = "".join(emoji_list)
        labels = [label for _, label in actions]
        time_value = round(float(frame.get("time", 0.0)), 3)

        if current and current["emoji"] == emojis:
            current["end"] = time_value
            current["labels"] = labels
            current["emoji_list"] = emoji_list
            current["frames"] += 1
            continue

        if current:
            segments.append(current)

        current = {
            "start": time_value,
            "end": time_value,
            "emoji": emojis,
            "emoji_list": emoji_list,
            "labels": labels,
            "frames": 1,
        }

    if current:
        segments.append(current)

    filtered_segments = []
    for segment in segments:
        duration = round(segment["end"] - segment["start"], 3)
        if duration < min_duration and filtered_segments:
            prev = filtered_segments[-1]
            prev["end"] = segment["end"]
            prev["frames"] += segment["frames"]
            continue

        filtered_segments.append(segment)

    for segment in filtered_segments:
        segment["duration"] = round(max(segment["end"] - segment["start"], 0.0), 3)
        for emoji in segment["emoji_list"]:
            emoji_counter[emoji] += 1
        segment.pop("emoji_list", None)

    return filtered_segments, emoji_counter


def analyze_pose_json(data: dict, min_duration: float) -> dict:
    frames = data.get("frames") or []
    segments, emoji_counter = build_segments(frames, min_duration=min_duration)

    timeline = " ".join(
        f"[{segment['start']:.2f}-{segment['end']:.2f}s]{segment['emoji']}"
        for segment in segments
    )

    return {
        "source_video": data.get("source_video"),
        "fps": data.get("fps"),
        "frame_count": len(frames),
        "segment_count": len(segments),
        "emoji_summary": dict(emoji_counter),
        "timeline": timeline,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(
        description="分析 MediaPipe Pose JSON，并把动作拆解成 emoji 时间段。"
    )
    parser.add_argument("input_json", help="输入的动作 JSON 路径")
    parser.add_argument(
        "-o",
        "--output",
        help="输出分析结果 JSON，默认写到输入文件同目录",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.3,
        help="最短片段时长，短于该值会并入前一个片段，默认 0.3 秒",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_emoji_analysis.json")
    )

    data = load_json(input_path)
    analysis = analyze_pose_json(data, min_duration=args.min_duration)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"分析完成: {output_path}")
    print(f"共 {analysis['segment_count']} 段动作")
    print(f"时间线: {analysis['timeline'][:300]}")


if __name__ == "__main__":
    main()
