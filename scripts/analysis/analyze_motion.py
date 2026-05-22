#!/usr/bin/env python3
"""Analyze a `*_pose.json` file produced by extract_pose.py and emit a
companion `*_motion.json` describing joint trajectories, motion segments and
hint cues.

Schema is documented in docs/motion-analysis-and-lesson-mode.md §1, mirrored in
TypeScript at web/src/core/motionTypes.ts. Keep both in sync when bumping
schema_version.

Usage:
    python3 scripts/analysis/analyze_motion.py wudao/angel_pose.json
    python3 scripts/analysis/analyze_motion.py wudao/angel_pose.json \
        --output wudao/angel_motion.json --min-segment 0.6 --max-segments 16
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2

# JointId → MediaPipe Pose landmark index. Mirror of JOINT_LANDMARK_INDEX in
# web/src/core/motionTypes.ts.
JOINT_INDEX: dict[str, int] = {
    "leftElbow": 13,
    "rightElbow": 14,
    "leftWrist": 15,
    "rightWrist": 16,
    "leftAnkle": 27,
    "rightAnkle": 28,
}

CARDINALS_8 = [
    "right",
    "upRight",
    "up",
    "upLeft",
    "left",
    "downLeft",
    "down",
    "downRight",
]
"""8-direction labels at 45° increments starting from +x (screen-right).

We FLIP the y-axis when computing direction so that 'up' corresponds to screen
up — natural for users to read."""


def load_pose(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_fps(data: dict[str, Any], frames: list[dict[str, Any]]) -> float:
    fps = data.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return float(fps)
    if len(frames) >= 2:
        dt_avg = (frames[-1].get("time", 0) - frames[0].get("time", 0)) / max(1, len(frames) - 1)
        if dt_avg > 0:
            return 1.0 / dt_avg
    return 30.0


# ---- Pose reference (Python mirror of web/src/core/reference.ts) -------------


def _normalize_vec(x: float, y: float) -> tuple[float, float] | None:
    length = math.hypot(x, y)
    if length == 0:
        return None
    return (x / length, y / length)


def get_pose_reference(points: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not points or len(points) < 29:
        return None
    try:
        ls = points[11]
        rs = points[12]
        lh = points[23]
        rh = points[24]
    except (IndexError, TypeError):
        return None
    if not all((ls, rs, lh, rh)):
        return None

    sc = ((ls["x"] + rs["x"]) / 2.0, (ls["y"] + rs["y"]) / 2.0)
    hc = ((lh["x"] + rh["x"]) / 2.0, (lh["y"] + rh["y"]) / 2.0)
    center = ((sc[0] + hc[0]) / 2.0, (sc[1] + hc[1]) / 2.0)

    shoulder_axis = _normalize_vec(rs["x"] - ls["x"], rs["y"] - ls["y"])
    hip_axis = _normalize_vec(rh["x"] - lh["x"], rh["y"] - lh["y"])
    torso = (hc[0] - sc[0], hc[1] - sc[1])

    x_axis = shoulder_axis or hip_axis or (1.0, 0.0)
    proj = torso[0] * x_axis[0] + torso[1] * x_axis[1]
    y_raw = (torso[0] - proj * x_axis[0], torso[1] - proj * x_axis[1])
    y_axis = _normalize_vec(*y_raw) or _normalize_vec(-x_axis[1], x_axis[0]) or (0.0, 1.0)

    scale = max(
        math.hypot(ls["x"] - rs["x"], ls["y"] - rs["y"]),
        math.hypot(lh["x"] - rh["x"], lh["y"] - rh["y"]),
        math.hypot(sc[0] - hc[0], sc[1] - hc[1]),
        0.001,
    )
    return {"center": center, "scale": scale, "x_axis": x_axis, "y_axis": y_axis}


def project_point(p: dict[str, Any], ref: dict[str, Any]) -> tuple[float, float]:
    nx = (p["x"] - ref["center"][0]) / ref["scale"]
    ny = (p["y"] - ref["center"][1]) / ref["scale"]
    return (nx, ny)


# ---- Trajectory math ---------------------------------------------------------


def smooth(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return values
    half = window // 2
    out: list[float | None] = []
    n = len(values)
    for i in range(n):
        bucket: list[float] = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            v = values[j]
            if v is not None:
                bucket.append(v)
        out.append(sum(bucket) / len(bucket) if bucket else None)
    return out


def quantize_direction(angle_deg: float) -> str:
    """Map an angle in degrees (y-flipped, 0=right) to one of CARDINALS_8."""
    bucket = int(((angle_deg % 360.0) + 22.5) // 45.0) % 8
    return CARDINALS_8[bucket]


def compute_trajectories(
    frames: list[dict[str, Any]],
    fps: float,
    *,
    visibility_threshold: float,
    smoothing_window: int,
    still_speed_threshold: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any] | None]]:
    """Return (trajectories, frame_refs) where frame_refs caches each frame's
    pose reference (for downstream segmentation)."""
    n = len(frames)
    frame_refs: list[dict[str, Any] | None] = []
    # raw_xy[joint][i] = (x, y) | None; visibility[joint][i] = float
    raw: dict[str, list[tuple[float, float] | None]] = {j: [None] * n for j in JOINT_INDEX}
    vis: dict[str, list[float]] = {j: [0.0] * n for j in JOINT_INDEX}
    times: list[float] = [0.0] * n

    for i, frame in enumerate(frames):
        t = float(frame.get("time", i / fps))
        times[i] = t
        pose = frame.get("pose_landmarks") or []
        ref = get_pose_reference(pose)
        frame_refs.append(ref)
        if not ref or not pose:
            continue
        for joint, idx in JOINT_INDEX.items():
            if idx >= len(pose):
                continue
            point = pose[idx]
            if not point:
                continue
            v = float(point.get("visibility", point.get("v", 1.0)))
            vis[joint][i] = v
            if v < visibility_threshold:
                continue
            try:
                x, y = project_point(point, ref)
            except Exception:
                continue
            raw[joint][i] = (x, y)

    trajectories: dict[str, dict[str, Any]] = {}
    for joint in JOINT_INDEX:
        xs: list[float | None] = [p[0] if p else None for p in raw[joint]]
        ys: list[float | None] = [p[1] if p else None for p in raw[joint]]
        sx = smooth(xs, smoothing_window)
        sy = smooth(ys, smoothing_window)

        # Central differences for speed; flip y so direction "up" maps to screen-up.
        samples: list[dict[str, Any]] = []
        speeds: list[float] = []
        for i in range(n):
            if sx[i] is None or sy[i] is None:
                continue
            # neighbor samples — fall back to forward / backward differences at edges
            left = i - 1
            right = i + 1
            while left >= 0 and (sx[left] is None or sy[left] is None):
                left -= 1
            while right < n and (sx[right] is None or sy[right] is None):
                right += 1
            if left < 0 and right >= n:
                vx = vy = 0.0
            elif left < 0:
                dt_seg = max(times[right] - times[i], 1e-6)
                vx = (sx[right] - sx[i]) / dt_seg  # type: ignore[operator]
                vy = (sy[right] - sy[i]) / dt_seg  # type: ignore[operator]
            elif right >= n:
                dt_seg = max(times[i] - times[left], 1e-6)
                vx = (sx[i] - sx[left]) / dt_seg  # type: ignore[operator]
                vy = (sy[i] - sy[left]) / dt_seg  # type: ignore[operator]
            else:
                dt_seg = max(times[right] - times[left], 1e-6)
                vx = (sx[right] - sx[left]) / dt_seg  # type: ignore[operator]
                vy = (sy[right] - sy[left]) / dt_seg  # type: ignore[operator]
            speed = math.hypot(vx, vy)
            angle_flipped = math.degrees(math.atan2(-vy, vx)) % 360.0
            cardinal = "still" if speed < still_speed_threshold else quantize_direction(angle_flipped)
            samples.append(
                {
                    "t": round(times[i], 4),
                    "x": round(sx[i], 5),  # type: ignore[arg-type]
                    "y": round(sy[i], 5),  # type: ignore[arg-type]
                    "vx": round(vx, 5),
                    "vy": round(vy, 5),
                    "speed": round(speed, 5),
                    "direction": round(angle_flipped, 2),
                    "cardinal": cardinal,
                    "isKeyFrame": False,
                    "visibility": round(vis[joint][i], 3),
                }
            )
            speeds.append(speed)

        # Mark key frames: local extrema in speed with min spacing of 0.18s.
        if samples:
            min_gap_idx = max(1, int(0.18 * fps))
            extrema = _find_extrema([s["speed"] for s in samples], min_gap_idx)
            for idx in extrema:
                samples[idx]["isKeyFrame"] = True

        # Stats
        total_dist = 0.0
        for k in range(1, len(samples)):
            total_dist += math.hypot(
                samples[k]["x"] - samples[k - 1]["x"],
                samples[k]["y"] - samples[k - 1]["y"],
            )
        peak = max(speeds, default=0.0)
        if speeds:
            mean = sum(speeds) / len(speeds)
            variance = sum((s - mean) ** 2 for s in speeds) / len(speeds)
        else:
            variance = 0.0

        trajectories[joint] = {
            "joint": joint,
            "samples": samples,
            "stats": {
                "totalDistance": round(total_dist, 4),
                "peakSpeed": round(peak, 4),
                "speedVariance": round(variance, 5),
                "coverage": round(len(samples) / max(1, n), 3),
            },
        }

    return trajectories, frame_refs


def _find_extrema(values: list[float], min_gap: int) -> list[int]:
    """Return indices of local maxima/minima with at least min_gap separation."""
    if len(values) < 3:
        return []
    candidates: list[int] = []
    for i in range(1, len(values) - 1):
        v = values[i]
        if (v > values[i - 1] and v >= values[i + 1]) or (v < values[i - 1] and v <= values[i + 1]):
            candidates.append(i)
    if not candidates:
        return []
    chosen: list[int] = []
    for idx in candidates:
        if not chosen or idx - chosen[-1] >= min_gap:
            chosen.append(idx)
    return chosen


# ---- Segmentation ------------------------------------------------------------


def _frame_dominant_direction(
    trajectories: dict[str, dict[str, Any]], time_index: dict[str, dict[float, dict[str, Any]]]
) -> Any:
    return None  # not used; placeholder for future DBSCAN extension.


def segment_motion(
    trajectories: dict[str, dict[str, Any]],
    duration: float,
    *,
    min_segment_duration: float,
    max_segments: int,
    beat_times: list[float] | None,
) -> list[dict[str, Any]]:
    """Rule-based v1 segmentation.

    Strategy:
      1. Pick the most-active trajectory (peakSpeed × coverage) as the spine.
      2. Find speed valleys on the spine (local minima below 35% of peak):
         each valley is a natural "hold" point and a strong segment boundary.
      3. Walk between valleys; if the dominant cardinal flips within the gap,
         insert a secondary cut at the flip point.
      4. Snap edges to nearest beat (±150 ms) when beat_times is provided.
      5. Merge segments shorter than `min_segment_duration` into the previous.
      6. Greedy-merge until len <= max_segments.
    """
    ranked = sorted(
        trajectories.items(),
        key=lambda kv: kv[1]["stats"]["peakSpeed"] * kv[1]["stats"]["coverage"],
        reverse=True,
    )
    if not ranked:
        return []
    primary_joint, primary_traj = ranked[0]
    samples = primary_traj["samples"]
    if not samples:
        return []

    # Step 2: speed valleys on the spine
    speeds = [s["speed"] for s in samples]
    times = [s["t"] for s in samples]
    peak = max(speeds) if speeds else 0.0
    valley_threshold = peak * 0.35
    min_gap_idx = max(3, int(0.4 * len(samples) / max(times[-1] - times[0], 1.0)))  # ~0.4s
    valley_idx: list[int] = []
    for i in range(1, len(speeds) - 1):
        if speeds[i] >= valley_threshold:
            continue
        if speeds[i] <= speeds[i - 1] and speeds[i] <= speeds[i + 1]:
            if not valley_idx or i - valley_idx[-1] >= min_gap_idx:
                valley_idx.append(i)

    # Build coarse boundaries from valleys (clip to [0, duration])
    boundaries: list[float] = [0.0]
    for vi in valley_idx:
        boundaries.append(times[vi])
    boundaries.append(duration)

    # Dedup + sort
    boundaries = sorted(set(round(b, 3) for b in boundaries))

    # Step 3: secondary cuts on cardinal flip inside long valley-to-valley gaps
    bin_size = 0.25
    refined: list[float] = []
    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        refined.append(a)
        if b - a < min_segment_duration * 2:
            continue
        # bin samples into 0.25s slots, take dominant non-still cardinal
        slot_dirs: list[tuple[float, str]] = []
        cur_start = a
        bucket: list[str] = []
        for s in samples:
            if s["t"] < a:
                continue
            if s["t"] >= b:
                break
            if s["t"] >= cur_start + bin_size:
                if bucket:
                    slot_dirs.append((cur_start, _majority(bucket)))
                cur_start = math.floor(s["t"] / bin_size) * bin_size
                bucket = []
            bucket.append(s["cardinal"])
        if bucket:
            slot_dirs.append((cur_start, _majority(bucket)))
        # Insert cuts where dominant flips between consecutive non-still slots
        last_dir = "still"
        for slot_t, slot_dir in slot_dirs:
            if slot_dir == "still":
                continue
            if last_dir != "still" and slot_dir != last_dir:
                refined.append(slot_t)
            last_dir = slot_dir
    refined.append(duration)
    refined = sorted(set(round(b, 3) for b in refined))

    # Build segments
    segments: list[tuple[float, float, str]] = []
    for i in range(len(refined) - 1):
        s_start = refined[i]
        s_end = refined[i + 1]
        if s_end <= s_start:
            continue
        # Dominant cardinal within span
        in_span = [s["cardinal"] for s in samples if s_start <= s["t"] < s_end]
        if not in_span:
            continue
        dom = _majority(in_span)
        segments.append((s_start, s_end, dom))

    if not segments:
        return []

    # Step 4: beat snapping
    if beat_times:
        snapped: list[tuple[float, float, str]] = []
        for s, e, c in segments:
            ns = _snap_to_beat(s, beat_times, tol=0.15)
            ne = _snap_to_beat(e, beat_times, tol=0.15)
            if ne <= ns:
                ne = e
            snapped.append((ns, ne, c))
        segments = snapped

    # Step 5: merge short segments forward
    merged: list[list[Any]] = []
    for s, e, c in segments:
        if merged and (e - s) < min_segment_duration:
            merged[-1][1] = e  # extend previous
        else:
            merged.append([s, e, c])
    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_segment_duration:
        merged[-2][1] = merged[-1][1]
        merged.pop()

    # Step 6: cap segment count by greedy merge of shortest adjacent pair
    while len(merged) > max_segments:
        idx = min(range(len(merged)), key=lambda i: merged[i][1] - merged[i][0])
        if idx == len(merged) - 1:
            merged[idx - 1][1] = merged[idx][1]
            merged.pop(idx)
        else:
            merged[idx + 1][0] = merged[idx][0]
            merged.pop(idx)

    # Emit MotionSegment dicts
    out: list[dict[str, Any]] = []
    for i, (s, e, c) in enumerate(merged):
        seg = build_segment(i, s, e, c, primary_joint, trajectories, beat_times)
        out.append(seg)
    return out


def _majority(items: list[str]) -> str:
    counts: dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    # 'still' loses ties so a real direction wins
    return max(counts.keys(), key=lambda k: (counts[k] - (0.5 if k == "still" else 0)))


def _snap_to_beat(t: float, beats: list[float], *, tol: float) -> float:
    if not beats:
        return t
    best = min(beats, key=lambda b: abs(b - t))
    return best if abs(best - t) <= tol else t


# ---- Segment metadata --------------------------------------------------------

# Template table for description / tip generation. Picked deliberately wide:
# (primary_joint, cardinal) keys; otherwise fall back on cardinal.
_TEMPLATES: dict[tuple[str, str], dict[str, str]] = {
    ("rightWrist", "up"): {
        "title": "右手由下向上挥",
        "tip": "肩膀放松，手腕领先",
    },
    ("rightWrist", "right"): {
        "title": "右手向右摆",
        "tip": "手肘高度不要塌",
    },
    ("rightWrist", "left"): {
        "title": "右手收回向左",
        "tip": "顺势带动身体重心",
    },
    ("rightWrist", "down"): {
        "title": "右手向下落",
        "tip": "落到位再起，不要顶肩",
    },
    ("leftWrist", "up"): {
        "title": "左手由下向上挥",
        "tip": "肩膀放松，手腕领先",
    },
    ("leftWrist", "left"): {
        "title": "左手向左摆",
        "tip": "手肘高度不要塌",
    },
    ("leftWrist", "right"): {
        "title": "左手收回向右",
        "tip": "顺势带动身体重心",
    },
    ("leftWrist", "down"): {
        "title": "左手向下落",
        "tip": "落到位再起，不要顶肩",
    },
    ("leftAnkle", "left"): {
        "title": "左脚向左跨",
        "tip": "膝盖弹起，脚尖落地",
    },
    ("rightAnkle", "right"): {
        "title": "右脚向右跨",
        "tip": "膝盖弹起，脚尖落地",
    },
}

_CARDINAL_FALLBACK: dict[str, dict[str, str]] = {
    "up": {"title": "动作向上", "tip": "手腕领先，眼随手走"},
    "down": {"title": "动作向下", "tip": "落到位再起，不要顶肩"},
    "left": {"title": "动作向左", "tip": "重心微沉，跟住节奏"},
    "right": {"title": "动作向右", "tip": "重心微沉，跟住节奏"},
    "upRight": {"title": "向右上发力", "tip": "用腰带动手臂"},
    "upLeft": {"title": "向左上发力", "tip": "用腰带动手臂"},
    "downRight": {"title": "向右下收回", "tip": "顺势放松"},
    "downLeft": {"title": "向左下收回", "tip": "顺势放松"},
    "still": {"title": "保持静止", "tip": "卡准节拍稳住"},
    "mixed": {"title": "节奏过渡", "tip": "跟住节拍，听准下一拍"},
}

_CARDINAL_EMOJI: dict[str, str] = {
    "up": "⬆️",
    "down": "⬇️",
    "left": "👈",
    "right": "👉",
    "upRight": "↗️",
    "upLeft": "↖️",
    "downRight": "↘️",
    "downLeft": "↙️",
    "still": "⏸",
    "mixed": "🎵",
}


def build_segment(
    index: int,
    start: float,
    end: float,
    cardinal: str,
    primary_joint_hint: str,
    trajectories: dict[str, dict[str, Any]],
    beat_times: list[float] | None,
) -> dict[str, Any]:
    duration = end - start
    # Determine primary joints by total in-segment distance.
    joint_distance: list[tuple[str, float]] = []
    for joint, traj in trajectories.items():
        in_seg = [s for s in traj["samples"] if start <= s["t"] < end]
        dist = 0.0
        for k in range(1, len(in_seg)):
            dist += math.hypot(
                in_seg[k]["x"] - in_seg[k - 1]["x"], in_seg[k]["y"] - in_seg[k - 1]["y"]
            )
        joint_distance.append((joint, dist))
    joint_distance.sort(key=lambda kv: kv[1], reverse=True)
    if not joint_distance or joint_distance[0][1] == 0:
        primary_joints = [primary_joint_hint]
    else:
        primary_joints = [j for j, d in joint_distance[:2] if d > 0]

    # Direction histogram inside the segment. The segmenter already passed in a
    # `cardinal` derived from the spine; we only fall back to recomputing when
    # that hint is "still" or "mixed".
    direction_hist: dict[str, int] = {}
    for joint in primary_joints:
        for s in trajectories[joint]["samples"]:
            if start <= s["t"] < end:
                direction_hist[s["cardinal"]] = direction_hist.get(s["cardinal"], 0) + 1
    total_dir = sum(direction_hist.values()) or 1
    sorted_dir = sorted(
        ((c, n / total_dir) for c, n in direction_hist.items() if c != "still"),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if cardinal not in (None, "still", "mixed") and direction_hist.get(cardinal, 0) > 0:
        primary_dir = cardinal
    elif sorted_dir and sorted_dir[0][1] >= 0.20:
        primary_dir = sorted_dir[0][0]
    elif direction_hist.get("still", 0) / total_dir > 0.55:
        primary_dir = "still"
    elif sorted_dir:
        # Even at low share, prefer naming a direction over "mixed" so the user
        # gets actionable copy. Reserve "mixed" for genuinely chaotic windows.
        primary_dir = sorted_dir[0][0]
    else:
        primary_dir = "mixed"

    # Difficulty: 4-factor heuristic, calibrated against the wudao/ corpus so
    # the typical segment lands on tier 2.
    direction_changes = 0
    for joint in primary_joints:
        prev = None
        for s in trajectories[joint]["samples"]:
            if not (start <= s["t"] < end):
                continue
            if s["cardinal"] != "still" and s["cardinal"] != prev:
                direction_changes += 1
                prev = s["cardinal"]
    peak_speed = 0.0
    for joint in primary_joints:
        peak_speed = max(
            peak_speed,
            max(
                (s["speed"] for s in trajectories[joint]["samples"] if start <= s["t"] < end),
                default=0.0,
            ),
        )
    active_joints = sum(1 for j, d in joint_distance if d > 0.05)
    inv_dur = 1.0 / max(duration, 0.2)

    score = (
        0.4 * min(1.0, direction_changes / 12.0)
        + 0.3 * min(1.0, peak_speed / 12.0)
        + 0.2 * min(1.0, active_joints / 6.0)
        + 0.1 * min(1.0, inv_dur / 5.0)
    )
    difficulty = 1 if score < 0.35 else (2 if score < 0.65 else 3)

    # Pick template
    primary_joint = primary_joints[0] if primary_joints else primary_joint_hint
    tpl = _TEMPLATES.get((primary_joint, primary_dir)) or _CARDINAL_FALLBACK.get(primary_dir, _CARDINAL_FALLBACK["mixed"])
    title = tpl["title"]
    tip = tpl["tip"]
    emoji = _CARDINAL_EMOJI.get(primary_dir, "🎵")

    # Description
    if primary_dir == "mixed":
        description = f"这一段以 {duration:.1f}s 完成节奏过渡，动作跨多个方向，按节拍跟住即可。"
    elif primary_dir == "still":
        description = f"这一段共 {duration:.1f}s，动作保持静止；需要稳住身体卡准节拍。"
    else:
        joint_label = _JOINT_ZH.get(primary_joint, "主导关节")
        description = f"这一段约 {duration:.1f}s，{joint_label}主要朝 {_CARDINAL_ZH.get(primary_dir, primary_dir)} 移动。"

    # Key frames (timestamps, dedup, sorted)
    key_frame_set: set[float] = set()
    for joint in primary_joints:
        for s in trajectories[joint]["samples"]:
            if s.get("isKeyFrame") and start <= s["t"] < end:
                key_frame_set.add(round(s["t"], 3))
    key_frames = sorted(key_frame_set)

    beat_indices = None
    if beat_times:
        beat_indices = [i for i, b in enumerate(beat_times) if start <= b < end]

    return {
        "id": f"seg-{index:03d}",
        "index": index,
        "startTime": round(start, 3),
        "endTime": round(end, 3),
        "duration": round(duration, 3),
        "emoji": emoji,
        "title": title,
        "description": description,
        "primaryJoints": primary_joints,
        "primaryDirection": primary_dir,
        "difficulty": difficulty,
        "keyFrames": key_frames,
        "tips": [tip],
        **({"beatIndices": beat_indices} if beat_indices else {}),
    }


_JOINT_ZH = {
    "leftWrist": "左手腕",
    "rightWrist": "右手腕",
    "leftElbow": "左手肘",
    "rightElbow": "右手肘",
    "leftAnkle": "左脚踝",
    "rightAnkle": "右脚踝",
}

_CARDINAL_ZH = {
    "up": "上方",
    "down": "下方",
    "left": "左侧",
    "right": "右侧",
    "upRight": "右上",
    "upLeft": "左上",
    "downRight": "右下",
    "downLeft": "左下",
    "still": "原地",
    "mixed": "多方向",
}


# ---- Hints -------------------------------------------------------------------


def build_hints(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for seg in segments:
        lead_ms = max(220.0, min(380.0, seg["duration"] * 1000.0 * 0.25))
        trigger = max(0.0, seg["startTime"] - lead_ms / 1000.0)
        hints.append(
            {
                "triggerTime": round(trigger, 3),
                "segmentId": seg["id"],
                "preview": f"下一拍：{seg['emoji']} {seg['title']}",
                "cue": "beatPrep" if seg.get("beatIndices") else "sectionStart",
                "leadMs": round(lead_ms, 1),
            }
        )
    return hints


# ---- Summary -----------------------------------------------------------------


def build_summary(
    trajectories: dict[str, dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    bpm: float | None,
    beat_times: list[float] | None,
) -> dict[str, Any]:
    primary_joints = [
        j
        for j, _ in sorted(
            trajectories.items(),
            key=lambda kv: kv[1]["stats"]["peakSpeed"] * kv[1]["stats"]["coverage"],
            reverse=True,
        )[:2]
    ]
    direction_counts: dict[str, int] = {}
    for seg in segments:
        d = seg["primaryDirection"]
        if d in ("still", "mixed"):
            continue
        direction_counts[d] = direction_counts.get(d, 0) + 1
    total = sum(direction_counts.values()) or 1
    dominant = sorted(
        (
            {"cardinal": c, "share": round(n / total, 3)}
            for c, n in direction_counts.items()
        ),
        key=lambda kv: kv["share"],
        reverse=True,
    )
    out: dict[str, Any] = {
        "primaryJoints": primary_joints,
        "dominantDirections": dominant,
    }
    if bpm:
        out["bpm"] = bpm
    if beat_times:
        out["beatTimes"] = [round(b, 3) for b in beat_times]
    return out


# ---- Per-beat slicing (schema v2) -------------------------------------------

# Pose-landmark indices reused for arm/leg pose detail.
_SHOULDER_IDX = {"left": 11, "right": 12}
_ELBOW_IDX = {"leftElbow": 13, "rightElbow": 14}
_WRIST_IDX = {"leftWrist": 15, "rightWrist": 16}
_HIP_IDX = {"left": 23, "right": 24}
_KNEE_IDX = {"leftAnkle": 25, "rightAnkle": 26}  # knees paired with each ankle
_ANKLE_IDX = {"leftAnkle": 27, "rightAnkle": 28}


def _samples_in(traj: dict[str, Any], start: float, end: float) -> list[dict[str, Any]]:
    return [s for s in traj["samples"] if start <= s["t"] < end]


def _displacement_distance(samples: list[dict[str, Any]]) -> tuple[float, tuple[float, float]]:
    """Return (path_length, net_vector). Net vector = end-position minus start-position."""
    if len(samples) < 2:
        return 0.0, (0.0, 0.0)
    path = 0.0
    for k in range(1, len(samples)):
        path += math.hypot(
            samples[k]["x"] - samples[k - 1]["x"],
            samples[k]["y"] - samples[k - 1]["y"],
        )
    net = (samples[-1]["x"] - samples[0]["x"], samples[-1]["y"] - samples[0]["y"])
    return path, net


def _avg_visibility(samples: list[dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    return sum(s.get("visibility", 0.0) for s in samples) / len(samples)


def build_beats(
    beat_times: list[float],
    duration: float,
    trajectories: dict[str, dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    still_speed_threshold: float,
) -> list[dict[str, Any]]:
    """Slice trajectories by beat boundaries; one entry per beat window.

    For each [beat_times[i], beat_times[i+1]) window we pick the joint with
    largest visibility-weighted net displacement and label its dominant
    cardinal. The last beat extends to `duration`.
    """
    if not beat_times:
        return []
    boundaries = list(beat_times)
    if boundaries[-1] < duration:
        boundaries.append(duration)
    out: list[dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        joint_stats: dict[str, dict[str, Any]] = {}
        best_joint: str | None = None
        best_score = -1.0
        for joint, traj in trajectories.items():
            inseg = _samples_in(traj, start, end)
            if not inseg:
                joint_stats[joint] = {
                    "distance": 0.0,
                    "peakSpeed": 0.0,
                    "cardinal": "still",
                    "visibility": 0.0,
                }
                continue
            path, net = _displacement_distance(inseg)
            net_len = math.hypot(*net)
            peak = max(s["speed"] for s in inseg)
            vis = _avg_visibility(inseg)
            if peak < still_speed_threshold:
                cardinal = "still"
                angle = None
            else:
                # Use net displacement direction when it dominates path; otherwise majority-vote.
                if net_len > 0.4 * path and net_len > 0.05:
                    angle = math.degrees(math.atan2(-net[1], net[0])) % 360.0
                    cardinal = quantize_direction(angle)
                else:
                    cardinal = _majority([s["cardinal"] for s in inseg])
                    angle = None
            joint_stats[joint] = {
                "distance": round(net_len, 4),
                "pathLength": round(path, 4),
                "peakSpeed": round(peak, 4),
                "cardinal": cardinal,
                "visibility": round(vis, 3),
                **({"angleDeg": round(angle, 1)} if angle is not None else {}),
            }
            score = net_len * max(vis, 0.0)
            if score > best_score:
                best_score = score
                best_joint = joint

        if best_joint is None or best_score <= 0:
            primary_joint = max(joint_stats, key=lambda j: joint_stats[j]["peakSpeed"])
            primary_dir = "still"
        else:
            primary_joint = best_joint
            primary_dir = joint_stats[best_joint]["cardinal"]

        tpl = _TEMPLATES.get((primary_joint, primary_dir)) or _CARDINAL_FALLBACK.get(
            primary_dir, _CARDINAL_FALLBACK["mixed"]
        )
        emoji = _CARDINAL_EMOJI.get(primary_dir, "🎵")

        # Tie back to enclosing segment by start time.
        seg_id = None
        for seg in segments:
            if seg["startTime"] <= start < seg["endTime"]:
                seg_id = seg["id"]
                break

        # Visibility warning if primary joint coverage is poor.
        warn = None
        prim_vis = joint_stats[primary_joint]["visibility"]
        if primary_joint in ("leftAnkle", "rightAnkle") and prim_vis < 0.4:
            warn = "lowFoot"
        elif primary_joint in ("leftWrist", "rightWrist") and prim_vis < 0.4:
            warn = "lowHand"

        out.append(
            {
                "index": i,
                "startTime": round(start, 3),
                "endTime": round(end, 3),
                "duration": round(end - start, 3),
                **({"segmentId": seg_id} if seg_id else {}),
                "primaryJoint": primary_joint,
                "primaryDirection": primary_dir,
                "emoji": emoji,
                "label": tpl["title"],
                "jointStats": joint_stats,
                **({"visibilityWarning": warn} if warn else {}),
            }
        )
    return out


def _frame_at_time(frames: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    if not frames:
        return None
    lo, hi = 0, len(frames) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frames[mid].get("time", 0.0) < t:
            lo = mid + 1
        else:
            hi = mid
    return frames[lo]


def _arm_pose(pose: list[dict[str, Any]], side: str) -> dict[str, Any] | None:
    if not pose or len(pose) < 33:
        return None
    s = pose[_SHOULDER_IDX[side]]
    e = pose[_ELBOW_IDX[f"{side}Elbow"]]
    w = pose[_WRIST_IDX[f"{side}Wrist"]]
    h = pose[_HIP_IDX[side]]
    if not all((s, e, w, h)):
        return None
    a = (s["x"] - e["x"], s["y"] - e["y"])
    b = (w["x"] - e["x"], w["y"] - e["y"])
    la = math.hypot(*a) or 1e-6
    lb = math.hypot(*b) or 1e-6
    cos = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (la * lb)))
    elbow_angle = math.degrees(math.acos(cos))
    # Wrist height bands (image y grows downward).
    sy = s["y"]
    hy = h["y"]
    midy = (sy + hy) / 2.0
    wy = w["y"]
    if wy < sy - 0.10:
        band = "overhead"
    elif wy < midy:
        band = "high"
    elif wy < hy:
        band = "mid"
    else:
        band = "low"
    return {"elbowAngleDeg": round(elbow_angle, 1), "wristHeightBand": band}


def _leg_pose(pose: list[dict[str, Any]], side: str) -> dict[str, Any] | None:
    if not pose or len(pose) < 33:
        return None
    h = pose[_HIP_IDX[side]]
    k = pose[_KNEE_IDX[f"{side}Ankle"]]
    a = pose[_ANKLE_IDX[f"{side}Ankle"]]
    other_h = pose[_HIP_IDX["right" if side == "left" else "left"]]
    if not all((h, k, a, other_h)):
        return None
    v1 = (h["x"] - k["x"], h["y"] - k["y"])
    v2 = (a["x"] - k["x"], a["y"] - k["y"])
    l1 = math.hypot(*v1) or 1e-6
    l2 = math.hypot(*v2) or 1e-6
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
    knee_angle = math.degrees(math.acos(cos))
    knee_bent = knee_angle < 150.0
    hip_center_x = (h["x"] + other_h["x"]) / 2.0
    if abs(a["x"] - hip_center_x) < 0.04:
        ankle_side = "center"
    elif (side == "left" and a["x"] < hip_center_x) or (side == "right" and a["x"] > hip_center_x):
        ankle_side = "outside"
    else:
        ankle_side = "inside"
    return {"kneeBent": knee_bent, "ankleSide": ankle_side}


def build_limbs(
    beats: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Per-joint per-beat slice. Adds armPose / legPose computed from the
    raw frame at the beat's mid-point."""
    limbs: dict[str, list[dict[str, Any]]] = {j: [] for j in trajectories}
    for beat in beats:
        mid_t = (beat["startTime"] + beat["endTime"]) / 2.0
        frame = _frame_at_time(frames, mid_t)
        pose = (frame or {}).get("pose_landmarks") or []
        for joint in trajectories:
            js = beat["jointStats"].get(joint, {})
            slice_obj: dict[str, Any] = {
                "beatIndex": beat["index"],
                "startTime": beat["startTime"],
                "endTime": beat["endTime"],
                "cardinal": js.get("cardinal", "still"),
                "emoji": _CARDINAL_EMOJI.get(js.get("cardinal", "still"), "🎵"),
                "distance": js.get("distance", 0.0),
                "peakSpeed": js.get("peakSpeed", 0.0),
            }
            if joint in ("leftWrist", "rightWrist"):
                arm = _arm_pose(pose, "left" if joint == "leftWrist" else "right")
                if arm:
                    slice_obj["armPose"] = arm
            elif joint in ("leftAnkle", "rightAnkle"):
                leg = _leg_pose(pose, "left" if joint == "leftAnkle" else "right")
                if leg:
                    slice_obj["legPose"] = leg
            limbs[joint].append(slice_obj)
    return limbs


def _bpm_to_beats(bpm: float, duration: float) -> list[float]:
    if bpm <= 0 or duration <= 0:
        return []
    step = 60.0 / bpm
    times: list[float] = []
    t = 0.0
    while t < duration:
        times.append(round(t, 3))
        t += step
    return times


# ---- Driver ------------------------------------------------------------------


def analyze_from_dict(
    pose_data: dict[str, Any],
    *,
    source_pose: str = "<memory>",
    min_segment_duration: float = 0.6,
    max_segments: int = 16,
    bpm: float | None = None,
    beat_times: list[float] | None = None,
) -> dict[str, Any]:
    """Pure in-memory motion analysis. Returns the full motion dict (schema v2)."""
    frames: list[dict[str, Any]] = pose_data.get("frames") or []
    fps = compute_fps(pose_data, frames)
    duration = frames[-1].get("time", 0.0) if frames else 0.0

    visibility_threshold = 0.4
    smoothing_window = 5
    still_speed_threshold = 0.15

    trajectories, _ = compute_trajectories(
        frames,
        fps,
        visibility_threshold=visibility_threshold,
        smoothing_window=smoothing_window,
        still_speed_threshold=still_speed_threshold,
    )

    # Beat fallback: when no explicit beats but we have a BPM, build a uniform grid.
    effective_beats = beat_times
    if not effective_beats and bpm:
        effective_beats = _bpm_to_beats(bpm, duration)

    segments = segment_motion(
        trajectories,
        duration,
        min_segment_duration=min_segment_duration,
        max_segments=max_segments,
        beat_times=effective_beats,
    )

    hints = build_hints(segments)

    beats = build_beats(
        effective_beats or [],
        duration,
        trajectories,
        segments,
        still_speed_threshold=still_speed_threshold,
    )
    limbs = build_limbs(beats, frames, trajectories)

    summary = build_summary(
        trajectories,
        segments,
        bpm=bpm,
        beat_times=effective_beats,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_pose": source_pose,
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "extract_config": {
            "stride": 1,
            "smoothingWindow": smoothing_window,
            "stillSpeedThreshold": still_speed_threshold,
            "visibilityThreshold": visibility_threshold,
            "minSegmentDuration": min_segment_duration,
            "maxSegments": max_segments,
        },
        "trajectories": trajectories,
        "segments": segments,
        "hints": hints,
        "beats": beats,
        "limbs": limbs,
        "summary": summary,
    }


def analyze(
    pose_path: Path,
    *,
    output: Path,
    min_segment_duration: float,
    max_segments: int,
    bpm: float | None,
    beat_times: list[float] | None,
) -> dict[str, Any]:
    data = load_pose(pose_path)
    motion = analyze_from_dict(
        data,
        source_pose=str(pose_path).replace("\\", "/"),
        min_segment_duration=min_segment_duration,
        max_segments=max_segments,
        bpm=bpm,
        beat_times=beat_times,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(motion, f, ensure_ascii=False)
    return motion


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Path to a *_pose.json from extract_pose.py")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: replace _pose.json with _motion.json next to the input)",
    )
    p.add_argument("--min-segment", type=float, default=0.6, help="Minimum segment duration in seconds")
    p.add_argument("--max-segments", type=int, default=16, help="Maximum number of segments")
    p.add_argument("--bpm", type=float, default=None, help="Optional BPM override")
    p.add_argument(
        "--beats",
        type=str,
        default=None,
        help="Optional comma-separated beat times in seconds (e.g. '0.62,1.24,1.86')",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"input pose JSON not found: {args.input}", file=sys.stderr)
        return 2
    output: Path = args.output or args.input.with_name(
        args.input.name.replace("_pose.json", "_motion.json")
    )
    if output == args.input:
        print("output would overwrite input; pass --output explicitly", file=sys.stderr)
        return 2
    beats: list[float] | None = None
    if args.beats:
        try:
            beats = [float(x.strip()) for x in args.beats.split(",") if x.strip()]
        except ValueError:
            print(f"failed to parse --beats: {args.beats}", file=sys.stderr)
            return 2
    motion = analyze(
        args.input,
        output=output,
        min_segment_duration=args.min_segment,
        max_segments=args.max_segments,
        bpm=args.bpm,
        beat_times=beats,
    )
    print(
        f"wrote {output} · segments={len(motion['segments'])} beats={len(motion.get('beats', []))} duration={motion['duration']}s primary_joints={motion['summary']['primaryJoints']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
