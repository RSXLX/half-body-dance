import argparse
import json
import os
import urllib.request
from collections import Counter

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarker, PoseLandmarker, RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core.image import Image as MpImage
from mediapipe.tasks.python.vision.core.image import ImageFormat

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
POSE_MODEL_FILE = "pose_landmarker.task"
HAND_MODEL_FILE = "hand_landmarker.task"
POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
POSE_REQUIRED_INDICES = (11, 12, 23, 24)
POSE_SHOULDER_INDICES = (11, 12)
POSE_HIP_INDICES = (23, 24)
HAND_LABELS = ("Left", "Right")
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


def download_model(model_path: str, url: str):
    if os.path.exists(model_path):
        return model_path

    print(f"模型文件不存在，正在下载: {url}")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, model_path)
    print(f"模型已下载到: {model_path}")
    return model_path


def serialize_pose_landmarks(pose_landmarks):
    if not pose_landmarks:
        return []
    return [
        {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "v": float(getattr(lm, "visibility", 0.0)),
        }
        for lm in pose_landmarks
    ]


def serialize_hand_landmarks(hand_landmarks):
    if not hand_landmarks:
        return []
    return [
        {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
        }
        for lm in hand_landmarks
    ]


def enhance_frame_for_retry(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(frame_bgr, 0.35, enhanced, 0.65, 0)


def get_pose_point(pose_landmarks, index):
    if len(pose_landmarks) != POSE_LANDMARK_COUNT:
        return None
    return pose_landmarks[index]


def clone_pose_landmarks(pose_landmarks):
    return [dict(point) for point in pose_landmarks] if pose_landmarks else []


def get_hand_weight(hand):
    if not hand or len(hand.get("landmarks", [])) != HAND_LANDMARK_COUNT:
        return 0.0
    anchor = hand["landmarks"][0]
    score = 1.0
    if 0.0 <= anchor.get("x", -1.0) <= 1.0 and 0.0 <= anchor.get("y", -1.0) <= 1.0:
        score += 0.5
    if hand.get("handedness") in HAND_LABELS:
        score += 0.5
    if hand.get("interpolated"):
        score *= 0.7
    return score


def distance(point_a, point_b):
    return float(np.hypot(point_a["x"] - point_b["x"], point_a["y"] - point_b["y"]))


def center_of_points(points):
    return {
        "x": sum(point["x"] for point in points) / len(points),
        "y": sum(point["y"] for point in points) / len(points),
    }


def is_hand_landmarks_complete(landmark_list):
    if len(landmark_list) != HAND_LANDMARK_COUNT:
        return False
    finger_tips = [4, 8, 12, 16, 20]
    return all(
        landmark_list[tip_index]["x"] >= 0 and landmark_list[tip_index]["x"] <= 1 and
        landmark_list[tip_index]["y"] >= 0 and landmark_list[tip_index]["y"] <= 1
        for tip_index in finger_tips
    )


def compute_pose_core_metrics(pose_landmarks):
    if len(pose_landmarks) != POSE_LANDMARK_COUNT:
        return {
            "visible_shoulders": 0,
            "visible_hips": 0,
            "core_average_visibility": 0.0,
            "core_min_visibility": 0.0,
        }

    shoulder_visibilities = [pose_landmarks[index]["v"] for index in POSE_SHOULDER_INDICES]
    hip_visibilities = [pose_landmarks[index]["v"] for index in POSE_HIP_INDICES]
    core_visibilities = shoulder_visibilities + hip_visibilities
    return {
        "visible_shoulders": sum(visibility >= 0.15 for visibility in shoulder_visibilities),
        "visible_hips": sum(visibility >= 0.15 for visibility in hip_visibilities),
        "core_average_visibility": float(sum(core_visibilities) / len(core_visibilities)),
        "core_min_visibility": float(min(core_visibilities)),
    }


def get_pose_reference_metrics(pose_landmarks):
    left_shoulder = get_pose_point(pose_landmarks, LEFT_SHOULDER)
    right_shoulder = get_pose_point(pose_landmarks, RIGHT_SHOULDER)
    left_hip = get_pose_point(pose_landmarks, 23)
    right_hip = get_pose_point(pose_landmarks, 24)
    if not all((left_shoulder, right_shoulder, left_hip, right_hip)):
        return None

    shoulder_center = {
        "x": (left_shoulder["x"] + right_shoulder["x"]) / 2.0,
        "y": (left_shoulder["y"] + right_shoulder["y"]) / 2.0,
    }
    hip_center = {
        "x": (left_hip["x"] + right_hip["x"]) / 2.0,
        "y": (left_hip["y"] + right_hip["y"]) / 2.0,
    }
    shoulder_width = distance(left_shoulder, right_shoulder)
    torso_height = distance(shoulder_center, hip_center)
    return {
        "shoulder_center": shoulder_center,
        "hip_center": hip_center,
        "shoulder_width": shoulder_width,
        "torso_height": torso_height,
    }


def evaluate_pose_quality(pose_landmarks, roi_min_visibility, final_min_visibility):
    metrics = compute_pose_core_metrics(pose_landmarks)
    roi_usable = (
        metrics["visible_shoulders"] >= 1
        and metrics["visible_hips"] >= 1
        and metrics["core_average_visibility"] >= roi_min_visibility
    )
    final_usable = (
        metrics["visible_shoulders"] >= 1
        and metrics["visible_hips"] >= 1
        and metrics["core_average_visibility"] >= final_min_visibility
    )
    return {
        **metrics,
        "roi_usable": roi_usable,
        "final_usable": final_usable,
    }


def can_track_pose(previous_pose_landmarks, candidate_pose_landmarks):
    if len(previous_pose_landmarks) != POSE_LANDMARK_COUNT or len(candidate_pose_landmarks) != POSE_LANDMARK_COUNT:
        return False

    previous_ref = get_pose_reference_metrics(previous_pose_landmarks)
    current_ref = get_pose_reference_metrics(candidate_pose_landmarks)
    if not previous_ref or not current_ref:
        return False

    previous_scale = max(previous_ref["shoulder_width"], previous_ref["torso_height"], 1e-6)
    current_scale = max(current_ref["shoulder_width"], current_ref["torso_height"], 1e-6)
    scale_ratio = current_scale / previous_scale
    if not 0.6 <= scale_ratio <= 1.6:
        return False

    center_shift = distance(previous_ref["shoulder_center"], current_ref["shoulder_center"])
    if center_shift > previous_scale * 0.9:
        return False

    tracked_indices = (LEFT_SHOULDER, RIGHT_SHOULDER, 23, 24, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)
    normalized_motion = []
    for index in tracked_indices:
        normalized_motion.append(distance(previous_pose_landmarks[index], candidate_pose_landmarks[index]) / previous_scale)
    average_motion = sum(normalized_motion) / len(normalized_motion)
    return average_motion <= 1.25


def blend_pose_landmarks(previous_pose_landmarks, candidate_pose_landmarks, blend_ratio):
    return [
        {
            key: previous_point[key] * (1.0 - blend_ratio) + current_point[key] * blend_ratio
            for key in ("x", "y", "z", "v")
        }
        for previous_point, current_point in zip(previous_pose_landmarks, candidate_pose_landmarks)
    ]


def estimate_hand_roi(pose_landmarks, side, frame_width, frame_height, expansion_factor):
    if side == "Left":
        wrist = get_pose_point(pose_landmarks, LEFT_WRIST)
        elbow = get_pose_point(pose_landmarks, LEFT_ELBOW)
        shoulder = get_pose_point(pose_landmarks, LEFT_SHOULDER)
    else:
        wrist = get_pose_point(pose_landmarks, RIGHT_WRIST)
        elbow = get_pose_point(pose_landmarks, RIGHT_ELBOW)
        shoulder = get_pose_point(pose_landmarks, RIGHT_SHOULDER)

    if not wrist or not elbow or not shoulder:
        return None

    wrist_xy = {"x": wrist["x"] * frame_width, "y": wrist["y"] * frame_height}
    elbow_xy = {"x": elbow["x"] * frame_width, "y": elbow["y"] * frame_height}
    shoulder_xy = {"x": shoulder["x"] * frame_width, "y": shoulder["y"] * frame_height}

    forearm = distance(wrist_xy, elbow_xy)
    upper_arm = distance(elbow_xy, shoulder_xy)
    base_size = max(forearm * 1.6, upper_arm * 0.9, min(frame_width, frame_height) * 0.12)
    half_size = int(round(base_size * expansion_factor))
    if half_size <= 4:
        return None

    center_x = int(round(wrist_xy["x"]))
    center_y = int(round(wrist_xy["y"]))
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(frame_width, center_x + half_size)
    y2 = min(frame_height, center_y + half_size)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def estimate_hand_roi_from_previous_hand(previous_hand, frame_width, frame_height, expansion_factor):
    landmarks = previous_hand.get("landmarks", [])
    if len(landmarks) != HAND_LANDMARK_COUNT:
        return None
    center = center_of_points(landmarks)
    root = landmarks[0]
    max_radius = max(distance(root, point) for point in landmarks)
    half_size = int(round(max(max_radius * frame_width, max_radius * frame_height, min(frame_width, frame_height) * 0.08) * expansion_factor * 1.8))
    center_x = int(round(center["x"] * frame_width))
    center_y = int(round(center["y"] * frame_height))
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(frame_width, center_x + half_size)
    y2 = min(frame_height, center_y + half_size)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def map_hand_landmarks_to_full_frame(landmarks, roi, frame_width, frame_height):
    roi_width = max(1, roi["x2"] - roi["x1"])
    roi_height = max(1, roi["y2"] - roi["y1"])
    mapped = []
    for point in landmarks:
        mapped.append(
            {
                "x": (roi["x1"] + point["x"] * roi_width) / frame_width,
                "y": (roi["y1"] + point["y"] * roi_height) / frame_height,
                "z": float(point["z"]),
            }
        )
    return mapped


def detect_missing_hands_with_roi(
    frame_bgr,
    pose_landmarks,
    hands,
    roi_hand_detector,
    frame_width,
    frame_height,
    roi_expansion_factor,
    previous_hands_by_label=None,
):
    if len(pose_landmarks) != POSE_LANDMARK_COUNT and not previous_hands_by_label:
        return []

    present_labels = {hand.get("handedness") for hand in hands}
    recovered_hands = []
    for side in HAND_LABELS:
        if side in present_labels:
            continue
        candidate_rois = []
        if len(pose_landmarks) == POSE_LANDMARK_COUNT:
            pose_roi = estimate_hand_roi(
                pose_landmarks,
                side,
                frame_width=frame_width,
                frame_height=frame_height,
                expansion_factor=roi_expansion_factor,
            )
            if pose_roi:
                candidate_rois.append(pose_roi)
        previous_hand = (previous_hands_by_label or {}).get(side)
        previous_roi = estimate_hand_roi_from_previous_hand(
            previous_hand or {},
            frame_width=frame_width,
            frame_height=frame_height,
            expansion_factor=roi_expansion_factor,
        )
        if previous_roi:
            candidate_rois.append(previous_roi)
        if not candidate_rois:
            continue
        best_hand = None
        best_score = -1.0
        target_root = get_pose_point(pose_landmarks, LEFT_WRIST if side == "Left" else RIGHT_WRIST)
        if not target_root and previous_hand:
            target_root = previous_hand["landmarks"][0]
        for roi in candidate_rois:
            crop_bgr = frame_bgr[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
            if crop_bgr.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_image = MpImage(ImageFormat.SRGB, np.ascontiguousarray(crop_rgb))
            roi_result = roi_hand_detector.detect(crop_image)
            if not roi_result.hand_landmarks:
                continue

            for hand_landmarks in roi_result.hand_landmarks:
                serialized = serialize_hand_landmarks(hand_landmarks)
                if not is_hand_landmarks_complete(serialized):
                    continue
                mapped = map_hand_landmarks_to_full_frame(
                    serialized,
                    roi=roi,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                root_distance = distance(mapped[0], target_root) if target_root else 0.05
                score = 1.0 / max(root_distance, 1e-6)
                if score > best_score:
                    best_score = score
                    best_hand = {
                        "handedness": side,
                        "landmarks": mapped,
                        "world_landmarks": [],
                        "finger_count": 5,
                        "roi_recovered": True,
                    }

        if best_hand:
            recovered_hands.append(best_hand)

    return recovered_hands


def smooth_pose_frames(frames, window_size):
    if window_size <= 1:
        return

    radius = window_size // 2
    valid_indices = [
        index for index, frame in enumerate(frames)
        if len(frame["pose_landmarks"]) == POSE_LANDMARK_COUNT
    ]
    valid_set = set(valid_indices)

    smoothed = {}
    for index in valid_indices:
        aggregated = []
        for landmark_index in range(POSE_LANDMARK_COUNT):
            weighted = {"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0}
            total_weight = 0.0
            for neighbor in range(index - radius, index + radius + 1):
                if neighbor not in valid_set:
                    continue
                point = frames[neighbor]["pose_landmarks"][landmark_index]
                weight = max(point.get("v", 0.0), 0.05)
                total_weight += weight
                for key in weighted:
                    weighted[key] += point[key] * weight
            if total_weight == 0.0:
                aggregated.append(dict(frames[index]["pose_landmarks"][landmark_index]))
                continue
            aggregated.append(
                {key: weighted[key] / total_weight for key in weighted}
            )
        smoothed[index] = aggregated

    for index, pose_landmarks in smoothed.items():
        frames[index]["pose_landmarks"] = pose_landmarks


def normalize_hands(frame):
    hands_by_label = {label: None for label in HAND_LABELS}
    extras = []
    for hand in frame.get("hands", []):
        label = hand.get("handedness")
        if label in hands_by_label and hands_by_label[label] is None:
            hands_by_label[label] = hand
        else:
            extras.append(hand)
    frame["_hands_by_label"] = hands_by_label
    frame["_hands_extra"] = extras


def interpolate_hand_sequence(frames, label, max_gap_frames):
    if max_gap_frames <= 0:
        return 0

    filled = 0
    index = 0
    total = len(frames)
    while index < total:
        if frames[index]["_hands_by_label"][label]:
            index += 1
            continue

        gap_start = index
        while index < total and not frames[index]["_hands_by_label"][label]:
            index += 1
        gap_end = index
        gap_size = gap_end - gap_start
        left_index = gap_start - 1
        right_index = gap_end
        if (
            gap_size > max_gap_frames
            or left_index < 0
            or right_index >= total
            or not frames[left_index]["_hands_by_label"][label]
            or not frames[right_index]["_hands_by_label"][label]
        ):
            continue

        left_hand = frames[left_index]["_hands_by_label"][label]
        right_hand = frames[right_index]["_hands_by_label"][label]
        for offset, frame_index in enumerate(range(gap_start, gap_end), start=1):
            alpha = offset / (gap_size + 1)
            landmarks = []
            for landmark_index in range(HAND_LANDMARK_COUNT):
                left_point = left_hand["landmarks"][landmark_index]
                right_point = right_hand["landmarks"][landmark_index]
                landmarks.append(
                    {
                        key: left_point[key] * (1.0 - alpha) + right_point[key] * alpha
                        for key in ("x", "y", "z")
                    }
                )

            world_landmarks = []
            if (
                len(left_hand.get("world_landmarks", [])) == HAND_LANDMARK_COUNT
                and len(right_hand.get("world_landmarks", [])) == HAND_LANDMARK_COUNT
            ):
                for landmark_index in range(HAND_LANDMARK_COUNT):
                    left_point = left_hand["world_landmarks"][landmark_index]
                    right_point = right_hand["world_landmarks"][landmark_index]
                    world_landmarks.append(
                        {
                            key: left_point[key] * (1.0 - alpha) + right_point[key] * alpha
                            for key in ("x", "y", "z")
                        }
                    )

            frames[frame_index]["_hands_by_label"][label] = {
                "handedness": label,
                "landmarks": landmarks,
                "world_landmarks": world_landmarks,
                "finger_count": 5,
                "interpolated": True,
            }
            filled += 1
    return filled


def smooth_hand_sequence(frames, label, window_size):
    if window_size <= 1:
        return

    radius = window_size // 2
    valid_indices = [
        index for index, frame in enumerate(frames)
        if frame["_hands_by_label"][label]
        and len(frame["_hands_by_label"][label]["landmarks"]) == HAND_LANDMARK_COUNT
    ]
    valid_set = set(valid_indices)
    smoothed = {}

    for index in valid_indices:
        source_hand = frames[index]["_hands_by_label"][label]
        hand_copy = {
            "handedness": source_hand["handedness"],
            "landmarks": [],
            "world_landmarks": [],
            "finger_count": source_hand.get("finger_count", 5),
        }
        if source_hand.get("interpolated"):
            hand_copy["interpolated"] = True

        for landmark_index in range(HAND_LANDMARK_COUNT):
            weighted = {"x": 0.0, "y": 0.0, "z": 0.0}
            total_weight = 0.0
            for neighbor in range(index - radius, index + radius + 1):
                if neighbor not in valid_set:
                    continue
                neighbor_hand = frames[neighbor]["_hands_by_label"][label]
                point = neighbor_hand["landmarks"][landmark_index]
                weight = get_hand_weight(neighbor_hand)
                total_weight += weight
                for key in weighted:
                    weighted[key] += point[key] * weight
            if total_weight == 0.0:
                hand_copy["landmarks"].append(dict(source_hand["landmarks"][landmark_index]))
            else:
                hand_copy["landmarks"].append({key: weighted[key] / total_weight for key in weighted})

        if len(source_hand.get("world_landmarks", [])) == HAND_LANDMARK_COUNT:
            for landmark_index in range(HAND_LANDMARK_COUNT):
                weighted = {"x": 0.0, "y": 0.0, "z": 0.0}
                total_weight = 0.0
                for neighbor in range(index - radius, index + radius + 1):
                    if neighbor not in valid_set:
                        continue
                    neighbor_hand = frames[neighbor]["_hands_by_label"][label]
                    if len(neighbor_hand.get("world_landmarks", [])) != HAND_LANDMARK_COUNT:
                        continue
                    point = neighbor_hand["world_landmarks"][landmark_index]
                    weight = get_hand_weight(neighbor_hand)
                    total_weight += weight
                    for key in weighted:
                        weighted[key] += point[key] * weight
                if total_weight == 0.0:
                    hand_copy["world_landmarks"].append(dict(source_hand["world_landmarks"][landmark_index]))
                else:
                    hand_copy["world_landmarks"].append({key: weighted[key] / total_weight for key in weighted})

        smoothed[index] = hand_copy

    for index, hand in smoothed.items():
        frames[index]["_hands_by_label"][label] = hand


def rebuild_frame_hands(frames):
    for frame in frames:
        hands = [frame["_hands_by_label"][label] for label in HAND_LABELS if frame["_hands_by_label"][label]]
        hands.extend(frame.get("_hands_extra", []))
        frame["hands"] = hands
        frame.pop("_hands_by_label", None)
        frame.pop("_hands_extra", None)


def interpolate_pose_gaps(frames, max_gap_frames):
    if max_gap_frames <= 0:
        return 0

    filled = 0
    index = 0
    total = len(frames)
    while index < total:
        if len(frames[index]["pose_landmarks"]) == POSE_LANDMARK_COUNT:
            index += 1
            continue

        gap_start = index
        while index < total and len(frames[index]["pose_landmarks"]) != POSE_LANDMARK_COUNT:
            index += 1
        gap_end = index
        gap_size = gap_end - gap_start
        left_index = gap_start - 1
        right_index = gap_end

        if (
            gap_size > max_gap_frames
            or left_index < 0
            or right_index >= total
            or len(frames[left_index]["pose_landmarks"]) != POSE_LANDMARK_COUNT
            or len(frames[right_index]["pose_landmarks"]) != POSE_LANDMARK_COUNT
        ):
            continue

        left_pose = frames[left_index]["pose_landmarks"]
        right_pose = frames[right_index]["pose_landmarks"]
        for offset, frame_index in enumerate(range(gap_start, gap_end), start=1):
            alpha = offset / (gap_size + 1)
            interpolated = []
            for landmark_index in range(POSE_LANDMARK_COUNT):
                left_point = left_pose[landmark_index]
                right_point = right_pose[landmark_index]
                interpolated.append(
                    {
                        key: left_point[key] * (1.0 - alpha) + right_point[key] * alpha
                        for key in ("x", "y", "z", "v")
                    }
                )
            frames[frame_index]["pose_landmarks"] = interpolated
            frames[frame_index]["pose_interpolated"] = True
            filled += 1

    return filled


def postprocess_frames(frames, pose_smoothing_window, hand_smoothing_window, max_gap_frames):
    for frame in frames:
        normalize_hands(frame)

    interpolated_count = interpolate_pose_gaps(frames, max_gap_frames)
    smooth_pose_frames(frames, pose_smoothing_window)

    interpolated_hands = {}
    for label in HAND_LABELS:
        interpolated_hands[label] = interpolate_hand_sequence(frames, label, max_gap_frames)
        smooth_hand_sequence(frames, label, hand_smoothing_window)

    rebuild_frame_hands(frames)
    return {
        "interpolated_pose_frames": interpolated_count,
        "interpolated_hand_frames": interpolated_hands,
    }


def maybe_upscale_frame(frame, video_width, video_height, min_short_side, max_long_side):
    if min_short_side <= 0 and max_long_side <= 0:
        return frame, 1.0

    short_side = min(video_width, video_height)
    long_side = max(video_width, video_height)
    if short_side <= 0 or long_side <= 0:
        return frame, 1.0

    scale = 1.0
    if min_short_side > 0 and short_side < min_short_side:
        scale = max(scale, min_short_side / short_side)
    if max_long_side > 0 and long_side * scale > max_long_side:
        scale = min(scale, max_long_side / long_side)
    if scale <= 1.0:
        return frame, 1.0

    new_width = max(1, int(round(video_width * scale)))
    new_height = max(1, int(round(video_height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC), scale


def build_quality_report(frames, stats, fps, detection_stride, upscale_applied, video_width, video_height):
    total_frames = len(frames)
    if total_frames == 0:
        return {"summary": "empty", "issues": ["no_frames"]}

    pose_present_frames = sum(1 for frame in frames if len(frame["pose_landmarks"]) == POSE_LANDMARK_COUNT)
    left_hand_frames = sum(
        1 for frame in frames
        if any(hand.get("handedness") == "Left" and len(hand.get("landmarks", [])) == HAND_LANDMARK_COUNT for hand in frame.get("hands", []))
    )
    right_hand_frames = sum(
        1 for frame in frames
        if any(hand.get("handedness") == "Right" and len(hand.get("landmarks", [])) == HAND_LANDMARK_COUNT for hand in frame.get("hands", []))
    )
    interpolated_pose_frames = sum(1 for frame in frames if frame.get("pose_interpolated"))
    interpolated_hands = Counter()
    low_visibility_frames = 0
    for frame in frames:
        if len(frame["pose_landmarks"]) == POSE_LANDMARK_COUNT:
            core_visibility = [frame["pose_landmarks"][index]["v"] for index in POSE_REQUIRED_INDICES]
            if sum(core_visibility) / len(core_visibility) < 0.35:
                low_visibility_frames += 1
        for hand in frame.get("hands", []):
            if hand.get("interpolated"):
                interpolated_hands[hand.get("handedness", "unknown")] += 1

    issues = []
    pose_coverage = pose_present_frames / total_frames
    if pose_coverage < 0.75:
        issues.append("pose_coverage_low")
    if left_hand_frames / total_frames < 0.35 and right_hand_frames / total_frames < 0.35:
        issues.append("hand_coverage_low")
    if low_visibility_frames / total_frames > 0.2:
        issues.append("occlusion_or_distance_high")
    if detection_stride > 1:
        issues.append("frame_sampling_enabled")
    if upscale_applied:
        issues.append("source_low_resolution")

    return {
        "summary": "good" if not issues else "needs_attention",
        "issues": issues,
        "metrics": {
            "fps": fps,
            "input_resolution": {"width": video_width, "height": video_height},
            "detection_stride": detection_stride,
            "pose_coverage": round(pose_coverage, 4),
            "left_hand_coverage": round(left_hand_frames / total_frames, 4),
            "right_hand_coverage": round(right_hand_frames / total_frames, 4),
            "low_visibility_ratio": round(low_visibility_frames / total_frames, 4),
            "interpolated_pose_ratio": round(interpolated_pose_frames / total_frames, 4),
            "interpolated_left_hand_ratio": round(interpolated_hands.get("Left", 0) / total_frames, 4),
            "interpolated_right_hand_ratio": round(interpolated_hands.get("Right", 0) / total_frames, 4),
            "pose_missing_raw": stats.get("pose_missing_raw", 0),
            "pose_filtered_low_visibility": stats.get("pose_filtered_low_visibility", 0),
            "hand_filtered_incomplete": stats.get("hand_filtered_incomplete", 0),
            "hand_filtered_out_of_range": stats.get("hand_filtered_out_of_range", 0),
            "roi_hand_recovered": stats.get("roi_hand_recovered", 0),
            "retry_pose_recovered": stats.get("retry_pose_recovered", 0),
            "pose_tracked": stats.get("pose_tracked", 0),
            "pose_tracked_hold": stats.get("pose_tracked_hold", 0),
            "retry_pose_tracked": stats.get("retry_pose_tracked", 0),
            "retry_hand_recovered": stats.get("retry_hand_recovered", 0),
        },
    }


def extract_pose_from_video(
    video_path,
    output_json_path,
    pose_model_path,
    hand_model_path,
    *,
    pose_detection_confidence,
    pose_presence_confidence,
    hand_detection_confidence,
    hand_presence_confidence,
    tracking_confidence,
    min_pose_visibility,
    pose_smoothing_window,
    hand_smoothing_window,
    interpolate_gap_frames,
    detection_stride,
    upsample_min_short_side,
    upsample_max_long_side,
    hand_roi_expansion_factor,
    max_frames,
):
    """
    离线动作提取工具
    读取舞蹈视频，使用 MediaPipe PoseLandmarker 和 HandLandmarker 提取关键点，并导出为 JSON 动作谱
    """
    pose_model_path = download_model(pose_model_path, POSE_MODEL_URL)
    hand_model_path = download_model(hand_model_path, HAND_MODEL_URL)

    pose_options = PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=pose_model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=pose_detection_confidence,
        min_pose_presence_confidence=pose_presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    hand_options = HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=hand_model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=hand_detection_confidence,
        min_hand_presence_confidence=hand_presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    roi_hand_options = HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=hand_model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=max(0.2, hand_detection_confidence - 0.1),
        min_hand_presence_confidence=max(0.2, hand_presence_confidence - 0.1),
        min_tracking_confidence=tracking_confidence,
    )
    retry_pose_options = PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=pose_model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=max(0.25, pose_detection_confidence - 0.1),
        min_pose_presence_confidence=max(0.25, pose_presence_confidence - 0.1),
        min_tracking_confidence=tracking_confidence,
    )
    retry_hand_options = HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=hand_model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=max(0.2, hand_detection_confidence - 0.1),
        min_hand_presence_confidence=max(0.2, hand_presence_confidence - 0.1),
        min_tracking_confidence=tracking_confidence,
    )

    with PoseLandmarker.create_from_options(pose_options) as pose, HandLandmarker.create_from_options(
        hand_options
    ) as hand, HandLandmarker.create_from_options(roi_hand_options) as roi_hand, PoseLandmarker.create_from_options(
        retry_pose_options
    ) as retry_pose, HandLandmarker.create_from_options(retry_hand_options) as retry_hand_detector:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0 / fps
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        aspect_ratio = (video_width / video_height) if video_width and video_height else None
        audio_source = os.path.basename(video_path)

        dance_data = {
            "fps": fps,
            "video_width": video_width,
            "video_height": video_height,
            "aspect_ratio": aspect_ratio,
            "frame_count_hint": total_frames_hint,
            "audio_source": audio_source,
            "source_video": os.path.basename(video_path),
            "extract_config": {
                "pose_detection_confidence": pose_detection_confidence,
                "pose_presence_confidence": pose_presence_confidence,
                "hand_detection_confidence": hand_detection_confidence,
                "hand_presence_confidence": hand_presence_confidence,
                "tracking_confidence": tracking_confidence,
                "min_pose_visibility": min_pose_visibility,
                "pose_smoothing_window": pose_smoothing_window,
                "hand_smoothing_window": hand_smoothing_window,
                "interpolate_gap_frames": interpolate_gap_frames,
                "detection_stride": detection_stride,
                "upsample_min_short_side": upsample_min_short_side,
                "upsample_max_long_side": upsample_max_long_side,
                "hand_roi_expansion_factor": hand_roi_expansion_factor,
            },
            "frames": [],
        }

        print(
            f"开始处理视频: {video_path}, FPS: {fps}, "
            f"分辨率: {video_width}x{video_height}, 比例: {aspect_ratio:.3f}"
            if aspect_ratio else f"开始处理视频: {video_path}, FPS: {fps}"
        )
        current_time = 0.0
        frame_index = 0
        stats = Counter()
        previous_hands_by_label = {label: None for label in HAND_LABELS}
        previous_pose_for_tracking = None
        missing_pose_streak = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            source_frame_index = frame_index
            frame_index += 1

            if max_frames is not None and source_frame_index >= max_frames:
                break
            if detection_stride > 1 and source_frame_index % detection_stride != 0:
                current_time += frame_interval
                stats["frames_skipped_by_stride"] += 1
                continue

            processed_frame, upscale_ratio = maybe_upscale_frame(
                frame,
                video_width,
                video_height,
                upsample_min_short_side,
                upsample_max_long_side,
            )
            if upscale_ratio > 1.0:
                stats["frames_upscaled"] += 1

            image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            image = MpImage(ImageFormat.SRGB, np.ascontiguousarray(image_rgb))
            timestamp_ms = int(round(current_time * 1000))

            pose_results = pose.detect_for_video(image, timestamp_ms)
            hand_results = hand.detect_for_video(image, timestamp_ms)

            pose_landmarks = []
            pose_for_roi = []
            pose_source = "missing"
            if pose_results.pose_landmarks:
                pose_candidate = serialize_pose_landmarks(pose_results.pose_landmarks[0])
                pose_quality = evaluate_pose_quality(
                    pose_candidate,
                    roi_min_visibility=max(0.12, min_pose_visibility * 0.7),
                    final_min_visibility=min_pose_visibility,
                )
                if pose_quality["roi_usable"]:
                    pose_for_roi = clone_pose_landmarks(pose_candidate)
                if pose_quality["final_usable"]:
                    pose_landmarks = clone_pose_landmarks(pose_candidate)
                    pose_source = "detected"
                    stats["pose_kept"] += 1
                else:
                    if previous_pose_for_tracking and pose_quality["roi_usable"] and can_track_pose(previous_pose_for_tracking, pose_candidate):
                        pose_landmarks = blend_pose_landmarks(previous_pose_for_tracking, pose_candidate, 0.65)
                        pose_for_roi = clone_pose_landmarks(pose_landmarks)
                        pose_source = "tracked"
                        stats["pose_tracked"] += 1
                    else:
                        stats["pose_filtered_low_visibility"] += 1
            else:
                stats["pose_missing_raw"] += 1
                if previous_pose_for_tracking and missing_pose_streak < 2:
                    pose_landmarks = clone_pose_landmarks(previous_pose_for_tracking)
                    pose_for_roi = clone_pose_landmarks(previous_pose_for_tracking)
                    pose_source = "tracked_hold"
                    stats["pose_tracked_hold"] += 1
                    missing_pose_streak += 1

            needs_retry = len(pose_landmarks) != POSE_LANDMARK_COUNT

            hands = []
            if hand_results.hand_landmarks:
                for i, hand_landmarks in enumerate(hand_results.hand_landmarks):
                    # 检查手部关键点质量
                    if len(hand_landmarks) < HAND_LANDMARK_COUNT:
                        stats["hand_filtered_incomplete"] += 1
                        continue  # 跳过不完整的手部数据

                    label = None
                    if hand_results.handedness and i < len(hand_results.handedness):
                        handedness_data = hand_results.handedness[i]
                        if isinstance(handedness_data, list) and len(handedness_data) > 0:
                            label = handedness_data[0].category_name
                        elif hasattr(handedness_data, 'category_name'):
                            label = handedness_data.category_name
                    hand_label = label or f"hand_{i}"

                    landmark_list = serialize_hand_landmarks(hand_landmarks)

                    if not is_hand_landmarks_complete(landmark_list):
                        stats["hand_filtered_out_of_range"] += 1
                        continue  # 跳过手指不完整的手部数据

                    world_list = []
                    if hand_results.hand_world_landmarks and i < len(hand_results.hand_world_landmarks):
                        for lm in hand_results.hand_world_landmarks[i]:
                            world_list.append({
                                "x": float(lm.x),
                                "y": float(lm.y),
                                "z": float(lm.z),
                            })

                    hands.append({
                        "handedness": hand_label,
                        "landmarks": landmark_list,
                        "world_landmarks": world_list,
                        "finger_count": 5,  # 明确标记五指完整
                    })
                    stats["hand_kept"] += 1

            if len(hands) < 2:
                needs_retry = True

            if needs_retry:
                retry_frame = enhance_frame_for_retry(processed_frame)
                retry_rgb = cv2.cvtColor(retry_frame, cv2.COLOR_BGR2RGB)
                retry_image = MpImage(ImageFormat.SRGB, np.ascontiguousarray(retry_rgb))
                retry_pose_results = retry_pose.detect(retry_image)
                retry_hand_results = retry_hand_detector.detect(retry_image)

                retry_pose_landmarks = []
                if retry_pose_results.pose_landmarks:
                    retry_pose_candidate = serialize_pose_landmarks(retry_pose_results.pose_landmarks[0])
                    retry_pose_quality = evaluate_pose_quality(
                        retry_pose_candidate,
                        roi_min_visibility=max(0.12, min_pose_visibility * 0.7),
                        final_min_visibility=min_pose_visibility,
                    )
                    if retry_pose_quality["roi_usable"] and len(pose_for_roi) != POSE_LANDMARK_COUNT:
                        pose_for_roi = clone_pose_landmarks(retry_pose_candidate)
                    if retry_pose_quality["final_usable"]:
                        retry_pose_landmarks = clone_pose_landmarks(retry_pose_candidate)
                    elif previous_pose_for_tracking and retry_pose_quality["roi_usable"] and can_track_pose(previous_pose_for_tracking, retry_pose_candidate):
                        retry_pose_landmarks = blend_pose_landmarks(previous_pose_for_tracking, retry_pose_candidate, 0.65)
                        if len(pose_for_roi) != POSE_LANDMARK_COUNT:
                            pose_for_roi = clone_pose_landmarks(retry_pose_landmarks)
                        stats["retry_pose_tracked"] += 1
                if len(pose_landmarks) != POSE_LANDMARK_COUNT and len(retry_pose_landmarks) == POSE_LANDMARK_COUNT:
                    pose_landmarks = clone_pose_landmarks(retry_pose_landmarks)
                    if len(pose_for_roi) != POSE_LANDMARK_COUNT:
                        pose_for_roi = clone_pose_landmarks(retry_pose_landmarks)
                    if pose_source == "missing":
                        pose_source = "retry"
                        stats["retry_pose_recovered"] += 1
                    else:
                        pose_source = "retry_tracked"

                retry_hands = []
                if retry_hand_results.hand_landmarks:
                    for i, retry_hand_landmarks in enumerate(retry_hand_results.hand_landmarks):
                        if len(retry_hand_landmarks) < HAND_LANDMARK_COUNT:
                            continue
                        label = None
                        if retry_hand_results.handedness and i < len(retry_hand_results.handedness):
                            handedness_data = retry_hand_results.handedness[i]
                            if isinstance(handedness_data, list) and len(handedness_data) > 0:
                                label = handedness_data[0].category_name
                            elif hasattr(handedness_data, "category_name"):
                                label = handedness_data.category_name
                        landmark_list = serialize_hand_landmarks(retry_hand_landmarks)
                        if not is_hand_landmarks_complete(landmark_list):
                            continue
                        retry_hands.append(
                            {
                                "handedness": label or f"retry_hand_{i}",
                                "landmarks": landmark_list,
                                "world_landmarks": [],
                                "finger_count": 5,
                                "retry_recovered": True,
                            }
                        )
                existing_labels = {hand.get("handedness") for hand in hands}
                for recovered_retry_hand in retry_hands:
                    if recovered_retry_hand.get("handedness") in HAND_LABELS and recovered_retry_hand.get("handedness") not in existing_labels:
                        hands.append(recovered_retry_hand)
                        existing_labels.add(recovered_retry_hand["handedness"])
                        stats["retry_hand_recovered"] += 1

            recovered_hands = detect_missing_hands_with_roi(
                processed_frame,
                pose_for_roi if len(pose_for_roi) == POSE_LANDMARK_COUNT else pose_landmarks,
                hands,
                roi_hand_detector=roi_hand,
                frame_width=processed_frame.shape[1],
                frame_height=processed_frame.shape[0],
                roi_expansion_factor=hand_roi_expansion_factor,
                previous_hands_by_label=previous_hands_by_label,
            )
            if recovered_hands:
                hands.extend(recovered_hands)
                stats["roi_hand_recovered"] += len(recovered_hands)

            current_hands_by_label = {label: None for label in HAND_LABELS}
            for hand_item in hands:
                label = hand_item.get("handedness")
                if label in current_hands_by_label and current_hands_by_label[label] is None:
                    current_hands_by_label[label] = hand_item
            for label in HAND_LABELS:
                if current_hands_by_label[label]:
                    previous_hands_by_label[label] = current_hands_by_label[label]
            if len(pose_landmarks) == POSE_LANDMARK_COUNT:
                previous_pose_for_tracking = clone_pose_landmarks(pose_landmarks)
                missing_pose_streak = 0

            dance_data["frames"].append(
                {
                    "time": round(current_time, 3),
                    "pose_landmarks": pose_landmarks,
                    "pose_source": pose_source,
                    "hands": hands,
                }
            )

            current_time += frame_interval

        cap.release()

    dance_data["postprocess"] = postprocess_frames(
        dance_data["frames"],
        pose_smoothing_window=pose_smoothing_window,
        hand_smoothing_window=hand_smoothing_window,
        max_gap_frames=interpolate_gap_frames,
    )
    dance_data["stats"] = {
        **dict(stats),
        "total_frames": len(dance_data["frames"]),
        "pose_frames_after_postprocess": sum(
            1
            for frame in dance_data["frames"]
            if len(frame["pose_landmarks"]) == POSE_LANDMARK_COUNT
        ),
    }
    dance_data["quality_report"] = build_quality_report(
        dance_data["frames"],
        stats=dance_data["stats"],
        fps=fps,
        detection_stride=detection_stride,
        upscale_applied=dance_data["stats"].get("frames_upscaled", 0) > 0,
        video_width=video_width,
        video_height=video_height,
    )

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dance_data, f, ensure_ascii=False, indent=2)

    print(
        "提取完成！"
        f" 共提取 {len(dance_data['frames'])} 帧数据，"
        f" pose 保留 {dance_data['stats'].get('pose_frames_after_postprocess', 0)} 帧，"
        f" pose 短缺口补帧 {dance_data['postprocess'].get('interpolated_pose_frames', 0)} 帧，"
        f" hand 补帧 L/R={dance_data['postprocess'].get('interpolated_hand_frames', {}).get('Left', 0)}/{dance_data['postprocess'].get('interpolated_hand_frames', {}).get('Right', 0)}，"
        f" 已保存至 {output_json_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="离线动作提取工具：读取视频并导出 MediaPipe Pose 与手部关键点 JSON。"
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default="wudao/angel.mp4",
        help="输入视频文件路径（默认 wudao/angel.mp4）",
    )
    parser.add_argument(
        "output_json_path",
        nargs="?",
        default="dance_data.json",
        help="输出 JSON 文件路径（默认 dance_data.json）",
    )
    parser.add_argument(
        "--pose_model",
        default=os.path.join(os.path.dirname(__file__), POSE_MODEL_FILE),
        help="PoseLandmarker 模型文件路径，默认会下载 pose_landmarker.task 到脚本目录。",
    )
    parser.add_argument(
        "--hand_model",
        default=os.path.join(os.path.dirname(__file__), HAND_MODEL_FILE),
        help="HandLandmarker 模型文件路径，默认会下载 hand_landmarker.task 到脚本目录。",
    )
    parser.add_argument("--pose_detection_confidence", type=float, default=0.45, help="Pose 检测阈值。")
    parser.add_argument("--pose_presence_confidence", type=float, default=0.45, help="Pose 存在阈值。")
    parser.add_argument("--hand_detection_confidence", type=float, default=0.35, help="Hand 检测阈值。")
    parser.add_argument("--hand_presence_confidence", type=float, default=0.35, help="Hand 存在阈值。")
    parser.add_argument("--tracking_confidence", type=float, default=0.35, help="视频跟踪阈值。")
    parser.add_argument("--min_pose_visibility", type=float, default=0.2, help="肩髋关键点最低可见度，低于该值视为无效 pose。")
    parser.add_argument("--pose_smoothing_window", type=int, default=5, help="Pose 平滑窗口，建议奇数，1 表示关闭。")
    parser.add_argument("--hand_smoothing_window", type=int, default=5, help="Hand 平滑窗口，建议奇数，1 表示关闭。")
    parser.add_argument("--interpolate_gap_frames", type=int, default=4, help="允许线性补齐的连续丢帧数。")
    parser.add_argument("--detection_stride", type=int, default=1, help="每隔多少帧做一次检测，1 表示逐帧。")
    parser.add_argument("--upsample_min_short_side", type=int, default=720, help="当短边低于该值时自动放大后再识别，0 表示关闭。")
    parser.add_argument("--upsample_max_long_side", type=int, default=1600, help="自动放大的长边上限，避免过大拖慢。")
    parser.add_argument("--hand_roi_expansion_factor", type=float, default=1.35, help="基于腕部 ROI 的手部二次检测区域放大系数。")
    parser.add_argument("--max_frames", type=int, default=None, help="仅处理前 N 帧，便于调试。")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(
            f"视频文件未找到：{args.video_path}。请提供一个有效的视频路径。"
        )

    extract_pose_from_video(
        args.video_path,
        args.output_json_path,
        args.pose_model,
        args.hand_model,
        pose_detection_confidence=args.pose_detection_confidence,
        pose_presence_confidence=args.pose_presence_confidence,
        hand_detection_confidence=args.hand_detection_confidence,
        hand_presence_confidence=args.hand_presence_confidence,
        tracking_confidence=args.tracking_confidence,
        min_pose_visibility=args.min_pose_visibility,
        pose_smoothing_window=max(1, args.pose_smoothing_window),
        hand_smoothing_window=max(1, args.hand_smoothing_window),
        interpolate_gap_frames=max(0, args.interpolate_gap_frames),
        detection_stride=max(1, args.detection_stride),
        upsample_min_short_side=max(0, args.upsample_min_short_side),
        upsample_max_long_side=max(0, args.upsample_max_long_side),
        hand_roi_expansion_factor=max(0.8, args.hand_roi_expansion_factor),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
