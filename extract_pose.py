import argparse
import json
import os
import urllib.request

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarker, PoseLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core.image import Image as MpImage
from mediapipe.tasks.python.vision.core.image import ImageFormat

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
POSE_MODEL_FILE = "pose_landmarker.task"
HAND_MODEL_FILE = "hand_landmarker.task"


def download_model(model_path: str, url: str):
    if os.path.exists(model_path):
        return model_path

    print(f"模型文件不存在，正在下载: {url}")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, model_path)
    print(f"模型已下载到: {model_path}")
    return model_path


def extract_pose_from_video(video_path, output_json_path, pose_model_path, hand_model_path):
    """
    离线动作提取工具
    读取舞蹈视频，使用 MediaPipe PoseLandmarker 和 HandLandmarker 提取关键点，并导出为 JSON 动作谱
    """
    pose_model_path = download_model(pose_model_path, POSE_MODEL_URL)
    hand_model_path = download_model(hand_model_path, HAND_MODEL_URL)

    with PoseLandmarker.create_from_model_path(pose_model_path) as pose, HandLandmarker.create_from_model_path(
        hand_model_path
    ) as hand:
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
            "frames": [],
        }

        print(
            f"开始处理视频: {video_path}, FPS: {fps}, "
            f"分辨率: {video_width}x{video_height}, 比例: {aspect_ratio:.3f}"
            if aspect_ratio else f"开始处理视频: {video_path}, FPS: {fps}"
        )
        current_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = MpImage(ImageFormat.SRGB, np.ascontiguousarray(image_rgb))

            pose_results = pose.detect(image)
            hand_results = hand.detect(image)

            pose_landmarks = []
            if pose_results.pose_landmarks:
                for pose_landmark_list in pose_results.pose_landmarks:
                    pose_landmarks.extend(
                        [
                            {
                                "x": float(lm.x),
                                "y": float(lm.y),
                                "z": float(lm.z),
                                "v": float(getattr(lm, "visibility", 0.0)),
                            }
                            for lm in pose_landmark_list
                        ]
                    )

            hands = []
            if hand_results.hand_landmarks:
                for i, hand_landmarks in enumerate(hand_results.hand_landmarks):
                    # 检查手部关键点质量
                    if len(hand_landmarks) < 21:
                        continue  # 跳过不完整的手部数据

                    label = None
                    if hand_results.handedness and i < len(hand_results.handedness):
                        handedness_data = hand_results.handedness[i]
                        if isinstance(handedness_data, list) and len(handedness_data) > 0:
                            label = handedness_data[0].category_name
                        elif hasattr(handedness_data, 'category_name'):
                            label = handedness_data.category_name
                    hand_label = label or f"hand_{i}"

                    # 提取五指关键点（21个点）
                    landmark_list = []
                    for lm in hand_landmarks:
                        landmark_list.append({
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                        })

                    # 验证五指关键点完整性
                    finger_tips = [4, 8, 12, 16, 20]  # 五个手指尖
                    finger_complete = all(
                        i < len(landmark_list) and
                        landmark_list[i]["x"] >= 0 and landmark_list[i]["x"] <= 1 and
                        landmark_list[i]["y"] >= 0 and landmark_list[i]["y"] <= 1
                        for i in finger_tips
                    )

                    if not finger_complete:
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

            dance_data["frames"].append(
                {
                    "time": round(current_time, 3),
                    "pose_landmarks": pose_landmarks,
                    "hands": hands,
                }
            )

            current_time += frame_interval

        cap.release()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dance_data, f, ensure_ascii=False)

    print(
        f"提取完成！共提取 {len(dance_data['frames'])} 帧数据，已保存至 {output_json_path}"
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
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(
            f"视频文件未找到：{args.video_path}。请提供一个有效的视频路径。"
        )

    extract_pose_from_video(args.video_path, args.output_json_path, args.pose_model, args.hand_model)


if __name__ == "__main__":
    main()
