import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.analysis import analyze_video_frames_stepfun as stepfun


class StepFunDanceFrameAnalysisTest(unittest.TestCase):
    def test_build_prompt_includes_terms_and_output_schema(self):
        bank = {
            "dance_terms": [
                {"term": "卡点", "meaning": "动作命中音乐重拍"},
                {"term": "wave", "meaning": "手臂或身体连续波浪"},
            ],
            "action_labels": [
                {"label": "抬手", "guidance": ["手腕或手肘明显上行"]},
            ],
            "output_schema_hint": {
                "video_summary": "整体描述",
                "frame_actions": [],
                "segments": [],
            },
        }

        prompt = stepfun.build_dance_motion_prompt(bank, "优先描述半身舞。")

        self.assertIn("逐帧观察", prompt)
        self.assertIn("卡点", prompt)
        self.assertIn("wave", prompt)
        self.assertIn("优先描述半身舞。", prompt)
        self.assertIn('"frame_actions"', prompt)

    def test_build_payload_uses_stepfun_chat_image_blocks(self):
        image = stepfun.FrameImage(
            index=1,
            timestamp_sec=0.5,
            path=Path("frame_000001.jpg"),
            data_url="data:image/jpeg;base64,abc",
        )

        payload = stepfun.build_chat_payload(
            model="step-3.6",
            prompt_text="分析动作",
            images=[image],
            detail="high",
            temperature=0.1,
        )

        self.assertEqual(payload["model"], "step-3.6")
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[0]["image_url"]["detail"], "high")
        self.assertEqual(content[-1], {"type": "text", "text": "分析动作"})

    def test_image_to_data_url_reads_jpeg(self):
        raw = b"\xff\xd8demo\xff\xd9"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            path.write_bytes(raw)

            data_url = stepfun.image_to_data_url(path)

        self.assertEqual(data_url, "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"))

    def test_extract_json_from_text_accepts_markdown_fence(self):
        parsed = stepfun.extract_json_from_text('```json\n{"segments": []}\n```')

        self.assertEqual(parsed, {"segments": []})

    def test_normalize_analysis_clamps_confidence_and_orders_frames(self):
        normalized = stepfun.normalize_analysis(
            {
                "video_summary": "上半身卡点舞",
                "frame_actions": [
                    {"frame_index": "2", "timestamp_sec": "0.20", "action": "落手", "confidence": 1.5},
                    {"frame_index": "1", "timestamp_sec": "0.10", "action": "抬手", "confidence": -1},
                ],
                "segments": [
                    {"start_sec": "0", "end_sec": "1.234", "label": "wave", "confidence": "0.8"}
                ],
            }
        )

        self.assertEqual([item["frame_index"] for item in normalized["frame_actions"]], [1, 2])
        self.assertEqual(normalized["frame_actions"][0]["confidence"], 0.0)
        self.assertEqual(normalized["frame_actions"][1]["confidence"], 1.0)
        self.assertEqual(normalized["segments"][0]["end_sec"], 1.23)

    def test_run_curl_sends_payload_from_temp_file(self):
        completed = Mock(returncode=0, stdout='{"choices":[{"message":{"content":"{}"}}]}', stderr="")
        with patch("scripts.analysis.analyze_video_frames_stepfun.subprocess.run", return_value=completed) as run:
            response = stepfun.run_curl(
                {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "a" * 10000}}]}]},
                "secret",
                "https://example.test/v1/chat/completions",
            )

        cmd = run.call_args.args[0]
        self.assertIn("--data-binary", cmd)
        data_arg = cmd[cmd.index("--data-binary") + 1]
        self.assertTrue(data_arg.startswith("@"))
        self.assertLess(sum(len(part) for part in cmd), 1000)
        self.assertEqual(response["choices"][0]["message"]["content"], "{}")


if __name__ == "__main__":
    unittest.main()
