# half-body-dance

半身舞动作匹配与分析工作台。

仓库包含三部分能力：

- `pose_viewer.html`：本地静态动作查看与对比工作台，内置 `wudao/` 示例数据。
- `extract_pose.py`：从舞蹈视频中提取姿态关键点并导出 JSON。
- `analyze_pose_emoji.py`、`analyze_video_emoji_volcengine.py`、`upload_and_analyze_video_volcengine.py`：对动作谱或视频做 emoji 时间线分析，其中后两者支持火山引擎 ARK / TOS。

## 目录说明

- `wudao/`：示例视频与对应姿态 JSON。
- `volcengine_emoji_prompt_bank.json`：视频动作分析提示词库。
- `App.jsx`：早期交互原型。

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 dev_server.py
```

打开 [http://localhost:4173/pose_viewer.html](http://localhost:4173/pose_viewer.html)。

说明：

- `dev_server.py` 会同时提供静态页面和 `/api/extract-pose` 接口。
- 准备页里上传短视频并自动识别生成 JSON，必须通过这个服务启动，不能再用纯 `http.server`。

## 常用命令

提取姿态：

```bash
python3 extract_pose.py wudao/angel.mp4 wudao/angel_pose.json
```

更稳的提取参数示例：

```bash
python3 extract_pose.py wudao/angel.mp4 wudao/angel_pose.json \
  --pose_detection_confidence 0.45 \
  --pose_presence_confidence 0.45 \
  --tracking_confidence 0.35 \
  --min_pose_visibility 0.2 \
  --pose_smoothing_window 5 \
  --hand_smoothing_window 5 \
  --detection_stride 1 \
  --upsample_min_short_side 720 \
  --upsample_max_long_side 1600 \
  --hand_roi_expansion_factor 1.35 \
  --interpolate_gap_frames 4
```

说明：

- 提取器已切换为 MediaPipe `VIDEO` 模式，识别会利用时序跟踪而不是逐帧独立检测。
- 导出前会对 `pose` 和 `hands` 做按时间轴的短时丢帧补点与平滑，手部按 `Left` / `Right` 分轨处理。
- 短边较小的视频会自动放大后再识别，降低远景、小手势漏检概率；可通过 `--upsample_min_short_side 0` 关闭。
- 当全图手部识别缺失时，会基于 pose 的腕/肘/肩位置估算局部 ROI 做二次手部检测，提升远景、遮挡和小手势场景下的命中率。
- `--detection_stride` 可控制采样步长。默认 `1` 逐帧识别；动作很慢时可以适当加大，动作很快建议保持 `1`。
- JSON 中新增 `extract_config`、`postprocess`、`stats`、`quality_report` 字段，原有 `frames[].pose_landmarks` / `hands` 结构保持兼容。
- `quality_report` 会汇总 pose/手部覆盖率、低可见度比例、补点比例，并给出 `issues` 提示，方便快速判断素材是否有远景、遮挡、跳帧过高等问题。
- 可用 `--max_frames 120` 先跑小样本调参数，再处理整段视频。

针对不同素材的建议：

- 远景/小人像：提高 `--upsample_min_short_side` 到 `900` 或 `1080`，并保持 `--detection_stride 1`。
- 手部经常漏检：适当增大 `--hand_roi_expansion_factor` 到 `1.5~1.8`，让腕部 ROI 搜索范围更宽。
- 遮挡较多：适当降低 `--pose_detection_confidence` / `--pose_presence_confidence` 到 `0.35~0.4`，同时保留 `--interpolate_gap_frames 4~6`。
- 快动作：保持 `--detection_stride 1`，并把 `--pose_smoothing_window` / `--hand_smoothing_window` 控制在 `3~5`，避免过度抹平动作峰值。

规则方式生成 emoji 时间线：

```bash
python3 analyze_pose_emoji.py wudao/angel_pose.json
```

分析公网可访问视频：

```bash
export ARK_API_KEY=...
python3 analyze_video_emoji_volcengine.py --video-url "https://example.com/demo.mp4"
```

上传本地视频到 TOS 后再分析：

```bash
export ARK_API_KEY=...
export TOS_ACCESS_KEY=...
export TOS_SECRET_KEY=...
export TOS_BUCKET=...
python3 upload_and_analyze_video_volcengine.py wudao/angel.mp4
```
