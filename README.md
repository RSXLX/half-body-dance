# half-body-dance

半身舞动作提取、跟练比对与 emoji 时间线分析工作台。

这个仓库把几类能力放在了一起：

- 本地跟练工作台：在浏览器里加载标准动作 JSON，并和摄像头实时动作做对齐与评分。
- 离线姿态提取：从舞蹈视频中提取 MediaPipe Pose / Hand 关键点，导出可复用 JSON。
- 动作语义分析：把动作谱或视频分析成 emoji 时间线，支持规则法和火山引擎视觉模型。
- 颜色时间轴分析：分析 emoji 视频里的颜色变化、音频结构和同步关系。

## 项目概览

项目当前的真实运行入口是 `pose_viewer.html`，不是 `App.jsx`。推荐工作流如下：

1. 用 `dev_server.py` 启动本地服务。
2. 在浏览器打开 `pose_viewer.html`。
3. 选择仓库内示例 JSON，或在准备页上传短视频自动生成动作结构。
4. 打开摄像头，开始跟练和动作评分。
5. 如需批量处理，再使用 Python 脚本离线提取姿态或做 emoji 分析。

仓库里的能力是通过 JSON 串起来的：

- `extract_pose.py` 负责从视频生成姿态 JSON。
- `pose_viewer.html` 负责加载 JSON 并做实时跟练。
- `analyze_pose_emoji.py` 和火山引擎脚本负责把 JSON 或视频进一步转成时间线分析结果。

## 核心文件

- `pose_viewer.html`：主界面，包含准备页、练习页、结果页三段流程。
- `dev_server.py`：本地开发服务，提供静态页面、`/api/health` 和 `/api/extract-pose`。
- `extract_pose.py`：离线姿态提取脚本，输出 MediaPipe Pose + Hand JSON。
- `analyze_pose_emoji.py`：基于规则的动作 emoji 时间线分析。
- `analyze_video_emoji_volcengine.py`：分析公网视频 URL 的动作 emoji 时间线。
- `upload_and_analyze_video_volcengine.py`：先上传本地视频到 TOS，再调用 ARK 分析。
- `analyze_emoji_color_timeline_volcengine.py`：分析 emoji 视频颜色变化与音频结构。
- `volcengine_emoji_prompt_bank.json`：火山引擎动作分析提示词库。
- `wudao/`：示例视频和对应姿态 JSON。
- `App.jsx`：早期 React 原型，不是当前主入口。

## 环境准备

### Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- `mediapipe`
- `numpy`
- `opencv-python`
- `tos`

仓库根目录已经带有 `pose_landmarker.task` 和 `hand_landmarker.task`。如果这两个模型文件缺失，`extract_pose.py` 首次运行时会自动下载。

### Node 脚本

`package.json` 里只有一个很薄的封装：

```bash
npm run dev
```

它实际调用的是 `./.venv/bin/python dev_server.py`，所以前提仍然是先创建并安装好 `.venv`。

## 快速开始

### 1. 启动本地跟练工作台

```bash
python3 dev_server.py
```

默认监听 `127.0.0.1:4173`，打开：

[http://127.0.0.1:4173/pose_viewer.html](http://127.0.0.1:4173/pose_viewer.html)

可选参数：

```bash
python3 dev_server.py --host 0.0.0.0 --port 4173
```

服务能力：

- 静态页面入口：`/pose_viewer.html`
- 健康检查：`/api/health`
- 上传短视频并提取姿态：`/api/extract-pose`

注意：

- 准备页里的“上传短视频并识别”依赖 `/api/extract-pose`，必须通过 `dev_server.py` 启动。
- 不能只用 `python3 -m http.server`，否则页面可以打开，但上传识别功能不可用。
- 上传接口限制为 80MB 以内的短视频。

### 2. 使用示例数据

仓库自带 `wudao/` 示例素材，包含：

- 原始视频：如 `wudao/angel.mp4`
- 提取结果：如 `wudao/angel_pose.json`

在页面中可直接加载这些 JSON 做跟练，也可以拿视频重新跑提取参数。

## 离线姿态提取

基础用法：

```bash
python3 extract_pose.py wudao/angel.mp4 wudao/angel_pose.json
```

如果不传参数，默认处理：

```bash
python3 extract_pose.py
```

等价于：

```bash
python3 extract_pose.py wudao/angel.mp4 dance_data.json
```

更稳的一组参数示例：

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

提取器当前的实现特点：

- 基于 MediaPipe Tasks 的 `PoseLandmarker` 和 `HandLandmarker`。
- 使用视频时序跟踪，而不是完全逐帧独立检测。
- 会对 `pose` 和 `hands` 做时间轴平滑与短时丢帧补点。
- 对远景、小手势场景支持自动放大后识别。
- 当全图手部检测缺失时，会基于腕/肘/肩估算 ROI 做二次手部检测。
- 输出结果里会附带 `extract_config`、`postprocess`、`stats`、`quality_report` 等元信息。

常用调参建议：

- 远景或人物太小：提高 `--upsample_min_short_side` 到 `900` 或 `1080`。
- 手部经常漏检：提高 `--hand_roi_expansion_factor` 到 `1.5` 到 `1.8`。
- 遮挡较多：适当降低 `--pose_detection_confidence` 和 `--pose_presence_confidence`。
- 快动作素材：保持 `--detection_stride 1`，并把平滑窗口控制在 `3` 到 `5`。
- 先试小样本：加 `--max_frames 120` 先验证参数，再处理完整视频。

## 浏览器跟练工作台

`pose_viewer.html` 的核心能力：

- 加载标准动作 JSON，并按时间轴回放。
- 用摄像头实时识别人像与手部动作。
- 把标准动作骨架重投影到当前用户身体参考系上，减少位置和身材差异的影响。
- 计算双臂和整体动作相似度，并给出实时评分。
- 当手部模型、对齐逻辑或评分逻辑异常时，会自动降级，而不是直接让页面崩掉。

准备页支持两种方式导入标准动作：

- 直接加载已有 JSON。
- 上传一个短视频，让本地服务调用 `extract_pose.py` 生成动作结构。

## emoji 动作时间线分析

### 1. 基于姿态 JSON 的规则分析

```bash
python3 analyze_pose_emoji.py wudao/angel_pose.json
```

可选输出路径：

```bash
python3 analyze_pose_emoji.py wudao/angel_pose.json -o wudao/angel_emoji_analysis.json
```

这个脚本会读取 `frames[].pose_landmarks`，按时间段生成：

- `timeline`
- `segments`
- `emoji_summary`

适合快速做规则型拆段和动作粗分类。

### 2. 分析公网可访问视频

```bash
export ARK_API_KEY=...
python3 analyze_video_emoji_volcengine.py --video-url "https://example.com/demo.mp4"
```

常见可选项：

- `--sampling-interval`
- `--max-sampling-frames`
- `--extra-guidance`
- `--prompt-bank`
- `--model`
- `-o/--output`

### 3. 上传本地视频到 TOS 后分析

```bash
export ARK_API_KEY=...
export TOS_ACCESS_KEY=...
export TOS_SECRET_KEY=...
export TOS_BUCKET=...
python3 upload_and_analyze_video_volcengine.py wudao/angel.mp4
```

这个脚本会：

1. 把本地 MP4 上传到火山 TOS。
2. 生成临时可访问 URL。
3. 调用 ARK Responses API 分析动作。
4. 输出结构化 JSON 结果。

如果只想预览 payload 和对象 key，不实际发请求，可以加：

```bash
python3 upload_and_analyze_video_volcengine.py wudao/angel.mp4 --dry-run
```

## emoji 颜色与音频时间轴分析

分析公网视频中的 emoji 颜色变化、音频段落和同步关系：

```bash
export ARK_API_KEY=...
python3 analyze_emoji_color_timeline_volcengine.py \
  --video-url "https://example.com/emoji.mp4"
```

输出结果会包含：

- `video_summary`
- `audio_summary`
- `emoji_color_timeline`
- `audio_structure`
- `sync_observations`

如果只想检查 prompt 和请求体：

```bash
python3 analyze_emoji_color_timeline_volcengine.py \
  --video-url "https://example.com/emoji.mp4" \
  --dry-run
```

## 火山引擎相关环境变量

动作分析与颜色时间轴分析涉及以下环境变量：

```bash
export ARK_API_KEY=...
export TOS_ACCESS_KEY=...
export TOS_SECRET_KEY=...
export TOS_BUCKET=...
export TOS_REGION=cn-beijing
export TOS_ENDPOINT=tos-cn-beijing.volces.com
```

其中：

- `ARK_API_KEY`：调用 ARK Responses API 必需。
- `TOS_ACCESS_KEY` / `TOS_SECRET_KEY` / `TOS_BUCKET`：上传本地视频到 TOS 时必需。
- `TOS_REGION` 和 `TOS_ENDPOINT`：可选，不传时脚本会按默认区域推导。

## 仓库结构

```text
.
├── README.md
├── dev_server.py
├── extract_pose.py
├── analyze_pose_emoji.py
├── analyze_video_emoji_volcengine.py
├── upload_and_analyze_video_volcengine.py
├── analyze_emoji_color_timeline_volcengine.py
├── pose_viewer.html
├── App.jsx
├── volcengine_emoji_prompt_bank.json
├── docs/
│   └── pose-viewer-frontend-refactor-plan.md
└── wudao/
    ├── *.mp4
    ├── *_pose.json
    └── emoji/
```

## 当前约定与限制

- 当前没有自动化测试，`npm test` 只会输出占位文本。
- `App.jsx` 是早期原型，当前功能迭代应以 `pose_viewer.html` 为主。
- 仓库里直接提交了示例视频、姿态 JSON 和模型文件，改动流程时要注意产物体积。
- 页面和脚本文案当前以中文为主，后续新增说明建议保持一致风格。

## 相关文档

- 前端改造规划：[docs/pose-viewer-frontend-refactor-plan.md](docs/pose-viewer-frontend-refactor-plan.md)
