# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 常用命令

- 启动浏览器端工作台（旧入口）：`npm run dev`（实际执行 `./.venv/bin/python dev_server.py`，必须先建好 `.venv`）。默认监听 `127.0.0.1:4173`，入口 `http://127.0.0.1:4173/pose_viewer.html`。项目当前没有打包/编译步骤，`pose_viewer.html` 是自包含的单文件应用。
- 启动新入口（Vite + TS · Phase 1 骨架）：`npm run web:dev`（监听 4300，`/api` 与 `/wudao` 反代到 4173 的 `dev_server.py`，所以两端要一起开）。`npm run web:build` 走 `tsc -b && vite build`，`npm run web:test` 跑 vitest。详见 `docs/refactor-roadmap.md`。
- 不要用 `python3 -m http.server` 代替 `dev_server.py`：前者能打开页面，但准备页的"上传短视频并识别"依赖 `dev_server.py` 暴露的 `/api/extract-pose`（另有 `/api/health`，上传上限 80MB）。
- Python 依赖通过随附的 `.venv` 运行：`source .venv/bin/activate`。关键依赖来自 `requirements.txt`：`mediapipe`、`numpy`、`opencv-python`、`tos`。
- 从视频离线提取动作 JSON：`python3 extract_pose.py <视频路径> <输出 json 路径>`（默认 `wudao/angel.mp4` → `dance_data.json`）。首次运行会自动从 Google 存储下载 `pose_landmarker.task` / `hand_landmarker.task`（两者已 checkin，删除后会被重下）。常用调参见 `README.md`（上采样、手部 ROI 扩张、平滑窗等）。
- 旧入口下 `npm test` 没用；要跑测试用 `npm run web:test`���vitest，仅覆盖 web/ 下已迁的纯函数模块）。

## 架构

两阶段 + 一条可选分析支线，全部通过 JSON 解耦。

1. **离线姿态提取 (`extract_pose.py`)**：`cv2.VideoCapture` 逐帧读取 → MediaPipe **Tasks** `PoseLandmarker` + `HandLandmarker`（`detect` 同步 API，不是 Solutions API，走视频时序跟踪而不是逐帧独立检测）→ 输出 `{fps, frames: [{time, pose_landmarks, hands: [{handedness, landmarks[21], world_landmarks, finger_count}]}], extract_config, postprocess, stats, quality_report}`。脚本强制校验 21 点手部完整性与坐标归一化范围、做时间轴平滑与短时丢帧补点；当全图手部检测缺失时会基于腕/肘/肩估算 ROI 做二次手部检测。
2. **在线比对 (`pose_viewer.html`)**：通过 jsDelivr CDN 加载 MediaPipe **Solutions** 的 `@mediapipe/pose@0.5.1675469404` 与 `@mediapipe/hands@0.4.1675469246`（注意是 Solutions 版本，与 Python 端的 Tasks 版本不同源，landmark 索引/schema 需逐处对齐）。用户可上传 JSON 作为"标准动作"，或在准备页上传短视频由 `dev_server.py` 调用 `extract_pose.py` 现场生成；摄像头实时姿态与该 JSON 按时间轴对齐后进行评分渲染。
3. **emoji/颜色时间线分析支线**（可选）：`scripts/analysis/analyze_pose_emoji.py`（基于规则，消费阶段 1 输出的 `frames[].pose_landmarks`）；`scripts/analysis/analyze_video_emoji_volcengine.py`（公网 URL 走火山 ARK Responses API）；`scripts/analysis/upload_and_analyze_video_volcengine.py`（先上传到火山 TOS 再分析，支持 `--dry-run`，会 `from analyze_video_emoji_volcengine import ...`，所以两脚本必须同目录）；`scripts/analysis/analyze_emoji_color_timeline_volcengine.py`（颜色+音频+同步分析）；`scripts/analysis/volcengine_emoji_prompt_bank.json` 是提示词库。分析产物（`*_volcengine_analysis.json` / `test_output.json` 等）默认写到 `output/`，已加到 `.gitignore`。这条支线不参与浏览器评分循环。

`wudao/` 是示例素材：同名 `*.mp4` 是原始视频，`*_pose.json` 是阶段 1 产物。加载 JSON 时 viewer 只依赖 JSON，无需视频文件。

## 评分算法关键点 (`pose_viewer.html`)

- **躯干参考系**：`getPoseReference` 基于肩中点、髋中点构造坐标轴与尺度 (`shoulder/hip center + xAxis + scale`)。`transformPointsWithReferences` 把标准动作 JSON 的坐标系重投影到当前用户躯干上，因此"白色导师骨架"会贴合用户位置/大小，而不是固定在原视频位置。
- **相似度**：`comparePoses` 计算多条骨骼向量（双臂、腿、躯干等索引来自 MediaPipe Pose 33 点约定）的 `(cosineSimilarity + 1) / 2`，按权重聚合为 0–100 分；手部通过 `normalizeHandLandmarks` 归一化后用 `findMatchingHand` 按 handedness 匹配再比对。
- **降级路径**：`disableHands` / `state.handsDisabledReason` / `state.matchDisabledReason` 构成统一降级语义——手部模型加载失败、JSON 对齐异常、评分异常都会被 `safeRefreshMatchScore` / `safeGetAlignedFrameData` 捕获并在 HUD 展示，而不是抛错中断渲染循环。新增评分/对齐逻辑时务必走这两层安全包装。
- **时间轴回放**：`getCurrentTargetFrame` 用 `performance.now()` 与 `playSpeed` 定位最接近的标准帧；修改播放逻辑要同时处理 `state.isPlaying`、`state.playbackStartAt`、`state.playbackStartFrame`。

## 火山引擎相关环境变量（仅分析支线需要）

```
ARK_API_KEY         # 调 ARK Responses API 必需
TOS_ACCESS_KEY      # 上传本地视频到 TOS 必需
TOS_SECRET_KEY
TOS_BUCKET
TOS_REGION          # 可选，默认按 bucket 推导
TOS_ENDPOINT        # 可选
```

## 其他约定

- `legacy/App.jsx` 是早期 React 原型（CDN 模式、无打包入口、不被 `pose_viewer.html` 引用），归档保留。新功能应做在 `pose_viewer.html` 上，或 Phase 1 之后的 `web/` Vite 工程上（见 `docs/refactor-roadmap.md`）。
- 大体量产物：`wudao/*.mp4` / `*.task` / `*_pose.json` 仍然 checkin 在工作目录；`output/` 和 `assets/frame*.jpg` 等中间产物已加 `.gitignore`。改动相关流程前先确认不要再把产物提回主仓。
- Python 端代码注释与 UI 文案均为中文，保持同一风格。
- MediaPipe Pose 关键点索引硬编码在多处（11/12 肩、13/14 肘、15/16 腕、23/24 髋等）；调整比对向量时要成对更新 `comparePoses` 与 `getPoseReference`。
- 目录规划：`scripts/analysis/`（分析支线 py）· `output/`（分析产物，gitignored）· `assets/`（本地调试图/音，部分 gitignored）· `legacy/`（归档）· `docs/`（路线图 + 改造计划）。
- 前端改造规划备忘在 `docs/pose-viewer-frontend-refactor-plan.md`；整体路线图与 Phase 拆分在 `docs/refactor-roadmap.md`；产品方向与优化评估在 `docs/product-and-optimization-roadmap.md`。
