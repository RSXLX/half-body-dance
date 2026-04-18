# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

- 启动浏览器端工作台：`npm run dev`（等价于 `python3 -m http.server 4173`），随后在浏览器打开 `http://localhost:4173/pose_viewer.html`。项目没有打包/编译步骤，`pose_viewer.html` 是自包含的单文件应用。
- 从视频离线提取动作 JSON：`python3 extract_pose.py <视频路径> <输出 json 路径>`（默认读取 `wudao/angel.mp4` → `dance_data.json`）。首次运行会自动从 Google 存储下载 `pose_landmarker.task` / `hand_landmarker.task` 到仓库根目录（这两个 `.task` 模型文件已 checkin，删除后会被重新下载）。
- Python 依赖需通过随附的 `.venv` 运行：`source .venv/bin/activate`。关键依赖是 `mediapipe`、`opencv-python`、`numpy`。
- `npm test` 仅输出占位文本；当前无自动化测试与 lint。

## 架构

两阶段流水线，通过 JSON 解耦：

1. **离线提取 (`extract_pose.py`)**：`cv2.VideoCapture` 逐帧读取 → MediaPipe Tasks `PoseLandmarker` + `HandLandmarker`（`detect` 同步 API，不是 Solutions API）→ 输出 `{fps, frames: [{time, pose_landmarks, hands: [{handedness, landmarks[21], world_landmarks, finger_count}]}]}`。脚本强制校验 21 点手部完整性与坐标归一化范围，丢弃残缺帧，避免污染下游评分。
2. **在线比对 (`pose_viewer.html`)**：通过 jsDelivr CDN 加载 MediaPipe Solutions 的 `@mediapipe/pose@0.5.1675469404` 与 `@mediapipe/hands@0.4.1675469246`（注意是 Solutions 版本，与 Python 端的 Tasks 版本不同源，landmark 索引/schema 需逐处对齐）。用户可上传 JSON 作为"标准动作"，摄像头实时姿态与该 JSON 按时间轴对齐后进行评分渲染。

`wudao/` 是示例素材：同名的 `*.mp4` 是原始视频，`*_pose.json` 是离线提取产物。加载 JSON 时 viewer 只依赖 JSON，无需视频文件。

## 评分算法关键点 (`pose_viewer.html`)

- **躯干参考系**：`getPoseReference` 基于肩中点、髋中点构造坐标轴与尺度 (`shoulder/hip center + xAxis + scale`)。`transformPointsWithReferences` 把标准动作 JSON 的坐标系重投影到当前用户躯干上，因此"白色导师骨架"会贴合用户位置/大小，而不是固定在原视频位置。
- **相似度**：`comparePoses` 计算多条骨骼向量（双臂、腿、躯干等索引来自 MediaPipe Pose 33 点约定）的 `(cosineSimilarity + 1) / 2`，按权重聚合为 0–100 分；手部通过 `normalizeHandLandmarks` 归一化后用 `findMatchingHand` 按 handedness 匹配再比对。
- **降级路径**：`disableHands` / `state.handsDisabledReason` / `state.matchDisabledReason` 构成统一降级语义——手部模型加载失败、JSON 对齐异常、评分异常都会被 `safeRefreshMatchScore` / `safeGetAlignedFrameData` 捕获并在 HUD 展示，而不是抛错中断渲染循环。新增评分/对齐逻辑时务必走这两层安全包装。
- **时间轴回放**：`getCurrentTargetFrame` 用 `performance.now()` 与 `playSpeed` 定位最接近的标准帧；修改播放逻辑要同时处理 `state.isPlaying`、`state.playbackStartAt`、`state.playbackStartFrame`。

## 其他约定

- `App.jsx` 是早期 React 原型（CDN 模式、无打包入口、不被 `pose_viewer.html` 引用），与现行 viewer 并行保留。新功能应做在 `pose_viewer.html` 上，除非明确要求迁移回 React。
- 大体量产物（`*.mp4`、`*.task`、`*_pose.json`、`test_output.json`）都直接 checkin 在工作目录，改动相关流程前先确认是否会导致仓库再次膨胀。
- Python 端代码注释与 UI 文案均为中文，保持同一风格。
- MediaPipe Pose 关键点索引硬编码在多处（11/12 肩、13/14 肘、15/16 腕、23/24 髋等）；调整比对向量时要成对更新 `comparePoses` 与 `getPoseReference`。
