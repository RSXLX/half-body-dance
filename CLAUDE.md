# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

- 安装 Python 依赖：`python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install -r requirements.txt`。
- 启动旧入口工作台：`npm run dev`（等价于 `./.venv/bin/python dev_server.py`），默认监听 `127.0.0.1:4173`，入口 `http://127.0.0.1:4173/pose_viewer.html`。
- 启动新入口（Vite + TS）：先开 `npm run dev`，再开 `npm run web:dev`。Vite 默认监听 `127.0.0.1:4300`，`/api` 与 `/wudao` 会反代到 4173 的 Python 服务。
- 构建新入口：`npm run web:build`（在 `web/` 内执行 `tsc -b && vite build`）。
- 跑新入口测试：`npm run web:test` 或 `npm test`（vitest run）。单个测试可用 `npm --prefix web run test -- <文件或名称片段>`，例如 `npm --prefix web run test -- goldenSamples`。
- 离线提取姿态 JSON：`python3 extract_pose.py <视频路径> <输出 json 路径>`；不传参时默认 `wudao/angel.mp4` → `dance_data.json`。常用参数见 `README.md`（上采样、手部 ROI 扩张、平滑窗、`--max_frames` 等）。
- 动作分析 JSON：`python3 scripts/analysis/analyze_motion.py <*_pose.json> --output <*_motion.json>`；规则型 emoji 分析：`python3 scripts/analysis/analyze_pose_emoji.py <*_pose.json>`。
- 不要用 `python3 -m http.server` 代替 `dev_server.py`：页面能打开，但准备页上传短视频、动作分析等 `/api/*` 能力不可用。

## 高层架构

项目由“离线提取 → JSON 数据 → 浏览器跟练/分析”串起来，旧单文件入口和新 Vite 入口并存。

1. **离线姿态提取 (`extract_pose.py`)**：`cv2.VideoCapture` 逐帧读取视频 → MediaPipe **Tasks** `PoseLandmarker` + `HandLandmarker` → 输出 `{fps, frames, extract_config, postprocess, stats, quality_report}`。脚本会校验 21 点手部完整性与坐标范围，做时间轴平滑、短时丢帧补点；手部漏检时会基于腕/肘/肩估算 ROI 做二次检测。
2. **本地 Python 服务 (`dev_server.py`)**：提供静态文件、`/api/health`、`/api/extract-pose` 和 `/api/analyze-motion`。上传接口限制 80MB，临时文件写入 `.tmp/`；`/api/analyze-motion` 调用 `scripts/analysis/analyze_motion.py`，生成 lesson/motion JSON。
3. **旧浏览器入口 (`pose_viewer.html`)**：自包含 HTML 应用，通过 jsDelivr CDN 加载 MediaPipe **Solutions** `@mediapipe/pose@0.5.1675469404` 与 `@mediapipe/hands@0.4.1675469246`。它消费 `extract_pose.py` 的 JSON 或上传视频实时生成的 JSON，按时间轴对齐摄像头姿态并评分。
4. **新浏览器入口 (`web/`)**：Vite + TypeScript 迁移工程。`web/src/core/` 放评分、参考系、手部归一化、节拍/动作分析等纯函数；`web/src/app/` 放控制器、练习循环、渲染和视图；`web/src/ui/` 放 UI 组件。`web/vite.config.ts` 将 `/api` 和 `/wudao` 代理到 4173。
5. **分析支线 (`scripts/analysis/`)**：`analyze_motion.py` 生成关节轨迹、动作片段和提示 cue；`analyze_pose_emoji.py` 做规则型 emoji 时间线；火山引擎脚本通过 ARK/TOS 分析公网或上传视频。分析产物默认放 `output/` 或显式输出路径，不参与实时评分循环。

`wudao/` 存示例素材：同名 `*.mp4` 是原始视频，`*_pose.json` 是姿态提取结果，`*_motion.json` 是动作/课程分析结果。浏览器加载 JSON 后不依赖原视频。

## 评分与动作数据关键点

- MediaPipe Pose 关键点索引硬编码在前后端多处（11/12 肩、13/14 肘、15/16 腕、23/24 髋、27/28 踝等）。调整比对向量、动作分析关节或 schema 时，要同步检查 `pose_viewer.html`、`web/src/core/*`、`scripts/analysis/analyze_motion.py`。
- 躯干参考系由肩中点、髋中点、肩/髋轴和尺度构造。旧入口函数是 `getPoseReference` / `transformPointsWithReferences`；新入口对应 `web/src/core/reference.ts`。标准动作会重投影到当前用户身体参考系，避免原视频位置/身材差异直接影响评分。
- 相似度使用骨骼向量 cosine 聚合到 0–100；手部先 `normalizeHandLandmarks`，再按 handedness 匹配。新入口核心实现在 `web/src/core/poseCompare.ts`、`similarity.ts`、`hands.ts`。
- 旧入口有统一降级路径：`disableHands`、`state.handsDisabledReason`、`state.matchDisabledReason`、`safeRefreshMatchScore`、`safeGetAlignedFrameData`。改评分/对齐逻辑时不要绕过这些包装。
- `scripts/analysis/analyze_motion.py` 的 `SCHEMA_VERSION` 和 `web/src/core/motionTypes.ts` 需要保持一致；schema 说明在 `docs/motion-analysis-and-lesson-mode.md`。

## 火山引擎相关环境变量（仅分析支线需要）

```bash
ARK_API_KEY
TOS_ACCESS_KEY
TOS_SECRET_KEY
TOS_BUCKET
TOS_REGION      # 可选，默认按 bucket 推导
TOS_ENDPOINT    # 可选
```

## 项目约定与当前状态

- `legacy/App.jsx` 是早期 React 原型，不是当前入口；`pose_viewer.html` 仍是可用旧入口，新功能优先判断应落在旧入口、`web/` 新入口，还是两边都要同步。
- `web/` 目录里存在 TypeScript 源文件及编译出的同名 `.js` 文件；修改逻辑时优先改 `.ts`，再通过 `npm run web:build` 让编译产物保持一致。
- Python 端代码注释与 UI 文案以中文为主，新增用户可见文案保持中文风格。
- 大体量素材和模型文件（`wudao/*.mp4`、`*.task`、`*_pose.json` 等）仍在仓库内；`output/`、`.tmp/`、部分 `assets/` 中间产物应保持 gitignored，不要把新分析产物带回主仓。
- 重要背景文档：`docs/refactor-roadmap.md`（迁移路线）、`docs/product-and-optimization-roadmap.md`（产品方向）、`docs/motion-analysis-and-lesson-mode.md`（motion/lesson schema）。
