# MediaPipe + 评分机制优化点

> 与 `docs/refactor-roadmap.md` Phase 2 协同推进。本文聚焦"识别管线 + 评分公式"两层的具体可优化点，按"落地难度 vs 收益"排序。每条都标了**何时做**——P0 = 当前 PracticeView 集成时一起做；P1 = MVP 上线前；P2 = 后续迭代。

## 1. MediaPipe 端

### 1.1 同源对齐到 Tasks Web（P1，影响最大）

- **现状**：浏览器走 MediaPipe **Solutions**（CDN：`@mediapipe/pose@0.5.1675469404`、`@mediapipe/hands@0.4.1675469246`），Python 离线走 **Tasks**（`PoseLandmarker.detect_for_video`）。两套 SDK 的 landmark schema、置信度字段名、坐标空间都有细微差别，是当前评分跨端不完全可比的根源。
- **改造**：浏览器迁到 `@mediapipe/tasks-vision`（PoseLandmarker / HandLandmarker），与 Python 端共用同一份 `.task` 模型文件；Vite 端 dynamic import + CDN-resolved fileset。
- **收益**：landmark 定义对齐 → 评分跨端一致；可与 Python 端共享黄金样本。
- **代价**：模型 + WASM 约 10–15 MB；首屏需要 lazy 加载与骨架占位。
- **当前 PracticeView 集成（本轮 P0）**：直接用 Tasks Web 起步，避开二次迁移。

### 1.2 GPU delegate 自动回落（P0）

- Tasks Web 支持 `delegate: 'GPU'`；桌面 Chrome / Safari 实测 2–3× 加速。
- 失败时回落 CPU；需要捕获 WASM 初始化错误并重建 detector。
- **当前 PracticeView 集成（本轮 P0）**：在 `poseDetector.ts` 的 create 路径里写"GPU first, fallback CPU"。

### 1.3 检测节流（detection stride）（P0）

- 摄像头 30 FPS，但人眼对评分稳定性的容忍 > 100 ms。
- **改造**：每 N 帧调一次 `detectForVideo`（默认 N=2），中间帧复用上次 landmark；评分本身改为按 100 ms 节流（`requestAnimationFrame` 里累计 ts，超过阈值才调 evaluate）。
- **当前 PracticeView 集成（本轮 P0）**：在 `practiceLoop.ts` 实现。
- 与 Python 端 `detection_stride` 参数对齐。

### 1.4 OffscreenCanvas + Worker（P2）

- Tasks Web 支持在 worker 里跑 `PoseLandmarker.detect`；主线程只负责 RAF 调度 + 绘制。
- **何时做**：当 detection stride + EMA 还压不住主线程长任务时再考虑。

### 1.5 模型加载体验（P1）

- 预热：进入 setup 页时就并行 `FilesetResolver.forVisionTasks(...)` 拉 wasm，不影响 setup 渲染。
- 加载中显示骨架占位 + 当前进度（fetch progress event）。
- 失败有明显降级（"未能加载姿态识别模型，进入观看模式"）。

### 1.6 可见度阈值动态化（P2）

- 当前评分时硬编码 `visibility >= 0.35`，低光场景容易把所有点过滤掉。
- 改为：取整帧 visibility 的 60% 分位数当阈值，避免 hard cutoff。

---

## 2. 评分机制

### 2.1 一阶 IIR 平滑（EMA）（P0）

- **现状**：`matchScore` 每帧抖 5–10 分，HUD 上字符乱跳。
- **改造**：`s = α·new + (1−α)·prev`，`α=0.3`；UI 显示前再 `Math.round`。
- **收益**：分数视觉稳定 + 不损失实时性（峰值在 ~3 帧内追平）。
- **当前 PracticeView 集成（本轮 P0）**：在 `scoreSmoothing.ts` 落地。

### 2.2 浮点贯穿，仅 UI round（P0）

- 当前 `compareArmPoses` 内部 `Math.round`，导致 EMA 在整数台阶上震荡。
- 改造：score 在内部保留浮点，传到 view 层再 round；这要求新建一份"内部"路径，不破坏 legacy 行为（legacy ���整数取整不动，避免 golden-sample 失效）。
- 在新 `scoringLoop.ts` 里增加 `evaluateFrameFloat` 路径供 PracticeView 用。

### 2.3 DTW 时间对齐（P1）

- **现状**：用户和老师按 `playSpeed` 死对齐，用户慢半拍直接 0 分。
- **改造**：在最近 ±300 ms 窗口内做 DTW，找最佳对齐帧；评分时用对齐后的 target。
- **代价**：每帧多 O(W²) 计算（W=窗口帧数 ≈ 9），可控。

### 2.4 拍点感知聚合（P1）

- 当前最终分是过程均分；舞蹈讲究"踩拍"。
- 改造：把整段切分为 beat segments（来自背景音 BPM），每段单独打分；最终分按"段命中率 + 段平均"双指标。

### 2.5 visibility 加权而不是阈值（P2）

- 现在 `visibility < 0.35` 直接跳过；不公平。
- 改造：把 visibility 当 segment 权重：`score *= avg(visibility)^β`。

### 2.6 方向 vs 长度的权重（P2）

- 用户身材小或离镜头远时，length 持续偏低，把分数压下去。
- 改造：当 `targetLength < threshold` 时把 lengthScore 权重降到 0.05。

### 2.7 分项扩到躯干/胯部（P2）

- 当前只评双臂；半身舞还有肩线扭动、胯部摆动。
- 改造：新增 `compareTorsoSway`、`compareHipSway`，权重可配置。

---

## 3. PracticeView render loop 集成顺序

按"先动起来，再调精度"的节奏推进：

1. **本轮**：
   - PoseDetector 抽象 + Tasks Web 默认实现（GPU→CPU 回落）。
   - EMA 平滑 + detection stride（每 2 帧）。
   - rAF 循环 + 离开 practice 时清理。
   - 接到 PracticeView：实时更新 `#scoreDialValue` / `#practiceScoreLabel`。
2. **下一轮**：
   - canvas 绘制层（teacher skeleton + user pose），用 `stageRenderer.ts` 的纯换算。
   - 背景音 + frame.time 同步。
3. **再下一轮**：
   - DTW 时间对齐。
   - beat-aware 评分聚合。

---

## 4. 测试策略

- **核心评分**：`web/src/core/__tests__/goldenSamples.test.ts` 已覆盖；将 EMA 与 stride 的等价性也加入回归（同一序列 + α=0、stride=1 时与原始等价）。
- **render loop**：用 mock detector + mock requestAnimationFrame 跑确定性测试。
- **PracticeView**：jsdom 已就位，所有点击/状态切换路径继续覆盖。
