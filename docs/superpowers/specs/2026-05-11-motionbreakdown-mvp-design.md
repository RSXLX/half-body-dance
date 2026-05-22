# MotionBreakdown MVP 设计

## 背景

仓库已经具备离线姿态提取、`analyze_motion.py` 规则分析、`*_motion.json` 示例产物，以及 `dev_server.py` 的 `/api/analyze-motion` 能力。当前缺口是：用户在新入口 `web/` 中还不能稳定看到“动作分解”产品形态，也不能把 preset 和上传视频统一串到可视化学习流程。

第一轮目标是做一个可交付的 MotionBreakdown MVP：同时支持 preset 和上传视频，使用规则法 motion 产物，不接 LLM，不做前端音频节拍分析。

## 目标

1. 新入口中提供“动作分解”视图，展示 motion 的拍点和片段。
2. Preset 路径能加载已有 `*_motion.json`，并显示动作分解入口。
3. 上传视频路径能在 `/api/extract-pose` 成功后调用 `/api/analyze-motion`，把返回的 motion 用同一视图展示。
4. 点击片段能进入 PracticeView，并定位到片段开始时间附近。
5. motion 缺失或分析失败时不阻塞普通跟练。

## 非目标

- 不调用 LLM 分析视频。
- 不让 LLM 参与动作、关节、方向、拍点或片段判定。
- 不做前端 Web Audio 节拍分析。
- 不做 `JointLane`、`KeyframeStrip`、关键帧缩略图。
- 不在本轮下线或归档 `pose_viewer.html`。
- 不重写 `analyze_motion.py` 的核心算法，除非发现与前端接入直接相关的小缺口。

## 路由与入口

在 `web/` 新入口中新增 `MotionBreakdownView`。它不是替代 PracticeView，而是位于 Setup 与 Practice 之间的学习视图。

- 用户选择 preset 后，SetupView 保持现有 pose 加载流程，并额外尝试加载 motion。
- 用户上传视频后，SetupView 先完成 pose 提取，再触发 motion 分析。
- motion 可用时，页面显示“动作分解”入口。
- 用户也可以跳过动作分解，直接进入普通练习。
- 在 MotionBreakdownView 点击片段时，切到 PracticeView，并传入该片段的 `startTime` 作为起始时间。

## 数据流

### Preset 路径

1. Preset 配置包含或推导出 `motionPath`。
2. 选择 preset 时加载 `posePath`。
3. 同时尝试加载 `motionPath`。
4. 成功后写入统一状态，例如 `state.motion`。
5. 失败时记录错误，并把 motion 状态置为不可用。

### 上传路径

1. 用户上传视频。
2. 前端调用 `/api/extract-pose`。
3. 如果 pose 成功，前端调用 `/api/analyze-motion`，payload 带 `poseJson`。
4. `/api/analyze-motion` 返回 `{ ok: true, motion }` 后写入统一状态。
5. 如果 motion 失败，保留 pose，允许继续普通练习。

### LLM 策略

本轮不接 LLM。规则法 `analyze_motion.py` 的产物是唯一真源。

未来如需接 LLM，只作为可选文案润色层，且只能改写 `segment.title`、`segment.tip`、`hint.preview` 一类展示文本；不得改写 `beats`、`segments`、`primaryJoint`、`direction`、时间戳或评分相关字段。

## 组件边界

### MotionBreakdownView

容器视图，负责：

- 接收 `motion`、当前 pose/preset 信息和回调。
- 渲染拍点概览与片段列表。
- 管理选中片段。
- 触发“进入练习”或“从片段开始练习”。

它不做动作算法判断，只消费 motion JSON。

### BeatStrip

展示 `motion.beats[]`。

- 一拍一格横向展示。
- 显示时间、emoji/方向、主关节等已有字段。
- 空 beats 时显示空态，而不是抛错。

### SegmentCard

展示 `motion.segments[]`。

- 显示标题、时间范围、emoji、难度或提示文案。
- 点击后调用 `onSelectSegment(segment)`。
- 不直接操作全局状态。

## 错误处理

- `*_motion.json` 加载失败：不影响 pose 加载和练习，只隐藏或禁用动作分解入口，并显示“动作分解暂不可用”。
- `/api/extract-pose` 成功但 `/api/analyze-motion` 失败：保留上传 pose，允许普通跟练，提示“动作已识别，但分解生成失败，可继续练习”。
- motion schema 不完整：前端轻量 guard；`beats` 或 `segments` 不是数组时视为不可用。
- 点击片段时缺少有效 `startTime`：留在分解页，不跳转。
- motion 空数据：显示空态，不阻塞用户返回 Setup 或进入普通 Practice。

## 测试计划

### 单元测试

- motion guard：合法 motion、缺 `beats`、缺 `segments`、字段类型错误。
- BeatStrip：给定 beats，渲染对应数量与核心文案。
- SegmentCard：点击时传回正确 segment。
- 上传 motion 链路函数：mock `/api/extract-pose` 与 `/api/analyze-motion`，覆盖成功、motion 失败、pose 失败。

### 视图测试

- preset 有 `motionPath` 且加载成功时，显示动作分解入口。
- preset motion 加载失败时，仍能进入普通练习。
- 上传视频 motion 分析失败时，仍能进入普通练习。
- 点击片段后进入 PracticeView，并使用片段开始时间。

### 验证命令

- `npm run web:test`
- `npm run web:build`
- 必要时运行：`python3 scripts/analysis/analyze_motion.py wudao/angel_pose.json --output /tmp/angel_motion_test.json`

## 第一轮交付判断

完成后应能验证：

1. 选择一个已有 preset 后，可以进入动作分解视图。
2. 上传一个短视频后，pose 与 motion 能串联生成；motion 失败时普通练习仍可用。
3. MotionBreakdownView 至少展示拍点条和片段卡片。
4. 点击片段可以从对应时间进入练习。
5. 所有新增前端测试通过，`web` 构建通过。
