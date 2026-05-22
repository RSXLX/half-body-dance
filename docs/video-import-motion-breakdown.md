# 视频导入 · 动作分解（手脚摆动 + 每拍动作） 可行性与开发文档

> 用户故事：我上传一段舞蹈/跟练视频，系统帮我把它**拆开来看**——哪只手怎么甩、哪只脚怎么动、**每一拍**的动作是什么；最好能在网页上像翻页一样一拍一拍过。
>
> 本文是落地这个功能的可行性 + 详细开发计划。它和 `docs/motion-analysis-and-lesson-mode.md`（学习/跟练模式总规划）正交：那一份是面向"看着分段去练"的产品形态；本文聚焦**"导入视频→分解→分析报告"**这一条独立流水线，且大部分能力已经在仓库里就绪，只是没有串成给用户看的入口。

---

## 1. 现状盘点（什么已经能用，什么是空白）

### 1.1 已经具备

| 能力 | 位置 | 现状 |
| --- | --- | --- |
| 视频上传 → 姿态 JSON | `dev_server.py` 的 `POST /api/extract-pose` + `extract_pose.py` | 上限 80MB；返回 `_pose.json`，含 33 点 pose + 21 点 hands + visibility + 时间轴 |
| 轨迹提取与方向标注 | `scripts/analysis/analyze_motion.py` | 输出 `_motion.json`，含 6 关节（左右手腕/手肘/脚踝）轨迹、`speed/direction/cardinal`、`segments`、`hints`、`summary` |
| 8 方向量化 | `analyze_motion.py::quantize_direction` | `up/down/left/right/up{Left,Right}/down{Left,Right}/still` |
| 段落切分 | `analyze_motion.py::segment_motion` | 速度谷 + 方向翻转 + 节拍吸附 + 短段合并 |
| 段落难度/标题/要领 | `analyze_motion.py::build_segment` + `_TEMPLATES` | 已经能输出"右手向右摆"这类中文短句 |
| 节拍/BPM 估计（前端） | `pose_viewer.html::analyzeAudioBeats` | 浏览器端 Web Audio，能给 `beatTimes[]` 与 BPM |
| TS 端 schema 镜像 | `web/src/core/motionTypes.ts` | 与 Python 字段一一对应 |
| 已生成的样例产物 | `wudao/*_motion.json` | angel / ladada / shoushi / 提莫队长 / 星奇摇2.0 / 米糕摇 / 花寂寥 / 魔法城堡 / 鸿门旋律 |

### 1.2 缺口（要做的事）

1. **没有"导入视频"这一条用户路径上的"动作分解"产物**：`/api/extract-pose` 只返回 `_pose.json`，前端不会去触发 `analyze_motion.py`，所以 `_motion.json` 仅对仓库内置素材存在。
2. **音频节拍没接到后端管线**：`analyze_motion.py` 接受 `--beats` 但没人喂它；`analyzeAudioBeats` 在浏览器里跑、结果没回流给后端。
3. **没有"每一拍"粒度的输出**：现有 `segments` 是粗粒度（默认 ≤16 段），不是"一拍一格"。用户问"每一拍的动作"需要一份 `beats[]` 数组：每个元素 = 一个节拍窗口 + 该窗内的主导关节/方向/简短描述。
4. **没有动作分解报告的前端视图**：上传视频后用户只能进入跟练 / 学习模式，没有一个"分析报告"页把段、拍、轨迹、缩略图摆出来给他翻看。
5. **没有关键帧缩略图**：只有时间戳（`keyFrames: [t]`），用户在报告里看不到画面。
6. **手臂"摆动"细节不足**：当前每帧只有 wrist 的速度方向，没有"手臂从体侧抬到头顶"这种**姿态层面**的描述（弯曲度、手臂指向）。脚同理。

> 结论：可行性高。真正要写的代码集中在 (a) 把 `analyze_motion` 接进上传流水线、(b) 增加按拍切片输出、(c) 增加分解报告视图。算法侧只做小幅增强，不需要换底座。

---

## 2. 数据契约扩展

完全保持向后兼容：在现有 `_motion.json` 顶层**新增**两个字段，老消费者忽略即可。

```jsonc
{
  "schema_version": 2,                  // ← 从 1 升到 2
  "source_pose": "wudao/angel_pose.json",
  "fps": 30.0, "duration": 36.4,
  "extracted_at": "...",
  "extract_config": { ... },
  "trajectories": { ... },              // 不变
  "segments":     [ ... ],              // 不变（仍是 ≤16 段的粗粒度叙事单元）
  "hints":        [ ... ],              // 不变
  "summary":      { ... },              // 不变；增加 bpm / beatTimes（已支持）

  // —— 新增 ——
  "beats": [ { /* BeatAction */ } ],
  "limbs": {                            // 关节级 per-beat 摘要，方便 UI 直接画
    "leftWrist":  [ { /* LimbBeatSlice */ } ],
    "rightWrist": [ ... ],
    "leftAnkle":  [ ... ],
    "rightAnkle": [ ... ]
  }
}
```

### 2.1 `BeatAction`（每一拍一条）

```ts
interface BeatAction {
  index: number;           // 第几拍，从 0 开始
  startTime: number;       // 秒，本拍开始（=beatTimes[i]）
  endTime: number;         // 秒，下一拍开始或视频结束
  duration: number;        // endTime - startTime
  segmentId?: string;      // 它落在哪个粗粒度 segment 里（可选回溯）
  primaryJoint: JointId;   // 本拍主导关节（按段内位移最大者选）
  primaryDirection: Cardinal8 | 'still';
  emoji: string;           // ⬆️ / ↗️ / 👉 / 👈 / ⏸ ...
  label: string;           // "右手向右挥"
  // 关节级位移 / 速度，用于报告里画小图
  jointStats: Record<JointId, {
    distance: number;      // 本拍内位移（躯干 scale 单位）
    peakSpeed: number;
    cardinal: Cardinal8 | 'still';
    angleDeg?: number;     // 主轴方向（0=右）
  }>;
  visibilityWarning?: 'lowFoot' | 'lowHand' | null;  // 关节遮挡时给前端打 tag
}
```

### 2.2 `LimbBeatSlice`（按关节聚合的同样信息，UI 友好）

```ts
interface LimbBeatSlice {
  beatIndex: number;
  startTime: number; endTime: number;
  cardinal: Cardinal8 | 'still';
  emoji: string;
  distance: number; peakSpeed: number;
  // 仅手臂相关：肘角度、手相对躯干的高度区间
  armPose?: { elbowAngleDeg: number; wristHeightBand: 'low'|'mid'|'high' | 'overhead' };
  // 仅腿相关
  legPose?: { kneeBent: boolean; ankleSide: 'inside'|'outside'|'center' };
}
```

> 选 8 方向 + 节拍粒度足够"用户读得懂"；继续细化（弧线类型、对称性）放到 v3 schema。

---

## 3. 后端管线

### 3.1 一键流水线

`dev_server.py` 现有 `/api/extract-pose` 只跑姿态提取。新增**链式分析**，避免增加新调用：

```
POST /api/extract-pose      （现有）  →  返回 _pose.json
POST /api/analyze-motion    （新增）  →  接 _pose.json + 可选 beats[]，返回 _motion.json
```

**为什么拆两步**：姿态提取是 CPU 密集型，可以提前完成；动作分解非常轻（毫秒级），允许前端在用户**确认了节拍/BPM**后再触发，否则按拍切片会被错估。

**`/api/analyze-motion` 接口**（新增）：

```
Content-Type: application/json
Body:
{
  "poseJson": { ... }  | { "path": "wudao/angel_pose.json" },
  "beatTimes": [0.62, 1.24, ...],   // 可选；浏览器侧 Web Audio 算的
  "bpm": 96,                         // 可选；用于在没有 beatTimes 时按 bpm 均匀切
  "minSegment": 0.6,
  "maxSegments": 16
}

Response: { ok: true, motion: {...}, beatsCount, segmentsCount }
```

实现要点：把 `analyze_motion.analyze` 改造为既能从文件路径加载、也能直接接受已 parse 的 `dict`（现已基本支持，仅需把 `load_pose` 提到上层）。

### 3.2 节拍来源（三选一，按可靠度降级）

| 方案 | 实现 | 适用 | 失败时降级 |
| --- | --- | --- | --- |
| A. 浏览器 Web Audio | 复用 `pose_viewer.html::analyzeAudioBeats`，`onset` 阈值法 | 有原声音乐的视频，UI 直接读 `<video>` 的音轨 | → B |
| B. 服务端 librosa | `dev_server.py` 新依赖 `librosa`，`librosa.beat.beat_track` | 上传视频也包含原声；浏览器解码失败时由后端兜底 | → C |
| C. BPM 均匀网格 | 给一个 `bpm` 输入框（或固定 96），按 `60/bpm` 步长生成 `beatTimes` | 视频无音轨/纯背景音 | → 没有 beats 时退化为只看 `segments` |

> v1 先做 A + C；B 写成可选模块（`requirements.txt` 加 `librosa>=0.10`），不强依赖。

### 3.3 `analyze_motion.py` 改造点

最小增量改动，按 PR 拆：

1. **暴露内存接口**：新增 `analyze_from_dict(pose_json: dict, *, beat_times, bpm, ...)`，把 IO 与算法解耦。`dev_server.py` 直接 import。（修改：`scripts/analysis/analyze_motion.py::analyze` → 拆成 `analyze_from_dict` + thin wrapper。）
2. **`build_beats(...)`** 新函数：以 `beat_times` 为切点，对每个区间执行和 `build_segment` 类似的"主导关节 + 主导方向"计算，但**不做合并**——一拍一条。落地在 segments 之后。
3. **`build_limbs(...)`** 新函数：把 `beats[]` 按关节透视一遍（每条轨迹各得一份 `LimbBeatSlice[]`）。手臂额外算 `elbowAngleDeg`（用 11/13/15 或 12/14/16 三点）和 `wristHeightBand`（手腕 y 相对肩高/胯高分桶）。腿用 23/25/27 三点。
4. **`schema_version` → 2**：新字段在 v2 才出现；`SCHEMA_VERSION = 2`。`web/src/core/motionTypes.ts` 同步更新（保持 v1 字段可选向后读）。
5. **可见度告警**：`jointStats[joint].visibility < 0.4` 持续 3 帧以上时本拍打 `visibilityWarning='lowFoot'/'lowHand'`，前端用 ⚠️ 图标提醒"脚被画面裁掉了，方向仅供参考"。
6. CLI 同步：`python3 scripts/analysis/analyze_motion.py wudao/angel_pose.json --beats $(cat wudao/angel_beats.txt)` 不变；新增 `--no-beats-fallback-bpm 96`。

### 3.4 缓存与重算

- `_motion.json` 文件内嵌 `extract_config` 哈希；前端如果发现 `bpm`/`minSegment` 改了就重新调一次 `/api/analyze-motion`，姿态那一步不重跑。
- 仓库内 `wudao/*_motion.json` 加一个一次性脚本 `scripts/analysis/regenerate_motion.py` 把存量产物升到 schema_version=2（不动 _pose.json）。

---

## 4. 前端：导入 → 分解报告

### 4.1 用户流（合并到现有 setup → practice 流程之外）

```
[Setup 卡片]
  ├─ 选仓库内置动作（已有）
  ├─ 上传短视频（已有 → 走 /api/extract-pose）
  └─ ★ 新增：「分析这段视频」按钮
        ↓
[Analyzing… 进度 UI]                    （poseExtract → audioBeats → analyzeMotion）
        ↓
[★ 动作分解报告 view（新视图）]
   ├─ 顶部：视频缩略 + BPM/拍数/总段数 摘要
   ├─ Tab 1「按拍」：beats[] 时间轴，每拍一格，点开看当帧
   ├─ Tab 2「按段」：segments[] 段卡片网格（复用 motion-analysis-and-lesson-mode 的卡片样式）
   ├─ Tab 3「按关节」：左/右手腕/脚踝四条 timeline，方向色块条
   └─ 底部 CTA：「跟练这一段」 → 进 PracticeView 并自动定位到该段
```

### 4.2 视图选址：旧入口 vs 新入口

按 `docs/refactor-roadmap.md`：
- **`pose_viewer.html`（旧入口）**：先在 setup 区加上传后自动跳一个**简单报告抽屉**（按拍 + emoji 列表），不引入新视图，避免拖延 P2 迁移。
- **`web/src/app/views/`（新入口）**：实现完整三 Tab 的 `MotionBreakdownView.ts`，配合 `motionTypes.ts` 已有类型。

> 推荐：先在新入口落地完整版（避免给 `pose_viewer.html` 再堆 1000 行），同时给旧入口做一个最小报告，保证用户体验闭环。

### 4.3 关键 UI 组件（新入口侧）

| 组件 | 文件 | 作用 |
| --- | --- | --- |
| `MotionBreakdownView` | `web/src/app/views/MotionBreakdownView.ts` | 容器视图，三 Tab 切换 |
| `BeatStrip` | `web/src/ui/BeatStrip.ts` | 横向时间轴，一拍一格，背景色按 `primaryDirection` 着色，hover 显示缩略帧 |
| `JointLane` | `web/src/ui/JointLane.ts` | 单关节方向条，8 色 + still=灰；用 `<canvas>` 画 |
| `SegmentCard` | `web/src/ui/SegmentCard.ts` | 段卡片（emoji + title + 时长 + 难度 + tip） |
| `KeyframeStrip` | `web/src/ui/KeyframeStrip.ts` | 段内关键帧缩略图（见 §5.2） |

### 4.4 状态机

```ts
type AnalysisStage =
  | 'idle' | 'uploading' | 'extractingPose'
  | 'detectingBeats' | 'analyzingMotion'
  | 'ready' | 'error';
```

每个阶段在 UI 上给一句话进度（"姿态识别中…"/"正在听节拍…"），让 30–60 秒的处理过程不冷场。

---

## 5. 算法增强（小幅）

### 5.1 每拍主导描述

`build_beats(beat_times, trajectories)`：
- 对每个 `[beat_times[i], beat_times[i+1])` 区间：
  1. 取每个关节在区间内的位移向量 = (x_end - x_start, y_end - y_start)（用平滑后的 samples）
  2. `primaryJoint = argmax(distance × visibility)`
  3. `primaryDirection = quantize_direction(angle_of_displacement)`；若 `peakSpeed < still_speed_threshold`，标 `still`
  4. 复用 `_TEMPLATES` / `_CARDINAL_FALLBACK` 生成 `label/emoji`
- 注意"位移"用区间起止两端的位置差，比"瞬时速度的 majority"更稳定（短拍内速度会反向）。

### 5.2 关键帧缩略图

仅当浏览器侧能解码视频时（Vite 入口拿到 `<video>` 元素），按 `keyFrames[]` 时间戳调 `video.currentTime = t` 后 `drawImage` 到 `<canvas>` → `toDataURL` 缓存到 IndexedDB。**不入库**，纯前端计算。无视频源时这一格留空。

### 5.3 手臂细节

```python
def elbow_angle(shoulder, elbow, wrist) -> float:
    a = (shoulder.x - elbow.x, shoulder.y - elbow.y)
    b = (wrist.x - elbow.x, wrist.y - elbow.y)
    cos = (a[0]*b[0]+a[1]*b[1]) / (hypot(*a)*hypot(*b)+1e-6)
    return degrees(acos(clamp(cos, -1, 1)))

def wrist_band(wrist_y, shoulder_y, hip_y) -> str:
    if wrist_y < shoulder_y - 0.1: return 'overhead'
    if wrist_y < (shoulder_y + hip_y)/2: return 'high'
    if wrist_y < hip_y: return 'mid'
    return 'low'
```

每拍取区间中点帧计算一次，写进 `LimbBeatSlice.armPose`。

### 5.4 脚部置信度

脚踝在自拍/手机竖屏里经常被裁。计算 `coverage = visibility 高于阈值的帧占比`：
- `coverage > 0.7` → 正常输出
- `0.3 < coverage ≤ 0.7` → 输出但本关节所有 beat 打 `visibilityWarning='lowFoot'`
- `≤ 0.3` → `limbs.leftAnkle = []`，UI 在脚 Tab 上显示"画面里看不全脚部"

---

## 6. 阶段性交付

按"能让用户上传视频看到分解"的最短路径排：

| Phase | 工作 | 验收 | 估时 |
| --- | --- | --- | --- |
| **M1** | 后端：`analyze_motion.py` 拆出 `analyze_from_dict` + 增加 `build_beats` + schema v2；`dev_server.py` 加 `/api/analyze-motion`；存量 `wudao/*_motion.json` 升级 | 命令行能用一条 `_pose.json` 跑出含 `beats[]/limbs{}` 的 v2 motion；`POST /api/analyze-motion` 返回符合 schema 的 JSON | 1 轮 |
| **M2** | 节拍：浏览器端 `analyzeAudioBeats` → 上传后回填到 `/api/analyze-motion`；提供"无音轨"时的 BPM 输入框 fallback | 上传 `wudao/angel.mp4` 等样例，能在前端拿到 ≥80% 与人工拍的拍点对齐的 `beatTimes` | 0.5 轮 |
| **M3** | 旧入口最小报告：`pose_viewer.html` 上传完成后在 setup 抽屉里渲染 `beats[]` 列表（一行 `[00:01.24] ⬆️ 右手向上挥`） | 用户上传任意视频后能看到全部拍点的方向标签 | 0.5 轮 |
| **M4** | 新入口完整视图：`MotionBreakdownView` + `BeatStrip` + `JointLane` + `SegmentCard` | 用户在新入口能切三 Tab，点段进入 PracticeView 自动定位 | 1–1.5 轮 |
| **M5** | 关键帧缩略图（§5.2） + 算法增强（§5.3 / §5.4 写入 schema） | 段卡片显示 3 张关键帧；遮挡严重的脚踝有 ⚠️ tag | 0.5 轮 |
| **M6** | 可选：火山 ARK 润色 segment.title / beat.label，让文案更口语化 | 可被开关；离线时退回模板 | 0.5 轮 |

> M1+M2+M3 = 用户已经能"导入视频→看到每拍动作"，是 MVP 切线。M4 是产品化形态，M5/M6 是质感增强。

---

## 7. 风险 / 降级路径

| 风险 | 触发条件 | 降级 |
| --- | --- | --- |
| 姿态完全识别失败 | `quality_report` 中 `pose_coverage < 0.3` | UI 报"画面中没有完整的人，建议正面拍摄/竖屏全身"；不进入分析 |
| 节拍误估（如纯人声） | `beatTimes` 间距 IQR 极大 | 弹出 BPM 输入框，让用户手动给 BPM；按 `60/bpm` 重切 |
| 上传超 80MB | `dev_server.py` 已有限制 | 已有错误提示；本计划不放宽（先解决长尾才考虑） |
| 浏览器解码 `<video>` 失败 → 缩略图缺 | 部分编码 | M5 优雅降级为只显示文字，不阻塞 M1–M4 |
| `analyze_motion` 时长爆炸 | 长视频 | 在 `dev_server.py` 给一个 `duration > 90s` 的硬上限，超时后告知"建议 ≤ 90 秒，长视频请剪辑后再传" |
| 多方向并行（双手同时反向） | 一拍内两手位移都很大 | `BeatAction.primaryJoint` 仍按 distance×visibility 选；新增可选字段 `secondaryJoint`，UI 可选展示 |

---

## 8. 验证用例（直接拿仓库素材跑回归）

| 视频 | 期望 | 用途 |
| --- | --- | --- |
| `wudao/angel.mp4` | 段落以双手挥动为主，`primaryJoints=['rightWrist','leftWrist']` | 基线 |
| `wudao/ladada.mp4` | 节奏明显，`beatTimes` 应落到鼓点 | 验证节拍方案 A |
| `wudao/魔法城堡.mp4` | 有大幅腿部动作 | 验证脚踝轨迹 / `lowFoot` 警告 |
| `wudao/星奇摇2.0.mp4` | 速度快、方向翻转密 | 段切分稳定性 |
| 用户自拍竖屏视频 | 脚常被裁，`coverage` 低 | `visibilityWarning='lowFoot'` 触发 |

每个用例存一份 `tests/fixtures/<name>.expected.json`（只断言 `beats.length / segments.length / 关键拍的 cardinal`），M1 收尾时跑 `pytest` 回归。

---

## 9. 文件改动清单（速查）

```
scripts/analysis/analyze_motion.py        # 拆 IO + 加 build_beats / build_limbs / schema v2
scripts/analysis/regenerate_motion.py     # 一次性脚本：存量 _motion.json 升级到 v2
dev_server.py                             # 新增 POST /api/analyze-motion
requirements.txt                          # 可选：librosa>=0.10（仅当走方案 B）
web/src/core/motionTypes.ts               # 加 BeatAction / LimbBeatSlice
web/src/app/views/MotionBreakdownView.ts  # 新视图（M4）
web/src/ui/BeatStrip.ts                   # 新组件
web/src/ui/JointLane.ts                   # 新组件
web/src/ui/SegmentCard.ts                 # 新组件
web/src/ui/KeyframeStrip.ts               # 新组件（M5）
pose_viewer.html                          # setup 抽屉里 inline 渲染 beats[]（M3，最小改动）
docs/video-import-motion-breakdown.md     # 本文
docs/motion-analysis-and-lesson-mode.md   # §1 数据契约同步加"v2 字段"备注
tests/fixtures/                           # 新增回归 fixture
```

---

## 10. 一句话总结

仓库里 90% 的零件已经在了：姿态提取 → 轨迹方向 → 段落切分 → 节拍估计都可用。**真正要新写的是**：(a) 一个把"姿态 + 节拍"喂回去得到"按拍方向标签"的 `build_beats` 函数（≈100 行 Python），(b) 一个在前端把这串标签渲染成可读报告的视图（≈ 一个新视图 + 三个组件），(c) 把上传管线串起来的一个新接口 `/api/analyze-motion`。预计 4–5 轮节奏内可以交付到 M4，且全程不阻塞现有 P2 重构与跟练评分链路。
