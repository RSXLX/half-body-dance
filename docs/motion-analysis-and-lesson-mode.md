# 动作分解 · 运动轨迹 · 学习模式 实施计划

> 在 `docs/refactor-roadmap.md`（前端重构）与 `docs/optimization-mediapipe-scoring.md`（识别 + 评分）之外，本文专门规划"用户上传视频 → 动作运动轨迹解析 → 手脚方向识别 → 下一步预告 → 动作分解 + 学习模式"这一条新增产品支线。每个 Phase 都有可验收点；与现有重构正交，不阻塞 Phase 2 收尾。

## 0. 产品视角

### 0.1 用户故事

1. 用户上传一段舞蹈视频，或选择仓库内的标准动作。
2. 系统在后台把视频拆成若干"动作段"（segment），每段附带：
   - 主导关节（哪只手 / 哪只脚）
   - 主要移动方向（上/下/左/右/对角）
   - 一句话要领（emoji + 中文）
   - 难度
   - 起止时间、关键帧
3. 进入 **学习模式**：
   - 段卡片网格 → 用户挑一段 → 进入循环播放该段
   - 可暂停、慢速（0.5x / 0.75x / 1x）、单段循环、回放上一段
   - 屏幕上叠加 "右手向上甩"、"左脚向左跨"等方向箭头与文字
4. 进入 **跟练模式**（PracticeView）：
   - 屏幕角落出现 **下一拍预告**（提前 200–500 ms）："👉 右手向右"
   - HUD 上方显示当前段标题 + 进度（"第 3/8 段：摆手向右"）
   - 跳过段、回到段头、整段循环

### 0.2 与现有能力的关系

| 现有能力 | 在本计划中的角色 |
| --- | --- |
| `extract_pose.py` | 输入端不变；新增的 motion 分析消费同一份 `_pose.json` |
| `scripts/analysis/analyze_pose_emoji.py` | **作为种子**：已经按时间段生成 `timeline / segments / emoji_summary`，将其升级为完整 motion 分析 |
| `web/src/core/poseCompare.ts` | 评分还是评分；分解是独立的另一条数据 |
| 火山引擎 ARK API（`scripts/analysis/analyze_video_emoji_volcengine.py`） | 可选：用大模型把规则法生成的段标题 / 要领润色得更口语化 |

**关键决策**：分解管线**离线生成**（Python 脚本写 JSON），Web 端只负责**消费 + 渲染**。这样模型/算法升级不会触发前端重构，模型推理也不挤占用户摄像头帧的算力。

---

## 1. 数据契约

### 1.1 文件命名

```
wudao/<name>.mp4              # 原始视频（已存在）
wudao/<name>_pose.json        # 姿态 JSON（已存在，extract_pose.py 输出）
wudao/<name>_motion.json      # 动作分解 JSON（新增本计划产物）
```

`_motion.json` 与 `_pose.json` 一一对应，通过同名前缀关联。Web 端在 setup/result/lesson 视图按需懒加载。

### 1.2 `_motion.json` 顶层结构

```jsonc
{
  "schema_version": 1,
  "source_pose": "wudao/angel_pose.json",
  "fps": 30.0,
  "duration": 36.4,
  "extracted_at": "2026-05-01T17:30:00Z",
  "extract_config": { "stride": 1, "smoothing_window": 5, "speed_quantile": 0.7 },

  "trajectories": {
    "leftWrist":  { /* JointTrajectory */ },
    "rightWrist": { /* JointTrajectory */ },
    "leftAnkle":  { /* JointTrajectory */ },
    "rightAnkle": { /* JointTrajectory */ },
    "leftElbow":  { /* JointTrajectory */ },
    "rightElbow": { /* JointTrajectory */ }
  },

  "segments": [ /* MotionSegment[] */ ],
  "hints":    [ /* MotionHint[] */ ],

  "summary": {
    "primary_joints": ["rightWrist", "leftWrist"],
    "dominant_directions": [ { "cardinal": "up", "share": 0.32 }, ... ],
    "bpm": 96,
    "beat_times": [0.62, 1.24, 1.86, ...]
  }
}
```

### 1.3 `JointTrajectory`

按躯干参考系（`getPoseReference`）归一化后存储；与 Python 的 `normalizeLandmarks` 一致。坐标单位是**躯干 scale**（约一肩宽 = 1）。

```ts
interface JointTrajectory {
  joint: JointId;            // "leftWrist" | "rightWrist" | "leftAnkle" | ...
  samples: TrajectorySample[];
  /** 整段轨迹的全局指标，便于 UI 排序 */
  stats: {
    totalDistance: number;   // 累计行程
    peakSpeed: number;       // 最大瞬时速度
    speedVariance: number;
  };
}

interface TrajectorySample {
  t: number;                 // 秒，帧时间
  x: number; y: number;      // 躯干参考系内归一化坐标
  vx: number; vy: number;    // dx/dt（已平滑）
  speed: number;             // sqrt(vx^2 + vy^2)
  direction: number;         // 0..360°，0° = +x（指向身体右）
  cardinal: Cardinal8;       // "up" | "upRight" | "right" | "downRight" | "down" | "downLeft" | "left" | "upLeft" | "still"
  /** 该样本是否是"关键帧"（局部速度极值） */
  isKeyFrame: boolean;
  /** 可见度低于阈值时整条轨迹用 "missing" 占位，不参与方向判断 */
  visibility: number;
}

type Cardinal8 =
  | 'up' | 'upRight' | 'right' | 'downRight'
  | 'down' | 'downLeft' | 'left' | 'upLeft'
  | 'still';
```

> **方向坐标系说明**：MediaPipe Pose 的 y 轴朝下；为了让 "up" 直觉上对应"屏幕上方"，方向计算时把 y 轴翻转，让 `cardinal === 'up'` 对应"向屏幕上方移动"。

### 1.4 `MotionSegment`

```ts
interface MotionSegment {
  id: string;                  // "seg-003"
  index: number;               // 顺序号，从 0 开始
  startTime: number;           // 秒
  endTime: number;             // 秒
  duration: number;
  emoji: string;               // 可视化首图，例如 "👉"
  title: string;               // "右手由下向上挥"
  description: string;         // 详细要领，2-3 句
  primaryJoints: JointId[];    // ["rightWrist"]
  primaryDirection: Cardinal8; // "up"
  /** 难度自动判定：方向变化数 + 速度峰值 + 涉及关节数 */
  difficulty: 1 | 2 | 3;
  /** 段内关键帧时间戳（局部速度极值） */
  keyFrames: number[];
  /** 学习模式提示词（最多 3 条） */
  tips: string[];
  /** 该段如果与拍点对齐，对齐到的拍点索引 */
  beatIndices?: number[];
  /** 由大模型润色后的可选文案（rule-based 文案保留在 description） */
  llmDescription?: string;
}
```

### 1.5 `MotionHint`

预告点：在 `triggerTime` 弹给用户，对应 `segmentId` 的开端。

```ts
interface MotionHint {
  /** 触发时间，通常 = segment.startTime - 0.4 */
  triggerTime: number;
  segmentId: string;
  /** 一行预览，例如 "下一拍：👉 右手向右" */
  preview: string;
  /** UI 渲染样式 */
  cue: 'beatPrep' | 'directionArrow' | 'sectionStart';
  /** 提前量，用于 PracticeView 显示倒计时 */
  leadMs: number;
}
```

---

## 2. 算法分层

### 2.1 关节轨迹（trajectory）

**输入**：`frames[].pose_landmarks` + 每帧的 `_reference`（已有，extract 时生成）。

**步骤**：

1. **归一化**：用每帧 `getPoseReference` 把目标关节投影到躯干坐标系。
2. **可见度过滤**：`visibility < 0.4` 的样本标记为缺失，**不**做插值（避免假动作），但相邻有效样本之间保留连接。
3. **平滑**：对 `(x, y)` 做 5 帧滑动平均（与 `extract_pose.py` 的 `pose_smoothing_window=5` 一致），减少识别抖动。
4. **速度**：中心差分 `v_t = (p_{t+1} - p_{t-1}) / (2*dt)`；首尾用前/后向差分。
5. **方向角**：`atan2(-vy, vx)` → 度数 → 8 方向 bin（每 45°，`still` 当 `speed < 0.05/scale_per_sec`）。
6. **关键帧**：速度的局部极小（停顿）或局部极大（爆发点）；用 1D 极值检测，最小邻居距离 = `0.18s × fps`。

**输出关节集合**（v1）：
- `leftWrist (15)`, `rightWrist (16)` — 半身舞最关键
- `leftElbow (13)`, `rightElbow (14)` — 用于检测挥臂幅度
- `leftAnkle (27)`, `rightAnkle (28)` — 当 visibility 足够时；半身视频通常缺失，输出 `null`

### 2.2 段切分（segmentation）

**目标**：把 N 帧切成 K 段，每段语义连贯（同一主导方向 / 同一主导关节）。

**算法选择 — 三选一渐进**：

1. **v1（必做）规则法**：
   - 候选边界：所有关节的"显著停顿"（连续 ≥ 3 帧 `cardinal === 'still'` 且关节速度峰值落差超过窗口最大值的 60%）。
   - 候选边界：关节主导方向发生跨 90° 切换。
   - 用 **节拍** 作为边界吸引子（如果有 BPM）：边界吸到最近的拍点，容差 ±150 ms。
   - 合并：相邻段时长 < 0.6s 与上一段合并，避免碎片。
   - 输出 4–12 段（典型 30–60s 半身舞）。
2. **v2（可选）DBSCAN**：在 (主导关节方向, 速度均值, 时间) 三维空间里做密度聚类，得到自然段。
3. **v3（远期）大模型分段**：把姿态 JSON + 视频帧采样喂给 ARK 多模态，用提示词要它给"教学段落"。

**复用现成代码**：`scripts/analysis/analyze_pose_emoji.py` 已经在做 emoji segment + timeline，要做的是把它升级到本计划的 schema（增加方向、关键帧、难度、tip）。

### 2.3 主导关节 + 方向识别

每段内：
- **主导关节**：累计行程（`Σ |Δp|`）排名前 1–2 的关节。
- **主导方向**：方向 bin 直方图，去掉 `still`，取占比最高的 bin（≥30% 才算"主导"，否则段被标记为 `mixed`）。
- 整段 `cardinal` 优先取主导方向；`mixed` 段渲染时用"自由发挥"的中性 emoji。

### 2.4 难度判定（启发式）

```
score = 0.4 * normalized(num_direction_changes)
      + 0.3 * normalized(peak_speed)
      + 0.2 * normalized(num_active_joints)   # 多关节同时动 → 难
      + 0.1 * normalized(segment_duration_inv)# 越短越难

difficulty = 1  if score < 0.33
           = 2  if 0.33 ≤ score < 0.66
           = 3  if score ≥ 0.66
```

阈值在 `extract_config` 内可调。

### 2.5 文案生成

#### 2.5.1 规则法（v1）

模板表（节选）：

| primary_joint × cardinal | 标题模板 | 要领 tip |
| --- | --- | --- |
| rightWrist × up | 右手由下向上挥 | 肩膀放松，手腕领先 |
| rightWrist × right | 右手向右摆 | 注意手肘高度不要塌 |
| leftWrist × leftDown | 左手向左下收回 | 顺势带动身体左倾 |
| both wrists × up | 双手向上举 | 手肘略外旋；眼睛看上方 |
| ankle × up | 踏步上跳 | 膝盖弹起，脚尖落地 |
| `mixed` | 节奏过渡 | 跟住节拍，听准下一拍 |

emoji 选择：`up=⬆️`、`right=👉`、`down=⬇️`、`left=👈`，对角加 ↗️↘️↙️↖️，`still=⏸`，`mixed=🎵`。

#### 2.5.2 大模型润色（v2，可选）

- 触发：用户在 setup 页勾选"AI 润色文案"或脚本带 `--llm`。
- 实现：把每段的 `(emoji, primary_joint, cardinal, duration, beats)` 拼成 prompt，调火山 ARK Responses API（已存在 `scripts/analysis/analyze_video_emoji_volcengine.py` 可改造）。
- 产物写到 `segment.llmDescription`，前端优先用，缺失时降级到 `description`。
- 配额限制：每段最多 1 次调用，整段视频最多 16 段调用，缓存到 `_motion.json`。

### 2.6 预告点（hints）生成

```
for seg in segments:
    leadMs = clamp(380, 220, seg.duration * 1000 * 0.25)   # 段越短预告越短
    triggerTime = max(0, seg.startTime - leadMs / 1000)
    hints.append({
      triggerTime,
      segmentId: seg.id,
      preview: f"下一拍：{seg.emoji} {seg.title}",
      cue: 'beatPrep' if seg.beatIndices else 'sectionStart',
      leadMs,
    })
```

---

## 3. 离线分析管线

### 3.1 入口脚本（新增）

`scripts/analysis/analyze_motion.py`：

```bash
python3 scripts/analysis/analyze_motion.py wudao/angel_pose.json \
  --output wudao/angel_motion.json \
  [--llm]                  # 可选：调火山 ARK 润色文案
  [--min-segment 0.6]      # 段最短时长（秒）
  [--max-segments 16]      # 切上限
  [--bpm 96]               # 手动指定 BPM；不传则尝试从 audio 估计
```

读 `_pose.json` → 输出 `_motion.json`。**不触动** `extract_pose.py` / `analyze_pose_emoji.py`（后者继续作为简化的 emoji 时间线产品）。

### 3.2 与现有脚本的关系

```
extract_pose.py            (无变更)
  └── _pose.json
        ├── analyze_pose_emoji.py    → _emoji.json   (现状，保留)
        └── analyze_motion.py        → _motion.json  (新增)
```

`analyze_motion.py` 内部可以**复用** `analyze_pose_emoji.py` 里现成的 timeline / emoji_summary 函数；以"它的 segments 为先验，本脚本细化方向 + 难度 + tip"的方式扩展，避免重复造轮。

### 3.3 BPM 估计（可选）

短时实现：用 `librosa.beat.beat_track` 处理同名 mp4 的音轨；失败则不输出 `bpm/beat_times`。本计划允许这一步缺失，分段不依赖它，只是吸附边界用。

### 3.4 上传短视频路径

`dev_server.py` 里已有 `/api/extract-pose`（上传 mp4 → 返回 `_pose.json`）。本计划在它后面追加一步：

```
POST /api/analyze-motion
body: { pose_json: <内容> | path: "wudao/xxx_pose.json" }
→ 调用 analyze_motion 模块（同进程），返回 _motion.json
```

整合到上传流程后，准备页"上传短视频并识别"按钮一次完成 pose + motion 双产物。

---

## 4. Web 端集成

### 4.1 类型与加载

新增模块（沿用 `web/src/core` 的纯函数风格）：

```
web/src/core/
├── motionTypes.ts          # MotionAnalysis / JointTrajectory / MotionSegment / MotionHint / Cardinal8
└── motionUtils.ts          # selectSegmentAt(t) / nextHint(t) / formatDirection(c) 等纯函数

web/src/app/data/
└── motion.ts               # fetchMotion(presetPath) — 把 _pose.json 路径换 _motion.json 路径并缓存
```

加载：在 `presets.ts` 增加 `motionPath`（默认派生自 `path.replace('_pose.json', '_motion.json')`）。如果 fetch 404，UI 自动退回"无分解模式"，不阻塞跟练。

### 4.2 视图改动

#### 4.2.1 SetupView（轻改）

- preset 卡片右下角新增一个角标 `🧩 N 段` —— 表示该 preset 已有 motion 分解。
- 加载 `_motion.json` 后弹一个"动作分解可用"toast，引导进入新增的 LessonView。

#### 4.2.2 LessonView（**新增**）

入口：从 SetupView 的 `🧩 N 段` 角标点击进入。

布局：
```
┌────────────────────────────────────────┐
│ ← 返回    《鸿门旋律》动作分解          │
├────────────────────────────────────────┤
│ ┌─段卡片─────┐ ┌─段卡片─────┐          │
│ │👉 右手向右 │ │⬆️ 双手向上 │  ...     │
│ │ 1.6 s · ★ │ │ 1.2 s · ★★│          │
│ └────────────┘ └────────────┘          │
│  ...                                   │
├────────────────────────────────────────┤
│  [当前选中段：👉 右手向右]             │
│  ╔═════════════════════════════╗       │
│  ║   骨架回放 + 方向箭头叠加    ║       │
│  ╚═════════════════════════════╝       │
│  要领：肩膀放松，手腕领先              │
│  [⏮ 上一段] [⏯ 单段循环] [⏭ 下一段]    │
│  [0.5x] [0.75x] [1x]   [📷 跟练这一段] │
└────────────────────────────────────────┘
```

行为：
- 点段卡片：选中并循环播放该段；其他段灰显。
- 单段循环、慢速、上下段切换都只影响"当前段"。
- 「跟练这一段」：跳到 PracticeView，但 `getCurrentTargetFrame` 只在该段范围内循环，不进入下一段。

#### 4.2.3 PracticeView（强化）

- HUD 顶部：`第 i / N 段：标题`；进度条按段切割（每段一段)。
- 浮层：当 `currentTime + leadMs >= nextHint.triggerTime` 时，左下角弹出 hint 卡片（持续 300 ms 进入 + 700 ms 显示）。
- 控制：
  - `[⏮ 段头]` 跳到当前段开头（替换原本的 `replay`）。
  - `[🔁 段循环]` 切换；开启时 `getCurrentTargetFrame` 在段内 wrap-around。

#### 4.2.4 ResultView（新增"段表现"块）

- 在分数下方新增"动作分解表现"卡片：每段一行 → 段标题 + 该段平均分。
- 点击行 → 跳回 LessonView 学习该段。

### 4.3 方向箭头可视化

在 LessonView 的回放画布上，对每个 `primaryJoint` 在关键帧位置画一条"轨迹尾迹"+ 末端箭头：

```
颜色：cardinal 决定（up=蓝，down=琥珀，left=粉，right=绿）
长度：0.12 × 躯干 scale
透明度：从 0.1 渐变到 0.9（指向当前帧）
```

PracticeView 上同样的箭头**用更小的尺寸渲染在 HUD 角落作为预览图**。

### 4.4 暂停 / 回放交互

LessonView 的"单段循环"配合：

- 暂停：`pauseAt(t)` —— 停在某个关键帧；再次点击从该帧继续。
- 慢速：维持 `playSpeed` 状态，影响 `getCurrentTargetFrame` 的 elapsed 计算。
- 关键帧步进：键盘 `→` / `←` 在 `segment.keyFrames` 之间跳；移动端长按 `[⏯]` 进入逐帧模式。

---

## 5. Phase 拆分（落地节奏）

| Phase | 内容 | 验收 | 预估 |
| --- | --- | --- | --- |
| **A0** | 数据契约定稿（`_motion.json` schema + Cardinal8 + JointId 枚举），写到本文件 §1 即可；Web 端先写空类型 stub | `web/src/core/motionTypes.ts` 编译通过 | 0.5 天 |
| **A1** | `analyze_motion.py` 规则法 v1：trajectories + segments + hints + 模板文案；为 `wudao/` 9 个 preset 全部跑一遍生成产物 | 9 份 `_motion.json` 文件，肉眼检查段数 4–12、方向标签合理 | 2 天 |
| **A2** | `dev_server.py` 增加 `/api/analyze-motion`；上传短视频流程一次返回 pose + motion | 上传 demo 视频可在浏览器 Network 看到两份 JSON | 0.5 天 |
| **B1** | LessonView 雏形：从 `_motion.json` 渲染段卡片网格；点击播放该段（用现有 PracticeView 的 stage） | 用户能选段循环看 | 1.5 天 |
| **B2** | LessonView 完整：方向箭头可视化、慢速、单段循环、关键帧步进 | 上线"学习模式" | 1 天 |
| **C1** | PracticeView 接入 hints + 段进度 HUD | 跟练时屏幕左下角能看到下一拍预告 | 1 天 |
| **C2** | ResultView 新增段表现行 | 结算页能看到每段平均分 | 0.5 天 |
| **D1**（可选） | 大模型润色文案 | `analyze_motion.py --llm` 写 `llmDescription`，前端优先取 | 1 天 |
| **D2**（可选） | DBSCAN 段切分 + BPM 吸附 | 段切分肉眼明显比 v1 好 | 1.5 天 |

总规模：A+B+C ≈ **6.5 天**，D 可选另加 2.5 天。

---

## 6. 与现有 roadmap 的协同

- 与 `docs/refactor-roadmap.md` 的 Phase 2 **正交**：本计划全部代码落在 `web/src/core/motion*` 与新增 `LessonView`，不动 SetupView / PracticeView 已迁移好的部分。
- 与 `docs/optimization-mediapipe-scoring.md` 的关系：分解和评分是**独立指标**——一个回答"用户有没有踩对动作"，一个回答"这段动作长什么样"。两条数据可以独立演进。
- 与 `docs/product-and-optimization-roadmap.md` 的关系：这是"方向 A（C 端跟练 App）"和"方向 C（emoji 谱面编辑器）"的桥梁——**离线分析管线**就是方向 C 的工具雏形；前端 LessonView 提升方向 A 的留存。

---

## 7. 风险与回滚

| 风险 | 概率 | 缓解 |
| --- | --- | --- |
| 段切分对低质量手机视频不稳 | 中 | v1 规则法保守输出（少切段），UI 容忍 N=1（"整段"也是合法分解） |
| `_motion.json` 体积膨胀（trajectory 全量样本） | 中 | 关节限定 6 个，每帧 7 字段；30 fps × 60s × 6 关节 × 7 字段 ≈ 75 KB / 段视频。可接受 |
| 大模型润色失败影响主路径 | 低 | LLM 仅写可选字段，主文案完全由规则生成 |
| MediaPipe 抖动让方向频繁切换 | 中 | 5 帧平滑 + 方向变化必须持续 ≥ 0.18 s 才计数；段最短 0.6 s |
| 用户上传视频里身体出框 / 遮挡 | 高 | visibility 过滤 + 缺失时段标 `mixed`，要领改为通用引导文 |

回滚：任一 Phase 失败 `git revert` 该 Phase 的合并提交。LessonView 是新加文件，删除即下线；`_motion.json` 缺失时前端自动退回"无分解模式"，不破坏现有体验。

---

## 8. 实施起手清单（Phase A0 + A1 第一波）

1. 在 `web/src/core/` 新建 `motionTypes.ts`，把 §1 的 TS 类型完整放进去（先空声明、不依赖任何运行时模块），让 `tsc --noEmit` 通过。
2. 在 `scripts/analysis/` 新建 `analyze_motion.py` 骨架：
   - 解析参数；
   - 读 `_pose.json`；
   - 调用 stub 函数 `compute_trajectories / segment_motion / build_hints`；
   - 写 `_motion.json`。
3. 实现 `compute_trajectories`：6 个关节 + 5 帧平滑 + 中心差分 + 方向 bin。
4. 实现 `segment_motion` v1：用 `analyze_pose_emoji.py` 现有 segment 输出做种子，本脚本只做"细化"。
5. 实现 `build_hints` + 模板文案表。
6. 给 9 份 preset 跑一遍，肉眼对照视频 review 段数 + 方向标签是否合理；调阈值。
7. 把 `_motion.json` 路径加到 `web/src/app/data/presets.ts`（只加字段，先不读）。

完成 1–7 即解锁 Phase B（LessonView）。

---

## 9. 设计原则备忘

- **离线产物即真理**：所有"段、方向、要领"全部走离线 JSON，前端零业务逻辑；LessonView 只做渲染与交互。这条原则保证算法迭代不需要前端 release。
- **优雅降级**：缺失 `_motion.json` 时 UI 退到现状；缺失 BPM 时段对齐失效但仍有段；缺失 ankle 数据时只渲染上肢。
- **分解 vs 评分独立**：分解是"教程视图"，评分是"考试视图"。两套数据互不强依赖，分别迭代。
- **可缓存**：`_motion.json` 是纯函数产物，文件名带 `schema_version`，破坏性升级时换名；旧版本继续可用。
