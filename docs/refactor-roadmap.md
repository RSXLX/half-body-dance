# 前端流程优化 + 项目结构重构 · 执行计划

> 本文对账 `docs/product-and-optimization-roadmap.md` 的 Phase 1，把"用户流线优化"与"引入 Vite+TS 模块化重构"合并成一条可落地路径。每个 Phase 结束都有可验证的验收点。

## 0. 当前痛点（流程视角）

从用户第一次打开到进入练习，现状步数：

1. 开页 → 看到 4 张卡片（选动作 / 设备 / 练习设置 / 准备状态）并列
2. 点一个示例动作（只有 3 个按钮）
3. 点"打开摄像头"
4. 滚动到底部点"开始练习"

主要问题：
- **步数冗余**：第 3、4 步可以合并为一键。
- **视觉噪音**：准备页的 4 张卡片同时曝光；"准备状态"的 6 项指标大部分是 runtime 状态（分数、节奏提示），准备阶段没意义。
- **动作库曝光不足**：仓库实际有 9 个 `_pose.json`，UI 只暴露 3 个。
- **CTA 不显眼**："开始练习"藏在 setup-footer 里，低端手机需要滚动。
- **顶部 tab 提前曝光**：practice / result 在未就绪时只是置灰，依然视觉占位。

## 1. 路线分三 Phase

| Phase | 内容 | 完成标志 | 预估 |
| --- | --- | --- | --- |
| **P0** | 流程优化（在现有 `pose_viewer.html` 直接改） + 仓库目录浅层整理 | 首屏更聚焦；一键开始可用；根目录噪音下降 | 本轮 + 1 轮 |
| **P1** | Vite + TS 工程骨架；抽出 `packages/pose-core` | 新入口 `npm run dev` 能跑起来；评分核心有类型与单测 | 1–2 轮 |
| **P2** | 逐页迁移（Setup → Practice → Result）到 Vite 工程；`pose_viewer.html` 归档为 legacy | `pose_viewer.html` 可删除；功能对齐 | 2–3 轮 |

三个 Phase 严格串行，但每个 Phase 内部可以并行。

---

## 2. Phase 0 详细清单

### 2.1 Setup 流程优化（直接改 `pose_viewer.html`）

| # | 改动 | 文件 | 依赖 |
| --- | --- | --- | --- |
| S1 | 动作库扩展到 9 个，示例按钮用网格而不是横向 carousel；卡片显示 emoji + 名字，首选项高亮 | pose_viewer.html | 无 |
| S2 | 合并"打开摄像头"与"开始练习"：点 CTA 时若 `cameraRunning=false` 自动 `await openCamera()` 再 `enterPracticeFlow`；失败时降级为只观看模式并弹出短提示 | pose_viewer.html | 无 |
| S3 | "开始练习"改 sticky CTA（`position: sticky; bottom: 16px`），未选动作时禁用并文案"先选择标准动作"；移除 `setup-footer` 整块 | pose_viewer.html | S1 |
| S4 | 顶部 `nav-tabs`：setup 时只显示"准备"一个 tab（其余隐藏而非置灰）；首次加载动作后渐显 practice，首次完结后渐显 result | pose_viewer.html | 无 |
| S5 | 准备状态卡片整块删除（迁到 debug drawer 或直接去掉；runtime 状态留给 practice HUD） | pose_viewer.html | S4 |
| S6 | 头部贴图 + 速度 + 准备状态收到"更多设置"折叠抽屉里，默认收起 | pose_viewer.html | S5 |
| S7 | URL 参数 `?preset=鸿门旋律&autoplay=1` 直达 practice（为后续分享做准备） | pose_viewer.html | S2 |

### 2.2 仓库目录浅层整理（不破坏 `npm run dev`）

根目录当前 30+ 文件，本轮只做低风险搬运：

```
.
├── pose_viewer.html        ← 入口保留不动（P2 再删）
├── dev_server.py           ← 入口保留
├── extract_pose.py         ← 入口保留
├── wudao/                  ← 保留
├── docs/
├── scripts/                ← 新建
│   └── analysis/           ← analyze_*.py、upload_and_analyze_*.py 迁入
├── output/                 ← 新建；*_volcengine_analysis.json、test_output.json 迁入
├── assets/                 ← 新建；frame5.jpg、contact.jpg、*.png、top_roi.jpg、tmp_audio.wav 迁入
└── .tmp/, tmp_frames/, emoji_crops/  ← 加到 .gitignore（不删除历史，避免破坏）
```

影响面：
- `scripts/analysis/*.py` 若被其他脚本导入，需同步改 import。实际检查：这些脚本互不 import（各自独立入口），安全。
- `README.md` 里的命令路径需要同步更新为 `python3 scripts/analysis/analyze_pose_emoji.py ...`。

**不做**：`pose_landmarker.task` / `hand_landmarker.task` / `wudao/*.mp4` / `*_pose.json` 先不动（`extract_pose.py` 会就地查找模型文件；迁 LFS 是 P1 的事）。

### 2.3 Phase 0 验收

- [ ] 首屏一屏可见完整流程（动作网格 + sticky 开始按钮），不需滚动
- [ ] 选动作后点一次"开始"就能进入 practice（摄像头自动请求）
- [ ] 9 个示例都可选
- [ ] 顶部 tab 不再出现灰色占位
- [ ] 根目录文件数 < 20
- [ ] `npm run dev` 与 `python3 scripts/analysis/analyze_pose_emoji.py` 均正常

---

## 3. Phase 1：Vite + TS 骨架

### 3.1 目标目录

```
web/                          ← 新建 Vite 工程（与 pose_viewer.html 并存一段时间）
├── index.html                ← 新入口
├── package.json
├── vite.config.ts
├── tsconfig.json
├── src/
│   ├── main.ts
│   ├── app/
│   │   ├── router.ts         ← setup/practice/result 三态
│   │   ├── state.ts          ← 对应当前 const state = {...}
│   │   └── views/
│   │       ├── SetupView.ts
│   │       ├── PracticeView.ts
│   │       └── ResultView.ts
│   ├── ui/                   ← 纯渲染：HUD、Sheet、Dial、BeatStrip
│   ├── media/                ← camera.ts、bgm.ts
│   └── integrations/
│       ├── mediapipe-pose.ts
│       └── mediapipe-hands.ts
├── packages/pose-core/       ← monorepo 子包（或 web 内 src/core/，择一）
│   ├── src/
│   │   ├── reference.ts      ← getPoseReference / transformPointsWithReferences
│   │   ├── similarity.ts     ← comparePoses / cosineSimilarity
│   │   ├── hands.ts          ← normalizeHandLandmarks / findMatchingHand
│   │   ├── rhythm.ts         ← beat 相关
│   │   └── types.ts
│   └── test/                 ← 黄金样本回归
└── public/
    └── wudao/                ← 通过 symlink 或 vite public 别名
```

### 3.2 选型

- **Vite 7.x** + **TypeScript strict**：零框架起步（原生 DOM + 手写模板字符串 / lit-html），避免一上来就背 React/Vue 的包袱。后续如需组件化，从 Preact 或 Svelte 增量引入。
- **包管理**：沿用 npm（仓库已有 package.json）。
- **测试**：`vitest` + `@vitest/ui`。pose-core 的黄金样本：加载仓库已有 `_pose.json` 作为 fixture，断言 `comparePoses` 输出分数落在区间内。
- **类型**：把 MediaPipe 的 `NormalizedLandmark`、仓库 JSON 的 `frames[]`、`extract_config` 全部打在 `types.ts`，Python 端用 `datamodel-code-generator` 从同一份 JSON Schema 反生成。

### 3.3 过渡策略

- 保留 `pose_viewer.html` 可访问（旧入口），新 Vite 入口挂 `/app/`。
- `dev_server.py` 的 `/api/extract-pose` 保持不变，Vite 通过 `server.proxy` 转发。
- Phase 1 结束时两套都能跑，但功能对齐以 Vite 端为准。

### 3.4 Phase 1 验收

- [ ] `npm run dev` 启动 Vite，访问 `/app/` 能看到空壳
- [ ] `packages/pose-core` 有类型定义 + 至少 3 条黄金样本回归测试通过
- [ ] `dev_server.py` 的 `/api/extract-pose` 能被 Vite 入口调用（proxy 配通）
- [ ] CI（或本地 `npm test`）跑通 vitest

---

## 4. Phase 2：逐页迁移

迁移顺序：SetupView（最简单，Phase 0 已经定型）→ ResultView（纯展示）→ PracticeView（最重，评分循环 + canvas）。

迁移完成后：
- `pose_viewer.html` 重命名为 `legacy/pose_viewer.html`，README 标注"已被 Vite 工程替代"。
- 分析支线脚本（`scripts/analysis/*.py`）保持 Python，不迁移。

### 4.1 Phase 2 验收

- [ ] 功能对齐：9 个示例 × 评分 / 节奏 / 贴图 / 降级路径全部可用
- [ ] 视觉无回归（对比 Phase 0 的截图）
- [ ] `pose_viewer.html` 从默认路径下线
- [ ] 仓库根目录文件数 < 15（进一步瘦身）

---

## 5. 风险与回滚

- **风险 A：Vite 迁移期间功能对齐有回归** → 用并行入口 + 对照截图；任何时点 `pose_viewer.html` 仍能服务用户。
- **风险 B：目录搬运破坏外部引用** → 先跑 `grep -r "analyze_pose_emoji" .` 等确认无跨文件硬引用；README 同步更新。
- **风险 C：Vite 工程引入新依赖污染 Python 环境** → Node 与 Python 通过 `dev_server.py` API 解耦，互不影响。
- **回滚**：任一 Phase 失败，`git revert` 该 Phase 的合并提交即可；Phase 0/1 都不删旧文件，回滚无数据损失。

## 6. 本次对话的交付

- 本文（`docs/refactor-roadmap.md`）。
- Phase 0 的 **2.1 Setup 流程优化** 全部落地到 `pose_viewer.html`（S1–S7）。
- Phase 0 的 **2.2 目录整理** 放到下一轮对话，避免单次改动面太大导致验收困难。

后续对话：按 Phase 1 骨架、Phase 2 逐页迁移继续。
