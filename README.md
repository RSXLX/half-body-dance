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
python3 -m http.server 4173
```

打开 [http://localhost:4173/pose_viewer.html](http://localhost:4173/pose_viewer.html)。

## 常用命令

提取姿态：

```bash
python3 extract_pose.py wudao/angel.mp4 wudao/angel_pose.json
```

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
