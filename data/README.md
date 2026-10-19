# data/ — 真机录制会话

> **这里放"删了就没了"的数据。** 与相邻目录的区别：

| 目录 | 语义 | 丢了怎么办 |
| ---- | ---- | ---------- |
| `outputs/` | 可再生的运行产物（标注视频、CSV、调试图） | 重跑一次就回来 |
| `assets/` | 入库的静态素材（标定板 PDF、示例图） | 从 git 恢复 |
| **`data/`（这里）** | **不可再生的真机录制** | **只能靠备份** |

## 一次采集长什么样

```
data/sessions/20261019-101500/
  session.json           ← 唯一权威元数据（相机 / 视频 / 标定 / 同步 / 环境）
  left.mkv               ← 左路原始视频（无损 FFV1）
  right.mkv              ← 右路原始视频（无损 FFV1）
  calib/                 ← 本次使用的标定快照（npz + json）
  derived/               ← CSV / 3D 图 / 标注视频（可再生产物，可删）
  notes.md               ← 现场备注（可选）
  left_times.csv         ← 逐帧采集时间戳（P1 录制器写；用来对齐/诊断）
  right_times.csv
```

- 目录名是 `YYYYMMDD-HHMMSS`，由 `stereo/sessions.py` 生成；重名自动加后缀。
- `session.json` 的字段见 `stereo/sessions.py` 的文档串与
  `docs/planbooks/2026-10-19_数据采集与结构整理.md` §4。
- **原始视频不要重命名、不要转码**；要标注/压缩就写到 `derived/`。

## 校验

```powershell
$PY = "$env:USERPROFILE\miniforge3\envs\myenv\python.exe"
& $PY apps\tools\check_dataset.py data\sessions\20261019-101500
& $PY apps\tools\check_dataset.py --root data\sessions          # 全部
```

校验内容：两路视频能否打开、分辨率是否与 `session.json`/标定一致、
帧数差、标定 sha256、基线（算法 vs 物理测量）偏差、同步偏移范围。
**录完立刻跑一次**；等处理时才发现录坏了，代价是重录。

## 备份

`data/` 默认被 `.gitignore` 忽略。**每次采集后至少复制到一块外部硬盘/网盘**；
不要只留在这台机器上。需要进版本库的只有元数据时，用
`check_dataset.py --export-meta <目标目录>`（计划 P1）；原始视频不进 git。

## 不要放

- 交付物（开题报告、答辩稿）→ `docs/` 或课程目录；
- 可再生的处理结果 → `outputs/` 或本会话的 `derived/`；
- 示例图片/标定板 → `assets/`。
