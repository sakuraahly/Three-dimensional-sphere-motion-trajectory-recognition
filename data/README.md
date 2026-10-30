# data/ — 真机录制会话

> **这里放"删了就没了"的数据。** 与相邻目录的区别：

| 目录 | 语义 | 丢了怎么办 |
| ---- | ---- | ---------- |
| `outputs/` | 可再生的运行产物（标注视频、CSV、调试图） | 重跑一次就回来 |
| `assets/` | 入库的静态素材（标定板 PDF、示例图） | 从 git 恢复 |
| **`data/`（这里）** | **不可再生的真机录制** | **只能靠备份** |

## 目录形态（2026-10-30 起）

```
data/stereo/<YYYYMMDD-HHMMSS>/    双目会话
data/mono/<YYYYMMDD-HHMMSS>/      单目会话
```

一个目录 = 一次采集，但**目录名按类型分**。以前双目放在 `data/sessions/` ——
`sessions` 是个通用词，里面却只有双目，而单目又另起 `data/mono`，
于是"会话"在磁盘上指了两样东西（用户原话："命名有歧义"）。
路径的唯一来源是 `sphere_tracker/paths.py`；旧位置 `data/sessions/` 仍被读取
（`stereo_roots()`），但不再写入，迁移用 `apps/tools/migrate_data_layout.py`。

## 双目会话长什么样

```
data/stereo/20261019-101500/
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

## 单目会话长什么样

```
data/mono/20261029-205522/
  clip.mkv               ← 原始视频（HuffYUV 无损，约 82 MB/s；10 秒 ≈ 800 MB）
  clip.mkv.json          ← 录制器写的实测值（分辨率 / 帧数 / 编码器）
  session.json           ← 会话元数据（相机、预期参数；实测值由 sync_from_take 补回）
  derived/               ← traj.csv / annotated.mp4 / visualization.json（可删）
```

- 单目**也要 `derived/visualization.json`** 才能进 3D 查看器；它由
  `run_video --vis-json` 写，**不写就只是"跑完了"，查看器里看不到**。
- 单目的深度靠球径先验（`Z = f·D/(2r)`），不是视差。`visualization.json` 里
  记着 `ball_diameter_mm` 与 `fx` —— 看到这两个数就该知道"这个 Z 是算出来的"。
- 录 10 秒约 800 MB，注意磁盘；`--codec mp4v` 可压小但丢无损。

## 校验

```powershell
$PY = "$env:USERPROFILE\miniforge3\envs\myenv\python.exe"
& $PY apps\tools\check_dataset.py data\stereo\20261019-101500
& $PY apps\tools\check_dataset.py --root data\stereo            # 全部
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
