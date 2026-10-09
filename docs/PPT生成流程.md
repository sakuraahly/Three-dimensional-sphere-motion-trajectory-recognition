# 开题报告 PPT 的生成流程

> 这份文档解决一个具体问题：**`apps/tools/` 下有两套名字很像的 PPT 工具，
> 该用哪一套、命令怎么写**。
>
> 相关文档：`docs/交接文档.md`（现状与踩坑）、`docs/文件结构.md`（目录职责）。

---

## 1. 先认清：PPT 不在本仓库

开题报告的 `.pptx` 放在**你自己的课程目录**里，不进版本库：

    D:\<课程资料目录>\大二上\普通物理学实验2\开题报告.pptx

原因有两个：

1. 它是**交付物**，不是可再生的构建产物 —— `outputs/` 被整个 gitignore，
   放进去等于 git 里没有副本，丢一次就找不回来。
2. 文稿本身在 `docs/开题报告_PPT逐页文稿.md` 与
   `docs/开题报告_研究思路四页文稿.md`，PPT 是可以随时重建的。

**所以流程是**：文稿（在仓库里）→ 脚本生成 pptx（落到 `outputs/`）→ 你手动拷到课程目录。

---

## 2. 两套工具的分工（别再混了）

| | `*_opening_report`（**主路径**） | `*_opening_ppt`（早期路径） |
|---|---|---|
| 作用对象 | **整份**开题报告（你的原稿） | 只有"研究思路"**四页** |
| 输入 | `开题报告.pptx`（25 页原稿） | `开题报告模版.pptx` |
| 输出 | 34 页完整版 | 独立的四页 pptx |
| 何时用 | **现在要交开题报告时** | 一般不用了，留作历史 |

> 判断口诀：**要交报告用 `_report`，只想看四页用 `_ppt`。**

---

## 3. 主路径：一键重建整份报告

### 3.1 前置

```powershell
$PY = "C:\Users\<USER>\miniforge3\envs\myenv\python.exe"
cd D:\MY_CODING_PROGRAM\Three-dimensional-sphere-motion-trajectory-recognition
```

> ⚠️ 别用裸 `python`，会解析到 msys64 那个没装 opencv 的解释器。

需要 `python-pptx` / `Pillow` / `lxml`（已装，见 `交接文档.md` §2）。
这些**只是做 PPT 用的**，跑 `apps/` 与 `tests/` 都不需要。

### 3.2 生成

```powershell
# 先空跑看它打算做什么（不写文件）
& $PY apps\tools\build_opening_report.py --src "D:\...\开题报告.pptx" --dry-run

# 正式生成
& $PY apps\tools\build_opening_report.py `
    --src "D:\<课程资料目录>\大二上\普通物理学实验2\开题报告.pptx" `
    --out outputs\开题报告_最终.pptx
```

脚本会：

1. 按**标题**定位「装置搭建 / 软件控制 / 数据采集 / 测量精度 / 时间规划 / 参考文献」
   六页（不靠硬编码页码，页序变了也不会错位）；
2. 按内容量拆页（25 页 → 34 页）；
3. 全篇套用《PPT字号等》规范：

   | 元素 | 规范 |
   |---|---|
   | 字体 | 中文**宋体** / 英文数字 **Times New Roman** |
   | 正文 | **26 号**、行距 **1.5 倍** |
   | 页左上角小标题 | **32 号** |
   | 表格 | 自行调整但 **≥ 20 号** |
   | 参考文献 | **22 号**、行距 **1 倍**、国标 **GB/T 7714** |
   | 物理量符号 | 斜体且全文统一（`ε_Z` 的 Z 用真下标） |

### 3.3 自检（**必做**）

```powershell
& $PY apps\tools\check_opening_report.py outputs\开题报告_最终.pptx
```

输出五项的通过情况：

```
[1] 空白内容占位符 : 0            <- 会显示成"单击此处添加文本"的元凶
[2] 小于 20 号     : 内容页 0
[3] 字体不符       : 内容页 0/282
[4] 正文行距非 1.5 : 内容页 0
[5] 总页数         : 34
```

以 `--out` 之外的路径调用也没问题（默认就是 `outputs/开题报告_最终.pptx`）。
**只要 [1]~[4] 有任何一项非 0，就不要交**。

### 3.4 肉眼看一遍排版

脚本判断不了"好不好看"，所以再渲染几张图确认：

```powershell
& $PY apps\tools\preview_pptx.py outputs\开题报告_最终.pptx `
    --slides 17,18,20,24,30 --out outputs\final_preview
```

PNG 落在 `outputs/final_preview/`。这是**近似**渲染（python-pptx + PIL），
不能替代 PowerPoint，但足以发现文字出界、元素重叠、内容被切。

粗略查"有没有文字顶到页面底边"：

```powershell
& $PY apps\tools\_overflow_lib.py outputs\开题报告_最终.pptx
```

### 3.5 交付

```powershell
copy "outputs\开题报告_最终.pptx" "D:\<课程资料目录>\大二上\普通物理学实验2\开题报告.pptx"
```

**拷之前先备份你课程目录里的那份**（同一条命令既是输入也是去向，很容易盖掉手工修改）。

---

## 4. 早期路径（`*_opening_ppt`，一般不用）

```powershell
& $PY apps\tools\build_opening_ppt.py       # 生成四页
& $PY apps\tools\check_opening_ppt.py       # 几何自检（出界/压页码/文字溢出）
& $PY apps\tools\preview_opening_ppt.py     # 预览到 outputs/ppt_preview/
```

它只处理"研究思路"四节、输出独立 pptx。整份报告已由主路径覆盖，
**保留它是为了历史上那版四页文稿还能复现**。

---

## 5. 踩过的坑（改脚本前先读）

### 5.1 `add_slide(layout)` 会克隆**全部**占位符

`python-pptx` 新建页时，会把版式上所有占位符都复制过来，
包括 `idx=13/14` 的内容占位符（`OBJECT` 类型）。
它们**不是空白形状** —— PowerPoint 会把它们渲染成
「单击此处添加文本」和插入图片/表格的图标。

- ❌ 只删"非占位符形状"**不够**，实测残留过 15 个，散在 9 页上；
- ✅ 正确做法：新页上**只保留标题占位符（`idx == 0`）**，其余占位符一律删。

`build_opening_report.py` 的 `clone_after()` 已按这个来写，
并且最后还有一道兜底扫描。**改这块时别把兜底删了。**

### 5.2 `prs.slides` 是即时计算的，不是快照

新增页之后，`prs.slides` 的索引会变。所以：

- 想"把新页插到第 22 页后面"**不要**反复调用"移动一页" —— 第二次调用时索引已经偏了（这个坑踩了两次）；
- ✅ 正确做法：**先把原始页序 `list(prs.slides)` 快照下来**，
  构造出完整的目标顺序，再用 `sldId.rId` 建映射**一次性重排**（`reorder()`）。

### 5.3 行高估算会偏小

按 `字号 × 行距 × 1.32` 估的行高，**比 PowerPoint 实际渲染偏小约 15%**
（宋体的行高系数比理论值大）。后果是最初参考文献第 11 条压到了底部注释上。

- 估版面时用 **≥1.4** 的系数，或直接用 `preview_pptx.py` 渲染出来看；
- 别把文字排到页面底边 95% 的位置，留余量。

### 5.4 中文与英文要分别设字体

OpenXML 用两个属性区分：`ea`（East Asian）与 `latin`。
只设 `run.font.name`（它写的是 `latin`）**中文不会变成宋体**。

```python
rPr = run._r.get_or_add_rPr()
rPr.find(f"{{...}}ea").set("typeface", "宋体")            # 中文
rPr.find(f"{{...}}latin").set("typeface", "Times New Roman")  # 英文数字
```

`check_opening_report.py` 的第 [3] 项就是查这个，两个属性都要对才算通过。

### 5.5 `~$` 锁文件

PPT 在 PowerPoint / WPS 里打开时，同目录会有 `~$开题报告.pptx`。
此时写入会失败或损坏文件。`.gitignore` 已忽略 `~$*`，
但**脚本本身不会帮你检测** —— 改文件前先确认已关闭。

---

## 6. 脚本清单

| 脚本 | 作用 |
|---|---|
| `build_opening_report.py` | **主**：整份报告一键重建 |
| `check_opening_report.py` | **主**：上面那份的合规自检 |
| `preview_pptx.py` | 任意 pptx 渲染成 PNG（排版检查） |
| `_preview_lib.py` | 上者的渲染实现（含连接线 off/ext + flip 处理） |
| `_overflow_lib.py` | 底部溢出扫描 |
| `build_opening_ppt.py` | 早期：只生成四页 |
| `check_opening_ppt.py` | 早期：四页的几何自检 |
| `preview_opening_ppt.py` | 早期：四页的预览 |

> 下划线开头的是内部模块，给上面的脚本 import 用，一般不用直接跑。
