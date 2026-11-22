# Git 分支与发布规范（LTS）

> 2026-10-24 起执行。目的：**成熟版本可长期回退，新硬件/新功能不污染主版本。**

## 1. 分支模型

| 分支 | 用途 | 规则 |
| --- | --- | --- |
| `main` | **LTS 稳定版** | 只接受已验收的发布合并；禁止直接开发/直接提交 |
| `lts/1080` | 1080p 成熟基线（只读备份） | 固定在 `2945252`，不继续开发 |
| `dev/hd-camera` | 1440p 新相机开发集成分支 | 当前开发主线；功能分支从这里切出并合回 |
| `feature/*` / `fix/*` | 单个功能/修复 | 从 `dev/hd-camera` 切出，完成后合回 dev |
| `release/*` | 发布候选 | 从 dev 切出，冻结后只修发布阻塞问题 |

当前标记：

```text
v1.0-lts    -> 2945252  （原 1080p 成熟版）
v2.0-dev.1  -> 77f4f10  （1440p 自动 FOV 开发快照，未验收）
```

## 2. 开发规则

1. **所有新工作先确认分支**；默认在 `dev/hd-camera`。
2. 不在 `main` 上直接改代码；`main` 只做发布合并。
3. 每个功能/修复一个分支，提交信息用英文 `type: summary`。
4. 合并前必须：
   - 全量 `run.bat test full` 26/26；
   - `check_standards.py` 通过；
   - 独立 Critic 复核；
   - 涉及真机的部分要有真机证据；
   - 更新交接/决策文档。
5. 合并到 dev 用普通 merge 或 squash；不得强推已共享分支。

## 3. 发布到 LTS 的门禁

`dev/hd-camera` → `release/hd-camera` → `main`：

```text
1. 从 dev 切 release/*
2. 冻结功能，只修发布阻塞 bug
3. 全量测试 + 规范检查 + 真机标定/录制/复检
4. 独立 Critic 给 PASS
5. 向用户汇报 release 候选与证据
6. 用户明确批准后，才允许合并 main
7. main 上打 annotated tag：vX.Y.Z-lts
```

**未经用户明确批准，不得合并到 main，不得 push。**

## 4. 回退

- `main` 始终指向最近一个已批准的 LTS tag。
- 出问题优先 `git switch main` 或 `git checkout v1.0-lts` 回退。
- 不删除 LTS 分支/tag；不 `reset --hard` 已发布 tag。
- 发布前创建本地 bundle 备份：

```powershell
git bundle create outputs/backups/repo-<tag>.bundle --all
```

## 5. 当前状态

- `main` / `lts/1080` = `2945252`（旧成熟版）。
- `dev/hd-camera` = 1440p 开发版，**尚未验收，不得合并**。
- 新相机相关提交只在 dev。

## 6. 制作发布包（2026-11-22）

发布包是**可分发的产物**，与"分支合并到 LTS"是两件事：
分支门禁（§3）管"代码能不能进 main"，本节管"怎么把当前 dev 打成能发给别人的包"。
**当前 dev 未验收，只允许制作 dev 快照包给自己/协作者试用，不得当正式版发布。**

### 6.1 两种包，按使用者选

| 形态 | 命令 | 产物 | 门槛 |
| --- | --- | --- | --- |
| 源码 zip | `python apps/tools/make_clean_bundle.py` | `outputs/clean_bundle_*.zip`（约 3 MiB） | 使用者需自装 Python 3.10+ |
| **Windows 安装程序（推荐）** | `installer\build_installer.bat` | `outputs/releases/SphereMotion-<版本>-setup.exe`（约 200 MB） | 制作机需 Inno Setup 6；**使用者不用装 Python**，向导里选自带或自带之外的解释器 |

> 原"免安装绿色包"方案已并入 Windows 安装程序：内嵌 Python 由向导的单选项
> "Use the bundled Python 3.14" 提供，选它即等价于绿色包体验（解压到用户目录即用）。

### 6.2 制作 setup.exe 的前置与核对

```bat
winget install --id JRSoftware.InnoSetup -e     :: 系统级安装，只做一次
python installer\verify_payload.py              :: 不装 Inno 也能先验源码 payload
python installer\build_runtime.py --check       :: 只校验内嵌运行时（不重建）
installer\build_installer.bat                   :: 一键出 setup.exe（含运行时下载/构建）
```

`build_installer.bat` 内部做四件事，缺一就停：
1. `make_clean_bundle.py` 出纯净源码包（**包内容只有这一份真相**，不在 .iss 里重写清单）；
2. `build_runtime.py` 构建内嵌 Python：下载 python-build-standalone 官方
   `3.14.10+20261110` install_only_stripped（**固定 sha256 `32b6cf90…efaab8f`**），
   装齐 `requirements.txt`，删 `__pycache__`/包自带 tests（约 390 MB）；有缓存就跳过；
3. 把 runtime 挪出 staging 后解压源码包、放入 `launcher.exe`、再挪回 runtime
   （几百 MB 产物不被 rmdir 误删）；
4. 调 `ISCC.exe installer\sphere_motion.iss` 产出 setup.exe。

**内嵌运行时的设计**：版本/URL/sha256/依赖/精简策略全集中在 `installer/build_runtime.py`
顶部（单一真相）；产物 `outputs/release_stage/runtime/` 是**构建产物，被 gitignore、
不入库、不进源码 zip**，只在 setup.exe 里出现。源码仓库因此保持轻量。

**向导里的解释器选择（Inno Pascal Script，`sphere_motion.iss` 的 `[Code]`）**：
- `CreateInputOptionPage` 单选：默认"用自带的 Python 3.14"，或"本机已有 python.exe"；
- `CreateInputFilePage` + `NextButtonClick`：选外部解释器时浏览选路径，并真跑
  `import cv2, numpy` 校验，不合格不让继续；
- `ShouldSkipPage`：选自带时跳过路径页；`Check: WantBundledPython`：选自带才复制 runtime；
- `CurStepChanged(ssPostInstall)`：写 `config/python_env.txt`（与 `setup_env.py`
  的 `write_pointer()` 同格式，`python=` 用相对路径 `runtime/python/python.exe`），
  Go launcher / bat / Python 三方同读这一份。以后换解释器改该文件的 `python=` 即可。

装到 `{localappdata}\SphereMotion`、免管理员；卸载用 Inno 自带卸载器，不用 `uninstall.bat`。

**2026-11-22 实测过的编译坑（别再踩）**：

| 坑 | 现象 | 正解 |
| --- | --- | --- |
| `StageDir` 相对路径 | ISCC 以 `.iss` 所在的 `installer\` 为基准，图标/payload 全找不到 | define 里写 `..\outputs\...` 回退一层 |
| `/S<dir>` | 以为能设 SourceDir，实际是 SignTool 选项 | 别用；用相对路径 |
| 中文语言包 | winget 国际版只有 29 种欧洲 `.isl`，无中文 | 用 `compiler:Default.isl`（英文向导）；要中文从官网下语言包 |
| Tasks 默认勾选 | 起初误判 `Flags: unchecked` 非法而全部删掉——后果是**静默装时默认全勾**，往用户桌面塞快捷方式 | `Flags: unchecked` / `Flags: checkedonce` 都是合法的（当时是别的语法错误）；`desktopicon` 必须 `unchecked` |
| Temp 被安全软件拦 | 双击 setup.exe 报 `Setup was unable to create the directory "...\Temp\is-XXXX.tmp" Error 5: 拒绝访问`；但同机 PowerShell 能建目录、Temp ACL 正常 | 360/火绒等 HIPS 拦「未签名 exe 在 Temp 释放可执行文件」。解法：加信任区/临时关防护，或先把 `TMP`/`TEMP` 指到别的盘再运行（实测 `/VERYSILENT` 可过）。根治要代码签名证书 |
| 日期格式化 | `FormatDateTime`/`DateToStr` 在 Inno 脚本子集里都不存在 | `created=` 写固定值 `by-installer`（只是元数据，非判据） |
| `CreateInputFilePage` 参数 | 只收 4 个参数（AfterID + 3 文案），多传一个空串报错 | 按 Inno 6 官方签名传 |
| winget 安装路径 | 不是 Program Files，而是 `%LOCALAPPDATA%\Programs\Inno Setup 6` | `build_installer.bat` 三个路径都试 |

编译结果：`outputs/releases/SphereMotion-0.3.0-setup.exe`（内嵌 Python 3.14.10，
93.9 MB，合法 PE，不入库）。

**2026-11-22 首次真机安装验证通过**（装到 `D:\SphereMotionTest\app`，避开 C 盘 Temp）：

| 验证项 | 结果 |
| --- | --- |
| 静默安装 | 退出码 0，日志 `Installation process succeeded`；6176 文件 / 406 MB |
| 环境指针 | `python=runtime/python/python.exe` / `kind=bundled` / `python_version=3.14.10` |
| 内嵌解释器 | `python 3.14.10` + `cv2 5.1.0` + `numpy 2.5.5` + `scipy 1.18.1` + `tkinter OK` |
| launcher 探测 | `--dry-run` 显示 `pointer/python/guiPython` 全部指向安装目录内 `runtime\python\*`（**tried 1 candidates**，未回落系统 Python） |
| 卸载 | 卸载器退出码 0，桌面 lnk / 开始菜单 / HKCU 注册表项全部清除（仅留空目录） |
| 快捷方式默认值 | 桌面 **不创建**（`Flags: unchecked`）、开始菜单创建（卸载入口） |

### 6.3 发布到 GitHub Release（需用户批准）

只有 §3 门禁全过、独立 Critic PASS、真机证据齐、用户明确批准后才做：

```bash
git tag -a v2.0-dev.N -m "dev 快照" dev/hd-camera   # dev 快照不需合 main
git push origin v2.0-dev.N
# 再把 outputs/releases/*-setup.exe 与 clean_bundle zip 传到对应 Release
```

未经用户批准不得打 tag / 上传 Release（外部可见、不可随意撤销）。
打 dev tag 不等于合 LTS；`main` 仍只接受 §3 的发布合并。
