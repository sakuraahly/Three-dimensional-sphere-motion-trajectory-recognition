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

### 6.1 三种包，按使用者选

| 形态 | 命令 | 产物 | 门槛 |
| --- | --- | --- | --- |
| 源码 zip | `python apps/tools/make_clean_bundle.py` | `outputs/clean_bundle_*.zip`（约 3 MiB） | 使用者需自装 Python 3.10+ |
| Windows 安装程序 | `installer\build_installer.bat` | `outputs/releases/SphereMotion-<版本>-setup.exe` | 制作机需 Inno Setup 6；使用者双击即可 |
| 免安装绿色包 | 规划中（内嵌 Python + 预装依赖） | 文件夹/zip，解压即用 | 制作时需处理 opencv/scipy 的 DLL |

### 6.2 制作 setup.exe 的前置与核对

```bat
winget install --id JRSoftware.InnoSetup -e     :: 系统级安装，只做一次
python installer\verify_payload.py              :: 不装 Inno 也能先验 payload
installer\build_installer.bat                   :: 一键出 setup.exe
```

`build_installer.bat` 内部做三件事，缺一就停：
1. `make_clean_bundle.py` 出纯净源码包（**包内容只有这一份真相**，不在 .iss 里重写清单）；
2. 解压到 `outputs/release_stage` 并放入根目录的 `launcher.exe`（没有就报错，提示先编译）；
3. 调 `ISCC.exe installer\sphere_motion.iss` 产出 setup.exe。

装到 `{localappdata}\SphereMotion`、免管理员；不自动装依赖（首次启动 launcher 引导）；
卸载用 Inno 自带卸载器，不用 `uninstall.bat`。

**2026-11-22 实测过的编译坑（别再踩）**：

| 坑 | 现象 | 正解 |
| --- | --- | --- |
| `StageDir` 相对路径 | ISCC 以 `.iss` 所在的 `installer\` 为基准，图标/payload 全找不到 | define 里写 `..\outputs\...` 回退一层 |
| `/S<dir>` | 以为能设 SourceDir，实际是 SignTool 选项 | 别用；用相对路径 |
| 中文语言包 | winget 国际版只有 29 种欧洲 `.isl`，无中文 | 用 `compiler:Default.isl`（英文向导）；要中文从官网下语言包 |
| Tasks 默认勾选 | `Flags: checked/unchecked`、`Checked: yes` 本机 6.7.3 全报错 | 两个 task 都不默认勾选，用户自己选 |
| winget 安装路径 | 不是 Program Files，而是 `%LOCALAPPDATA%\Programs\Inno Setup 6` | `build_installer.bat` 三个路径都试 |

编译结果：`outputs/releases/SphereMotion-0.2.0-setup.exe`（4.54 MB，合法 PE，不入库）。
安装未在本机执行（安全策略拦了执行安装包）——编译自检 + payload 自检全过，
真正安装验证请由人双击 setup.exe 完成。

### 6.3 发布到 GitHub Release（需用户批准）

只有 §3 门禁全过、独立 Critic PASS、真机证据齐、用户明确批准后才做：

```bash
git tag -a v2.0-dev.N -m "dev 快照" dev/hd-camera   # dev 快照不需合 main
git push origin v2.0-dev.N
# 再把 outputs/releases/*-setup.exe 与 clean_bundle zip 传到对应 Release
```

未经用户批准不得打 tag / 上传 Release（外部可见、不可随意撤销）。
打 dev tag 不等于合 LTS；`main` 仍只接受 §3 的发布合并。
