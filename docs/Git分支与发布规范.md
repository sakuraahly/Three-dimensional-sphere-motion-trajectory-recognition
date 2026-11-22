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
