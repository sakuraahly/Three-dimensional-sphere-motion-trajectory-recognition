# SphereMotion 0.3.3

## 本版重点

- **管理员 launcher 保留**：`launcher.exe` 继续通过 `requireAdministrator` 请求提权，
  相机在管理员权限下可用；不显示终端窗口（`run.bat guiw` + `pythonw`）。
- **安装器快速开始可打开**：安装完成页不再依赖 `.md` 文件关联，改为用
  `notepad.exe` 打开 `README.md`。
- **新装首次启动自动建档案**：启动时如果没有 active 装置档案，自动创建并选中
  「平行装置」，门厅状态条会提示去「装置档案」查看/切换，并引导下一步标定。
  已有用户档案时不会重复创建或覆盖。
- **覆盖升级**：同一 `AppId` 装回上次目录，升级前关闭旧程序；只删代码目录，
  `config/`、`data/`、`logs/`、`outputs/`、`models/` 保留。

## 安装

运行 `SphereMotion-0.3.3-setup.exe`，按向导选择 Python（推荐自带运行时）。

## 产物

- `SphereMotion-0.3.3-setup.exe`
  - SHA-256：`3C804D05B0639605E4A47149D927721B2523D3D47CFD6DF077B884ECA5E9D82C`
  - 大小：`91,809,687` bytes
