# SphereMotion 0.3.2

## 本版重点

- **launcher 默认管理员权限**：`launcher.exe` 嵌入 `requireAdministrator` manifest。
  实测普通权限下所有相机（含内置相机）都会 `isOpened=False`；管理员权限下恢复出图。
- **不再创建终端窗口**：launcher 通过隐藏的 `cmd` 调用 `run.bat guiw` →
  `pythonw.exe`，不会出现控制台窗口。
- **安装器支持覆盖升级**：同一 `AppId` 直接覆盖旧版本，沿用上次安装目录；
  升级前只删除代码目录，`config/`、`data/`、`logs/`、`outputs/`、`models/` 原样保留。
- 安装包继续自带完整 Python 3.14 运行时与依赖，无需用户另装 Python。

## 安装 / 升级

直接运行 `SphereMotion-0.3.2-setup.exe`：

- **全新安装**：默认装到 `%LOCALAPPDATA%\Programs\SphereMotion`。
- **覆盖升级**：已经装过旧版时，安装器提示覆盖，装回原目录并保留用户数据。

## 产物

- `SphereMotion-0.3.2-setup.exe`
  - SHA-256：`0817E8CD67592F91C26198C9F4952A67A880A284E9DF5A2446B40D7075216512`
  - 大小：91,805,265 bytes
