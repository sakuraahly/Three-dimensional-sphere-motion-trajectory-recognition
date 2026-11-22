; ===========================================================================
; 球形物体三维运动轨迹识别 —— Windows 安装程序（Inno Setup 6）
;
; 由 installer\build_installer.bat 一键编译；payload 来自
; apps\tools\make_clean_bundle.py 的纯净包 + 根目录 launcher.exe。
;
; 设计原则（别改错）：
;   1. 装到 {localappdata}，不写 Program Files → 免管理员、无 UAC、无签名摩擦；
;   2. 卸载用 Inno 自带卸载器（可靠、可回滚）；uninstall.bat 只作为文件保留给
;      高级用户，不作为卸载入口 —— 两套删除逻辑并存必然打架；
;   3. 安装过程**不自动装 Python 依赖** —— 首次启动 launcher.exe 会检测环境，
;      缺则弹窗引导跑 install.bat。安装器保持"复制文件"这一件事，失败面最小。
; ===========================================================================

#define AppName "球形物体三维运动轨迹识别"
#define AppVer "0.2.0"
#define Publisher "SphereMotion"
#define AppId "6E4A1B7C-2D9F-4A3E-9B11-5C8D7E2F0A44"
; NOTE: ISCC 把脚本所在目录（installer\）当作相对路径基准，所以回退一层到仓库根。
#define StageDir "..\outputs\release_stage"
#define OutDir "..\outputs\releases"
#define OutBase "SphereMotion-" + AppVer + "-setup"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher={#Publisher}
DefaultDirName={localappdata}\SphereMotion
DefaultGroupName={#AppName}
OutputDir={#OutDir}
OutputBaseFilename={#OutBase}
Compression=lzma2/max
SolidCompression=yes
; 免管理员：不写 Program Files、不写 HKLM
PrivilegesRequired=lowest
; 图标来自 Go launcher 的源码资源（纯净包里已带）
SetupIconFile={#StageDir}\native\launcher\icon.ico
UninstallDisplayIcon={app}\launcher.exe
; 中文界面（Inno Setup 6 自带）
ShowLanguageDialog=no
LanguageDetectionMethod=locale
; 升级覆盖：同一 AppId 重装时提示
DisableDirPage=auto

[Languages]
; 用内置英文界面：winget 装的国际版只带 29 种欧洲语言，不含中文 .isl。
; 想要中文向导，从 https://jrsoftware.org/download.php/islang.exe 下载后
; 把 ChineseSimplified.isl 放进 Inno 的 Languages\ 目录，再把下一行换成：
;   Name: "chinese"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
; 两个快捷方式任务都默认不勾选，由用户自己决定 —— 避免依赖版本敏感的
; "Checked" 参数/flag（本机 Inno 6.7.3 两者都不接受，见 build 记录）。
Name: "desktopicon"; Description: "Create desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "startmenu"; Description: "Create Start Menu shortcuts"; GroupDescription: "Shortcuts:"

[Files]
; 整个 payload 原样装入（源码 + 文档 + 标定板 + launcher.exe）
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\launcher.exe"; Tasks: startmenu
Name: "{group}\安装环境与依赖"; Filename: "{app}\install.bat"; Tasks: startmenu
Name: "{group}\卸载"; Filename: "{uninstallexe}"; Tasks: startmenu
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\launcher.exe"; Tasks: desktopicon

[Run]
; 不自动启动：安装完成后由用户自己双击（首次启动会引导装环境）
; 这里只写一个"打开说明书"的可选勾选项
Filename: "{app}\README.md"; Description: "查看快速开始说明"; Flags: postinstall skipifsilent unchecked

[UninstallDelete]
; 运行期产物不入库但会生成在用户目录，卸载时一并清掉（只删安装目录内）
Type: filesandordirs; Name: "{app}\config\python_env.txt"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\outputs"

[Messages]
english.WelcomeLabel2=安装 {#AppName} {#AppVer} to your user directory.{newline}{newline}Setup only copies files; it never touches your system Python. On first launch, the app will guide you through installing the runtime if missing.
