; ===========================================================================
; 球形物体三维运动轨迹识别 —— Windows 安装程序（Inno Setup 6）
;
; 由 installer\build_installer.bat 一键编译；payload 来自
; apps\tools\make_clean_bundle.py 的纯净包 + launcher.exe + 内嵌 Python 运行时。
;
; 设计原则（别改错）：
;   1. 装到用户级 Programs（{userpf}\SphereMotion，C 盘、免管理员、无 UAC）；
;      不写 Program Files，避免运行时无法写 data/config；
;   2. 卸载用 Inno 自带卸载器（可靠、可回滚），卸载时让用户选择
;      「只删程序」或「程序+用户数据」；uninstall.bat 只作为文件保留给
;      高级用户，不作为卸载入口 —— 两套删除逻辑并存必然打架；
;   3. 安装向导里让用户选解释器：默认用安装包自带的 Python 3.14（开箱即用，
;      不用自己装 Python），也可选本机已有 python.exe。选定后写入
;      config/python_env.txt —— 与 setup_env.py 的 write_pointer 同格式，
;      Go launcher / bat / Python 三方同读这一份真相；
;   4. 内嵌运行时按选择装：选"自带"才复制到 {localappdata}\SphereMotion\runtime
;      （几百 MB，固定 C: 盘；Windows 相机权限会拒绝非系统盘进程），
;      选"自己的"就不复制，避免白占磁盘。
; ===========================================================================

#define AppName "球形物体三维运动轨迹识别"
#define AppVer "0.3.1"
#define Publisher "SphereMotion"
#define AppId "6E4A1B7C-2D9F-4A3E-9B11-5C8D7E2F0A44"
; ISCC 把脚本所在目录（installer\）当相对路径基准，回退一层到仓库根。
#define StageDir "..\outputs\release_stage"
#define OutDir "..\outputs\releases"
#define OutBase "SphereMotion-" + AppVer + "-setup"
#define PointerRel "config\python_env.txt"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher={#Publisher}
; {userpf} = C:\Users\<用户>\AppData\Local\Programs（用户级“Programs”）。
; 不写死盘符；管理员模式也不改这里，保持可预测。
DefaultDirName={userpf}\SphereMotion
DefaultGroupName={#AppName}
OutputDir={#OutDir}
OutputBaseFilename={#OutBase}
Compression=lzma2/max
SolidCompression=yes
; 免管理员：不写 Program Files、不写 HKLM
PrivilegesRequired=lowest
SetupIconFile={#StageDir}\native\launcher\icon.ico
UninstallDisplayIcon={app}\launcher.exe
ShowLanguageDialog=no
LanguageDetectionMethod=locale
DisableDirPage=auto
; 需要展示自定义向导页（单选项 + 文件选择），不能用 /VERYSILENT 时静默跳过
; —— 但 silent 安装时 NextButtonClick 仍会被调用，默认值可用。

[Languages]
; 中文语言文件随仓库提供（installer/ChineseSimplified.isl，来自 jrsoftware/issrc），
; 构建机不需要另外安装 Inno 语言包。英文保留为回退。
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"

[Tasks]
; 桌面快捷方式默认**不勾选**：安装器不该往桌面塞东西，用户要就自己勾。
; 开始菜单默认勾选（卸载入口在那里，没有它用户找不到怎么卸）。
Name: "desktopicon"; Description: "创建桌面快捷方式 / Create desktop shortcut"; GroupDescription: "快捷方式 / Shortcuts:"; Flags: unchecked
Name: "startmenu"; Description: "创建开始菜单快捷方式 / Create Start Menu shortcuts"; GroupDescription: "快捷方式 / Shortcuts:"; Flags: checkedonce

[Files]
; 源码 + 文档 + 标定板 + launcher.exe（永远装）
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "runtime\*"
; 内嵌 Python 运行时：只有用户选"用自带的"才复制（几百 MB，按需）。
; **必须放 C: 盘**：Windows 相机权限在部分机器上会拒绝 D: 等非系统盘上的进程，
; MSMF/DSHOW 都会报“can't be used to capture by index”（内置相机也一样）。
; 应用本身允许装在其他盘，但 Python 进程固定到 {localappdata}（默认 C:）。
Source: "{#StageDir}\runtime\*"; DestDir: "{localappdata}\SphereMotion\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: WantBundledPython

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\launcher.exe"; Tasks: startmenu
Name: "{group}\安装环境与依赖"; Filename: "{app}\install.bat"; Tasks: startmenu
Name: "{group}\卸载"; Filename: "{uninstallexe}"; Tasks: startmenu
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\launcher.exe"; Tasks: desktopicon

[Run]
; 完成后可选打开说明书（不强制启动 GUI）
Filename: "{app}\README.md"; Description: "查看快速开始说明 / View quick start"; Flags: postinstall skipifsilent unchecked

[InstallDelete]
; 旧版本把 runtime 放在 {app}；升级时清掉，避免 D: 上的旧 Python 被误用。
Type: filesandordirs; Name: "{app}\runtime"

[UninstallDelete]
; 运行时是程序本身，永远删（现在固定在 C: 的 localappdata）。
; 用户数据（config / data / logs / outputs）由 [Code] 的卸载选项决定：
; 「只删程序」保留；「程序+数据」才 DelTree。
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{localappdata}\SphereMotion\runtime"

[Code]
// ---------------------------------------------------------------------------
// 解释器选择：两个标准向导页 + 写环境指针。
// 用 CreateInputOptionPage / CreateInputFilePage / ShouldSkipPage /
// NextButtonClick / CurStepChanged 全部是 Inno 6 官方 API（见 is6help）。
// ---------------------------------------------------------------------------
var
  ChoicePage: TInputOptionWizardPage;
  PathPage: TInputFileWizardPage;
  ChosenPython: String;
  ChosenKind: String;
  DeleteUserData: Boolean;

function IsChinese: Boolean;
begin
  Result := ActiveLanguage = 'chinesesimplified';
end;

procedure InitializeWizard;
begin
  // 单选页：绑在"选择安装位置"之后。Exclusive=True → 单选按钮。
  if IsChinese then
  begin
    ChoicePage := CreateInputOptionPage(wpSelectDir,
      'Python 运行环境',
      '选择要使用的 Python 解释器',
      '安装包自带可用的 Python 3.14 运行时和全部依赖（OpenCV、NumPy、' +
        'SciPy、Tkinter）。也可以使用本机已经安装的 Python。' +
        '之后可以编辑 config\python_env.txt 更改。',
      True, False);
    ChoicePage.Add('使用安装包自带的 Python 3.14（推荐，无需另外安装）');
    ChoicePage.Add('使用本机已安装的 Python 解释器');
    ChoicePage.SelectedValueIndex := 0;

    PathPage := CreateInputFilePage(ChoicePage.ID,
      '选择 Python 解释器',
      '定位 python.exe',
      '选择要使用的 python.exe。它必须能够 import cv2 和 numpy。');
    PathPage.Add('python.exe：', 'Python executable|python.exe|All files|*.*', '.exe');
  end
  else
  begin
    ChoicePage := CreateInputOptionPage(wpSelectDir,
      'Python runtime',
      'Choose which Python interpreter to use',
      'This package includes a ready-to-use Python 3.14 runtime with all dependencies ' +
        '(OpenCV, NumPy, SciPy, Tkinter). You may also use a Python already installed ' +
        'on this computer. You can change this later by editing config\python_env.txt.',
      True, False);
    ChoicePage.Add('Use the bundled Python 3.14 (recommended, nothing else to install)');
    ChoicePage.Add('Use a Python interpreter already installed on this computer');
    ChoicePage.SelectedValueIndex := 0;

    PathPage := CreateInputFilePage(ChoicePage.ID,
      'Select Python interpreter',
      'Locate python.exe',
      'Select the python.exe to use. It must be able to import cv2 and numpy.');
    PathPage.Add('python.exe:', 'Python executable|python.exe|All files|*.*', '.exe');
  end;
  PathPage.Values[0] := '';
end;

// 选"自带的"时跳过文件选择页。
function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  if (PageID = PathPage.ID) and (ChoicePage.SelectedValueIndex = 0) then
    Result := True;
end;

// [Files] 的 Check：选"自带的"才复制 runtime\。
function WantBundledPython: Boolean;
begin
  Result := ChoicePage.SelectedValueIndex = 0;
end;

// 校验用户选的 python.exe：存在 + 能 import cv2/numpy。
// 静默安装（/VERYSILENT）时也会走到这里，默认值=0 不需要校验路径。
function NextButtonClick(CurPageID: Integer): Boolean;
var
  py: String;
  rc: Integer;
begin
  Result := True;
  if CurPageID <> PathPage.ID then Exit;
  if ChoicePage.SelectedValueIndex = 0 then Exit;

  py := Trim(PathPage.Values[0]);
  if (py = '') or not FileExists(py) then
  begin
    if IsChinese then
      MsgBox('请选择有效的 python.exe 路径。', mbError, MB_OK)
    else
      MsgBox('Please select a valid python.exe path.', mbError, MB_OK);
    Result := False;
    Exit;
  end;
  // -c 探针：真跑一次 import，不靠"路径看起来对"。SW_HIDE 避免弹黑框。
  if Exec(py, '-c "import cv2, numpy"', '', SW_HIDE, ewWaitUntilTerminated, rc) then
  begin
    if rc <> 0 then
    begin
      if IsChinese then
        MsgBox('这个 Python 无法 import cv2 和 numpy。' + #13#10 +
               '请换一个解释器，或使用安装包自带的 Python。', mbError, MB_OK)
      else
        MsgBox('This Python cannot import cv2 and numpy.' + #13#10 +
               'Please choose another interpreter, or use the bundled one.', mbError, MB_OK);
      Result := False;
    end;
  end
  else
  begin
    if IsChinese then
      MsgBox('无法运行所选的 Python。请检查路径后重试。', mbError, MB_OK)
    else
      MsgBox('Could not run the selected Python. Check the path and try again.', mbError, MB_OK);
    Result := False;
  end;
end;

// 安装完成时写 config/python_env.txt。
// 格式与 apps/setup_env.py 的 write_pointer() 一致：
//   python=<相对或绝对路径>  kind=...  created=...  python_version=...  verified=...
function PyVer: String;
begin
  if WantBundledPython then Result := '3.14.10' else Result := 'external';
end;

function VerifiedList: String;
begin
  if WantBundledPython then Result := 'cv2,numpy,scipy,tkinter' else Result := 'cv2,numpy';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  lines: TStringList;
begin
  if CurStep <> ssPostInstall then Exit;

  if WantBundledPython then
  begin
    // 绝对路径：运行时不跟 {app} 走，固定放 C:（见 [Files] 的说明）。
    ChosenPython := ExpandConstant('{localappdata}\SphereMotion\runtime\python\python.exe');
    ChosenKind := 'bundled';
  end
  else
  begin
    ChosenPython := Trim(PathPage.Values[0]);
    ChosenKind := 'adopted';
  end;

  lines := TStringList.Create;
  try
    lines.Add('# sphere-python-env v1 - generated by installer; do not edit');
    lines.Add('python=' + ChosenPython);
    lines.Add('kind=' + ChosenKind);
    lines.Add('created=by-installer');
    lines.Add('python_version=' + PyVer);
    lines.Add('verified=' + VerifiedList);
    // 确保 config\ 存在（源码包里有 .gitkeep，已随 [Files] 装好）
    if not DirExists(ExpandConstant('{app}\config')) then
      CreateDir(ExpandConstant('{app}\config'));
    lines.SaveToFile(ExpandConstant('{app}\' + '{#PointerRel}'));
  finally
    lines.Free;
  end;
end;

// ---------------------------------------------------------------------------
// 卸载选项：只删程序 / 程序 + 用户数据。默认保留数据（最安全的默认）。
// 这不是安装向导页；Inno 卸载器的 [Code] 以 InitializeUninstall 的返回值
// 决定是否继续，所以用一次经典 Yes/No/Cancel 确认。
// ---------------------------------------------------------------------------
function InitializeUninstall(): Boolean;
var
  ans: Integer;
  msg: String;
begin
  // 静默卸载不弹窗，默认保留用户数据，避免脚本/部署队列误删。
  if UninstallSilent then
  begin
    DeleteUserData := False;
    Result := True;
    Exit;
  end;

  if IsChinese then
    msg := '卸载 球形物体三维运动轨迹识别' + #13#10 + #13#10 +
      '是：删除程序 + 用户数据（data / config / logs / outputs）' + #13#10 +
      '否：只删除程序，保留用户数据' + #13#10 +
      '取消：退出卸载'
  else
    msg := 'Uninstall SphereMotion' + #13#10 + #13#10 +
      'Yes: remove program + user data (data / config / logs / outputs)' + #13#10 +
      'No: remove program only, keep user data' + #13#10 +
      'Cancel: abort uninstall';

  ans := MsgBox(msg, mbConfirmation, MB_YESNOCANCEL);
  if ans = IDCANCEL then
  begin
    Result := False;
    Exit;
  end;
  DeleteUserData := ans = IDYES;
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and DeleteUserData then
  begin
    DelTree(ExpandConstant('{app}\data'), True, True, True);
    DelTree(ExpandConstant('{app}\config'), True, True, True);
    DelTree(ExpandConstant('{app}\logs'), True, True, True);
    DelTree(ExpandConstant('{app}\outputs'), True, True, True);
  end;
end;
