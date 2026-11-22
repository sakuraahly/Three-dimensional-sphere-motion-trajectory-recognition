@echo off
chcp 65001 >nul
rem ==========================================================================
rem  一键编译 Windows 安装程序（setup.exe）
rem
rem  前置：装好 Inno Setup 6（https://jrsoftware.org/isinfo.php）
rem        winget 用户可先跑： winget install --id JRsoftware.InnoSetup -e
rem  用法：双击本文件，或命令行 build_installer.bat
rem  产物：outputs\releases\SphereMotion-0.2.0-setup.exe
rem
rem  设计要点（与项目其它入口保持一致）：
rem   - 源码/入口不是这里复制的，来自 apps/tools/make_clean_bundle.py 的纯净包，
rem     再放进刚编译的 launcher.exe —— 单一真相，不另写一份文件清单。
rem   - 安装到用户目录（{localappdata}），不要管理员权限、不写 Program Files。
rem ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0.."

where ISCC.exe >nul 2>nul
if not errorlevel 1 (
    set "ISCC=ISCC.exe"
    goto :found_iscc
)
rem winget 默认装到用户目录；老式安装可能在 Program Files。三个都试。
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" (
    set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
) else (
    echo [失败] 没找到 Inno Setup 6 的编译器 ISCC.exe。
    echo        装好后重新双击本文件；或用 winget:
    echo        winget install --id JRSoftware.InnoSetup -e
    pause
    exit /b 1
)
:found_iscc

rem 找 Python：优先外部传入的 PY 环境变量，其次 PATH 里的 python
if not defined PY set "PY=python"
echo [1/3] 生成纯净源码包（用 %PY%）...
"%PY%" apps\tools\make_clean_bundle.py --out outputs\release_payload_src.zip
if errorlevel 1 (
    echo [失败] 生成源码包失败。若 python 不在 PATH，请先设 PY，例如：
    echo        set PY=C:\path\to\python.exe
    goto :fail
)

echo [2/3] 展开到 staging 并放入 launcher.exe...
if not exist "launcher.exe" (
    echo [失败] 根目录没有 launcher.exe，请先编译：
    echo        cd native\launcher ^&^& build.bat
    goto :fail
)
if exist outputs\release_stage rmdir /s /q outputs\release_stage
mkdir outputs\release_stage
pushd outputs\release_stage
tar -xf ..\release_payload_src.zip
copy /y ..\..\launcher.exe launcher.exe >nul
popd

echo [3/3] 编译 setup.exe...
rem .iss 里的相对路径已带 ..\ 回退到仓库根（ISCC 以脚本所在目录为基准）
"%ISCC%" installer\sphere_motion.iss
if errorlevel 1 goto :fail

echo.
echo   完成： outputs\releases\SphereMotion-0.2.0-setup.exe
echo   只生成本地文件，未发布到 GitHub。
pause
exit /b 0

:fail
echo.
echo [失败] 安装程序未生成，请看上面的报错。
pause
exit /b 1
