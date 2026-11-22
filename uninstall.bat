@echo off
chcp 65001 >nul
rem ==========================================================================
rem  卸载程序（分发版用）—— 把整个安装目录送进 Windows 回收站
rem
rem  安全设计（2026-11-05，用户明确要求"绝不能在开发仓库里直接运行"）：
rem    1) 只认自己所在的目录，不接受外部传入的任意目标路径；
rem    2) 目录必须同时有 run.bat 与 apps\gui_main.py，才认为是本程序；
rem    3) 出现 .git / .workbuddy / AGENTS.md 任一开发仓库标记，立即拒绝；
rem       —— 分发出去的纯净包里没有这三者，所以不会拦正常卸载；
rem    4) 列出将删内容，要求手输 UNINSTALL 二次确认；
rem    5) 只走回收站（可还原），**绝不永久删除**；回收站失败就停下来；
rem    6) 脚本先复制一份到 %TEMP% 再执行，否则删不掉正在运行的自己。
rem
rem  注意：正式安装目录里如果存在 unins000.exe，会优先转交 Inno 卸载器：
rem  那里可以选「只删程序」或「程序+数据」。只有找不到官方卸载器时，
rem  下面的“整目录送回收站”兜底逻辑才会执行。
rem ==========================================================================
setlocal EnableExtensions

set "TARGET=%~dp0"
set "TARGET=%TARGET:~0,-1%"
>> "%TEMP%\sphere_uninstall_last.log" echo [entry] args=%* cwd=%CD%

rem ---------- 第二遍：由 %TEMP% 里的副本执行 ----------
if /i "%~1"=="--execute" goto :execute

rem ---------- 第一遍：交互确认 ----------
echo ============================================================
echo   卸载 · 球形物体三维运动轨迹识别
echo ============================================================
echo.
if not exist "%TARGET%\run.bat" goto :not_install
if not exist "%TARGET%\apps\gui_main.py" goto :not_install
if exist "%TARGET%\.git" goto :dev_repo
if exist "%TARGET%\.workbuddy" goto :dev_repo
if exist "%TARGET%\AGENTS.md" goto :dev_repo

rem 正式安装包用 Inno 卸载器；转交后它会弹出「只删程序 / 程序+数据」选择。
if exist "%TARGET%\unins000.exe" (
    echo   检测到官方卸载器，转交；可选择只删程序或连用户数据一起删。
    start "" "%TARGET%\unins000.exe"
    exit /b 0
)

echo   将卸载的目录：%TARGET%
echo.
echo   顶层内容：
for /d %%D in ("%TARGET%\*") do echo     [目录] %%~nxD
for %%F in ("%TARGET%\*") do echo     [文件] %%~nxF
echo.
echo   ** 此操作非常危险，可能导致不可逆的数据丢失！**
echo   该目录下全部内容（含 .venv、config、logs 以及 data 里你采集的数据）
echo   都会被送进 Windows 回收站。回收站可还原，但请先确认没有要留的东西。
echo   本脚本不会做永久删除，也不会碰这个目录之外的任何位置。
echo.
set "ANS="
set /p "ANS=确认请输入 UNINSTALL（其他任何输入都取消）： "
if /i not "%ANS%"=="UNINSTALL" (
    echo.
    echo   已取消，没有改动任何文件。
    pause
    exit /b 0
)

rem 为什么必须"另起一个独立进程"（2026-11-05 实测踩到）：cmd 执行 .bat 时
rem 会一直握着脚本文件句柄，而且本窗口的工作目录就是安装目录 —— 两者都会让
rem DeleteDirectory 报「文件被另一个进程占用」。所以：先复制到 TEMP，本窗口
rem 立刻退出释放锁，再由工作目录为 TEMP 的独立窗口执行删除。
set "UTAG=%RANDOM%%RANDOM%"
set "COPY=%TEMP%\sphere_uninstall_%UTAG%.bat"
rem 目标传参文件也带上同一个随机后缀：并发卸载时各自读写各自的文件，不再互相覆盖。
set "TARGETFILE=%TEMP%\sphere_uninstall_target.txt.%UTAG%"
copy /y "%~f0" "%COPY%" >nul
if errorlevel 1 (
    echo   [失败] 无法把卸载脚本复制到临时目录，已取消。
    pause
    exit /b 1
)
echo.
echo   已确认。卸载将在新窗口继续，本窗口关闭以释放安装目录。
rem 目标路径写进临时文件而不是命令行参数：路径里可能有空格，经 start 转一手
rem 引号极易丢失；文件读取没有这个坑。副本只收 --execute 标记 + 纯数字文件名后缀
rem（<%RANDOM%> 串联，并发卸载各读各的），不含任何带空格的路径。
> "%TARGETFILE%" echo %TARGET%
>> "%TEMP%\sphere_uninstall_last.log" echo [first] copy=%COPY% target=%TARGET% targetfile=%TARGETFILE%
start "卸载 · 球形物体三维运动轨迹识别" /MIN /D "%TEMP%" "%COPY%" --execute %UTAG%
exit /b 0

:not_install
echo   [拒绝] 这个目录不像是本程序的安装位置（缺 run.bat 或 apps\gui_main.py）。
echo   目标：%TARGET%
echo.
pause
exit /b 1

:dev_repo
echo   [拒绝] 检测到开发仓库标记（.git / .workbuddy / AGENTS.md）。
echo   卸载器只允许在分发出去的纯净包副本上运行，不能在开发仓库里执行。
echo   目标：%TARGET%
echo.
pause
exit /b 1

rem ---------- 执行删除：只由上面的临时副本调用 ----------
:execute
set "TARGETFILE=%TEMP%\sphere_uninstall_target.txt.%2"
set "TARGET="
if exist "%TARGETFILE%" set /p TARGET=< "%TARGETFILE%"
if "%TARGET%"=="" (
    echo   [拒绝] 读不到卸载目标（临时传参文件缺失），已取消，未删任何东西。
    >> "%TEMP%\sphere_uninstall_last.log" echo [second] refuse: no target file=%TARGETFILE%
    exit /b 1
)
>> "%TEMP%\sphere_uninstall_last.log" echo [second] entered target=%TARGET% cwd=%CD%
rem ---- 第二遍复核：目标来自 %TEMP% 文件，可能被篡改；删前必须重新验明正身 ----
if not exist "%TARGET%\run.bat" goto :execute_refuse
if not exist "%TARGET%\apps\gui_main.py" goto :execute_refuse
if exist "%TARGET%\.git" goto :execute_refuse
if exist "%TARGET%\.workbuddy" goto :execute_refuse
if exist "%TARGET%\AGENTS.md" goto :execute_refuse
rem 盘符根拒删：C: / C:\ 这类根送进回收站等于清空整个盘。
if /i "%TARGET%"=="%SystemDrive%" goto :execute_root
if /i "%TARGET%"=="%SystemDrive%\" goto :execute_root
if "%TARGET:~1,1%"==":" if "%TARGET:~2,1%"=="" goto :execute_root
if "%TARGET:~1,1%"==":" if "%TARGET:~2,1%"=="\" if "%TARGET:~3,1%"=="" goto :execute_root
rem 等上一个窗口彻底退出（它的文件句柄释放有一小段延迟），再开始删。
timeout /t 1 >nul
set "SPHERE_UNINSTALL_TARGET=%TARGET%"
echo   正在送入回收站：%TARGET%
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ok=$false; $msg=''; for($i=0; $i -lt 6 -and -not $ok; $i++){ try { Add-Type -AssemblyName Microsoft.VisualBasic; [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($env:SPHERE_UNINSTALL_TARGET, [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs, [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin); $ok=$true } catch { $msg=$_.Exception.Message; Start-Sleep -Milliseconds 700 } }; if(-not $ok){ Write-Host ('FAIL: ' + $msg); exit 1 }; Write-Host 'RECYCLED'" >> "%TEMP%\sphere_uninstall_last.log" 2>&1
set "PSRC=%ERRORLEVEL%"
>> "%TEMP%\sphere_uninstall_last.log" echo [second] ps_rc=%PSRC%
if not "%PSRC%"=="0" (
    echo   [失败] 回收站操作没成功，目录仍在原处。没有做任何永久删除。
    timeout /t 8 >nul
    exit /b 1
)
if exist "%TARGET%" (
    echo   [失败] 目录仍然存在，未判定为成功。
    timeout /t 8 >nul
    exit /b 1
)
echo   卸载完成。如需还原，请到 Windows 回收站找回该目录。
timeout /t 8 >nul
rem 收掉 TEMP 里的临时副本与目标文件（删不掉也不影响结果，交给系统清理 TEMP）。
del /q "%~f0" >nul 2>nul
del /q "%TARGETFILE%" >nul 2>nul
exit /b 0

:execute_refuse
echo   [拒绝] 第二遍复核不通过：目标缺少 run.bat / apps\gui_main.py，
echo          或含 .git / .workbuddy / AGENTS.md 开发仓库标记，已取消删除。
>> "%TEMP%\sphere_uninstall_last.log" echo [second] refuse: identity check failed target=%TARGET%
del /q "%TARGETFILE%" >nul 2>nul
timeout /t 8 >nul
exit /b 1

:execute_root
echo   [拒绝] 第二遍复核不通过：目标是盘符根目录（%TARGET%），已取消删除。
>> "%TEMP%\sphere_uninstall_last.log" echo [second] refuse: drive root target=%TARGET%
del /q "%TARGETFILE%" >nul 2>nul
timeout /t 8 >nul
exit /b 1
