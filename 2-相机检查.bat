@echo off
chcp 65001 >nul
rem ============================================================
rem  正文用 UTF-8 + 开头 chcp 65001：这样中文提示在任何代码页下
rem  都能正确显示（cmd.exe 按当前代码页解析 .bat 正文，不切码页会乱码）。
rem  纯 ASCII 正文是另一条路（见 启动程序.bat），但就没法显示中文提示了。
rem  行尾必须 CRLF。
rem ============================================================
setlocal
cd /d "%~dp0"

set "PY=C:\Users\<USER>\miniforge3\envs\myenv\python.exe"
if not exist "%PY%" set "PY=python"

rem 有固定好的角色就用角色名（换 USB 口也不漂），否则自动挑两台。
set "CAM=--auto-cameras"
if exist "config\cameras.json" set "CAM=--left left --right right"
echo ============================================================
echo   第 2 步 · 相机检查：编号有没有漂移
echo ============================================================
echo.
echo   [1/2] 比对配置与当前设备……
echo.
"%PY%" apps\tools\cameras.py check
set "RC=%ERRORLEVEL%"

echo.
echo   [2/2] 逐个编号试开，报实际分辨率 / 是否全黑……
echo.
"%PY%" apps\tools\cameras.py list --probe

echo.
echo ============================================================
if "%RC%"=="2" (
    echo   【有危险项】请重新双击  1-相机设置.bat  确认一遍。
) else (
    echo   【没有危险项】可以继续做标定了。
)
echo ============================================================
pause
endlocal