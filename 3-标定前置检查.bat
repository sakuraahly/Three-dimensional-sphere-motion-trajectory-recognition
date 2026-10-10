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
echo   第 3 步 · 标定前置检查：两路都要看得到棋盘，而且视野重叠
echo ============================================================
echo.
echo   把打印好的棋盘格举到【两路都能看到】的位置。
echo   板必须平整（贴在硬纸板 / 亚克力上），否则标定白做。
echo.
echo   本次相机选择：%CAM%
echo ============================================================
echo.

"%PY%" apps\tools\check_stereo_setup.py %CAM% --cols 8 --rows 5 --square 25 --seconds 8
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
if "%RC%"=="0" (
    echo   【通过】两路都看到棋盘了，可以去做  4-双目标定.bat
) else (
    echo   【没通过】先解决取景 / 重叠 / 光照问题。
    echo   如果某一路始终看不到棋盘，说明两台相机没对着同一片区域。
)
echo ============================================================
pause
endlocal