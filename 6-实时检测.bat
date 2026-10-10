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
echo   第 6 步 · 实时检测（单目）
echo ============================================================
echo.
echo   相机：auto（自动挑最优：优先外接、跳过标了「无关」的、跳过全黑流）
echo   方法：fusion（颜色 + 梯度 + 运动 投票融合）
echo.
echo   在视频窗口里按  q  或  ESC  结束。
echo.
echo   如果三维距离看着不对，先标定焦距：
echo       "%PY%" apps\tools\calibrate_focal.py --distance 1.0
echo ============================================================
echo.

"%PY%" apps\run_realtime.py --camera auto --method fusion --ball-diameter 0.22

echo.
echo ============================================================
echo   已停止。
echo ============================================================
pause
endlocal