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
if not exist "config\stereo_calibration.npz" (
    echo ============================================================
    echo   找不到标定结果 config\stereo_calibration.npz
    echo   请先双击  4-双目标定.bat
    echo ============================================================
    pause
    endlocal
    exit /b 1
)

echo ============================================================
echo   第 5 步 · 标定复检：装置有没有被碰动
echo ============================================================
echo.
echo   把棋盘格举进画面并保持不动。
echo   相机：%CAM%
echo.
echo   参考值：装置没动约 0.08 px；碰动 2 度会到 27.9 px（差 250 倍）。
echo   每次正式拍摄之前都跑一遍这一步。
echo ============================================================
echo.
pause

"%PY%" apps\verify_stereo.py --calib config/stereo_calibration %CAM% --cols 8 --rows 5 --square 25
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
if "%RC%"=="0" (
    echo   【通过】装置没动。把这次的数字记下来当作基线。
) else (
    echo   【没通过】装置动过了，请重新跑  4-双目标定.bat
)
echo ============================================================
pause
endlocal