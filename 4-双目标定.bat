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
rem >>> 打印后用尺子量方格边长，量到多少就把这里改成多少 <<<
set "SQUARE=25"

echo ============================================================
echo   第 4 步 · 双目标定
echo ============================================================
echo.
echo   棋盘格：8x5 内角点，方格 %SQUARE% mm
echo   相机  ：%CAM%
echo.
echo   把板举到两路都能看到的位置，慢慢移动，每个姿势【停稳 1 秒】。
echo   程序只收停稳的帧，晃着不动它是不收的。
echo.
echo   看结果：极线误差小于 0.3 px 就算好。
echo.
echo   注意：做这一步之前，两台相机必须已经【刚性固定】在同一根横梁上。
echo         装置一动，解出来的 R/T 立刻作废，只能重标。
echo ============================================================
echo.
pause

"%PY%" apps\calibrate_stereo.py %CAM% --cols 8 --rows 5 --square %SQUARE%
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
if "%RC%"=="0" (
    echo   【成功】结果写在 config\stereo_calibration.npz / .json
    echo   下一步：双击  5-标定复检.bat ，把复检基线值记下来。
) else (
    echo   【没成功】标定没有完成，上面有原因。常见三种：
    echo     - 收不到帧        -^> 两路视野不重叠，重新对一下取景
    echo     - 老说棋盘在移动   -^> 没停稳，每个姿势停 1 秒
    echo     - 相机打不开      -^> 双击  2-相机检查.bat  看看设备
)
echo ============================================================
pause
endlocal