@echo off
chcp 65001 >nul
rem ==========================================================================
rem  球形物体三维运动轨迹识别 —— 【唯一入口】双击本文件即可
rem
rem  它做三件事：
rem    1) 找到一台"确实装了 opencv"的 Python（不是随便一个 python.exe）
rem    2) 缺依赖就问一句，然后自动装（清华镜像，失败退回官方源）
rem    3) 启动图形界面；这台机器没有图形界面就自动换成命令行界面
rem
rem  可选参数（平时不用管，给习惯命令行的自己用）：
rem    cli        强制用命令行界面
rem    setup      直接做相机设置（左 / 右 / 无关）
rem    check      直接做相机检查
rem    precheck   直接做标定前置检查
rem    calibrate  直接做双目标定
rem    verify     直接做标定复检
rem    run        直接开始实时检测
rem    diag       相机画质 / 链路诊断（"某一路不清晰"先跑这个）
rem    test       跑回归测试
rem
rem  正文是 UTF-8，所以开头必须 chcp 65001 —— cmd.exe 按当前代码页解析
rem  .bat 正文，不切码页中文提示会变成乱码。行尾 CRLF。
rem ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title 球形物体三维运动轨迹识别 · 启动器

set "MODE=%~1"
if "%MODE%"=="" set "MODE=gui"

rem 打印出来的棋盘格量到多少就改成多少（毫米）
set "SQUARE=25"

rem 有固定好的角色就用角色名（换 USB 口也不漂），否则自动挑两台
set "CAM=--auto-cameras"
if exist "config\cameras.json" set "CAM=--left left --right right"

echo ============================================================
echo   球形物体三维运动轨迹识别 · 启动器
echo ============================================================
echo.
echo [1/3] 检查 Python 环境 ...

rem ---------- 1) 找一台真的能 import cv2 的 Python ----------
rem 光看路径不算数：msys64 那个 python 在 PATH 里但没装 opencv，
rem 所以每个候选都必须真跑一次 import 才算通过。
set "PY="
call :probe "%USERPROFILE%\miniforge3\envs\myenv\python.exe"
call :probe "%LOCALAPPDATA%\miniforge3\envs\myenv\python.exe"
call :probe "%USERPROFILE%\miniforge3\envs\myenv\python.exe"
call :probe "%ProgramData%\miniforge3\envs\myenv\python.exe"
call :probe "%USERPROFILE%\anaconda3\envs\myenv\python.exe"
call :probe "%LOCALAPPDATA%\anaconda3\envs\myenv\python.exe"
for /f "delims=" %%I in ('where python 2^>nul') do call :probe "%%I"
for /f "delims=" %%I in ('where python3 2^>nul') do call :probe "%%I"

if not defined PY goto :need_install

echo       找到：%PY%
"%PY%" -c "import sys,cv2,numpy;print('      Python '+sys.version.split()[0]+' / opencv '+cv2.__version__+' / numpy '+numpy.__version__)"
echo.
goto :launch

rem ---------- 2) 没找到就装依赖 ----------
:need_install
echo       没有找到装好 opencv / numpy 的 Python。
echo.
echo [2/3] 安装依赖 ...
echo.

set "BASE="
call :probe_base "%USERPROFILE%\miniforge3\envs\myenv\python.exe"
call :probe_base "%LOCALAPPDATA%\miniforge3\envs\myenv\python.exe"
call :probe_base "%USERPROFILE%\miniforge3\envs\myenv\python.exe"
call :probe_base "%ProgramData%\miniforge3\envs\myenv\python.exe"
call :probe_base "%USERPROFILE%\anaconda3\envs\myenv\python.exe"
call :probe_base "%LOCALAPPDATA%\anaconda3\envs\myenv\python.exe"
for /f "delims=" %%I in ('where python 2^>nul') do call :probe_base "%%I"

if not defined BASE (
    echo       连一个能运行的 python 都没找到，没法自动装。
    echo.
    echo       请先装 Python（推荐 Miniforge），装完重新双击本文件：
    echo         https://github.com/conda-forge/miniforge/releases
    echo.
    echo       装好后执行这三条：
    echo         conda create -n myenv python=3.12 -y
    echo         conda activate myenv
    echo         pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo       用这个 Python 装：%BASE%
echo       （不想装就直接关掉这个窗口）
echo.
set "ANS="
set /p "ANS=现在安装依赖吗？[Y/N] "
if /i not "%ANS%"=="Y" (
    echo.
    echo       已跳过。装好之后重新双击本文件即可。
    pause
    exit /b 1
)

echo.
echo       正在安装（清华镜像）...
"%BASE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo.
    echo       镜像源失败，改用官方源再试一次 ...
    "%BASE%" -m pip install -r requirements.txt
)

"%BASE%" -c "import cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo.
    echo       装完仍然 import 不了 cv2，请看上面的报错。
    pause
    exit /b 1
)
set "PY=%BASE%"
echo.
echo       依赖装好了。
echo.

rem ---------- 3) 启动 ----------
:launch
if /i "%MODE%"=="gui"       goto :ui_gui
if /i "%MODE%"=="cli"       goto :ui_cli
if /i "%MODE%"=="setup"     goto :do_setup
if /i "%MODE%"=="check"     goto :do_check
if /i "%MODE%"=="precheck"  goto :do_precheck
if /i "%MODE%"=="calibrate" goto :do_calibrate
if /i "%MODE%"=="verify"    goto :do_verify
if /i "%MODE%"=="run"       goto :do_run
if /i "%MODE%"=="diag"      goto :do_diag
if /i "%MODE%"=="test"      goto :do_test
echo       不认识的参数「%MODE%」，改用图形界面。
goto :ui_gui

:ui_gui
echo [3/3] 启动图形界面 ...
"%PY%" -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo       这台机器没有 tkinter，自动改用命令行界面。
    goto :ui_cli
)
rem 自动定位界面脚本（根目录里 import tkinter 的那个），不写死中文文件名
"%PY%" -c "import glob,runpy; f=[p for p in glob.glob('*.py') if 'tkinter' in open(p,encoding='utf-8').read()][0]; runpy.run_path(f,run_name='__main__')"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo       启动器异常退出（返回码 %RC%）。请看上面的报错。
    pause
)
exit /b %RC%

:ui_cli
echo [3/3] 启动命令行界面 ...
"%PY%" -c "import glob,runpy; f=[p for p in glob.glob('*.py') if 'tkinter' not in open(p,encoding='utf-8').read()][0]; runpy.run_path(f,run_name='__main__')"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo       启动器异常退出（返回码 %RC%）。请看上面的报错。
    pause
)
exit /b %RC%

rem ---------- 直接执行某个步骤（命令行参数用）----------
:do_setup
set "DIRECT=apps\tools\cameras.py identify"
set "DTITLE=相机设置（逐个画面确认 左 / 右 / 无关）"
goto :direct

:do_check
set "DIRECT=apps\tools\cameras.py check"
set "DTITLE=相机检查（配置与当前设备有没有漂移）"
goto :direct

:do_precheck
set "DIRECT=apps\tools\check_stereo_setup.py %CAM% --cols 8 --rows 5 --square %SQUARE% --seconds 8"
set "DTITLE=标定前置检查（两路是否都看到棋盘）"
goto :direct

:do_calibrate
set "DIRECT=apps\calibrate_stereo.py %CAM% --cols 8 --rows 5 --square %SQUARE%"
set "DTITLE=双目标定（每个姿势停稳 1 秒）"
goto :direct

:do_verify
set "DIRECT=apps\verify_stereo.py --calib config/stereo_calibration %CAM% --cols 8 --rows 5 --square %SQUARE%"
set "DTITLE=标定复检（装置有没有被碰动）"
goto :direct

:do_run
set "DIRECT=apps\run_realtime.py --camera auto --method fusion --ball-diameter 0.22"
set "DTITLE=实时检测（窗口里按 q 或 ESC 结束）"
goto :direct

:do_diag
set "DIRECT=apps\tools\camera_quality.py %CAM% --together"
set "DTITLE=相机画质 / 链路诊断（先分清是带宽不足还是光学/曝光问题）"
goto :direct

:do_test
set "DIRECT=tests\selftest_synthetic.py"
set "DTITLE=环境自测（不需要摄像头，约 1 分钟）"
goto :direct

:direct
echo [3/3] %DTITLE%
echo.
"%PY%" %DIRECT%
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo 完成（返回码 0）。
) else (
    echo 结束（返回码 %RC%）。上面有原因。
)
pause
exit /b %RC%

rem ==========================================================================
rem  子过程
rem ==========================================================================
:probe
rem 这个候选存在、而且真的能 import cv2+numpy 吗？通过就定为 %PY%
if defined PY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import cv2, numpy" >nul 2>nul
if errorlevel 1 exit /b 0
set "PY=%~1"
exit /b 0

:probe_base
rem 退一步：只要这个候选能跑 python 就行（拿来 pip install）
if defined BASE exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys" >nul 2>nul
if errorlevel 1 exit /b 0
set "BASE=%~1"
exit /b 0
