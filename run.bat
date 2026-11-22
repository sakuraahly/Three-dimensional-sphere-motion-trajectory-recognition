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
rem    doctor     相机体检（真开一遍：隐私开关/后端/能不能出图；只报告）
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

rem 环境安装器单独路由：此刻可能连 Python 都还没配好，不能走下面的探测。
rem 根目录不新增第四个脚本（AGENTS.md 规则 5）——"只做某一步"就给 bat 加参数。
if /i "%MODE%"=="install" goto :do_install

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

rem ---------- 0) 安装器登记过的环境优先 ----------
rem config\python_env.txt 是纯文本 key=value（bat / Go / Python 三方都能读），
rem 项目内写相对路径。命中后仍要真跑一次 import 才算数：.venv 被删了、
rem 或整个仓库搬到别的机器，指针可能只剩一个空壳。
if exist "config\python_env.txt" (
    for /f "usebackq tokens=1,* delims==" %%A in ("config\python_env.txt") do (
        if "%%A"=="python" set "PY=%%B"
    )
)
if defined PY set "PY=%PY:/=\%"
if defined PY if exist "%PY%" (
    "%PY%" -c "import cv2,numpy" >nul 2>nul
    if not errorlevel 1 (
        echo       找到：%PY%（安装器登记的环境）
        echo.
        goto :launch
    )
)
set "PY="

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
echo       推荐改用一键安装器：run.bat install（建隔离环境，不污染全局 Python）
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
if /i "%MODE%"=="doctor"    goto :do_doctor
if /i "%MODE%"=="test"      goto :do_test
if /i "%MODE%"=="standards" goto :do_standards
if /i "%MODE%"=="gate"      goto :do_gate
if /i "%MODE%"=="legacy"    goto :ui_legacy
echo       不认识的参数「%MODE%」，改用图形界面。
goto :ui_gui

:ui_gui
echo [3/3] 启动图形界面 ...
"%PY%" -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo       这台机器没有 tkinter，自动改用命令行界面。
    goto :ui_cli
)
rem 界面脚本固定在 apps/ 下（英文名）。以前靠 glob 扫根目录里"含 tkinter 的 .py"
rem 来猜哪个是 GUI —— 两个启动脚本搬进 apps/ 之后根目录没有 .py 了，那条路必然失效。
"%PY%" apps\gui_main.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo       启动器异常退出（返回码 %RC%）。请看上面的报错。
    pause
)
exit /b %RC%

:ui_legacy
rem 经典单窗口界面已于 2026-11-20 废止（决策见 docs/决策记录.md）。
rem 保留这个参数只为给老用户一句明确解释，**不再启动任何程序**。
echo.
echo [已废止] 经典界面不再提供。
echo       新界面是唯一入口：直接运行 run.bat（它打开门厅 + 三个模式窗）。
exit /b 0

:ui_cli
echo [3/3] 启动命令行界面 ...
"%PY%" apps\cli_main.py
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

:do_doctor
set "DIRECT=apps\tools\cameras.py doctor"
set "DTITLE=相机体检（隐私开关 / 后端 / 逐台能不能出图；只报告）"
goto :direct

:do_test
rem 回归测试统一走 apps\run_tests.py（项目里**唯一**跑全套测试的地方）。
rem   默认 fast：跳过"已知慢的"测试，只跑其余（名单来自 timings.json 的实测中位数，不写死）。
rem   当前跳过了哪些、各耗时多少：跑一次 apps\run_tests.py --list 看，别在这里写死数字。
rem   要全套：run.bat test full
rem 分层依据是 outputs\test_results\timings.json 里的**实测中位数**，不写死名单。
set "TIER=%~2"
if "%TIER%"=="" set "TIER=fast"
if /i "%TIER%"=="full" (
    set "DIRECT=apps\run_tests.py full"
    set "DTITLE=回归测试 · full（全部测试都跑，最慢的一档）"
) else (
    set "DIRECT=apps\run_tests.py fast"
    set "DTITLE=回归测试 · fast（跳过已知慢的）"
)
goto :direct

:do_standards
rem 规范落差自查：文档引用 / 规范来源 / 派生事实计数 / 待办清单 是否与磁盘一致。
set "DIRECT=apps\tools\check_standards.py"
set "DTITLE=规范落差自查（引用 / 来源 / 计数 / 待办）"
goto :direct

:do_gate
rem 提交前的自动门（`.githooks/pre-push` 调它）：规范自查 + 回归 fast（并行）。
rem
rem ⚠️ **这里绝不能 pause**：它是被 git 钩子**非交互**调用的，
rem    等输入就等于把 push 卡死，而且看起来像"卡住了"而不是"在等回车"。
rem 形状与 `:direct` 不同（连跑两条、不要标题横幅），所以单独写而不是 goto :direct。
echo [gate 1/2] 规范落差自查（引用 / 来源 / 计数 / 待办）...
"%PY%" apps\tools\check_standards.py --quiet
if errorlevel 1 goto :gate_fail
echo.
echo [gate 2/2] 回归测试 fast（-j auto 并行；名单来自实测中位数，不写死）...
"%PY%" apps\run_tests.py fast -j auto
if errorlevel 1 goto :gate_fail
echo.
echo [gate] 通过。
exit /b 0

:gate_fail
echo.
echo [gate] 不通过 —— 已拦下。修完再推；确有急事用 `git push --no-verify`，
echo        并在提交信息里说明为什么跳过（别默认它就是"能用"）。
exit /b 1

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

rem ---------- 环境安装器（run.bat install / launcher.exe install）----------
:do_install
echo ============================================================
echo   一键环境安装器
echo ============================================================
echo.
echo   建隔离环境 / 复用现成环境，装齐依赖，并登记给所有启动入口。
echo   详细日志： logs\misc\
echo.
set "BOOT="
rem 引导解释器：已登记的 > PATH 里的 python > py launcher。
rem 它只要能跑标准库即可；真正装依赖的是它建出来的那个 .venv。
if exist "config\python_env.txt" (
    for /f "usebackq tokens=1,* delims==" %%A in ("config\python_env.txt") do (
        if "%%A"=="python" set "BOOT=%%B"
    )
)
if defined BOOT set "BOOT=%BOOT:/=\%"
if defined BOOT if not exist "%BOOT%" set "BOOT="
if not defined BOOT for /f "delims=" %%I in ('where python 2^>nul') do if not defined BOOT set "BOOT=%%I"
rem py launcher 可能指向很老的 Python（连本文件都解析不了），候选先验版本再收。
if not defined BOOT for /f "delims=" %%I in ('where py 2^>nul') do if not defined BOOT (
    "%%I" -c "import sys; assert sys.version_info>=(3,9)" >nul 2>nul
    if not errorlevel 1 set "BOOT=%%I"
)
if not defined BOOT (
    echo   [失败] 这台机器连一个能运行的 Python 都没有。
    echo          先装 Python 3.10+ ： https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo   引导解释器：%BOOT%
echo.
rem install 之后的参数要**原样**转给 setup_env.py：install.bat 用 call 转来时引号还在，
rem 旧写法会把引号剥掉（带空格路径被拆成多个参数），而且只取前 4 个会静默丢参数。
rem 这里逐个 shift + 重新包引号：带空格路径不拆、参数数量不限、开关型参数也能过。
set "INSTALL_ARGS="
:collect_arg
shift
if "%~1"=="" goto :collect_done
set "INSTALL_ARGS=%INSTALL_ARGS% "%~1""
goto :collect_arg
:collect_done
"%BOOT%" apps\setup_env.py %INSTALL_ARGS%
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo   环境已就绪。双击 run.bat，或 install.bat 所在目录的启动入口即可。
) else (
    echo   安装未完成（返回码 %RC%）。请看上面的报错与 logs\misc\
)
echo.
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
