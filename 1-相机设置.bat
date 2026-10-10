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
echo   第 1 步 · 相机设置：哪台是左、哪台是右、哪台无关
echo ============================================================
echo.
echo   接下来会为【每一个相机编号】依次弹出一个窗口。
echo   看画面，在那个窗口里按一个键：
echo.
echo       l  =  这台是【左】相机（双目的一台）
echo       r  =  这台是【右】相机（双目的一台）
echo       n  =  【无关】—— 这台永远不要用
echo             （笔记本内置相机、红外相机就按这个）
echo       m  =  单目【主】相机
echo       s  =  跳过（不改动配置）
echo       q  =  退出
echo.
echo   左 / 右 / 无关 三者互斥：改标一个，旧的标记会自动摘掉。
echo.
echo   提示：两台外接相机长得一模一样，对着其中一台镜头挥挥手，
echo         看哪个窗口在动，就知道该按哪个键。
echo ============================================================
echo.
pause

"%PY%" apps\tools\cameras.py identify

echo.
echo ============================================================
echo   完成。结果已写入 config\cameras.json
echo   下一步：双击  2-相机检查.bat  确认没有漂移。
echo ============================================================
pause
endlocal