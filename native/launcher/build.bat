@echo off
chcp 65001 >nul
rem ==========================================================================
rem  编译根目录的「启动器.exe」—— 原生 Windows GUI 启动器，双击无黑框
rem
rem  源码 main.go / 图标 icon.ico + icon.rc
rem  需要 Go（go build）；有 windres（msys64/mingw64）才嵌图标，没有也能编译。
rem  产物不入库（编译产物、平台相关），源码入库，随时可重建。
rem ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

where go >nul 2>nul
if errorlevel 1 (
    echo [失败] 没找到 Go 编译器。装好 Go 后重新双击本文件： https://go.dev/dl/
    pause
    exit /b 1
)

set "HAS_ICON="
if exist "icon.ico" if exist "icon.rc" (
    where windres >nul 2>nul
    if not errorlevel 1 (
        windres -i icon.rc -o icon.syso -O coff
        if not errorlevel 1 (
            set "HAS_ICON=1"
        ) else (
            echo [警告] 图标资源编译失败，改出无图标版本。
            del /q icon.syso 2>nul
        )
    ) else (
        echo [提示] 没找到 windres，跳过图标，exe 仍可正常编译。
    )
)

set "CGO_ENABLED=0"
go build -ldflags="-s -w -H windowsgui" -o "..\..\启动器.exe" .
if errorlevel 1 (
    echo [失败] 编译失败，请看上面的报错。
    pause
    exit /b 1
)

echo.
echo   已生成： ..\..\启动器.exe
if defined HAS_ICON (echo   图标：   已嵌入) else (echo   图标：   未嵌入)
echo.
echo   双击根目录的 启动器.exe 即可打开图形界面；
echo   带参数时（如 启动器.exe test）原样转给 启动程序.bat。
echo.
pause
