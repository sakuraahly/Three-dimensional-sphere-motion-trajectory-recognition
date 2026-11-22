@echo off
chcp 65001 >nul
rem ==========================================================================
rem  环境安装器（分发版用）—— 转发给主入口的 install 参数
rem
rem  参数路由的唯一产地仍是 run.bat，这里不复制第二份安装逻辑。
rem  双击本文件 = 建隔离环境 / 复用现成环境、装齐依赖、写环境指针。
rem  高级参数会原样透传，例如：install.bat --isolate  或  install.bat --ai
rem ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

call "%~dp0run.bat" install %*
set "RC=%ERRORLEVEL%"
exit /b %RC%
