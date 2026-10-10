@echo off
rem ============================================================
rem  Spherical object 3D trajectory recognition - CLI launcher
rem  Double-click this file to start the colored console launcher.
rem  (Kept ASCII-only on purpose: a Chinese .bat body would be
rem   mis-decoded by cmd.exe on non-UTF8 code pages.)
rem ============================================================
setlocal
cd /d "%~dp0"

set "PY=C:\Users\<USER>\miniforge3\envs\myenv\python.exe"
if not exist "%PY%" set "PY=python"

rem Auto-locate the CLI script: the root .py file that does NOT import tkinter.
"%PY%" -c "import glob,runpy; f=[p for p in glob.glob('*.py') if 'tkinter' not in open(p,encoding='utf-8').read()][0]; runpy.run_path(f,run_name='__main__')"

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start the CLI launcher.
    echo Please check the Python environment: %PY%
    pause
)
endlocal
