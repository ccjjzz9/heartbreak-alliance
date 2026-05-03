@echo off
chcp 65001 >/dev/null
title 呓 v2.0 - Web

echo ========================================
echo              呓 v2.0 (Web)
echo ========================================
echo.
echo 正在启动服务器...
echo 浏览器将自动打开 http://127.0.0.1:8088
echo 按 Ctrl+C 停止服务
echo ========================================

set "USB=%~dp0"
set "PYTHON=%USB%pytorch-env\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

start http://127.0.0.1:8088
"%PYTHON%" "%USB%app\server.py"
pause
