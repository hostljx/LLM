@echo off
chcp 65001 >nul
title Redis 8.10.3 (127.0.0.1:6379)

set "REDIS_HOME=D:\PythonProjectAll\LLM\redis-bin\Redis-8.10.3-Windows-x64-msys2"
set "DATA_DIR=%~dp0..\data\redis"

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

echo.
echo   Redis 启动中,监听 127.0.0.1:6379
echo   数据目录: %DATA_DIR%
echo   关闭本窗口或按 Ctrl+C 停止
echo ------------------------------------------------------------
"%REDIS_HOME%\redis-server.exe" --port 6379 --bind 127.0.0.1 --protected-mode yes --appendonly yes --appendfsync everysec --dir "%DATA_DIR%"
echo ------------------------------------------------------------
echo Redis 已退出.
pause
