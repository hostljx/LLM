@echo off
chcp 65001 >nul
title Redis 8.10.3  (127.0.0.1:6379)
cd /d "%~dp0redis-bin\Redis-8.10.3-Windows-x64-msys2"

echo.
echo   Redis 启动中，监听 127.0.0.1:6379
echo.
echo   >> 关闭本窗口 或 按 Ctrl+C 即停止 Redis ^<^<
echo.
echo ------------------------------------------------------------
redis-server.exe redis.conf
echo ------------------------------------------------------------
echo.
echo Redis 已退出。
pause
