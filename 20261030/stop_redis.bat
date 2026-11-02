@echo off
chcp 65001 >nul
cd /d "%~dp0redis-bin\Redis-8.10.3-Windows-x64-msys2"

redis-cli.exe ping >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Redis 当前没有在运行。
) else (
    redis-cli.exe shutdown save
    echo.
    echo   Redis 已停止，数据已保存到 dump.rdb
)
echo.
pause
