@echo off
chcp 65001 >nul
title DingdingRobot

cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set "REDIS_CLI=D:\PythonProjectAll\LLM\redis-bin\Redis-8.10.3-Windows-x64-msys2\redis-cli.exe"
"%REDIS_CLI%" -p 6379 ping >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [错误] Redis 没有启动,机器人无法连接队列.
    echo   请先双击 scripts\start_redis.bat,等窗口稳定后再运行本脚本.
    echo.
    pause
    exit /b 1
)

echo.
echo   启动钉钉机器人(gateway + worker)
echo   停止:Ctrl+C
echo ------------------------------------------------------------
".venv\Scripts\python.exe" -m dingdingrobot.main all
echo ------------------------------------------------------------
echo 机器人已退出.
pause
