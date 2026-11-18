@echo off
chcp 65001 >nul

set "REDIS_HOME=D:\PythonProjectAll\LLM\redis-bin\Redis-8.10.3-Windows-x64-msys2"

"%REDIS_HOME%\redis-cli.exe" -p 6379 ping >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Redis 当前没有在运行.
) else (
    "%REDIS_HOME%\redis-cli.exe" -p 6379 shutdown save
    echo.
    echo   Redis 已停止,数据已保存.
)
echo.
pause
