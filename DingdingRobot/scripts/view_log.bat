@echo off
chcp 65001 >nul
title DingdingRobot 日志(Ctrl+C 退出)

set "LOG_FILE=%~dp0..\data\bot.log"

if not exist "%LOG_FILE%" (
    echo.
    echo   日志文件还不存在:%LOG_FILE%
    echo   请先启动机器人(scripts\start_bot.bat).
    echo.
    pause
    exit /b 1
)

echo.
echo   实时查看:%LOG_FILE%
echo   按 Ctrl+C 退出
echo ------------------------------------------------------------
powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG_FILE%' -Encoding UTF8 -Wait -Tail 80"
echo ------------------------------------------------------------
pause
