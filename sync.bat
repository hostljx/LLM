@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title LLM 仓库同步

set "BASH="

rem 1) 标准 Git for Windows
if not defined BASH if exist "C:\Program Files\Git\bin\bash.exe" set "BASH=C:\Program Files\Git\bin\bash.exe"
rem 2) 用户目录安装
if not defined BASH if exist "%LOCALAPPDATA%\Programs\Git\bin\bash.exe" set "BASH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"
rem 3) 便携版（遍历版本目录取最后一个）
if not defined BASH (
  for /d %%d in ("%USERPROFILE%\.workbuddy\binaries\PortableGit\versions\*") do (
    if exist "%%~fd\bin\bash.exe" set "BASH=%%~fd\bin\bash.exe"
  )
)

if not defined BASH (
  echo.
  echo [错误] 找不到 bash.exe
  echo 请安装 Git for Windows：https://git-scm.com/download/win
  echo.
  pause
  exit /b 1
)

"%BASH%" "%~dp0sync.sh" %*

echo.
pause
