@echo off
setlocal

rem BirdCrop Drag & Drop launcher for Windows.
rem Drop one or more image files or folders onto this file or its desktop shortcut.
rem The interpreter is resolved relative to this installed package when possible.

if "%~1"=="" (
    echo BirdCrop Drag ^& Drop
    echo.
    echo Usage: drag one or more image files or folders onto this file.
    echo Example: select a folder in Explorer and drag it onto the BirdCrop shortcut.
    echo.
    echo To process a folder from a command prompt, use:
    echo   birdcrop "C:\path\to\images"
    exit /b 0
)

set "python_exe=%~dp0..\..\..\Scripts\python.exe"
if not exist "%python_exe%" set "python_exe=%~dp0..\..\..\python.exe"

if exist "%python_exe%" (
    "%python_exe%" -m birdcrop --confidence 0.33 -- %*
) else (
    rem Fallback for unusual installations where the interpreter is on PATH.
    python -m birdcrop --confidence 0.33 -- %*
)
set "exit_code=%ERRORLEVEL%"

if not "%exit_code%"=="0" (
    echo.
    echo BirdCrop finished with exit code %exit_code%.
    pause
)

exit /b %exit_code%
