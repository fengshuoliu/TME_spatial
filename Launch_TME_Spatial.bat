@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "POWERSHELL_SCRIPT=%SCRIPT_DIR%launch_tme_spatial.ps1"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if not exist "%POWERSHELL_SCRIPT%" (
    echo Could not find "%POWERSHELL_SCRIPT%".
    echo Press any key to close.
    pause >nul
    exit /b 1
)

if not exist "%POWERSHELL_EXE%" (
    echo Could not find Windows PowerShell at "%POWERSHELL_EXE%".
    echo Press any key to close.
    pause >nul
    exit /b 1
)

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%POWERSHELL_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Launcher failed with exit code %EXIT_CODE%.
    echo Press any key to close.
    pause >nul
)

exit /b %EXIT_CODE%
