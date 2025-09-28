@echo off
REM Face Recognition System - Windows Deployment Script

setlocal enabledelayedexpansion

REM Check if PowerShell is available
powershell -Command "Get-Host" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PowerShell is not available
    echo Please install PowerShell to run this deployment script
    pause
    exit /b 1
)

REM Run the PowerShell deployment script
echo Starting Face Recognition System Deployment...
echo.

if "%1"=="" (
    powershell -ExecutionPolicy Bypass -File "scripts\deploy.ps1" deploy
) else (
    powershell -ExecutionPolicy Bypass -File "scripts\deploy.ps1" %*
)

if %errorlevel% neq 0 (
    echo.
    echo Deployment encountered errors. Please check the output above.
    pause
    exit /b %errorlevel%
)

echo.
echo Deployment completed! 
echo.
echo You can now access:
echo   Frontend: http://localhost:3000
echo   Backend API: http://localhost:8000/api
echo   API Documentation: http://localhost:8000/api/docs
echo.
echo Default login: admin / admin123
echo.
pause