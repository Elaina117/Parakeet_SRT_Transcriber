@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
echo ============================================================
echo Parakeet SRT Transcriber v12r20
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
 echo .venv not found. Run install_windows.bat first.
 pause
 exit /b 1
)
set "PYTHONPATH=%~dp0;%~dp0.venv\Lib\site-packages;%PYTHONPATH%"

rem CUDA diagnostic must pass before the server starts.
.venv\Scripts\python.exe diagnose_cuda.py
if errorlevel 1 (
 echo.
 echo CUDA diagnostic failed. Server will NOT start.
 pause
 exit /b 1
)

echo.
echo Starting server at http://127.0.0.1:5173
echo CUDA is the primary execution provider; CPU is secondary for unsupported graph nodes.
.venv\Scripts\python.exe server.py
set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
exit /b %RC%
