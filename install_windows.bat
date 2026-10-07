@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "LOG=%~dp0install_log.txt"
>"%LOG%" echo ============================================================
>>"%LOG%" echo Parakeet SRT Transcriber v12r18 installer
>>"%LOG%" echo Started: %DATE% %TIME%
>>"%LOG%" echo ============================================================

echo Parakeet SRT Transcriber v12r18
 echo CUDA 11.8 + cuDNN 8 exact-match install + DLL diagnostic
 echo.
set "PY="
for %%V in (3.12 3.11 3.10 3.13) do (
  if not defined PY (
    for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)" 2^>nul') do (
      set "PY=%%P"
      set "PYVER=%%V"
    )
  )
)
if not defined PY (
 echo Supported Python not found.
 >>"%LOG%" echo Supported Python not found.
 pause
 exit /b 1
)
echo Selected Python %PYVER%: %PY%
>>"%LOG%" echo Selected Python %PYVER%: %PY%

rem Remove conflicting CUDA 12 / cuDNN 9 NVIDIA Python runtime packages from earlier revisions.
.venv\Scripts\python.exe -m pip uninstall -y nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12 >>"%LOG%" 2>&1

if not exist ".venv\Scripts\python.exe" (
 "%PY%" -m venv .venv >>"%LOG%" 2>&1
 if errorlevel 1 goto FAIL
)

rem Remove stale startup customization from all earlier revisions.
if exist ".venv\Lib\site-packages\sitecustomize.py" del /q ".venv\Lib\site-packages\sitecustomize.py" >>"%LOG%" 2>&1
if exist ".venv\Lib\site-packages\sitecustomize.pyc" del /q ".venv\Lib\site-packages\sitecustomize.pyc" >>"%LOG%" 2>&1

.venv\Scripts\python.exe -m pip install --upgrade pip >>"%LOG%" 2>&1
if errorlevel 1 goto FAIL
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt >>"%LOG%" 2>&1
if errorlevel 1 goto FAIL

rem Install Sherpa-ONNX with CUDA 11.8 support for optional DPDFNet speech denoising.
set "SHERPA_VERSION=1.13.8+cuda"
for /f "delims=" %%V in ('.venv\Scripts\python.exe -c "import sys; print(sys.version_info.minor)"') do set "VENV_PY_MINOR=%%V"
if "%VENV_PY_MINOR%"=="13" set "SHERPA_VERSION=1.13.7+cuda"
.venv\Scripts\python.exe -m pip install --no-deps --verbose "sherpa-onnx==%SHERPA_VERSION%" --no-index -f https://k2-fsa.github.io/sherpa/onnx/cuda.html >>"%LOG%" 2>&1
if errorlevel 1 goto FAIL

set "PYTHONPATH=%~dp0;%PYTHONPATH%"
.venv\Scripts\python.exe diagnose_cuda.py >>"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto FAIL

echo.
echo ============================================================
echo INSTALL SUCCESS - CUDA DIAGNOSTIC PASSED
echo ============================================================
echo Log: %LOG%
pause
exit /b 0

:FAIL
if not defined RC set "RC=%ERRORLEVEL%"
>>"%LOG%" echo FAILED with exit code %RC% at %DATE% %TIME%
echo.
echo ============================================================
echo INSTALL FAILED - exit code %RC%
echo ============================================================
echo See install_log.txt
pause
exit /b %RC%
