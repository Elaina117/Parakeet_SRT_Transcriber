@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "LOG=%~dp0install_moss_log.txt"
set "TEMP=%~dp0tmp\moss-install-temp"
set "TMP=%TEMP%"
set "PIP_CACHE_DIR=%~dp0tmp\moss-pip-cache"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
>"%LOG%" echo MOSS-Transcribe-Diarize optional runtime installer
>>"%LOG%" echo Started: %DATE% %TIME%

if not exist ".moss-venv\Scripts\python.exe" (
  py -3.12 -m venv .moss-venv >>"%LOG%" 2>&1
  if errorlevel 1 goto FAIL
)

echo Installing PyTorch CUDA 12.6 runtime for Pascal GPUs...
.moss-venv\Scripts\python.exe -m pip install --upgrade pip >>"%LOG%" 2>&1
if errorlevel 1 goto FAIL
.moss-venv\Scripts\python.exe -m pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu126 >>"%LOG%" 2>&1
if errorlevel 1 goto FAIL

echo Installing the official OpenMOSS inference package...
.moss-venv\Scripts\python.exe -m pip install "git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git" >>"%LOG%" 2>&1
if errorlevel 1 goto FAIL

.moss-venv\Scripts\python.exe -c "import torch, transformers, moss_transcribe_diarize; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); print('transformers', transformers.__version__)" >>"%LOG%" 2>&1
if errorlevel 1 goto FAIL
>".moss-venv\.install-complete" echo installed %DATE% %TIME%

echo.
echo MOSS runtime installed. The model weights will download the first time MOSS is selected.
echo Log: %LOG%
pause
exit /b 0

:FAIL
set "RC=%ERRORLEVEL%"
>>"%LOG%" echo FAILED with exit code %RC% at %DATE% %TIME%
echo.
echo MOSS runtime installation failed with exit code %RC%.
echo See install_moss_log.txt
pause
exit /b %RC%
