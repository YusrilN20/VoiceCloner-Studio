@echo off
title VoiceCloner Studio - Setup
cd /d "%~dp0"
where uv >nul 2>nul || (echo [!] uv belum terpasang: https://docs.astral.sh/uv/ & pause & exit /b 1)
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail

rem torch CUDA (~2.6 GB). curl -C - can resume; uv's cache rename gets blocked by antivirus here.
if not exist .wheels mkdir .wheels
set "T=torch-2.14.1+cu126-cp311-cp311-win_amd64.whl"
set "A=torchaudio-2.11.0+cu126-cp311-cp311-win_amd64.whl"
curl -L -C - --retry 10 -o ".wheels\%T%" "https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%%2Bcu126-cp311-cp311-win_amd64.whl" || goto :fail
curl -L -C - --retry 10 -o ".wheels\%A%" "https://download-r2.pytorch.org/whl/cu126/torchaudio-2.11.0%%2Bcu126-cp311-cp311-win_amd64.whl" || goto :fail
uv pip install --python .venv\Scripts\python.exe --no-deps --reinstall ".wheels\%T%" ".wheels\%A%" || goto :fail
".venv\Scripts\python.exe" -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo Selesai. Jalankan Jalankan_VoiceCloner.bat
pause
exit /b 0
:fail
echo [!] Setup gagal. Lihat pesan di atas.
pause
exit /b 1
