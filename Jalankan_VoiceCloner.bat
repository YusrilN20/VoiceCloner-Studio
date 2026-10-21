@echo off
title VoiceCloner Studio
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [!] Environment belum ada. Jalankan Setup_VoiceCloner.bat dulu.
  pause
  exit /b 1
)
echo VoiceCloner Studio - membuka http://127.0.0.1:7865
echo Jangan tutup jendela ini selama aplikasi dipakai.
echo.
set "HF_HOME=%LOCALAPPDATA%\OmniVoice\hf_cache"
set "PYTHONUTF8=1"
".venv\Scripts\python.exe" app.py
pause
