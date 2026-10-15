@echo off
title VoiceCloner Studio (Dedicated OmniVoice Engine)
color 0b
echo ========================================================
echo         VoiceCloner Studio - Voice Cloning Dedicated
echo ========================================================
echo.
echo Membuka antarmuka di browser: http://127.0.0.1:7865 ...
echo Jangan tutup jendela terminal ini selama aplikasi berjalan.
echo.

cd /d "D:\VoiceCloner"

set "PYTHON_EXE=C:\Users\ASUS TUF\AppData\Roaming\VoiceStudio\runtime\project\.venv\Scripts\python.exe"

"%PYTHON_EXE%" app.py
pause
