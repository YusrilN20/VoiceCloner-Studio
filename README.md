# 🎙️ VoiceCloner Studio

A lightweight, dedicated, and offline Voice Cloning Text-to-Speech (TTS) studio powered by **k2-fsa/OmniVoice** and **faster-whisper**.

Designed specifically for content creators who need fast, local voice cloning without the bloat of video dubbing or heavy background processes.

---

## ✨ Features

- **⚡ Fast & Optimized:** Tuned to run smoothly even on 4GB VRAM GPUs (e.g. NVIDIA GTX 1650).
- **👤 Persistent Voice Profiles:** Save your voice sample once; use it forever without re-uploading.
- **🎙️ Automatic Transcription (ASR):** Uses local Whisper AI to automatically transcribe reference voice clips.
- **⚙️ Full Production Controls:** Fine-tune `Steps`, `CFG Guidance`, `Speed`, `t_shift`, `Pos/Class Temperature`, `Layer Penalty`, `Denoise`, and `Postprocess`.
- **💾 Auto-Save Settings:** Remembers your favorite parameters across app restarts.
- **🔒 Privacy-First & Offline:** 100% runs locally on your PC. No audio or text ever leaves your machine.

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support (recommended)
- FFmpeg installed in system PATH

### 2. Installation
```bash
git clone https://github.com/YusrilN20/VoiceCloner-Studio.git
cd VoiceCloner-Studio

pip install -r requirements.txt
```

### 3. Run the Studio
```bash
python app.py
```
Or simply double-click `Jalankan_VoiceCloner.bat` on Windows.

The studio will launch in your browser at `http://127.0.0.1:7865`.

---

## 💡 Best Practices for Voice Cloning

1. **Reference Audio:** Use 5–10 seconds of clear speech (no background music/noise).
2. **Speed Setting:** Set `0.85x – 0.95x` for calm monologue / storytelling tone.
3. **Punctuation:** Use commas (`,`) and ellipses (`...`) in your script to naturally control pauses and breathing.

---

## 📜 License
MIT License. Feel free to use and customize for your own content creation!
