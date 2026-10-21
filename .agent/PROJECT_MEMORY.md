# VoiceCloner Studio — project memory

## Apa
Gradio app (app.py) kloning suara lokal: OmniVoice (k2-fsa, fp16 di GTX 1650 4 GB) + faster-whisper large-v3-turbo (CPU int8) untuk transkrip referensi. Port 7865.

## Environment
- `.venv` (uv, Python 3.11) di folder proyek. Runtime VoiceStudio lama (`AppData\Roaming\VoiceStudio\runtime`) sudah tidak ada.
- Model cache: `%LOCALAPPDATA%\OmniVoice\hf_cache` (OmniVoice snapshot c5fdb5cc, whisper turbo ct2). Path ditulis di app.py, fallback repo id HF.
- torch CUDA: cu126 (driver 555.97 = CUDA 12.5 OK). uv sering gagal "Access is denied" saat rename cache (antivirus?) → unduh wheel pakai curl ke `.wheels/` lalu `uv pip install` file.

## Keputusan
- Referensi maks 10 dtk (OmniVoice dilatih 3–10 dtk; >20 dtk ada warning). Upload panjang → `pick_window` pilih jendela ≤10 dtk dengan ucapan terbanyak, potong di jeda.
- Whisper: `language` dipaksa + initial_prompt bertanda baca (tempo OmniVoice dihitung dari bobot ref_text ÷ durasi).
- Cache `VoiceClonePrompt` per profil di `profiles/_cache/<nama>_<hash>.pt` (hash = mtime+size audio + transkrip).
- Default param = rekomendasi OmniVoice (cfg 2.0, class_temp 0, pos_temp 5, steps 24 "Seimbang").
- UI: tema gelap hangat, aksen oranye #f08a3c, kontras teks ≥6:1 (dicek).

## TODO / belum
- Tes dengar A/B oleh user (lama vs baru) — file bench sudah dihapus; generate ulang bila perlu.
- Hasil ukur (opel, ±16 dtk audio): Seimbang 30 dtk, Cepat 20 dtk, Kualitas 40 dtk; VRAM puncak 2.2 GiB.

## Kinerja GTX 1650
- fp16 GEMM 0.34 TFLOPS vs fp32 1.68 → `_upcast_linear_compute`: bobot fp16 disimpan tertranspose, hitung fp32 (`x.float() @ W^T.float()`). Deteksi otomatis `_fp16_matmul_is_slow()`.
- Kecepatan berfluktuasi ±20% menurut panjang sekuens (heuristik kernel cuBLAS), bukan bug.
- `create_voice_clone_prompt` WAJIB di dalam `torch.inference_mode()` (tanpa itu puncak VRAM 4.3 GiB → spill ke shared RAM).
