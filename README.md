# VoiceCloner Studio

Studio kloning suara (TTS) lokal & offline berbasis **k2-fsa/OmniVoice** + **faster-whisper**. Dioptimalkan untuk GPU 4 GB (GTX 1650).

## Fitur

- **Profil suara permanen**: unggah rekaman sekali, pakai terus.
- **Potong referensi otomatis**: rekaman sepanjang apa pun diambil maksimal 10 detik bagian dengan ucapan terbanyak, dipotong di jeda (bukan di tengah kata), mono 24 kHz, volume dinormalkan.
- **Transkrip akurat & bisa diedit**: Whisper dipaksa ke bahasa yang dipilih dan menghasilkan tanda baca; transkrip bisa dikoreksi lalu disimpan.
- **Cache suara referensi**: referensi di-encode sekali per profil (`profiles/_cache/*.pt`), generate berikutnya langsung memakainya.
- **Model dimuat saat aplikasi dibuka**, bukan saat klik pertama.
- **Mode Cepat / Seimbang / Kualitas** (16 / 24 / 32 langkah) + pengaturan lanjutan.
- **Angka dibaca sebagai kata** (2025 -> "dua ribu dua puluh lima").
- 100% lokal, tidak ada audio/teks yang keluar dari komputer.

## Instalasi (Windows)

Butuh: [uv](https://docs.astral.sh/uv/), GPU NVIDIA (disarankan), FFmpeg di PATH.

1. Klik dua kali `Setup_VoiceCloner.bat` (membuat `.venv`, memasang torch CUDA).
2. Klik dua kali `Jalankan_VoiceCloner.bat`, lalu buka `http://127.0.0.1:7865`.

## Tips kemiripan suara

1. **Referensi 6-10 detik**, bersih, satu orang, tanpa musik/gema. Lebih panjang tidak membuat lebih mirip (OmniVoice dilatih dengan contoh 3-10 detik) dan memperlambat proses.
2. **Transkrip harus persis** sama dengan ucapan di audio, termasuk tanda baca. Tempo hasil dihitung dari transkrip / durasi referensi.
3. **Gaya bicara ikut isi referensi**: mau gaya bercerita, pakai potongan saat orangnya bercerita. Satu orang boleh punya beberapa profil (`opel_santai`, `opel_serius`).
4. **Jeda diatur lewat tanda baca di naskah**: `,` jeda pendek, `.` jeda kalimat, `...` jeda ragu.
5. Class temp 0 = paling mirip & stabil. Naikkan ke 0.1-0.3 hanya bila hasil terasa datar.

## Tes

```bash
python tests/test_pick_window.py
```

## Lisensi

MIT. OmniVoice: Apache-2.0. Kloning suara hanya dengan izin pemilik suara.
