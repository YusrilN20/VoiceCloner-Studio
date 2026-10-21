import os
import json
import time
import hashlib
import threading

import numpy as np
import soundfile as sf
import gradio as gr
import torch

# Paths - Dynamic & Portable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
PROMPT_CACHE_DIR = os.path.join(PROFILES_DIR, "_cache")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PROFILES_JSON = os.path.join(BASE_DIR, "profiles.json")
SETTINGS_JSON = os.path.join(BASE_DIR, "settings.json")

# Local cache paths or Hugging Face repo IDs (auto-download on new machines)
DEFAULT_LOCAL_WHISPER = r"C:\Users\ASUS TUF\AppData\Local\OmniVoice\hf_cache\models--deepdml--faster-whisper-large-v3-turbo-ct2\snapshots\4df90f75321148c3a29a9e2351b7ddf8f5b115a8"
DEFAULT_LOCAL_OMNIVOICE = r"C:\Users\ASUS TUF\AppData\Local\OmniVoice\hf_cache\models--k2-fsa--OmniVoice\snapshots\c5fdb5ccb189668d56333f77ba2629f4cd7535f4"

WHISPER_PATH = DEFAULT_LOCAL_WHISPER if os.path.exists(DEFAULT_LOCAL_WHISPER) else "deepdml/faster-whisper-large-v3-turbo-ct2"
OMNIVOICE_PATH = DEFAULT_LOCAL_OMNIVOICE if os.path.exists(DEFAULT_LOCAL_OMNIVOICE) else "k2-fsa/OmniVoice"

# OmniVoice is trained on 3-10 s prompts; longer refs cost compute every step and can hurt similarity.
REF_MAX_SEC = 10.0
REF_MIN_SEC = 3.0
SAMPLE_RATE = 24000

os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

PRESETS = {"Cepat": 16, "Seimbang": 24, "Kualitas": 32}

DEFAULT_SETTINGS = {
    "preset": "Seimbang",
    "steps": 24,
    "cfg": 2.0,
    "speed": 1.0,
    "t_shift": 0.10,
    "pos_temp": 5.0,
    "class_temp": 0.0,
    "layer_penalty": 5.0,
    "language": "id",
    "denoise": True,
    "postprocess": True,
    "normalize_numbers": True,
    "selected_profile": None,
}

LANGUAGES = [("Indonesia", "id"), ("English", "en"), ("Jawa", "jv"), ("Sunda", "su"),
             ("Melayu", "ms"), ("日本語", "ja"), ("中文", "zh"), ("Otomatis", "Auto")]


# ---------------------------------------------------------------- storage

def _read_json(path, fallback):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return fallback


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: a crash mid-write can't wipe profiles


def load_settings():
    res = DEFAULT_SETTINGS.copy()
    res.update(_read_json(SETTINGS_JSON, {}))
    return res


def save_settings(**kwargs):
    curr = load_settings()
    curr.update(kwargs)
    try:
        _write_json(SETTINGS_JSON, curr)
    except Exception as e:
        print("Gagal menyimpan settings:", e)


def get_profiles():
    """Profiles whose audio file exists (broken entries are hidden, not deleted)."""
    profiles = _read_json(PROFILES_JSON, {})
    return {k: v for k, v in profiles.items() if os.path.exists(v.get("audio_path", ""))}


def save_profiles(profiles):
    _write_json(PROFILES_JSON, profiles)


def audio_seconds(path):
    try:
        return sf.info(path).duration
    except Exception:
        return 0.0


def safe_name(name):
    keep = "".join(c if c.isalnum() or c in "-_ " else "" for c in (name or "")).strip()
    return keep.replace(" ", "_")


def _drop_prompt_cache(profile_name):
    prefix = safe_name(profile_name) + "_"
    for old in os.listdir(PROMPT_CACHE_DIR):
        if old.startswith(prefix):
            try:
                os.remove(os.path.join(PROMPT_CACHE_DIR, old))
            except OSError:
                pass


# ---------------------------------------------------------------- models

_omnivoice_model = None
_whisper_model = None
_model_lock = threading.Lock()
_prompt_cache = {}


def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import faster_whisper
        # CPU keeps the 4 GB GPU free for OmniVoice; transcription only runs when saving a profile.
        _whisper_model = faster_whisper.WhisperModel(WHISPER_PATH, device="cpu", compute_type="int8")
    return _whisper_model


def _fp16_matmul_is_slow():
    """GTX 16xx (Turing without tensor cores) runs fp16 GEMM ~5x slower than fp32."""
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = a.float()

    def t(x):
        x @ x
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(4):
            x @ x
        torch.cuda.synchronize()
        return time.time() - s

    slow = t(a) > 1.3 * t(b)
    del a, b
    return slow


def _upcast_linear_compute(module):
    """Run each fp16 Linear as fp32 `x @ W^T` (GPUs with weak fp16 math, e.g. GTX 16xx).

    W^T is stored once, contiguous and still fp16, so VRAM stays the same; cuBLAS picks
    faster kernels for that layout (~20-45% per GEMM on a GTX 1650 vs F.linear).
    """
    n = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.float16:
            wt = m.weight.detach().t().contiguous()
            bias = m.bias.detach().float() if m.bias is not None else None
            del m.weight
            m.register_buffer("weight_t", wt, persistent=False)

            def fwd(x, _m=m, _b=bias):
                y = x.float() @ _m.weight_t.float()
                if _b is not None:
                    y = y + _b
                return y.to(x.dtype)

            m.forward = fwd
            n += 1
    return n


def load_omnivoice():
    global _omnivoice_model
    with _model_lock:
        if _omnivoice_model is None:
            from omnivoice import OmniVoice
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = OmniVoice.from_pretrained(
                OMNIVOICE_PATH,
                device_map=device,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                load_asr=False,
            )
            if device == "cuda" and _fp16_matmul_is_slow():
                n = _upcast_linear_compute(model)
                print(f"[VoiceCloner] GPU fp16 lambat: {n} layer dijalankan dalam fp32 (~3.7x lebih cepat).")
            _omnivoice_model = model
    return _omnivoice_model


def warmup():
    try:
        load_omnivoice()
        print("[VoiceCloner] Model siap.")
    except Exception as e:
        print("[VoiceCloner] Gagal memuat model:", e)


def device_label():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU (lambat)"


# ---------------------------------------------------------------- reference audio

def pick_window(spans, total_ms, max_ms, min_ms):
    """Choose (start, end) in ms: the window cut at pauses holding the most speech, <= max_ms."""
    if spans[-1][1] - spans[0][0] <= max_ms:
        return spans[0][0], spans[-1][1]
    best = None
    for i in range(len(spans)):
        speech = 0
        for j in range(i, len(spans)):
            if spans[j][1] - spans[i][0] > max_ms:
                break
            speech += spans[j][1] - spans[j][0]
            if spans[j][1] - spans[i][0] >= min_ms and (best is None or speech > best[0]):
                best = (speech, spans[i][0], spans[j][1])
    if best is None:  # one long phrase without pauses: hard cut
        return spans[0][0], min(total_ms, spans[0][0] + max_ms)
    return best[1], best[2]


def prepare_reference(src_path, dest_path):
    """Mono 24 kHz, edge silence trimmed, cut at a pause to <= REF_MAX_SEC, peak-normalised.

    Returns (seconds, note).
    """
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    seg = AudioSegment.from_file(src_path).set_channels(1).set_frame_rate(SAMPLE_RATE)
    if seg.dBFS == float("-inf"):
        raise ValueError("Audio senyap.")
    thresh = max(seg.dBFS - 16, -50)
    spans = detect_nonsilent(seg, min_silence_len=250, silence_thresh=thresh, seek_step=10)
    if not spans:
        raise ValueError("Audio tidak berisi suara (terlalu senyap).")

    start, end = pick_window(spans, len(seg), int(REF_MAX_SEC * 1000), int(REF_MIN_SEC * 1000))
    clip = seg[max(0, start - 120): min(len(seg), end + 200)]
    note = ""
    if len(seg) - len(clip) > 1500:
        note = f"Diambil {len(clip) / 1000:.1f} dtk terbaik dari {len(seg) / 1000:.1f} dtk (dipotong di jeda)."
    clip = clip.apply_gain(-1.0 - clip.max_dBFS)
    clip.export(dest_path, format="wav")
    return len(clip) / 1000, note


def normalize_id_numbers(text):
    """Indonesian number formats OmniVoice's per-integer num2words pass mangles.

    "Rp25.000" -> "dua puluh lima ribu rupiah", "1.500.000" -> one integer, "3,5" -> "tiga koma lima",
    "07.30" -> "tujuh lewat tiga puluh", "50%" -> "lima puluh persen". Plain integers are left to OmniVoice.
    """
    import re
    try:
        from num2words import num2words
    except ImportError:
        return text

    def w(n):
        return num2words(int(n), lang="id")

    def money(m):
        return f"{w(m.group(1).replace('.', ''))} rupiah"

    text = re.sub(r"\bRp\.?\s?(\d{1,3}(?:\.\d{3})+|\d+)(?:,-|,00)?", money, text, flags=re.I)
    text = re.sub(r"\b(\d{1,2})[.:](\d{2})\b",
                  lambda m: f"{w(m.group(1))} lewat {w(m.group(2))}"
                  if int(m.group(1)) < 24 and int(m.group(2)) < 60 else m.group(0), text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", lambda m: w(m.group(0).replace(".", "")), text)
    text = re.sub(r"\b(\d+),(\d+)\b", lambda m: f"{w(m.group(1))} koma {' '.join(w(d) for d in m.group(2))}", text)
    text = re.sub(r"(\d+)\s?%", lambda m: f"{w(m.group(1))} persen", text)
    return text


def transcribe_audio(audio_path, language="id"):
    if not audio_path or not os.path.exists(audio_path):
        return ""
    asr = load_whisper()
    lang = None if language in (None, "", "Auto") else language
    # A punctuated prompt nudges Whisper to output punctuation, which drives OmniVoice's pacing.
    prompt = "Halo, apa kabar? Saya baik-baik saja, terima kasih." if lang == "id" else None
    segments, _ = asr.transcribe(
        audio_path,
        language=lang,
        beam_size=5,
        condition_on_previous_text=False,
        initial_prompt=prompt,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def _prompt_key(profile_name, audio_path, transcript):
    stat = os.stat(audio_path)
    raw = f"{profile_name}|{stat.st_mtime_ns}|{stat.st_size}|{transcript}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def get_voice_prompt(model, profile_name, prof):
    """Encode the reference once per (audio, transcript) and reuse it on every generate."""
    from omnivoice.models.omnivoice import VoiceClonePrompt

    key = _prompt_key(profile_name, prof["audio_path"], prof["transcript"])
    if key in _prompt_cache:
        return _prompt_cache[key], True
    cache_file = os.path.join(PROMPT_CACHE_DIR, f"{safe_name(profile_name)}_{key}.pt")
    if os.path.exists(cache_file):
        try:
            prompt = VoiceClonePrompt.load(cache_file)
            _prompt_cache[key] = prompt
            return prompt, True
        except Exception:
            pass
    with torch.inference_mode():  # without this autograd buffers push the 4 GB card into shared RAM
        prompt = model.create_voice_clone_prompt(
            ref_audio=prof["audio_path"],
            ref_text=prof["transcript"] or None,
            preprocess_prompt=True,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # audio tokenizer encode peaks high; free it before generation
    _drop_prompt_cache(profile_name)
    try:
        prompt.save(cache_file)
    except Exception as e:
        print("Gagal menyimpan cache prompt:", e)
    _prompt_cache[key] = prompt
    return prompt, False


# ---------------------------------------------------------------- UI handlers

def profile_info_md(name):
    profiles = get_profiles()
    if not name or name not in profiles:
        return "<span class='vc-meta'>Belum ada profil. Tambahkan rekaman di panel <b>Profil baru</b>.</span>"
    p = profiles[name]
    sec = audio_seconds(p["audio_path"])
    warn = ""
    if sec < REF_MIN_SEC:
        warn = " · <span class='vc-warn'>terlalu pendek, idealnya 6–10 dtk</span>"
    elif sec > REF_MAX_SEC + 1:
        warn = " · <span class='vc-warn'>lebih dari 10 dtk, lebih lambat; buat ulang profil agar dipotong otomatis</span>"
    elif not p.get("transcript"):
        warn = " · <span class='vc-warn'>transkrip kosong</span>"
    return f"<span class='vc-meta'>{sec:.1f} dtk · dibuat {p.get('created_at', '-')}</span>{warn}"


def on_select_profile(profile_name):
    profiles = get_profiles()
    if profile_name and profile_name in profiles:
        save_settings(selected_profile=profile_name)
        p = profiles[profile_name]
        return p["audio_path"], p["transcript"], profile_info_md(profile_name)
    return None, "", profile_info_md(None)


def save_new_profile(profile_name, audio_file, manual_transcript, language):
    clean_name = safe_name(profile_name)
    if not clean_name:
        return gr.update(), "<span class='vc-err'>Isi nama profil (huruf/angka).</span>"
    if not audio_file:
        return gr.update(), "<span class='vc-err'>Unggah atau rekam audio dulu.</span>"

    dest_audio = os.path.join(PROFILES_DIR, f"{clean_name}.wav")
    try:
        seconds, note = prepare_reference(audio_file, dest_audio)
    except Exception as e:
        return gr.update(), f"<span class='vc-err'>Audio gagal diproses: {e}</span>"
    if seconds < REF_MIN_SEC:
        note += " Rekaman di bawah 3 dtk; kemiripan akan rendah."

    transcript = (manual_transcript or "").strip()
    if not transcript:
        try:
            transcript = transcribe_audio(dest_audio, language)
        except Exception as e:
            return gr.update(), f"<span class='vc-err'>Transkripsi gagal: {e}</span>"

    profiles = _read_json(PROFILES_JSON, {})
    profiles[clean_name] = {
        "audio_path": dest_audio,
        "transcript": transcript,
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    save_profiles(profiles)
    save_settings(selected_profile=clean_name)
    _drop_prompt_cache(clean_name)

    msg = (f"Profil <b>{clean_name}</b> tersimpan ({seconds:.1f} dtk). {note} "
           "Dengarkan referensinya dan cek transkrip; perbaiki kata yang salah lalu klik <b>Simpan transkrip</b>.")
    return gr.update(choices=list(get_profiles().keys()), value=clean_name), msg


def save_transcript(profile_name, transcript):
    profiles = _read_json(PROFILES_JSON, {})
    if not profile_name or profile_name not in profiles:
        return "<span class='vc-err'>Pilih profil dulu.</span>"
    profiles[profile_name]["transcript"] = (transcript or "").strip()
    save_profiles(profiles)
    return "Transkrip disimpan. Generate berikutnya memakai teks ini."


def retranscribe(profile_name, language):
    profiles = get_profiles()
    if not profile_name or profile_name not in profiles:
        return gr.update(), "<span class='vc-err'>Pilih profil dulu.</span>"
    text = transcribe_audio(profiles[profile_name]["audio_path"], language)
    return text, "Hasil Whisper ditampilkan, <b>belum disimpan</b>. Periksa, lalu klik <b>Simpan transkrip</b>."


def delete_profile(profile_name):
    profiles = _read_json(PROFILES_JSON, {})
    if profile_name not in profiles:
        return gr.update(), "Profil tidak ditemukan."
    del profiles[profile_name]
    save_profiles(profiles)
    _drop_prompt_cache(profile_name)
    choices = list(get_profiles().keys())
    new_val = choices[0] if choices else None
    save_settings(selected_profile=new_val)
    return gr.update(choices=choices, value=new_val), f"Profil <b>{profile_name}</b> dihapus (file audio tetap di folder profiles)."


def generate_voice(profile_name, text_to_speak, language, steps, cfg, speed, t_shift,
                   pos_temp, class_temp, layer_penalty, denoise, postprocess, normalize_numbers,
                   preset, progress=gr.Progress()):
    if not text_to_speak or not text_to_speak.strip():
        return None, "<span class='vc-err'>Tulis naskah dulu.</span>"
    profiles = get_profiles()
    if not profile_name or profile_name not in profiles:
        return None, "<span class='vc-err'>Pilih profil suara dulu.</span>"

    save_settings(
        selected_profile=profile_name, language=language, preset=preset, steps=int(steps),
        cfg=float(cfg), speed=float(speed), t_shift=float(t_shift), pos_temp=float(pos_temp),
        class_temp=float(class_temp), layer_penalty=float(layer_penalty), denoise=bool(denoise),
        postprocess=bool(postprocess), normalize_numbers=bool(normalize_numbers),
    )

    progress(0.05, desc="Memuat model")
    model = load_omnivoice()
    prof = profiles[profile_name]

    with _model_lock:
        progress(0.15, desc="Menyiapkan suara referensi")
        voice_prompt, cached = get_voice_prompt(model, profile_name, prof)

        gen_kw = {
            "num_step": int(steps),
            "guidance_scale": float(cfg),
            "speed": float(speed),
            "t_shift": float(t_shift),
            "position_temperature": float(pos_temp),
            "class_temperature": float(class_temp),
            "layer_penalty_factor": float(layer_penalty),
            "denoise": bool(denoise),
            "postprocess_output": bool(postprocess),
        }
        if language and language != "Auto":
            gen_kw["language"] = language

        script = text_to_speak.strip()
        if normalize_numbers and language == "id":
            script = normalize_id_numbers(script)

        progress(0.3, desc="Membuat suara")
        start = time.time()
        try:
            audios = model.generate(
                text=script,
                voice_clone_prompt=voice_prompt,
                normalize_text=bool(normalize_numbers),
                **gen_kw,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None, ("<span class='vc-err'>VRAM penuh.</span> Tutup aplikasi lain yang memakai GPU, "
                          "atau pecah naskah jadi beberapa bagian.")
        elapsed = time.time() - start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    audio = audios[0] if isinstance(audios, (list, tuple)) else audios
    arr = audio.detach().cpu().float().numpy() if torch.is_tensor(audio) else np.asarray(audio, dtype=np.float32)
    arr = np.clip(arr.squeeze(), -1.0, 1.0)
    sr = int(getattr(model, "sampling_rate", SAMPLE_RATE))

    out_path = os.path.join(OUTPUTS_DIR, f"{safe_name(profile_name)}_{time.strftime('%Y%m%d_%H%M%S')}.wav")
    sf.write(out_path, arr, sr)
    dur = len(arr) / sr
    rtf = elapsed / dur if dur else 0
    ref_note = "referensi dari cache" if cached else "referensi diproses lalu di-cache"
    return out_path, (f"<span class='vc-ok'>Selesai</span> · {dur:.1f} dtk audio dalam {elapsed:.1f} dtk "
                      f"({rtf:.2f}× durasi) · {ref_note} · <code>{os.path.basename(out_path)}</code>")


# ---------------------------------------------------------------- UI

saved_cfg = load_settings()
all_profile_names = list(get_profiles().keys())
initial_profile = saved_cfg.get("selected_profile")
if initial_profile not in all_profile_names:
    initial_profile = all_profile_names[0] if all_profile_names else None
init_audio, init_transcript, init_info = on_select_profile(initial_profile)

ACCENT = "#f08a3c"
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fff7ed", c100="#ffedd5", c200="#fed7aa", c300="#fdba74", c400="#fb923c",
        c500=ACCENT, c600="#d9732a", c700="#b45a1f", c800="#8c461b", c900="#713b19", c950="#3d1d0a",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f6f5f3", c100="#e9e7e3", c200="#d3d0ca", c300="#b3afa7", c400="#8f8a81",
        c500="#6f6a62", c600="#57534c", c700="#3f3c37", c800="#2a2825", c900="#1c1b19", c950="#131211",
    ),
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "Segoe UI", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
)
_theme_vars = dict(
    body_background_fill="#131211", body_text_color="#ecebe8", body_text_color_subdued="#a29d94",
    background_fill_primary="#1c1b19", background_fill_secondary="#232120",
    block_background_fill="#1f1d1b", block_border_color="#35322e", border_color_primary="#3a3733",
    block_label_text_color="#b3aea5", block_title_text_color="#ecebe8", block_info_text_color="#b3aea5",
    block_label_background_fill="#1f1d1b", block_label_border_color="#35322e",
    input_background_fill="#151413", input_border_color="#3a3733", input_border_color_focus=ACCENT,
    button_primary_background_fill=ACCENT, button_primary_background_fill_hover="#f59e58",
    button_primary_text_color="#1a0f06", button_primary_border_color=ACCENT,
    button_secondary_background_fill="#26241f", button_secondary_background_fill_hover="#312e29",
    button_secondary_text_color="#ecebe8", button_secondary_border_color="#34312d",
    slider_color=ACCENT, checkbox_background_color_selected=ACCENT, checkbox_border_color_selected=ACCENT,
    color_accent=ACCENT, color_accent_soft="#3a2616", loader_color=ACCENT,
    block_shadow="none", shadow_drop="none",
)
# Same palette in light and dark so the app never flips to a washed-out light theme.
_no_dark = {"color_accent", "shadow_drop"}  # gradio has no *_dark variant for these
theme = theme.set(**_theme_vars, **{k + "_dark": v for k, v in _theme_vars.items() if k not in _no_dark})

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; padding: 28px 24px 48px !important;
  box-sizing: border-box !important; width: 100% !important; }
footer { display: none !important; }
.vc-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px;
  padding-bottom: 20px; margin-bottom: 4px; border-bottom: 1px solid #2e2c29; }
.vc-head h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; line-height: 1.1; color: #ecebe8; }
.vc-head p { margin: 6px 0 0; color: #a29d94; font-size: 0.95rem; max-width: 60ch; }
.vc-chip { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.78rem; color: #c9c5be;
  border: 1px solid #34312d; border-radius: 6px; padding: 6px 10px; white-space: nowrap; }
.vc-chip b { color: #f08a3c; font-weight: 600; }
.vc-section { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #a29d94; }
.vc-meta { color: #a29d94; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
.vc-warn { color: #f2b766; font-size: 0.85rem; }
.vc-err { color: #f2836b; }
.vc-ok { color: #8fd18b; font-weight: 600; }
.vc-hint, .vc-hint p { color: #a29d94; font-size: 0.86rem; line-height: 1.55; margin: 0; }
.vc-hint b, .vc-status b { color: #ecebe8; }
.vc-status { min-height: 1.5em; font-variant-numeric: tabular-nums; color: #c9c5be; }
.vc-status code, .vc-hint code { font-size: 0.8rem; }
#vc-generate { min-height: 52px; font-size: 1.05rem; font-weight: 700; letter-spacing: 0.01em; }
button.secondary { font-weight: 500 !important; }
#vc-mode label { white-space: nowrap; }
input[type=range]::-webkit-slider-runnable-track {
  background: linear-gradient(to right, #f08a3c var(--range_progress), #3a3733 var(--range_progress)) !important; }
input[type=range]::-moz-range-track { background: #3a3733 !important; }
.vc-status code { white-space: nowrap; }
#vc-delete { background: transparent !important; border: 1px solid #7a3b2e !important; color: #f2836b !important; }
#vc-delete:hover { background: #3a1d17 !important; }
.vc-ok, .vc-err, .vc-warn { font-weight: 600; }
button { transition: background-color .18s ease, border-color .18s ease, transform .08s ease; cursor: pointer; }
button:active { transform: translateY(1px); }
button:focus-visible, textarea:focus-visible, input:focus-visible { outline: 2px solid #f08a3c !important; outline-offset: 2px; }
#vc-script textarea { font-size: 1.02rem; line-height: 1.6; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (max-width: 760px) {
  .gradio-container { padding: 16px 12px 32px !important; }
  .vc-head { flex-direction: column; align-items: flex-start; } }
"""

with gr.Blocks(title="VoiceCloner Studio") as demo:
    gr.HTML(f"""
    <header class="vc-head">
      <div>
        <h1>VoiceCloner Studio</h1>
        <p>Kloning suara lokal dengan OmniVoice. Semua audio diproses di komputer ini.</p>
      </div>
      <span class="vc-chip">Perangkat: <b>{device_label()}</b></span>
    </header>
    """)

    with gr.Row(equal_height=False):
        # LEFT: voice profile
        with gr.Column(scale=5, min_width=320):
            gr.HTML("<div class='vc-section'>Profil suara</div>")
            profile_dropdown = gr.Dropdown(label="Suara", choices=all_profile_names,
                                           value=initial_profile, interactive=True)
            profile_info = gr.HTML(init_info)
            loaded_audio_preview = gr.Audio(label="Referensi", value=init_audio, interactive=False)
            transcript_edit = gr.Textbox(
                label="Transkrip referensi",
                info="Harus sama persis dengan ucapan di audio. Transkrip salah = kemiripan & tempo turun.",
                value=init_transcript, lines=3, interactive=True,
            )
            with gr.Row():
                btn_save_transcript = gr.Button("Simpan transkrip", variant="secondary")
                btn_retranscribe = gr.Button("Transkrip ulang", variant="secondary")
            transcript_msg = gr.HTML(elem_classes="vc-hint")

            with gr.Accordion("Profil baru", open=not all_profile_names):
                gr.HTML("<p class='vc-hint'>Unggah rekaman berapa pun panjangnya. Aplikasi mengambil "
                        "<b>maksimal 10 detik</b> bagian dengan ucapan terbanyak, dipotong di jeda. "
                        "Pakai rekaman bersih tanpa musik, satu orang, dengan gaya bicara yang ingin ditiru.</p>")
                new_profile_name = gr.Textbox(label="Nama profil", placeholder="contoh: opel_santai")
                new_audio_input = gr.Audio(label="Rekaman suara", type="filepath",
                                           sources=["upload", "microphone"])
                new_transcript_input = gr.Textbox(label="Transkrip (opsional)",
                                                  placeholder="Kosongkan agar ditranskripsi otomatis")
                btn_save_profile = gr.Button("Simpan profil", variant="primary")
                profile_status_msg = gr.HTML(elem_classes="vc-hint")

            with gr.Accordion("Hapus profil", open=False):
                gr.HTML("<p class='vc-hint'>Menghapus profil dari daftar. File audionya tetap disimpan.</p>")
                btn_delete_profile = gr.Button("Hapus profil terpilih", variant="stop", elem_id="vc-delete")
                delete_status_msg = gr.HTML(elem_classes="vc-hint")

        # RIGHT: script + generate
        with gr.Column(scale=7, min_width=360):
            gr.HTML("<div class='vc-section'>Naskah</div>")
            text_input = gr.Textbox(
                label="Teks yang diucapkan", elem_id="vc-script", lines=8,
                placeholder="Tulis naskah di sini. Koma (,) memberi jeda pendek, titik (.) jeda kalimat, "
                            "elipsis (...) jeda ragu.",
            )
            with gr.Row():
                lang_dropdown = gr.Dropdown(label="Bahasa", choices=LANGUAGES,
                                            value=saved_cfg.get("language", "id"), scale=2, min_width=160)
                preset_radio = gr.Radio(label="Mode (jumlah langkah)",
                                        choices=[(f"{k} · {v}", k) for k, v in PRESETS.items()],
                                        value=saved_cfg.get("preset", "Seimbang"), scale=5, min_width=320,
                                        elem_id="vc-mode")
            slider_speed = gr.Slider(label="Kecepatan bicara (1 = sama dengan referensi)", minimum=0.7,
                                     maximum=1.3, step=0.05, value=saved_cfg.get("speed", 1.0))

            btn_generate = gr.Button("Buat suara", variant="primary", elem_id="vc-generate")
            gen_status = gr.HTML("<span class='vc-meta'>Hasil tersimpan otomatis di folder <code>outputs</code>.</span>",
                                 elem_classes="vc-status")
            audio_output = gr.Audio(label="Hasil", interactive=False)

            with gr.Accordion("Pengaturan lanjutan", open=False):
                gr.HTML("<p class='vc-hint'>Nilai bawaan mengikuti rekomendasi OmniVoice. "
                        "Class temp 0 = paling stabil & mirip referensi; naikkan sedikit (0.1–0.3) bila hasil datar.</p>")
                with gr.Row():
                    slider_steps = gr.Slider(label="Steps", minimum=8, maximum=48, step=1,
                                             value=saved_cfg.get("steps", 24))
                    slider_cfg = gr.Slider(label="CFG", minimum=1.0, maximum=4.0,
                                           step=0.1, value=saved_cfg.get("cfg", 2.0))
                    slider_t_shift = gr.Slider(label="t_shift", minimum=0.0, maximum=1.0, step=0.05,
                                               value=saved_cfg.get("t_shift", 0.10))
                with gr.Row():
                    slider_pos_temp = gr.Slider(label="Pos. temp", minimum=0.0, maximum=10.0, step=0.5,
                                                value=saved_cfg.get("pos_temp", 5.0))
                    slider_class_temp = gr.Slider(label="Class temp", minimum=0.0, maximum=2.0, step=0.05,
                                                  value=saved_cfg.get("class_temp", 0.0))
                    slider_layer_penalty = gr.Slider(label="Layer pen.", minimum=0.0, maximum=10.0, step=0.5,
                                                     value=saved_cfg.get("layer_penalty", 5.0))
                with gr.Row():
                    toggle_denoise = gr.Checkbox(label="Denoise", value=saved_cfg.get("denoise", True))
                    toggle_postprocess = gr.Checkbox(label="Buang jeda panjang",
                                                     value=saved_cfg.get("postprocess", True))
                    toggle_numbers = gr.Checkbox(label="Angka jadi kata",
                                                 value=saved_cfg.get("normalize_numbers", True))

    # ---- events
    profile_outputs = [loaded_audio_preview, transcript_edit, profile_info]
    profile_dropdown.change(on_select_profile, [profile_dropdown], profile_outputs)
    btn_save_transcript.click(save_transcript, [profile_dropdown, transcript_edit], [transcript_msg])
    btn_retranscribe.click(retranscribe, [profile_dropdown, lang_dropdown], [transcript_edit, transcript_msg])
    btn_save_profile.click(
        save_new_profile, [new_profile_name, new_audio_input, new_transcript_input, lang_dropdown],
        [profile_dropdown, profile_status_msg],
    ).then(on_select_profile, [profile_dropdown], profile_outputs)
    btn_delete_profile.click(delete_profile, [profile_dropdown], [profile_dropdown, delete_status_msg]
                             ).then(on_select_profile, [profile_dropdown], profile_outputs)
    preset_radio.change(lambda p: PRESETS.get(p, 24), [preset_radio], [slider_steps])
    btn_generate.click(
        generate_voice,
        [profile_dropdown, text_input, lang_dropdown, slider_steps, slider_cfg, slider_speed, slider_t_shift,
         slider_pos_temp, slider_class_temp, slider_layer_penalty, toggle_denoise, toggle_postprocess,
         toggle_numbers, preset_radio],
        [audio_output, gen_status],
        concurrency_limit=1,
    )

if __name__ == "__main__":
    threading.Thread(target=warmup, daemon=True).start()  # model loads while the UI opens
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7865, inbrowser=True, theme=theme, css=CSS,
    )
