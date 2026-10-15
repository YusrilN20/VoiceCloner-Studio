import os
import sys
import json
import shutil
import time
import numpy as np
import soundfile as sf
import gradio as gr
import torch

# Paths - Dynamic & Portable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PROFILES_JSON = os.path.join(BASE_DIR, "profiles.json")
SETTINGS_JSON = os.path.join(BASE_DIR, "settings.json")

# Local cache paths or Hugging Face repo IDs (auto-download on new machines)
DEFAULT_LOCAL_WHISPER = r"C:\Users\ASUS TUF\AppData\Local\OmniVoice\hf_cache\models--deepdml--faster-whisper-large-v3-turbo-ct2\snapshots\4df90f75321148c3a29a9e2351b7ddf8f5b115a8"
DEFAULT_LOCAL_OMNIVOICE = r"C:\Users\ASUS TUF\AppData\Local\OmniVoice\hf_cache\models--k2-fsa--OmniVoice\snapshots\c5fdb5ccb189668d56333f77ba2629f4cd7535f4"

WHISPER_PATH = DEFAULT_LOCAL_WHISPER if os.path.exists(DEFAULT_LOCAL_WHISPER) else "deepdml/faster-whisper-large-v3-turbo-ct2"
OMNIVOICE_PATH = DEFAULT_LOCAL_OMNIVOICE if os.path.exists(DEFAULT_LOCAL_OMNIVOICE) else "k2-fsa/OmniVoice"

os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Default settings
DEFAULT_SETTINGS = {
    "steps": 22,
    "cfg": 1.9,
    "speed": 0.9,
    "t_shift": 0.10,
    "pos_temp": 5.5,
    "class_temp": 0.2,
    "layer_penalty": 5.0,
    "language": "id",
    "denoise": True,
    "postprocess": True,
    "selected_profile": "nurhukam"
}

def load_settings():
    if os.path.exists(SETTINGS_JSON):
        try:
            with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)
                res = DEFAULT_SETTINGS.copy()
                res.update(saved)
                return res
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings(**kwargs):
    curr = load_settings()
    curr.update(kwargs)
    try:
        with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(curr, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Gagal menyimpan settings:", e)

# Global model placeholders
_omnivoice_model = None
_whisper_model = None

def get_profiles():
    if os.path.exists(PROFILES_JSON):
        try:
            with open(PROFILES_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_profiles(profiles):
    with open(PROFILES_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

def get_profile_names():
    profiles = get_profiles()
    return list(profiles.keys())

def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import faster_whisper
        _whisper_model = faster_whisper.WhisperModel(WHISPER_PATH, device="cpu", compute_type="int8")
    return _whisper_model

def load_omnivoice():
    global _omnivoice_model
    if _omnivoice_model is None:
        from omnivoice import OmniVoice
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _omnivoice_model = OmniVoice.from_pretrained(
            OMNIVOICE_PATH,
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            load_asr=False
        )
    return _omnivoice_model

def transcribe_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return ""
    asr = load_whisper()
    segments, info = asr.transcribe(audio_path)
    text = " ".join([s.text for s in segments]).strip()
    return text

def save_new_profile(profile_name, audio_file, manual_transcript):
    if not profile_name or not profile_name.strip():
        return gr.update(), "⚠️ Error: Masukkan nama profil!", ""
    if not audio_file:
        return gr.update(), "⚠️ Error: Masukkan file sampel audio!", ""
    
    clean_name = profile_name.strip()
    dest_audio = os.path.join(PROFILES_DIR, f"{clean_name}.wav")
    
    # Process/cut audio to max 8s
    try:
        data, sr = sf.read(audio_file)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        max_samples = int(sr * 8)
        if len(data) > max_samples:
            data = data[:max_samples]
        sf.write(dest_audio, data, sr)
    except Exception:
        shutil.copy2(audio_file, dest_audio)
    
    final_transcript = manual_transcript.strip() if manual_transcript and manual_transcript.strip() else ""
    if not final_transcript:
        final_transcript = transcribe_audio(dest_audio)
    
    profiles = get_profiles()
    profiles[clean_name] = {
        "audio_path": dest_audio,
        "transcript": final_transcript,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_profiles(profiles)
    save_settings(selected_profile=clean_name)
    
    choices = list(profiles.keys())
    msg = f"✅ Profil '{clean_name}' berhasil disimpan! Transkrip: \"{final_transcript}\""
    return gr.update(choices=choices, value=clean_name), msg, final_transcript

def delete_profile(profile_name):
    profiles = get_profiles()
    if profile_name in profiles:
        del profiles[profile_name]
        save_profiles(profiles)
        choices = list(profiles.keys())
        new_val = choices[0] if choices else None
        save_settings(selected_profile=new_val)
        return gr.update(choices=choices, value=new_val), f"🗑️ Profil '{profile_name}' berhasil dihapus."
    return gr.update(), "Profil tidak ditemukan."

def on_select_profile(profile_name):
    profiles = get_profiles()
    if profile_name and profile_name in profiles:
        save_settings(selected_profile=profile_name)
        p = profiles[profile_name]
        return p["audio_path"], p["transcript"]
    return None, ""

# Murni Voice Cloning Generik Asli (Paling Otentik)
def generate_voice(
    profile_name,
    text_to_speak,
    language,
    steps,
    cfg,
    speed,
    t_shift,
    pos_temp,
    class_temp,
    layer_penalty,
    denoise,
    postprocess
):
    if not text_to_speak or not text_to_speak.strip():
        return None, "⚠️ Error: Ketikkan teks naskah yang ingin diucapkan!"
    
    profiles = get_profiles()
    if not profile_name or profile_name not in profiles:
        return None, "⚠️ Error: Pilih profil suara terlebih dahulu!"
    
    # Save the current settings persistently
    save_settings(
        selected_profile=profile_name,
        language=language,
        steps=int(steps),
        cfg=float(cfg),
        speed=float(speed),
        t_shift=float(t_shift),
        pos_temp=float(pos_temp),
        class_temp=float(class_temp),
        layer_penalty=float(layer_penalty),
        denoise=bool(denoise),
        postprocess=bool(postprocess)
    )
    
    prof = profiles[profile_name]
    ref_audio = prof["audio_path"]
    ref_text = prof["transcript"]
    
    if not os.path.exists(ref_audio):
        return None, f"⚠️ Error: File audio referensi tidak ditemukan di {ref_audio}"
    
    model = load_omnivoice()
    
    gen_kw = {
        "num_step": int(steps),
        "guidance_scale": float(cfg),
        "speed": float(speed),
        "t_shift": float(t_shift),
        "position_temperature": float(pos_temp),
        "class_temperature": float(class_temp),
        "layer_penalty_factor": float(layer_penalty),
        "denoise": bool(denoise),
        "postprocess_output": bool(postprocess)
    }
    if language and language.strip() and language != "Auto":
        gen_kw["language"] = language.strip()
        
    start_time = time.time()
    audios = model.generate(
        text=text_to_speak.strip(),
        ref_audio=ref_audio,
        ref_text=ref_text,
        **gen_kw
    )
    elapsed = time.time() - start_time
    
    audio = audios[0] if isinstance(audios, (list, tuple)) else audios
    sample_rate = int(getattr(model, "sampling_rate", 24000))
    
    # Detach & convert
    arr = audio.detach().to("cpu").float().numpy().squeeze()
    if arr.ndim > 1:
        arr = arr.mean(axis=int(np.argmin(arr.shape)))
    arr = np.clip(arr, -1.0, 1.0)
    
    out_filename = f"gen_{int(time.time())}.wav"
    out_path = os.path.join(OUTPUTS_DIR, out_filename)
    sf.write(out_path, arr, sample_rate)
    
    return out_path, f"🎉 Selesai dalam {elapsed:.1f} detik! Output tersimpan di: {out_path}"

# Initial configurations
saved_cfg = load_settings()
all_profile_names = get_profile_names()
initial_profile = saved_cfg.get("selected_profile")
if initial_profile not in all_profile_names:
    initial_profile = all_profile_names[0] if all_profile_names else None

init_audio = None
init_transcript = ""
if initial_profile and initial_profile in get_profiles():
    init_audio = get_profiles()[initial_profile]["audio_path"]
    init_transcript = get_profiles()[initial_profile]["transcript"]

theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="slate"
)

with gr.Blocks(title="VoiceCloner Studio") as demo:
    gr.Markdown("# 🎙️ VoiceCloner Studio (Pure Voice Cloning)")
    gr.Markdown("Kloning suara murni dan otentik berbasis OmniVoice & faster-whisper. **Profil dan setting tersimpan permanen.**")
    
    with gr.Row():
        # LEFT COLUMN: Voice Profile Management
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Profil Suara")
            with gr.Row():
                profile_dropdown = gr.Dropdown(
                    label="Pilih Suara:",
                    choices=all_profile_names,
                    value=initial_profile,
                    interactive=True,
                    scale=4
                )
                btn_refresh = gr.Button("🔄", scale=1, min_width=40)
            
            loaded_audio_preview = gr.Audio(label="Preview Suara Profil", value=init_audio, interactive=False)
            loaded_transcript_view = gr.Textbox(label="Transkrip Terdaftar", value=init_transcript, interactive=False, lines=3)
            
            with gr.Accordion("➕ Tambah Profil Suara Baru (Cukup 1x Upload)", open=False):
                new_profile_name = gr.Textbox(label="Nama Profil (Contoh: nurhikam, opel)", placeholder="Ketik nama di sini...")
                new_audio_input = gr.Audio(label="Upload Rekaman Suara (WAV/MP3, ideal 5-8 detik)", type="filepath")
                new_transcript_input = gr.Textbox(
                    label="Transkrip (Boleh Dikosongkan = Otomatis Ditranskripsi)",
                    placeholder="Biarkan kosong agar sistem otomatis mentranskripsi dengan Whisper AI..."
                )
                btn_save_profile = gr.Button("Simpan Profil Suara", variant="primary")
                profile_status_msg = gr.Markdown()
            
            with gr.Accordion("🗑️ Kelola Profil", open=False):
                btn_delete_profile = gr.Button("Hapus Profil Terpilih", variant="stop")
                delete_status_msg = gr.Markdown()
        
        # RIGHT COLUMN: Main TTS Workspace
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Naskah Script")
            text_input = gr.Textbox(
                label="Ketik atau paste naskah cerita di sini:",
                placeholder="Tulis naskah yang ingin disuarakan...",
                lines=6
            )
            
            with gr.Accordion("⚙️ Setting Suara (Production Overrides)", open=True):
                with gr.Row():
                    slider_steps = gr.Slider(label="Steps", minimum=10, maximum=40, step=1, value=saved_cfg.get("steps", 22))
                    slider_cfg = gr.Slider(label="CFG (Guidance)", minimum=1.0, maximum=4.0, step=0.1, value=saved_cfg.get("cfg", 1.9))
                    slider_speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, step=0.05, value=saved_cfg.get("speed", 0.9))
                with gr.Row():
                    slider_t_shift = gr.Slider(label="t_shift", minimum=0.0, maximum=1.0, step=0.05, value=saved_cfg.get("t_shift", 0.10))
                    slider_pos_temp = gr.Slider(label="Pos temp", minimum=1.0, maximum=10.0, step=0.5, value=saved_cfg.get("pos_temp", 5.5))
                    slider_class_temp = gr.Slider(label="Class temp", minimum=0.0, maximum=5.0, step=0.1, value=saved_cfg.get("class_temp", 0.2))
                with gr.Row():
                    slider_layer_penalty = gr.Slider(label="Layer penalty", minimum=0.0, maximum=10.0, step=0.5, value=saved_cfg.get("layer_penalty", 5.0))
                    lang_dropdown = gr.Dropdown(label="Language", choices=["Auto", "id", "en", "ja", "zh"], value=saved_cfg.get("language", "id"))
                with gr.Row():
                    toggle_denoise = gr.Checkbox(label="Denoise", value=saved_cfg.get("denoise", True))
                    toggle_postprocess = gr.Checkbox(label="Postprocess", value=saved_cfg.get("postprocess", True))
            
            btn_generate = gr.Button("⚡ Generate Suara", variant="primary", size="lg")
            gen_status = gr.Markdown()
            audio_output = gr.Audio(label="Hasil Suara", interactive=False)

    # Wire helpers
    def refresh_ui():
        profiles = get_profiles()
        choices = list(profiles.keys())
        cfg = load_settings()
        val = cfg.get("selected_profile")
        if val not in choices:
            val = choices[0] if choices else None
        audio_p = profiles[val]["audio_path"] if val and val in profiles else None
        trans_p = profiles[val]["transcript"] if val and val in profiles else ""
        return gr.update(choices=choices, value=val), audio_p, trans_p

    # Wire events
    btn_refresh.click(
        fn=refresh_ui,
        inputs=[],
        outputs=[profile_dropdown, loaded_audio_preview, loaded_transcript_view]
    )
    
    profile_dropdown.change(
        fn=on_select_profile,
        inputs=[profile_dropdown],
        outputs=[loaded_audio_preview, loaded_transcript_view]
    )
    
    btn_save_profile.click(
        fn=save_new_profile,
        inputs=[new_profile_name, new_audio_input, new_transcript_input],
        outputs=[profile_dropdown, profile_status_msg, loaded_transcript_view]
    ).then(
        fn=on_select_profile,
        inputs=[profile_dropdown],
        outputs=[loaded_audio_preview, loaded_transcript_view]
    )
    
    btn_delete_profile.click(
        fn=delete_profile,
        inputs=[profile_dropdown],
        outputs=[profile_dropdown, delete_status_msg]
    ).then(
        fn=on_select_profile,
        inputs=[profile_dropdown],
        outputs=[loaded_audio_preview, loaded_transcript_view]
    )
    
    btn_generate.click(
        fn=generate_voice,
        inputs=[
            profile_dropdown,
            text_input,
            lang_dropdown,
            slider_steps,
            slider_cfg,
            slider_speed,
            slider_t_shift,
            slider_pos_temp,
            slider_class_temp,
            slider_layer_penalty,
            toggle_denoise,
            toggle_postprocess
        ],
        outputs=[audio_output, gen_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7865, inbrowser=True, theme=theme)
