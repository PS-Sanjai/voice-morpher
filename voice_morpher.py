"""
Voice Morpher
-------------
1. Build a voice: upload/record a few samples of the target voice.
2. Speak (any language) into the mic or upload a clip.
3. Get back the same words spoken in the target voice.

Powered by Seed-VC (zero-shot voice conversion). Runs 100% locally.
"""

import os
import sys
import re
import json
import time
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import gradio as gr

# ---------------------------------------------------------------- paths
APP_DIR = Path(__file__).resolve().parent
SEED_DIR = APP_DIR / "seed-vc"
VOICES_DIR = APP_DIR / "voices"
OUTPUT_DIR = APP_DIR / "outputs"
VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

if not SEED_DIR.exists():
    sys.exit("seed-vc folder not found. Run setup first (setup.bat on Windows, ./setup.sh on Mac/Linux).")

# Seed-VC uses relative imports and a ./checkpoints cache, so run from inside it.
sys.path.insert(0, str(SEED_DIR))
os.chdir(SEED_DIR)

REF_SR = 22050          # Seed-VC speech model sample rate
MAX_REF_SECONDS = 25    # Seed-VC only uses the first ~25 s of reference audio

# ---------------------------------------------------------------- model (lazy)
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            print("Loading voice models (first run downloads ~2 GB, please wait)...")
            from seed_vc_wrapper import SeedVCWrapper
            _model = SeedVCWrapper()
            print(f"Models ready on {_model.device}.")
    return _model


# ---------------------------------------------------------------- voice library
def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\- ]", "", (name or "").strip()).strip().replace(" ", "_")
    return name[:40]


def list_voices():
    return sorted(p.name for p in VOICES_DIR.iterdir()
                  if p.is_dir() and (p / "reference.wav").exists())


def voice_info(name):
    if not name:
        return "No voices yet."
    meta = VOICES_DIR / name / "meta.json"
    if not meta.exists():
        return ""
    m = json.loads(meta.read_text())
    return (f"**{name}** — {m['num_samples']} sample(s), "
            f"{m['reference_seconds']:.1f}s of clean speech used as the voice reference.")


def build_reference(paths):
    """Merge samples into one clean reference: drop silences, normalise, cap at 25 s."""
    pieces, gap = [], np.zeros(int(0.15 * REF_SR), dtype=np.float32)
    for p in paths:
        y, _ = librosa.load(p, sr=REF_SR, mono=True)
        if y.size == 0:
            continue
        for start, end in librosa.effects.split(y, top_db=35):
            seg = y[start:end]
            if len(seg) > 0.2 * REF_SR:          # skip clicks / tiny blips
                pieces += [seg, gap]
    if not pieces:
        raise gr.Error("Couldn't find any speech in those samples. Try clearer recordings.")
    ref = np.concatenate(pieces)[: MAX_REF_SECONDS * REF_SR]
    peak = np.abs(ref).max()
    if peak > 0:
        ref = ref / peak * 0.95
    return ref.astype(np.float32)


def create_voice(name, files, mic_sample):
    name = safe_name(name)
    if not name:
        raise gr.Error("Give the voice a name first.")
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    if mic_sample:
        paths.append(mic_sample)
    if not paths:
        raise gr.Error("Add at least one audio sample (upload or record).")

    vdir = VOICES_DIR / name
    (vdir / "samples").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        y, sr = librosa.load(p, sr=None, mono=True)
        sf.write(vdir / "samples" / f"sample_{int(time.time())}_{i}.wav", y, sr)

    # Rebuild from every sample this voice has (lets you add more later).
    all_samples = sorted(str(p) for p in (vdir / "samples").glob("*.wav"))
    ref = build_reference(all_samples)
    sf.write(vdir / "reference.wav", ref, REF_SR)
    secs = len(ref) / REF_SR
    (vdir / "meta.json").write_text(json.dumps(
        {"num_samples": len(all_samples), "reference_seconds": secs}))

    tip = "" if secs >= 8 else "  ⚠️ Under 8s of speech — add more samples for a closer match."
    voices = list_voices()
    return (f"✅ Saved voice **{name}** ({secs:.1f}s of speech).{tip}",
            gr.update(choices=voices, value=name),
            gr.update(choices=voices, value=name),
            str(vdir / "reference.wav"))


def delete_voice(name):
    if not name:
        raise gr.Error("Pick a voice to delete.")
    import shutil
    shutil.rmtree(VOICES_DIR / name, ignore_errors=True)
    voices = list_voices()
    first = voices[0] if voices else None
    return (f"🗑️ Deleted **{name}**.",
            gr.update(choices=voices, value=first),
            gr.update(choices=voices, value=first),
            None)


def preview_voice(name):
    if not name:
        return None, ""
    return str(VOICES_DIR / name / "reference.wav"), voice_info(name)


# ---------------------------------------------------------------- conversion
def morph(source, voice, quality, keep_intonation, pitch_shift, speed,
          progress=gr.Progress()):
    if not source:
        raise gr.Error("Record or upload what you want to say first.")
    if not voice:
        raise gr.Error("Create / pick a target voice first (Voices tab).")

    progress(0.05, desc="Loading models (slow only the first time)...")
    model = get_model()

    # Clean the input: trim leading/trailing silence.
    progress(0.2, desc="Preparing your recording...")
    y, sr = librosa.load(source, sr=None, mono=True)
    y, _ = librosa.effects.trim(y, top_db=40)
    if len(y) < 0.3 * sr:
        raise gr.Error("The recording is too short or silent.")
    src_path = OUTPUT_DIR / "_last_input.wav"
    sf.write(src_path, y, sr)

    progress(0.4, desc="Converting voice...")
    result = None
    for _, full_audio in model.convert_voice(
        source=str(src_path),
        target=str(VOICES_DIR / voice / "reference.wav"),
        diffusion_steps=int(quality),
        length_adjust=1.0 / float(speed),
        inference_cfg_rate=0.7,
        f0_condition=bool(keep_intonation),
        auto_f0_adjust=True,
        pitch_shift=int(pitch_shift),
        stream_output=True,
    ):
        if full_audio is not None:
            result = full_audio

    if result is None:
        raise gr.Error("Conversion produced no audio. Try a longer recording.")
    out_sr, audio = result
    out_file = OUTPUT_DIR / f"{voice}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    sf.write(out_file, audio, out_sr)
    progress(1.0, desc="Done")
    return str(out_file), str(out_file)


# ---------------------------------------------------------------- UI
CSS = """
#title {text-align:center; margin-bottom:0}
#subtitle {text-align:center; opacity:.7; margin-top:4px}
"""

with gr.Blocks(title="Voice Morpher", css=CSS, theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ Voice Morpher", elem_id="title")
    gr.Markdown("Speak in any language — hear it back in your saved voice. Runs entirely on your computer.",
                elem_id="subtitle")

    voices_now = list_voices()

    with gr.Tab("1 · Speak & Morph"):
        with gr.Row():
            with gr.Column():
                source = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                  label="What you want to say (record or upload)")
                voice_pick = gr.Dropdown(voices_now, value=voices_now[0] if voices_now else None,
                                         label="Target voice")
                with gr.Accordion("Settings", open=False):
                    quality = gr.Slider(4, 50, value=25, step=1,
                                        label="Quality (diffusion steps) — higher is better but slower")
                    keep_intonation = gr.Checkbox(
                        False, label="Keep my exact intonation/pitch contour (44 kHz mode, good for expressive speech or singing)")
                    pitch_shift = gr.Slider(-12, 12, value=0, step=1,
                                            label="Pitch shift in semitones (only with intonation mode)")
                    speed = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Speed")
                go = gr.Button("✨ Morph my voice", variant="primary", size="lg")
            with gr.Column():
                out_audio = gr.Audio(label="Result", type="filepath", interactive=False)
                out_file = gr.File(label="Download WAV")
                gr.Markdown("Tip: speak clearly in a quiet room. Background noise and music "
                            "make the result less accurate.")

    with gr.Tab("2 · Voices"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Create or add to a voice\n"
                            "Upload a few clips of the voice (10–25 s of total speech is ideal). "
                            "Using the same name again adds the new samples to that voice.")
                vname = gr.Textbox(label="Voice name", placeholder="e.g. my_voice")
                vfiles = gr.File(file_count="multiple", file_types=["audio"],
                                 label="Audio samples (wav, mp3, m4a, ogg, flac)")
                vmic = gr.Audio(sources=["microphone"], type="filepath",
                                label="…or record a sample now")
                save_btn = gr.Button("💾 Save voice", variant="primary")
                status = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Your voices")
                lib_pick = gr.Dropdown(voices_now, value=voices_now[0] if voices_now else None,
                                       label="Voice")
                info = gr.Markdown(voice_info(voices_now[0]) if voices_now else "No voices yet.")
                ref_preview = gr.Audio(
                    value=str(VOICES_DIR / voices_now[0] / "reference.wav") if voices_now else None,
                    label="Reference used for cloning", interactive=False, type="filepath")
                del_btn = gr.Button("🗑️ Delete this voice", variant="stop")
        gr.Markdown("_Only use voices you have permission to use._")

    # wiring
    go.click(morph, [source, voice_pick, quality, keep_intonation, pitch_shift, speed],
             [out_audio, out_file])
    save_btn.click(create_voice, [vname, vfiles, vmic],
                   [status, voice_pick, lib_pick, ref_preview]).then(
        voice_info, lib_pick, info)
    lib_pick.change(preview_voice, lib_pick, [ref_preview, info])
    del_btn.click(delete_voice, lib_pick, [status, voice_pick, lib_pick, ref_preview]).then(
        voice_info, lib_pick, info)

if __name__ == "__main__":
    app.queue().launch(inbrowser=True, allowed_paths=[str(VOICES_DIR), str(OUTPUT_DIR)])
