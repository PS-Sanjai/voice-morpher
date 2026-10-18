# 🎙️ Voice Morpher

Give it samples of a voice. Then speak in **any language** — it gives you back the
same words in that voice. Runs fully on your own computer (nothing is uploaded).

Built on [Seed-VC](https://github.com/Plachtaa/seed-vc), an open-source zero-shot
voice conversion model. Because it converts the *sound* of the voice rather than
re-reading text, it works with Tamil, Hindi, English, or anything else.

## Requirements
- **Python 3.10** (https://www.python.org/downloads/release/python-31011/)
- **Git** (https://git-scm.com)
- **FFmpeg** if you want to upload mp3/m4a files (`winget install ffmpeg` on Windows)
- An **NVIDIA GPU** is strongly recommended (works on CPU / Apple Silicon, just slower)
- ~6 GB free disk space (dependencies + models)

## Setup (once)
**Windows:** double-click `setup.bat`
**Mac / Linux:** `chmod +x setup.sh run.sh && ./setup.sh`

## Run
**Windows:** double-click `run.bat`  **Mac / Linux:** `./run.sh`

Your browser opens at http://127.0.0.1:7860. The first conversion downloads the
models (~2 GB) — after that it's quick.

## How to use
1. **Voices tab** — type a name, upload 2–5 clips of the voice (10–25 s of total
   speech is ideal), click **Save voice**. Clean audio, one speaker, no music.
2. **Speak & Morph tab** — record yourself (or upload a clip), pick the voice,
   click **Morph my voice**. Play or download the WAV.

### Settings
| Setting | What it does |
|---|---|
| Quality | Diffusion steps. 10 = fast, 25 = balanced, 40–50 = best |
| Keep my exact intonation | Preserves your pitch contour (44 kHz model). Good for expressive speech/singing |
| Pitch shift | Semitones up/down — only in intonation mode |
| Speed | Speaks faster/slower than your recording |

Results are saved in the `outputs/` folder; voices in `voices/`.

## Tips for the best match
- More reference speech (up to 25 s) = closer voice.
- Record your input in a quiet room, close to the mic.
- If it sounds a bit like you still, raise Quality and add more samples.
