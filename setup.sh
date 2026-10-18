#!/usr/bin/env bash
# Voice Morpher - one-time setup for macOS / Linux
set -e
cd "$(dirname "$0")"

command -v git >/dev/null || { echo "Please install git first."; exit 1; }
PY=$(command -v python3.10 || command -v python3)
echo "Using $($PY --version)"

[ -d seed-vc ] || git clone --depth 1 https://github.com/Plachtaa/seed-vc.git
[ -d venv ] || "$PY" -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

if [[ "$(uname)" == "Darwin" ]]; then
  pip install -r seed-vc/requirements-mac.txt
else
  pip install -r seed-vc/requirements.txt
fi
pip install gradio soundfile librosa

echo
echo "Setup complete! Start the app with:  ./run.sh"
