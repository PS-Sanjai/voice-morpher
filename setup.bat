@echo off
REM Voice Morpher - one-time setup for Windows
cd /d "%~dp0"

where git >nul 2>nul || (echo Please install Git from https://git-scm.com and run this again. & pause & exit /b 1)
where py >nul 2>nul || (echo Please install Python 3.10 from https://www.python.org and run this again. & pause & exit /b 1)

if not exist seed-vc (
    echo Downloading Seed-VC...
    git clone --depth 1 https://github.com/Plachtaa/seed-vc.git
)

if not exist venv (
    echo Creating Python 3.10 environment...
    py -3.10 -m venv venv || (echo Python 3.10 is required. Install it from python.org. & pause & exit /b 1)
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing dependencies (this takes a while, PyTorch is large)...
pip install -r seed-vc\requirements.txt
pip install gradio soundfile librosa

echo.
echo Setup complete! Double-click run.bat to start Voice Morpher.
pause
