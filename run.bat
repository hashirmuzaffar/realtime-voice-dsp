@echo off
REM Launches the voice changer GUI from this folder.
cd /d "%~dp0"
python -m voice_changer %*
pause
