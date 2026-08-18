@echo off
REM One-time setup: install Python dependencies.
cd /d "%~dp0"
python -m pip install -r requirements.txt
pause
