#!/usr/bin/env bash
# One-time setup on macOS / Linux.
#
# macOS prerequisites (run once, manually):
#   1. Install Homebrew if you don't have it:
#        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#   2. Install BlackHole (the macOS equivalent of VB-CABLE):
#        brew install blackhole-2ch
#      Reboot, or log out/in, after install.
#   3. Install PortAudio (sounddevice's audio backend):
#        brew install portaudio
#
# Linux: install pulseaudio + portaudio via your distro's package manager,
# then create a virtual sink: `pactl load-module module-null-sink sink_name=virt`.

set -e

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ from https://www.python.org/downloads/ or via brew install python"
  exit 1
fi

echo "Installing Python deps..."
python3 -m pip install --user -r requirements.txt

echo
echo "Done. Now run:  ./run.sh"
echo
echo "macOS reminder: System Settings -> Privacy & Security -> Microphone -> enable Terminal (or your IDE) so the voice changer can read your mic."
