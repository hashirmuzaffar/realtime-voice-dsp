#!/usr/bin/env bash
# Launches the voice changer GUI. Pass --list to see device numbers.
set -e
cd "$(dirname "$0")"
python3 -m voice_changer "$@"
