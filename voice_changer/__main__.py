"""Entry point: `python -m voice_changer`."""

import argparse
import sys

from .audio import AudioEngine, list_input_devices, list_output_devices
from .dsp import Pipeline


SAMPLE_RATE = 48000
BLOCK = 1024


def cli_list_devices():
    print("=== Input devices ===")
    for i, d in list_input_devices():
        print(f"  [{i:2d}] {d['name']}  ({d['max_input_channels']} ch)")
    print("\n=== Output devices ===")
    for i, d in list_output_devices():
        print(f"  [{i:2d}] {d['name']}  ({d['max_output_channels']} ch)")


def cli_run_headless(args):
    pipeline = Pipeline(sample_rate=SAMPLE_RATE)
    pipeline.vocoder.set_pitch_semitones(args.semitones)
    pipeline.vocoder.set_formant(args.formant)
    pipeline.gate.set_threshold_db(args.gate_db)
    pipeline.output_gain_db = args.gain_db

    engine = AudioEngine(pipeline, sample_rate=SAMPLE_RATE, block_size=BLOCK)
    engine.start(args.in_dev, args.out_dev)
    print(f"running: in={args.in_dev} out={args.out_dev} "
          f"semitones={args.semitones} formant={args.formant}")
    print("press ctrl-c to stop.")
    try:
        import threading
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        engine.stop()


def main():
    p = argparse.ArgumentParser(prog="voice_changer")
    p.add_argument('--list', action='store_true',
                   help='list audio devices and exit')
    p.add_argument('--headless', action='store_true',
                   help='no GUI; needs --in/--out')
    p.add_argument('--in', dest='in_dev', type=int)
    p.add_argument('--out', dest='out_dev', type=int)
    p.add_argument('--semitones', type=float, default=-4.0)
    p.add_argument('--formant', type=float, default=0.85)
    p.add_argument('--gate-db', type=float, default=-50.0)
    p.add_argument('--gain-db', type=float, default=0.0)
    args = p.parse_args()

    if args.list:
        cli_list_devices()
        return

    if args.headless:
        if args.in_dev is None or args.out_dev is None:
            print("--headless requires --in and --out indices "
                  "(use --list to see them)", file=sys.stderr)
            sys.exit(2)
        cli_run_headless(args)
        return

    # GUI mode
    from .gui import VoiceChangerGUI
    pipeline = Pipeline(sample_rate=SAMPLE_RATE)
    engine = AudioEngine(pipeline, sample_rate=SAMPLE_RATE, block_size=BLOCK)
    gui = VoiceChangerGUI(pipeline, engine)
    gui.run()


if __name__ == "__main__":
    main()
