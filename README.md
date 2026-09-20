# realtime-voice-dsp

Real-time pitch and formant shifting for a live microphone, written from
scratch in NumPy and SciPy. No ML, no external audio libraries beyond
PortAudio bindings: an STFT phase vocoder, a cepstral formant warper, a noise
gate and a limiter, running inside a low-latency audio callback.

Routes to a virtual audio device, so the processed signal can be used as a
microphone by any application that accepts one.

```
mic ──► high-pass ──► noise gate ──► phase vocoder ──► limiter ──► virtual device
        60 Hz DC       -50 dB         pitch + formant   -1 dBFS
        + rumble       120 ms hold    shift             ceiling
```

---

## The interesting part

**Pitch and formant are shifted independently.** Naive pitch shifting scales
the whole spectrum, which drags the formants along with it. That is why
resampled speech sounds like a chipmunk or a monster rather than a different
person. Here the spectral envelope is separated from the excitation and warped
on its own:

1. Estimate the spectral envelope by cepstral liftering.
2. Divide the spectrum by the envelope, leaving the excitation residual.
3. Warp the envelope by `formant_ratio`.
4. Multiply the warped envelope back onto the residual.

So pitch can move down seven semitones while the formants move only a little,
or the reverse. Shifting both by matched amounts is what keeps the result
plausible as a human voice.

**Phase is reconstructed, not copied.** Per bin, the true instantaneous
frequency is recovered from the phase advance between frames, then magnitude is
reassigned to the bin *pair* straddling `k * ratio` with linear interpolation
rather than rounded to a single bin. Synthesis phase accumulates the new
instantaneous frequency. Rounding to one bin is the usual shortcut and it is
audible as a metallic edge.

**Latency is bounded by dropping, not blocking.** Input and output run as two
independent streams joined by a bounded queue. If the queue fills, the oldest
block is dropped and the newest pushed, so latency cannot grow unboundedly
during a long call. If the queue is empty, the output callback emits silence
rather than waiting, because a conferencing app tolerates a glitch far better than a
stalled stream.

**A bypass guard, on by default.** If pitch and formant are both within 1% of
unity the chain is a no-op, and the pipeline outputs silence instead of the dry
signal. Passing an unprocessed microphone straight through to a virtual device
is almost never what you want; this makes that state audible immediately.

---

## Signal chain

| Stage | |
|---|---|
| `Biquad` | 60 Hz high-pass, RBJ coefficients, persistent state. Removes DC, mains hum and desk rumble before the vocoder, where they would otherwise smear the envelope estimate. 60 Hz rather than 80 Hz keeps low male fundamentals intact. |
| `NoiseGate` | RMS threshold with a 120 ms hold, so the tail of a word is not clipped by a gate that closes too eagerly. |
| `PhaseVocoder` | 2048-point FFT, 512 hop (4× overlap), Hann window with overlap-add normalisation. |
| `Limiter` | −1 dBFS ceiling. The formant warp can add gain in-band; this stops it reaching the output device hot. |

Defaults: 48 kHz, 1024-sample blocks, mono, float32.

---

## Install

Needs Python 3.10+, PortAudio, and a virtual audio device to route into.

**macOS**
```bash
brew install portaudio blackhole-2ch   # reboot or log out/in after BlackHole
./install.sh
```
Then allow microphone access for your terminal under
System Settings → Privacy & Security → Microphone.

**Linux**
```bash
# portaudio + pulseaudio from your package manager, then:
pactl load-module module-null-sink sink_name=virt
./install.sh
```

**Windows**: install [VB-CABLE](https://vb-audio.com/Cable/), then `install.bat`.

## Run

```bash
./run.sh                    # GUI
python -m voice_changer --list     # enumerate audio devices

python -m voice_changer --headless \
    --in 1 --out 4 --semitones -4 --formant 0.85 --gate-db -50
```

In the GUI, pick input and output devices, choose a preset or set pitch and
formant by hand, and press start. Space bar is a hard mute.

Point your conferencing app's microphone at the virtual device
(BlackHole / VB-CABLE / the null sink) rather than at your real one.

---

## Presets

| | Pitch | Formant | |
|---|---:|---:|---|
| Natural | −0.7 st | 0.97 | gentlest shift, cleanest reconstruction |
| Subtle shift | −1.2 st | 0.95 | |
| Deep | −4.0 st | 0.84 | |
| Very deep | −7.0 st | 0.74 | most artifacts |
| Higher | +3.5 st | 1.16 | |
| Very high | +6.5 st | 1.28 | |
| Muffled | −1.0 st | 0.85 | lower gate, distant character |

There is deliberately no bypass preset; use mute instead.

---

## Layout

```
voice_changer/dsp.py        Biquad, PhaseVocoder, NoiseGate, Limiter, Pipeline
voice_changer/audio.py      sounddevice streams, bounded queue, device discovery
voice_changer/gui.py        Tk control panel, level meters, test playback
voice_changer/presets.py    parameter sets
voice_changer/settings.py   JSON persistence in ~/.voice_changer.json
voice_changer/__main__.py   CLI: --list, --headless, GUI default
```

`dsp.py` has no dependency on the audio or GUI layers. It is plain NumPy in,
NumPy out, so the processing can be tested offline without an audio device.

## Measured behaviour

Feeding a 140 Hz synthetic tone with a harmonic through the pipeline offline:

```
 -7.0 st  ->  peak at   93.3 Hz   (expected 93)
  0.0 st  ->  peak at  140.0 Hz   (expected 140)
 +6.5 st  ->  peak at  204.0 Hz   (expected 204)
```

All presets output finite, non-clipping audio; the bypass guard measures
exactly zero RMS at unity settings.

## Limitations

Mono only. The phase vocoder introduces the usual transient smearing: plosives
and sharp consonants soften as the shift grows, which is inherent to the method
rather than a tuning problem; a transient-preserving variant would need onset
detection and phase reset. Large shifts beyond roughly ±7 semitones stop
sounding plausible. There is no test suite in the repository, though `dsp.py` is
structured to make one straightforward.

## Licence

MIT, see [LICENSE](LICENSE).
