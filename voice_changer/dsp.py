"""Real-time voice DSP modules.

All processors are stateful, mono, float32, and operate at a fixed sample
rate set at construction time. Callers can pass arbitrary block sizes;
internal frame-based processors buffer as needed.
"""

import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# Biquad IIR filter (used for high-pass / DC removal)
# ---------------------------------------------------------------------------

class Biquad:
    """Single biquad section with persistent state, vectorized via scipy."""

    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        self.b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._zi = np.zeros(2, dtype=np.float64)

    def set_highpass(self, freq_hz: float, q: float = 0.707):
        w0 = 2.0 * np.pi * freq_hz / self.sr
        cosw, sinw = np.cos(w0), np.sin(w0)
        alpha = sinw / (2.0 * q)
        b0 = (1.0 + cosw) / 2.0
        b1 = -(1.0 + cosw)
        b2 = (1.0 + cosw) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha
        self.b = np.array([b0, b1, b2]) / a0
        self.a = np.array([1.0, a1 / a0, a2 / a0])
        self._zi = np.zeros(2, dtype=np.float64)

    def reset(self):
        self._zi = np.zeros(2, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        y, self._zi = lfilter(self.b, self.a, x, zi=self._zi)
        return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase vocoder: pitch shift with optional formant warping
# ---------------------------------------------------------------------------

class PhaseVocoder:
    """STFT pitch shifter with independent formant control.

    Pitch shifting:
        - STFT analysis at hop H over Hann-windowed frames of size F.
        - Per bin, recover true frequency from phase advance.
        - Reassign magnitude to bin pair (floor, ceil) of k*ratio with
          linear interpolation — smoother than rounding to one bin.
        - Synthesize phase by accumulating the new instantaneous frequency.
        - Inverse STFT with overlap-add and window-norm compensation.

    Formant control:
        - Estimate spectral envelope via cepstral liftering.
        - Divide spectrum by envelope (residual = excitation).
        - Warp the envelope by formant_ratio.
        - Multiply warped envelope back onto residual.
    """

    def __init__(self, sample_rate=48000, fft_size=2048, hop=512):
        self.sr = sample_rate
        self.fft = fft_size
        self.hop = hop
        self.n_bins = fft_size // 2 + 1
        self.window = np.hanning(fft_size).astype(np.float32)
        self._win_norm = self._compute_window_norm()

        # Buffers
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._frame_start = 0
        self._out_buf = np.zeros(fft_size * 2, dtype=np.float32)
        self._out_read = 0
        self._out_write = 0

        # Phase tracking
        self._last_phase = np.zeros(self.n_bins, dtype=np.float64)
        self._sum_phase = np.zeros(self.n_bins, dtype=np.float64)

        # Bin frequency lookup
        self._bin_idx = np.arange(self.n_bins, dtype=np.float64)
        self._expected_phase = 2.0 * np.pi * self.hop * self._bin_idx / self.fft

        # Smooth Hann-shaped cepstral lifter — eliminates "plastic" ringing
        # caused by a hard rectangular cutoff.
        self._lifter = self._make_lifter(fade_in=30, fade_out=70)

        # Exponential smoothing of the spectral envelope across frames —
        # eliminates the frame-to-frame envelope jitter that produces a
        # "modulated / processed" timbre.
        self._env_state = None
        self._env_alpha = 0.45

        # Live parameters
        self.pitch_ratio = 1.0
        self.formant_ratio = 1.0

    def _make_lifter(self, fade_in: int, fade_out: int) -> np.ndarray:
        lifter = np.zeros(self.fft, dtype=np.float64)
        lifter[:fade_in] = 1.0
        lifter[-fade_in:] = 1.0
        n = fade_out - fade_in
        if n > 0:
            fade = 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / n))
            lifter[fade_in:fade_out] = fade
            lifter[-fade_out:-fade_in] = fade[::-1]
        return lifter

    def set_pitch_semitones(self, semitones: float):
        self.pitch_ratio = float(2.0 ** (semitones / 12.0))

    def set_formant(self, ratio: float):
        self.formant_ratio = float(max(0.4, min(2.5, ratio)))

    def _compute_window_norm(self) -> float:
        win = self.window.astype(np.float64)
        accum = np.zeros(self.fft * 4, dtype=np.float64)
        for i in range(0, len(accum) - self.fft, self.hop):
            accum[i:i + self.fft] += win * win
        mid = len(accum) // 2
        return float(accum[mid]) if accum[mid] > 0 else 1.0

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        windowed = frame * self.window
        spec = np.fft.rfft(windowed)
        mag = np.abs(spec).astype(np.float64)
        phase = np.angle(spec).astype(np.float64)

        # --- Formant warp via cepstral envelope ----------------------------
        if abs(self.formant_ratio - 1.0) > 1e-3:
            log_mag = np.log(mag + 1e-9)
            cep = np.fft.irfft(log_mag, n=self.fft)
            env_log = np.fft.rfft(cep * self._lifter, n=self.fft).real
            env = np.exp(env_log)

            if self._env_state is None or self._env_state.shape != env.shape:
                self._env_state = env.copy()
            else:
                a = self._env_alpha
                self._env_state = a * env + (1.0 - a) * self._env_state
            env = self._env_state

            residual = mag / (env + 1e-9)
            idx = self._bin_idx / self.formant_ratio
            warped = np.interp(idx, self._bin_idx, env,
                               left=env[0], right=env[-1])
            mag = warped * residual

        # --- Phase advance from analysis ----------------------------------
        delta = phase - self._last_phase
        self._last_phase = phase
        delta -= self._expected_phase
        delta = np.mod(delta + np.pi, 2.0 * np.pi) - np.pi
        omega_in = self._expected_phase + delta  # advance per hop, input bins

        # --- Pitch shift via spectral gather ------------------------------
        # For each output bin q, sample input bin q/ratio with linear interp.
        # We also gather omega_in (scaled by ratio) and the input phase
        # itself (via complex interp, to handle wrapping) — the gathered
        # phase is the reference for phase-locking below.
        if abs(self.pitch_ratio - 1.0) > 1e-3:
            src = self._bin_idx / self.pitch_ratio
            s_lo = np.floor(src).astype(np.int64)
            s_hi = s_lo + 1
            frac = src - s_lo

            valid = (s_lo >= 0) & (s_hi < self.n_bins)
            s_lo_c = np.clip(s_lo, 0, self.n_bins - 1)
            s_hi_c = np.clip(s_hi, 0, self.n_bins - 1)

            mag_out = mag[s_lo_c] * (1.0 - frac) + mag[s_hi_c] * frac
            omega_out = (omega_in[s_lo_c] * (1.0 - frac) +
                         omega_in[s_hi_c] * frac) * self.pitch_ratio

            # Complex interp of unit phase (preserves wrapping)
            z_in = np.exp(1j * phase)
            z_at = z_in[s_lo_c] * (1.0 - frac) + z_in[s_hi_c] * frac
            ref_phase = np.angle(z_at)

            mag_out[~valid] = 0.0
            omega_out[~valid] = 0.0
            ref_phase[~valid] = 0.0
        else:
            mag_out = mag
            omega_out = omega_in
            ref_phase = phase

        # --- Standard phase accumulation ----------------------------------
        self._sum_phase += omega_out

        # --- Phase locking (Laroche-Dolson rigid) -------------------------
        # Without locking, every output bin's phase drifts independently,
        # which sounds smeary / "phasey" / underwater. With locking, we find
        # spectral peaks (harmonics of the voice) and force surrounding bins
        # to keep the same relative phase pattern they had in the input.
        # The voice keeps its time-domain coherence — sounds like a real
        # voice instead of a phase vocoder.
        max_mag = float(np.max(mag_out)) if mag_out.size else 0.0
        if max_mag > 1e-7 and self.n_bins > 5:
            thr = max_mag * 0.005
            is_peak = np.zeros(self.n_bins, dtype=bool)
            is_peak[2:-2] = (
                (mag_out[2:-2] > mag_out[1:-3]) &
                (mag_out[2:-2] > mag_out[3:-1]) &
                (mag_out[2:-2] > mag_out[:-4]) &
                (mag_out[2:-2] > mag_out[4:]) &
                (mag_out[2:-2] > thr)
            )
            peak_idx = np.where(is_peak)[0]

            if peak_idx.size >= 2:
                midpoints = (peak_idx[:-1] + peak_idx[1:]) // 2
                pos = np.searchsorted(midpoints, np.arange(self.n_bins))
                pos = np.clip(pos, 0, peak_idx.size - 1)
                nearest = peak_idx[pos]

                rel = ref_phase - ref_phase[nearest]
                rel = np.mod(rel + np.pi, 2.0 * np.pi) - np.pi
                locked = self._sum_phase[nearest] + rel
                mask = ~is_peak
                self._sum_phase[mask] = locked[mask]
            elif peak_idx.size == 1:
                nearest = peak_idx[0]
                rel = ref_phase - ref_phase[nearest]
                rel = np.mod(rel + np.pi, 2.0 * np.pi) - np.pi
                locked = self._sum_phase[nearest] + rel
                mask = np.arange(self.n_bins) != nearest
                self._sum_phase[mask] = locked[mask]

        self._sum_phase = np.mod(self._sum_phase + np.pi, 2.0 * np.pi) - np.pi

        spec_out = mag_out * np.exp(1j * self._sum_phase)
        out = np.fft.irfft(spec_out, n=self.fft)
        return (out * self.window).astype(np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = len(x)

        self._in_buf = np.concatenate([self._in_buf, x])

        while self._frame_start + self.fft <= len(self._in_buf):
            frame = self._in_buf[self._frame_start:self._frame_start + self.fft]
            synth = self._process_frame(frame)

            end = self._out_write + self.fft
            if end > len(self._out_buf):
                pad = end - len(self._out_buf) + self.fft
                self._out_buf = np.concatenate([
                    self._out_buf, np.zeros(pad, dtype=np.float32)
                ])
            self._out_buf[self._out_write:end] += synth / self._win_norm

            self._frame_start += self.hop
            self._out_write += self.hop

        if self._frame_start > 8 * self.fft:
            self._in_buf = self._in_buf[self._frame_start:].copy()
            self._frame_start = 0

        avail = self._out_write - self._out_read
        if avail < n:
            out = np.zeros(n, dtype=np.float32)
            if avail > 0:
                out[:avail] = self._out_buf[self._out_read:self._out_read + avail]
                self._out_read += avail
        else:
            out = self._out_buf[self._out_read:self._out_read + n].copy()
            self._out_read += n

        if self._out_read > 8 * self.fft:
            tail = self._out_buf[self._out_read:].copy()
            self._out_buf = np.concatenate([
                tail, np.zeros(self.fft * 2, dtype=np.float32)
            ])
            self._out_write -= self._out_read
            self._out_read = 0

        return out


# ---------------------------------------------------------------------------
# Block-based noise gate
# ---------------------------------------------------------------------------

class NoiseGate:
    """RMS-based noise gate with hold and per-block linear ramp."""

    def __init__(self, sample_rate=48000, threshold_db=-50.0, hold_ms=120.0):
        self.sr = sample_rate
        self.threshold = 10.0 ** (threshold_db / 20.0)
        self._hold_total = int(sample_rate * hold_ms / 1000.0)
        self._hold_remaining = 0
        self._gain = 0.0

    def set_threshold_db(self, db: float):
        self.threshold = 10.0 ** (db / 20.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms > self.threshold:
            target = 1.0
            self._hold_remaining = self._hold_total
        elif self._hold_remaining > 0:
            target = 1.0
            self._hold_remaining = max(0, self._hold_remaining - len(x))
        else:
            target = 0.0
        ramp = np.linspace(self._gain, target, len(x), dtype=np.float32)
        self._gain = target
        return x * ramp


# ---------------------------------------------------------------------------
# Soft limiter
# ---------------------------------------------------------------------------

class Limiter:
    """tanh-based soft clipper. Cheap, transparent at low signal levels."""

    def __init__(self, ceiling_db=-1.0):
        self.ceiling = 10.0 ** (ceiling_db / 20.0)

    def set_ceiling_db(self, db: float):
        self.ceiling = 10.0 ** (db / 20.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        c = self.ceiling
        return (c * np.tanh(x / c)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline that the GUI talks to
# ---------------------------------------------------------------------------

class Pipeline:
    SHIFT_TOL = 1e-2  # below this, the vocoder is effectively a pass-through

    def __init__(self, sample_rate=48000, record_seconds: float = 6.0):
        self.sr = sample_rate

        # Pre-vocoder filtering. 60 Hz keeps voice warmth (male fundamentals
        # sit around 80–180 Hz; 80 Hz cutoff was eating bass) while still
        # removing AC hum and rumble.
        self.hpf = Biquad(sample_rate)
        self.hpf.set_highpass(60.0, q=0.707)

        self.gate = NoiseGate(sample_rate)
        self.vocoder = PhaseVocoder(sample_rate)
        self.limiter = Limiter()

        self.input_gain_db = 0.0
        self.output_gain_db = 0.0
        self.dry_wet = 1.0
        self.mute = False           # hard mute — outputs silence
        self.safety_lock = True     # mute when the chain is a no-op

        self.input_level = 0.0
        self.output_level = 0.0

        # Circular tap of processed output for the test-playback button.
        self._rec_size = int(sample_rate * record_seconds)
        self._rec_buf = np.zeros(self._rec_size, dtype=np.float32)
        self._rec_pos = 0

    # -- bypass detection ---------------------------------------------------

    def is_processing_active(self) -> bool:
        """True when pitch or formant differs enough from unity to matter."""
        v = self.vocoder
        return (abs(v.pitch_ratio - 1.0) > self.SHIFT_TOL or
                abs(v.formant_ratio - 1.0) > self.SHIFT_TOL)

    # -- recording tap ------------------------------------------------------

    def _record_append(self, x: np.ndarray):
        n = len(x)
        if n >= self._rec_size:
            self._rec_buf[:] = x[-self._rec_size:]
            self._rec_pos = 0
            return
        end = self._rec_pos + n
        if end <= self._rec_size:
            self._rec_buf[self._rec_pos:end] = x
        else:
            split = self._rec_size - self._rec_pos
            self._rec_buf[self._rec_pos:] = x[:split]
            self._rec_buf[:n - split] = x[split:]
        self._rec_pos = end % self._rec_size

    def get_recent(self, seconds: float) -> np.ndarray:
        n = min(int(self.sr * seconds), self._rec_size)
        # Ring buffer in chronological order ending at _rec_pos
        start = (self._rec_pos - n) % self._rec_size
        if start + n <= self._rec_size:
            return self._rec_buf[start:start + n].copy()
        first = self._rec_size - start
        return np.concatenate([
            self._rec_buf[start:], self._rec_buf[:n - first]
        ]).copy()

    # -- main entry point ---------------------------------------------------

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if self.input_gain_db != 0.0:
            x = x * (10.0 ** (self.input_gain_db / 20.0))
        self.input_level = float(np.sqrt(np.mean(x * x) + 1e-12))

        # Hard mute — never lets anything through.
        if self.mute:
            silence = np.zeros_like(x)
            self.output_level = 0.0
            self._record_append(silence)
            return silence

        # Bypass guard — do not pass the dry signal through unprocessed.
        if self.safety_lock and not self.is_processing_active():
            silence = np.zeros_like(x)
            self.output_level = 0.0
            self._record_append(silence)
            return silence

        # High-pass / DC removal first — kills mic rumble + room hum.
        x = self.hpf.process(x)

        gated = self.gate.process(x)
        wet = self.vocoder.process(gated)

        if len(wet) != len(gated):
            if len(wet) > len(gated):
                wet = wet[:len(gated)]
            else:
                wet = np.concatenate([
                    wet, np.zeros(len(gated) - len(wet), dtype=np.float32)
                ])

        mixed = self.dry_wet * wet + (1.0 - self.dry_wet) * gated
        if self.output_gain_db != 0.0:
            mixed = mixed * (10.0 ** (self.output_gain_db / 20.0))

        out = self.limiter.process(mixed)
        self.output_level = float(np.sqrt(np.mean(out * out) + 1e-12))
        self._record_append(out)
        return out
