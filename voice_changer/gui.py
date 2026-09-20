"""tkinter GUI for the voice changer.

All audio processing happens in sounddevice's PortAudio threads.
The GUI thread:
    - sets pipeline parameters (atomic float assignment, GIL safe enough)
    - polls input/output level meters via after()
    - starts/stops the AudioEngine
    - persists settings on close
"""

import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd

from . import audio, presets, settings


class VoiceChangerGUI:
    METER_INTERVAL_MS = 50

    def __init__(self, pipeline, engine):
        self.pipeline = pipeline
        self.engine = engine
        self._saved = settings.load()

        self.root = tk.Tk()
        self.root.title("Voice Changer")
        self.root.geometry("580x680")
        self.root.minsize(540, 660)

        self._build()
        self._populate_devices()
        self._restore_settings()
        self._tick_meters()

        self.root.bind_all("<space>", lambda e: self._toggle_mute())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        pad = dict(padx=10, pady=6)

        # --- Devices ---
        dev = ttk.LabelFrame(self.root, text="Devices")
        dev.pack(fill="x", **pad)
        ttk.Label(dev, text="Input (your mic):").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.in_combo = ttk.Combobox(dev, width=55, state="readonly")
        self.in_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Label(dev, text="Output (CABLE Input):").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.out_combo = ttk.Combobox(dev, width=55, state="readonly")
        self.out_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        dev.columnconfigure(1, weight=1)

        # --- Transport ---
        trans = ttk.Frame(self.root)
        trans.pack(fill="x", **pad)
        self.start_btn = ttk.Button(trans, text="Start", command=self._toggle_engine)
        self.start_btn.pack(side="left", padx=4)
        ttk.Button(trans, text="Refresh devices",
                   command=self._populate_devices).pack(side="left", padx=4)
        ttk.Button(trans, text="Test playback (5s)",
                   command=self._test_playback).pack(side="left", padx=4)
        self.status_label = ttk.Label(trans, text="stopped", foreground="grey")
        self.status_label.pack(side="left", padx=12)

        # --- Mute (Panic) ---
        mute_frame = ttk.Frame(self.root)
        mute_frame.pack(fill="x", **pad)
        self.mute_btn = tk.Button(
            mute_frame, text="MUTE  (Space)",
            command=self._toggle_mute,
            bg="#3a7", fg="white", activebackground="#2a5",
            font=("Segoe UI", 12, "bold"), height=2, relief="raised", bd=2,
        )
        self.mute_btn.pack(fill="x", padx=4)

        # --- Controls ---
        ctrl = ttk.LabelFrame(self.root, text="Controls")
        ctrl.pack(fill="x", **pad)

        self.pitch_var   = tk.DoubleVar(value=0.0)
        self.formant_var = tk.DoubleVar(value=1.0)
        self.gate_var    = tk.DoubleVar(value=-50.0)
        self.gain_var    = tk.DoubleVar(value=0.0)
        self.drywet_var  = tk.DoubleVar(value=1.0)

        self._pitch_label   = self._slider(ctrl, 0, "Pitch (semitones)", -12, 12,
                                           self.pitch_var,   self._on_pitch)
        self._formant_label = self._slider(ctrl, 1, "Formant",            0.5, 1.7,
                                           self.formant_var, self._on_formant, fmt="{:.2f}")
        self._gate_label    = self._slider(ctrl, 2, "Noise gate (dB)",   -70, 0,
                                           self.gate_var,    self._on_gate)
        self._gain_label    = self._slider(ctrl, 3, "Output gain (dB)",  -12, 12,
                                           self.gain_var,    self._on_gain)
        self._drywet_label  = self._slider(ctrl, 4, "Dry/Wet",             0, 1,
                                           self.drywet_var,  self._on_drywet, fmt="{:.2f}")

        self.safety_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl, text="Bypass guard: silence output when no shift is applied",
            variable=self.safety_var, command=self._on_safety,
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=4)

        ctrl.columnconfigure(1, weight=1)

        # --- Presets ---
        pre = ttk.LabelFrame(self.root, text="Presets")
        pre.pack(fill="x", **pad)
        names = list(presets.PRESETS.keys())
        cols = 3
        for i, name in enumerate(names):
            ttk.Button(pre, text=name,
                       command=lambda n=name: self._apply_preset(n)
                       ).grid(row=i // cols, column=i % cols,
                              sticky="ew", padx=4, pady=4)
        for c in range(cols):
            pre.columnconfigure(c, weight=1)

        # --- Meters ---
        met = ttk.LabelFrame(self.root, text="Levels")
        met.pack(fill="x", **pad)
        ttk.Label(met, text="In ").grid(row=0, column=0, padx=6, pady=2)
        self.in_meter = ttk.Progressbar(met, length=420, maximum=100)
        self.in_meter.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(met, text="Out").grid(row=1, column=0, padx=6, pady=2)
        self.out_meter = ttk.Progressbar(met, length=420, maximum=100)
        self.out_meter.grid(row=1, column=1, sticky="ew", padx=6, pady=2)
        met.columnconfigure(1, weight=1)

    def _slider(self, parent, row, label, mn, mx, var, callback, fmt="{:.1f}"):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           padx=6, pady=3)
        val_label = ttk.Label(parent, text=fmt.format(var.get()), width=7,
                              anchor="e")
        val_label.grid(row=row, column=2, padx=6, pady=3)

        def on_change(v):
            f = float(v)
            val_label.configure(text=fmt.format(f))
            callback(f)

        scale = ttk.Scale(parent, from_=mn, to=mx, variable=var,
                          orient="horizontal", command=on_change)
        scale.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        return val_label

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------

    def _populate_devices(self):
        ins = audio.list_input_devices()
        outs = audio.list_output_devices()

        self._in_map = {f"[{i}] {d['name']}": i for i, d in ins}
        self._out_map = {f"[{i}] {d['name']}": i for i, d in outs}

        self.in_combo['values'] = list(self._in_map.keys())
        self.out_combo['values'] = list(self._out_map.keys())

        if self._in_map and not self.in_combo.get():
            self.in_combo.current(0)

        if self._out_map and not self.out_combo.get():
            preferred = audio.find_default_virtual_cable(outs)
            if preferred is not None:
                for label, idx in self._out_map.items():
                    if idx == preferred:
                        self.out_combo.set(label)
                        break
            else:
                self.out_combo.current(0)

    # ------------------------------------------------------------------
    # Transport / mute / test
    # ------------------------------------------------------------------

    def _toggle_engine(self):
        if self.engine.running:
            self.engine.stop()
            self.start_btn.configure(text="Start")
            self.status_label.configure(text="stopped", foreground="grey")
            return

        in_sel = self.in_combo.get()
        out_sel = self.out_combo.get()
        if not in_sel or not out_sel:
            self.status_label.configure(text="pick devices first",
                                        foreground="red")
            return
        try:
            self.engine.start(self._in_map[in_sel], self._out_map[out_sel])
            self.start_btn.configure(text="Stop")
            self.status_label.configure(text="running", foreground="green")
        except Exception as e:
            self.status_label.configure(text=f"error: {e}", foreground="red")

    def _toggle_mute(self):
        self.pipeline.mute = not self.pipeline.mute
        if self.pipeline.mute:
            self.mute_btn.configure(text="MUTED, press Space to unmute",
                                    bg="#c33", activebackground="#a22")
        else:
            self.mute_btn.configure(text="MUTE  (Space)",
                                    bg="#3a7", activebackground="#2a5")

    def _test_playback(self):
        """Play back the last 5 seconds of processed output on the system
        default speaker, NOT through CABLE, so you can hear yourself."""
        clip = self.pipeline.get_recent(5.0)
        if np.max(np.abs(clip)) < 1e-4:
            self.status_label.configure(
                text="no audio captured yet, start engine and talk first",
                foreground="orange")
            return

        def _play():
            try:
                sd.play(clip, samplerate=self.pipeline.sr)
                sd.wait()
            except Exception as e:
                self.status_label.configure(text=f"playback error: {e}",
                                            foreground="red")
        threading.Thread(target=_play, daemon=True).start()
        self.status_label.configure(
            text="playing 5s of processed audio...", foreground="blue")

    # ------------------------------------------------------------------
    # Param callbacks
    # ------------------------------------------------------------------

    def _on_pitch(self, v):     self.pipeline.vocoder.set_pitch_semitones(v)
    def _on_formant(self, v):   self.pipeline.vocoder.set_formant(v)
    def _on_gate(self, v):      self.pipeline.gate.set_threshold_db(v)
    def _on_gain(self, v):      self.pipeline.output_gain_db = v
    def _on_drywet(self, v):    self.pipeline.dry_wet = v
    def _on_safety(self):       self.pipeline.safety_lock = self.safety_var.get()

    def _apply_preset(self, name):
        p = presets.PRESETS[name]
        self.pitch_var.set(p['pitch']);     self._on_pitch(p['pitch'])
        self.formant_var.set(p['formant']); self._on_formant(p['formant'])
        self.gate_var.set(p['gate_db']);    self._on_gate(p['gate_db'])
        self.gain_var.set(p['gain_db']);    self._on_gain(p['gain_db'])
        self.drywet_var.set(p['dry_wet']);  self._on_drywet(p['dry_wet'])

        self._pitch_label.configure(text=f"{p['pitch']:.1f}")
        self._formant_label.configure(text=f"{p['formant']:.2f}")
        self._gate_label.configure(text=f"{p['gate_db']:.1f}")
        self._gain_label.configure(text=f"{p['gain_db']:.1f}")
        self._drywet_label.configure(text=f"{p['dry_wet']:.2f}")

        self._saved["last_preset"] = name

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _restore_settings(self):
        s = self._saved

        # Devices: match by name if still present.
        for label in self._in_map:
            if s.get("input_device_name") and s["input_device_name"] in label:
                self.in_combo.set(label); break
        for label in self._out_map:
            if s.get("output_device_name") and s["output_device_name"] in label:
                self.out_combo.set(label); break

        self.pitch_var.set(s["pitch"]);     self._on_pitch(s["pitch"])
        self.formant_var.set(s["formant"]); self._on_formant(s["formant"])
        self.gate_var.set(s["gate_db"]);    self._on_gate(s["gate_db"])
        self.gain_var.set(s["gain_db"]);    self._on_gain(s["gain_db"])
        self.drywet_var.set(s["dry_wet"]);  self._on_drywet(s["dry_wet"])
        self.safety_var.set(s["safety_lock"]); self._on_safety()

        self._pitch_label.configure(text=f"{s['pitch']:.1f}")
        self._formant_label.configure(text=f"{s['formant']:.2f}")
        self._gate_label.configure(text=f"{s['gate_db']:.1f}")
        self._gain_label.configure(text=f"{s['gain_db']:.1f}")
        self._drywet_label.configure(text=f"{s['dry_wet']:.2f}")

    def _persist_settings(self):
        in_label = self.in_combo.get()
        out_label = self.out_combo.get()
        # Strip the "[idx] " prefix so device-index changes don't break matching
        in_name = in_label.split("] ", 1)[1] if "] " in in_label else in_label
        out_name = out_label.split("] ", 1)[1] if "] " in out_label else out_label

        settings.save({
            "input_device_name": in_name,
            "output_device_name": out_name,
            "pitch": float(self.pitch_var.get()),
            "formant": float(self.formant_var.get()),
            "gate_db": float(self.gate_var.get()),
            "gain_db": float(self.gain_var.get()),
            "dry_wet": float(self.drywet_var.get()),
            "safety_lock": bool(self.safety_var.get()),
            "last_preset": self._saved.get("last_preset", ""),
        })

    # ------------------------------------------------------------------
    # Meters
    # ------------------------------------------------------------------

    def _tick_meters(self):
        self.in_meter['value'] = self._rms_to_pct(self.pipeline.input_level)
        self.out_meter['value'] = self._rms_to_pct(self.pipeline.output_level)
        self.root.after(self.METER_INTERVAL_MS, self._tick_meters)

    @staticmethod
    def _rms_to_pct(rms: float) -> float:
        if rms <= 1e-6:
            return 0.0
        db = 20.0 * np.log10(rms)
        return float(max(0.0, min(100.0, (db + 60.0) * 100.0 / 60.0)))

    # ------------------------------------------------------------------

    def _on_close(self):
        try:
            self._persist_settings()
        finally:
            self.engine.stop()
            self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.engine.stop()
