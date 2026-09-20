"""Preset parameter sets. Each is a dict matching the GUI control names.

Tuning notes:
    - Smaller pitch and formant shifts sound cleaner. Large shifts cost
      naturalness because the phase vocoder has more to reconstruct.
    - Pitch is in semitones; formant is a spectral-envelope scale factor.
      Shifting both together keeps the result plausible as a human voice;
      pitch alone produces the classic chipmunk/monster artifact.
    - gate_db is the noise-gate threshold, gain_db the output trim, and
      dry_wet the processed/unprocessed blend.

There is no bypass preset by design; use MUTE (or spacebar) to silence the
output instead of routing the dry signal through.
"""

PRESETS = {
    # Gentlest shift, cleanest reconstruction.
    "Natural":        dict(pitch=-0.7, formant=0.97, gate_db=-52, gain_db=0, dry_wet=1.0),
    "Subtle shift":   dict(pitch=-1.2, formant=0.95, gate_db=-52, gain_db=0, dry_wet=1.0),
    "Deep":           dict(pitch=-4.0, formant=0.84, gate_db=-50, gain_db=0, dry_wet=1.0),
    "Very deep":      dict(pitch=-7.0, formant=0.74, gate_db=-50, gain_db=2, dry_wet=1.0),
    "Higher":         dict(pitch= 3.5, formant=1.16, gate_db=-50, gain_db=0, dry_wet=1.0),
    "Very high":      dict(pitch= 6.5, formant=1.28, gate_db=-50, gain_db=0, dry_wet=1.0),
    "Muffled":        dict(pitch=-1.0, formant=0.85, gate_db=-45, gain_db=0, dry_wet=1.0),
}
