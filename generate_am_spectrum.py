#!/usr/bin/env python3
"""
Script 1 — AM Spectrum Composer
================================
Reads stations.toml, auto-discovers MP3 files in data/, converts stereo to
mono, aligns all sources to the longest MP3 duration (looping shorter ones),
AM-modulates each on its carrier, sums them and adds AWGN noise.

Output
------
  data/am_composite.wav   — composite RF signal at fs_sim Hz (float32)
  data/am_composite.png   — power-spectrum plot

Configuration
-------------
  Edit stations.toml to add / remove stations or change simulation parameters.
"""

import glob
import json
import tomllib
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pydub import AudioSegment

# ── Load configuration ────────────────────────────────────────────────────────
with open("stations.toml", "rb") as f:
    cfg = tomllib.load(f)

sim      = cfg["simulation"]
FS_SIM   = int(sim["fs_sim"])
MAX_DUR  = float(sim["max_duration"])
MOD_IDX  = float(sim["mod_index"])
SNR_DB   = float(sim["snr_db"])
AUDIO_BW = float(sim["audio_bw"])
STATIONS = cfg["station"]           # list of dicts

# ── Discover and load MP3s (stereo → mono) ────────────────────────────────────
print("Scanning data/ for MP3 files …")
mp3_paths = sorted(glob.glob("data/*.mp3"))
if not mp3_paths:
    raise FileNotFoundError("No .mp3 files found in data/")

recordings = []
for path in mp3_paths:
    print(f"  Loading {Path(path).name} …", end=" ", flush=True)
    seg = AudioSegment.from_mp3(path)
    raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels == 2:
        raw = raw.reshape(-1, 2).mean(axis=1)   # stereo → mono
    else:
        raw = raw.flatten()
    raw /= 32768.0
    dur = len(raw) / seg.frame_rate
    print(f"{dur:.1f} s  {'stereo→mono' if seg.channels==2 else 'mono'}")
    recordings.append({"name": Path(path).stem, "samples": raw, "fs": seg.frame_rate})

# ── Determine master duration ─────────────────────────────────────────────────
longest_s = max(len(r["samples"]) / r["fs"] for r in recordings)
duration  = min(longest_s, MAX_DUR)
n_sim     = int(FS_SIM * duration)
print(f"\nLongest MP3 : {longest_s:.1f} s")
print(f"Master dur  : {duration:.1f} s  (cap={MAX_DUR:.0f} s)")
print(f"Sim samples : {n_sim:,}  @{FS_SIM//1000} kHz\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def loop_to_n(x, n):
    """Repeat x until it has at least n samples, then trim."""
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]

def resample_audio(x, fs_in, fs_out, bw):
    n_out = int(len(x) * fs_out / fs_in)
    x_r   = signal.resample(x, n_out)
    sos   = signal.butter(8, bw / (fs_out / 2), btype="low", output="sos")
    return signal.sosfilt(sos, x_r)

t = np.arange(n_sim) / FS_SIM

def make_audio(source_key, auto_idx_ref):
    """Return audio array of length n_sim at FS_SIM Hz."""

    if source_key == "auto" or source_key.endswith(".mp3"):
        # Pick next unassigned recording, or the named file
        if source_key == "auto":
            idx = auto_idx_ref[0]
            auto_idx_ref[0] += 1
            if idx >= len(recordings):
                # Fall back to pink noise if we run out of recordings
                source_key = "pink"
            else:
                rec = recordings[idx]
        else:
            match = [r for r in recordings if r["name"] + ".mp3" == source_key
                     or Path(source_key).stem == r["name"]]
            if not match:
                raise FileNotFoundError(f"Recording not found: {source_key}")
            rec = match[0]

        if source_key != "pink":
            n_native = int(rec["fs"] * duration)
            seg      = loop_to_n(rec["samples"], n_native)
            audio    = resample_audio(seg, rec["fs"], FS_SIM, AUDIO_BW)
            return audio[:n_sim]

    if source_key == "tone":
        return (0.6 * np.sin(2 * np.pi * 440 * t)
              + 0.4 * np.sin(2 * np.pi * 880 * t))

    if source_key == "pink":
        white = np.random.default_rng(42).standard_normal(n_sim)
        sos   = signal.butter(2,
                              [20 / (FS_SIM / 2), AUDIO_BW / (FS_SIM / 2)],
                              btype="band", output="sos")
        pink  = signal.sosfilt(sos, white)
        pink /= np.max(np.abs(pink)) + 1e-12
        return pink

    raise ValueError(f"Unknown source: '{source_key}'")

# ── AM modulate and sum ───────────────────────────────────────────────────────
print("Generating composite AM signal …")
composite   = np.zeros(n_sim)
auto_cursor = [0]          # mutable counter passed by reference into make_audio

for st in STATIONS:
    src   = st["source"]
    freq  = int(st["freq"])
    label = st["label"]

    audio = make_audio(src, auto_cursor)
    peak  = np.max(np.abs(audio)) + 1e-12
    audio = audio / peak

    carrier   = np.cos(2 * np.pi * freq * t)
    composite += (1.0 + MOD_IDX * audio) * carrier

    src_desc = recordings[auto_cursor[0]-1]["name"] \
               if src == "auto" and auto_cursor[0] > 0 and src == "auto" \
               else src
    print(f"  {label} — {freq/1e3:.0f} kHz  [{src}]")

composite /= len(STATIONS)

# ── Add AWGN noise ────────────────────────────────────────────────────────────
sig_pwr   = np.mean(composite ** 2)
noise_pwr = sig_pwr / (10 ** (SNR_DB / 10))
composite += np.random.default_rng(0).standard_normal(n_sim) * np.sqrt(noise_pwr)

print(f"\nSNR target : {SNR_DB} dB  |  "
      f"signal RMS {np.sqrt(sig_pwr):.4f}  |  "
      f"noise RMS {np.sqrt(noise_pwr):.4f}")

# ── Save WAV ──────────────────────────────────────────────────────────────────
out_wav = "data/am_composite.wav"
composite_norm = composite / (np.max(np.abs(composite)) + 1e-12)
wavfile.write(out_wav, FS_SIM, (composite_norm * 32767).astype(np.int16))
print(f"\nSaved: {out_wav}  ({FS_SIM/1e3:.0f} kHz, int16, {duration:.1f} s)")

# ── Plot power spectrum ───────────────────────────────────────────────────────
print("Plotting spectrum …")
nperseg    = 2 ** 16
freqs, psd = signal.welch(composite, fs=FS_SIM, nperseg=nperseg,
                           window="hann", scaling="density")

f_lo, f_hi = 80_000, 170_000
mask    = (freqs >= f_lo) & (freqs <= f_hi)
f_kHz   = freqs[mask] / 1e3
p_db    = 10 * np.log10(psd[mask] + 1e-30)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(f_kHz, p_db, color="steelblue", linewidth=0.8)
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("PSD (dB/Hz)")
ax.set_title(f"Composite AM spectrum — {len(STATIONS)} stations + AWGN  "
             f"({duration:.1f} s @ {FS_SIM//1000} kHz)")
ax.grid(True, alpha=0.3)

colors = plt.cm.tab10(np.linspace(0, 0.9, len(STATIONS)))
ylim   = ax.get_ylim()
for st, c in zip(STATIONS, colors):
    fc = int(st["freq"]) / 1e3
    ax.axvline(fc, color=c, linewidth=1.2, linestyle="--", alpha=0.85)
    ax.text(fc, ylim[1], f"{st['label']} {fc:.0f}k",
            color=c, fontsize=7.5, ha="center", va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

plt.tight_layout()
out_png = "data/am_composite.png"
plt.savefig(out_png, dpi=150)
plt.show()
print(f"Saved: {out_png}")

# ── Save spectrum.json for the TypeScript demodulator ─────────────────────────
# Subsample to ~2000 points so the JSON stays small
stride  = max(1, len(f_kHz) // 2000)
spec_json = {
    "freqsKHz": f_kHz[::stride].tolist(),
    "psdDb":    p_db[::stride].tolist(),
    "stations": [{"label": st["label"], "freq": int(st["freq"])} for st in STATIONS],
}
out_json = "data/spectrum.json"
with open(out_json, "w") as f:
    json.dump(spec_json, f, separators=(",", ":"))
print(f"Saved: {out_json}  ({len(spec_json['freqsKHz'])} points)")
