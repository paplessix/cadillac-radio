#!/usr/bin/env python3
"""
Script 2 — Interactive RLC Series Demodulator
==============================================
Loads the composite AM signal produced by generate_am_spectrum.py and lets you
tune a series RLC circuit to select one station at a time.

Physics
-------
Series RLC (voltage measured across R) acts as a 2nd-order bandpass filter:

        H(s) = (R/L) · s
               ─────────────────────────
               s² + (R/L)·s + 1/(LC)

Resonant frequency : f₀ = 1 / (2π √(LC))
Quality factor     : Q  = (1/R) √(L/C) = f₀/BW₃dB

Interactive controls
--------------------
  L slider — inductance  (µH)    shifts f₀
  C slider — capacitance (pF)    shifts f₀
  R slider — resistance  (Ω)     widens/narrows the passband (changes Q)
  "Save demodulated audio" button — writes data/demodulated.wav at 44.1 kHz

Usage
-----
  python rlc_demodulator.py [--wav data/am_composite.wav]
"""

import argparse
import tomllib
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ── Load config ───────────────────────────────────────────────────────────────
with open("stations.toml", "rb") as _f:
    _cfg = tomllib.load(_f)
STATION_FREQS  = [int(s["freq"]) for s in _cfg["station"]]
STATION_LABELS = [s["label"]     for s in _cfg["station"]]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Interactive RLC AM demodulator")
parser.add_argument("--wav", default="data/am_composite.wav",
                    help="Input composite WAV file (default: data/am_composite.wav)")
args = parser.parse_args()

# ── Load composite signal ─────────────────────────────────────────────────────
print(f"Loading {args.wav} …")
fs, x = wavfile.read(args.wav)
if x.ndim == 2:
    x = x[:, 0]
x = x.astype(np.float64)
x /= np.max(np.abs(x)) + 1e-12
t  = np.arange(len(x)) / fs
print(f"  Sample rate : {fs/1e3:.0f} kHz")
print(f"  Duration    : {len(x)/fs:.2f} s")

# ── Pre-compute spectrum for background display ───────────────────────────────
nperseg  = 2 ** 16
freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, window="hann", scaling="density")
psd_db   = 10 * np.log10(psd + 1e-30)
freq_kHz = freqs / 1e3

# Band of interest
F_LO, F_HI = 80_000, 170_000        # Hz
mask = (freqs >= F_LO) & (freqs <= F_HI)

# ── Default RLC values ────────────────────────────────────────────────────────
# Target station C at 125 kHz → f₀ = 125 kHz
# Choose L = 1 mH, solve C = 1/(L·ω₀²)
L0_uH = 1000.0          # µH  = 1 mH
f_target = 125_000.0
C0_pF  = 1e12 / (L0_uH * 1e-6 * (2 * np.pi * f_target) ** 2)   # pF
R0     = 10.0            # Ω

# ── RLC digital filter ────────────────────────────────────────────────────────
def rlc_filter(x_in, L_uH, C_pF, R_ohm, fs):
    """Apply a series-RLC bandpass (voltage across R) to x_in."""
    L = L_uH * 1e-6
    C = C_pF * 1e-12
    omega0 = 1.0 / np.sqrt(L * C)
    f0     = omega0 / (2 * np.pi)
    bw     = R_ohm / L                    # rad/s bandwidth
    Q      = omega0 / bw

    # Guard: f0 must sit below Nyquist with margin
    f_ny = fs / 2.0
    if f0 >= 0.98 * f_ny or f0 <= 100:
        return np.zeros_like(x_in), f0, Q

    # 2nd-order bandpass IIR (iirpeak = unit-gain at f0)
    b, a = signal.iirpeak(f0, Q, fs)
    y    = signal.lfilter(b, a, x_in)
    return y, f0, Q

def filter_freq_response(L_uH, C_pF, R_ohm, fs, n=8192):
    """Return (freq_Hz, H_dB) for the RLC filter."""
    L = L_uH * 1e-6
    C = C_pF * 1e-12
    omega0 = 1.0 / np.sqrt(L * C)
    bw     = R_ohm / L
    Q      = omega0 / bw
    f0     = omega0 / (2 * np.pi)

    f_ny = fs / 2.0
    if f0 >= 0.98 * f_ny or f0 <= 100:
        freqs = np.linspace(F_LO, F_HI, n)
        return freqs, np.full(n, -60.0)

    b, a   = signal.iirpeak(f0, Q, fs)
    w, H   = signal.freqz(b, a, worN=n, fs=fs)
    H_dB   = 20 * np.log10(np.abs(H) + 1e-12)
    # Restrict to band of interest
    keep   = (w >= F_LO) & (w <= F_HI)
    return w[keep], H_dB[keep]

def envelope_detect(y_filtered, fs, audio_bw=5000):
    """Rectify and low-pass → AM envelope."""
    env = np.abs(y_filtered)
    sos = signal.butter(4, audio_bw / (fs / 2), btype="low", output="sos")
    env = signal.sosfilt(sos, env)
    # Remove DC
    env -= np.mean(env)
    peak = np.max(np.abs(env)) + 1e-12
    return env / peak

# ── Initial computation ───────────────────────────────────────────────────────
y_filt, f0_init, Q_init = rlc_filter(x, L0_uH, C0_pF, R0, fs)
env_init = envelope_detect(y_filt, fs)
fr_init, fH_init = filter_freq_response(L0_uH, C0_pF, R0, fs)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("#1a1a2e")

AX_SPEC    = fig.add_axes([0.07, 0.55, 0.88, 0.37])  # spectrum
AX_ENV     = fig.add_axes([0.07, 0.20, 0.88, 0.27])  # demodulated audio
AX_L       = fig.add_axes([0.15, 0.13, 0.70, 0.025])
AX_C       = fig.add_axes([0.15, 0.09, 0.70, 0.025])
AX_R       = fig.add_axes([0.15, 0.05, 0.70, 0.025])
AX_BTN     = fig.add_axes([0.78, 0.005, 0.18, 0.03])

for ax in [AX_SPEC, AX_ENV]:
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="0.8")
    ax.xaxis.label.set_color("0.8")
    ax.yaxis.label.set_color("0.8")
    ax.title.set_color("0.9")
    for sp in ax.spines.values():
        sp.set_edgecolor("0.4")

# ── Spectrum plot ─────────────────────────────────────────────────────────────
AX_SPEC.plot(freq_kHz[mask], psd_db[mask],
             color="#4fc3f7", linewidth=0.8, label="Composite signal")
fh_line, = AX_SPEC.plot(fr_init / 1e3,
                          fH_init + np.max(psd_db[mask]),  # overlay on PSD scale
                          color="#ff6b6b", linewidth=2.0, label="RLC response")
vline_spec = AX_SPEC.axvline(f0_init / 1e3, color="#ffd93d",
                              linewidth=1.0, linestyle="--")
title_spec = AX_SPEC.set_title(
    f"Power spectrum  |  f₀ = {f0_init/1e3:.2f} kHz  |  Q = {Q_init:.1f}",
    fontsize=11)
AX_SPEC.set_xlabel("Frequency (kHz)")
AX_SPEC.set_ylabel("PSD (dB/Hz)")
AX_SPEC.legend(loc="upper right", fontsize=8,
               facecolor="#1a1a2e", labelcolor="0.85", framealpha=0.8)

# Station markers (from stations.toml)
_cmap = plt.cm.tab10(np.linspace(0, 0.9, len(STATION_FREQS)))
ylim_spec = AX_SPEC.get_ylim()
for fc_hz, lab, col in zip(STATION_FREQS, STATION_LABELS, _cmap):
    fc = fc_hz / 1e3
    AX_SPEC.axvline(fc, color=col, linewidth=0.8, linestyle=":", alpha=0.6)
    AX_SPEC.text(fc, ylim_spec[1], f"{lab} {fc:.0f}k", color=col, fontsize=7,
                 ha="center", va="top")

# ── Demodulated audio plot ────────────────────────────────────────────────────
# Downsample preview to reduce drawing time
DS   = max(1, fs // 44100)
t_ds = t[::DS]
e_ds = env_init[::DS]

env_line, = AX_ENV.plot(t_ds, e_ds, color="#a29bfe", linewidth=0.6)
AX_ENV.set_xlabel("Time (s)")
AX_ENV.set_ylabel("Amplitude")
AX_ENV.set_title("Demodulated audio (envelope detected)")
AX_ENV.set_xlim(0, t[-1])
AX_ENV.set_ylim(-1.05, 1.05)

# ── Sliders ───────────────────────────────────────────────────────────────────
slider_kw = dict(color="#4fc3f7", track_color="#333355")

# L: 100 µH – 5000 µH (logarithmic feel, but linear widget)
sl_L = Slider(AX_L, "L (µH)", 100.0, 5000.0, valinit=L0_uH, **slider_kw)
# C: 1 pF – 5000 pF
sl_C = Slider(AX_C, "C (pF)", 1.0,   5000.0, valinit=C0_pF, **slider_kw)
# R: 1 Ω – 200 Ω
sl_R = Slider(AX_R, "R (Ω)",  1.0,   200.0,  valinit=R0,    **slider_kw)

for sl in [sl_L, sl_C, sl_R]:
    sl.label.set_color("0.85")
    sl.valtext.set_color("0.85")

# ── Update callback ───────────────────────────────────────────────────────────
def update(_):
    L = sl_L.val
    C = sl_C.val
    R = sl_R.val

    y_f, f0, Q = rlc_filter(x, L, C, R, fs)
    env         = envelope_detect(y_f, fs)
    fr, fH      = filter_freq_response(L, C, R, fs)

    # Scale RLC response to overlay on PSD
    psd_max = np.max(psd_db[mask])
    fH_scaled = fH + psd_max

    fh_line.set_xdata(fr / 1e3)
    fh_line.set_ydata(fH_scaled)
    vline_spec.set_xdata([f0 / 1e3, f0 / 1e3])
    AX_SPEC.set_title(
        f"Power spectrum  |  f₀ = {f0/1e3:.2f} kHz  |  Q = {Q:.1f}",
        fontsize=11, color="0.9")

    e_ds_new = env[::DS]
    env_line.set_ydata(e_ds_new)

    update._last = (env, f0, Q)
    fig.canvas.draw_idle()

update._last = (env_init, f0_init, Q_init)
sl_L.on_changed(update)
sl_C.on_changed(update)
sl_R.on_changed(update)

# ── Save button ───────────────────────────────────────────────────────────────
btn = Button(AX_BTN, "Save demodulated", color="#2d3436", hovercolor="#636e72")
btn.label.set_color("white")

def save_audio(_):
    env, f0, _ = update._last
    # Downsample to 44100 Hz
    fs_out   = 44_100
    n_out    = int(len(env) * fs_out / fs)
    env_out  = signal.resample(env, n_out).astype(np.float32)
    out_path = "data/demodulated.wav"
    wavfile.write(out_path, fs_out, env_out)
    print(f"Saved: {out_path}  (f₀ = {f0/1e3:.2f} kHz, 44.1 kHz, float32)")

btn.on_clicked(save_audio)

# ── Info text ─────────────────────────────────────────────────────────────────
fig.text(0.07, 0.165,
         "Tune L and C to shift the resonant frequency onto a station carrier.  "
         "Decrease R to narrow the passband (raise Q) and reject adjacent stations.",
         color="0.6", fontsize=8)

plt.suptitle("RLC Series Demodulator — Interactive AM Tuner",
             fontsize=13, color="0.95", y=0.98)

print("\nInteractive window open.  Adjust sliders to tune the RLC circuit.")
print("Keyboard: q / Ctrl-W to close.")
plt.show()
