# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An AM radio simulation and demodulation project with two layers:

1. **Python pipeline** — generates a composite AM signal from MP3 files and a synthetic RF spectrum
2. **TypeScript browser app** — an interactive RLC-circuit AM tuner that demodulates the generated signal in real time

## Commands

### Python scripts (run from repo root)

```bash
# Generate composite AM signal (reads stations.toml + data/*.mp3)
python generate_am_spectrum.py
# Outputs: data/am_composite.wav, data/am_composite.png, data/spectrum.json

# Interactive RLC demodulator (matplotlib GUI)
python rlc_demodulator.py
python rlc_demodulator.py --wav data/am_composite.wav
```

Python deps: `numpy`, `scipy`, `matplotlib`, `pydub`.

### Browser app (demodulator_app/)

```bash
cd demodulator_app
npm install
npm run dev       # dev server at localhost:5173
npm run build     # tsc + vite build → dist/
npm run preview   # serve dist/
```

The dev/preview server must serve files from `demodulator_app/dist/` (or the Vite root) because `main.ts` fetches `/spectrum.json` and `/am_composite.wav` at runtime. Run `generate_am_spectrum.py` first so those files exist in `data/`; Vite's `vite.config.ts` should alias or copy them, or run `npm run build` which copies them to `dist/`.

## Architecture

### Data flow

```
stations.toml + data/*.mp3
        │
generate_am_spectrum.py
        │
        ├─▶ data/am_composite.wav   (500 kHz, float32 — composite RF signal)
        ├─▶ data/am_composite.png   (power-spectrum plot)
        └─▶ data/spectrum.json      (downsampled PSD + station list for the UI)
                │
        demodulator_app/
                │
        main.ts  ──fetch──▶  /spectrum.json   (renders spectrum canvas)
                └──fetch──▶  /am_composite.wav (streams with progress bar)
                                    │
                             worker.ts (Web Worker)
                             DSP pipeline: biquad BPF → rectify → IIR LP
                             → decimate 500k→5k → resample→44.1k
                                    │
                             AudioContext playback
```

### Key files

| File | Role |
|---|---|
| `stations.toml` | Single source of truth for carrier frequencies, station labels, audio sources, and simulation parameters |
| `generate_am_spectrum.py` | AM modulator + spectrum composer; controlled entirely by `stations.toml` |
| `rlc_demodulator.py` | Matplotlib interactive tuner (Python-only, no browser required) |
| `demodulator_app/src/main.ts` | UI, canvas rendering, slider events, Web Audio playback |
| `demodulator_app/src/worker.ts` | All DSP math runs off the main thread; receives `DspRequest`, returns `DspResponse` via `postMessage` with buffer transfer |

### RLC physics

Resonant frequency: `f₀ = 1 / (2π √(LC))`  
Quality factor: `Q = (1/R) √(L/C) = f₀ / BW₃dB`

Sliders: L (µH, 100–5000), C (pF, 1–5000), R (Ω, 1–200). Station-preset buttons fix L=1 mH and solve for C. Clicking the spectrum canvas also tunes C directly.

### Simulation parameters (stations.toml)

- `fs_sim = 500_000` — RF sample rate in Hz
- `mod_index` — AM modulation depth (0–1)
- `snr_db` — AWGN noise level
- `audio_bw` — per-station audio bandwidth in Hz (also hardcoded as `AUDIO_BW = 4500` in `main.ts`)

Station `source` can be `"auto"` (next unassigned MP3), a filename like `"foo.mp3"`, `"tone"` (440+880 Hz), or `"pink"` (band-limited noise).
