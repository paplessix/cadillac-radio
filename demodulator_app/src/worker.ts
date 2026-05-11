/**
 * DSP Web Worker — IQ demodulation
 *
 * Replaces the original BPF + rectify pipeline with IQ (complex) downconversion.
 *
 * Why IQ?
 * A real biquad BPF centred at f0 has identical gain at its digital mirror
 * frequency (fs/2 − f0).  At fs=500 kHz, tuning to 107 kHz produces an equal
 * response at 143 kHz and vice-versa — audible aliasing tones.
 * IQ mixing multiplies by e^{-j2πf0n/fs}, which shifts f0 to DC and moves the
 * mirror to −2f0 (way out of band), so a simple lowpass on I and Q is all that
 * is needed for both selectivity and envelope detection.
 */

export interface DspRequest {
  samples:    Float32Array
  sampleRate: number   // 500 000
  f0:         number   // resonant frequency (Hz)
  audioBw:    number   // LP cutoff = half RF bandwidth = f0/(2Q) = R/(4πL)
}

export interface DspResponse {
  audio:      Float32Array  // 44 100 Hz
  sampleRate: number
}

// ── Biquad 2nd-order Butterworth LP (Audio EQ Cookbook) ────────────
function lpfCoeffs(fc: number, fs: number): [number[], number[]] {
  const w0     = 2 * Math.PI * fc / fs
  const alpha  = Math.sin(w0) / (2 * 0.7071)
  const a0     = 1 + alpha
  const cosw0  = Math.cos(w0)
  const b = [(1 - cosw0) / 2 / a0, (1 - cosw0) / a0, (1 - cosw0) / 2 / a0]
  const a = [1, -2 * cosw0 / a0, (1 - alpha) / a0]
  return [b, a]
}

// Direct Form II Transposed biquad
function biquad(x: Float32Array, b: number[], a: number[]): Float32Array {
  const y  = new Float32Array(x.length)
  let s1 = 0, s2 = 0
  const b0 = b[0], b1 = b[1], b2 = b[2], a1 = a[1], a2 = a[2]
  for (let i = 0; i < x.length; i++) {
    const xi = x[i], yi = b0 * xi + s1
    s1 = b1 * xi - a1 * yi + s2
    s2 = b2 * xi - a2 * yi
    y[i] = yi
  }
  return y
}

// ── IQ downconversion ──────────────────────────────────────────────
// Recursive complex oscillator: O(N) mults, renormalised every 4096 steps.
function iqMix(x: Float32Array, f0: number, fs: number): [Float32Array, Float32Array] {
  const N  = x.length
  const I  = new Float32Array(N)
  const Q  = new Float32Array(N)
  const w  = 2 * Math.PI * f0 / fs
  const dRe = Math.cos(w), dIm = -Math.sin(w)
  let re = 1, im = 0

  for (let i = 0; i < N; i++) {
    I[i] = x[i] * re
    Q[i] = x[i] * im
    const nr = re * dRe - im * dIm
    const ni = re * dIm + im * dRe
    re = nr; im = ni
    if ((i & 4095) === 4095) { const r = Math.sqrt(re*re + im*im); re /= r; im /= r }
  }
  return [I, Q]
}

// Integer decimation
function decimate(x: Float32Array, factor: number): Float32Array {
  const n = Math.floor(x.length / factor)
  const y = new Float32Array(n)
  for (let i = 0; i < n; i++) y[i] = x[i * factor]
  return y
}

// Linear-interpolation resample
function resampleLinear(x: Float32Array, inRate: number, outRate: number): Float32Array {
  const ratio = inRate / outRate
  const n = Math.floor(x.length / ratio)
  const y = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const pos = i * ratio, idx = Math.floor(pos), frac = pos - idx
    y[i] = x[idx] * (1 - frac) + x[Math.min(idx + 1, x.length - 1)] * frac
  }
  return y
}

function removeDC(x: Float32Array): Float32Array {
  let mean = 0
  for (let i = 0; i < x.length; i++) mean += x[i]
  mean /= x.length
  const y = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) y[i] = x[i] - mean
  return y
}

function normalise(x: Float32Array): Float32Array {
  let peak = 0
  for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a }
  if (peak < 1e-9) return x
  const y = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) y[i] = x[i] / peak
  return y
}

// ── Main pipeline ──────────────────────────────────────────────────
self.onmessage = (e: MessageEvent<DspRequest>) => {
  const { samples, sampleRate, f0, audioBw } = e.data
  const fs = sampleRate

  // 1. IQ downconvert — moves f0→DC, mirror→−2f0 (eliminated by LP below)
  const [I, Q] = iqMix(samples, f0, fs)

  // 2. LP on both channels at audioBw  →  acts as BPF centred at f0
  const [bLP, aLP] = lpfCoeffs(audioBw, fs)
  const I_lp = biquad(I, bLP, aLP)
  const Q_lp = biquad(Q, bLP, aLP)

  // 3. Envelope = |I + jQ|
  const envelope = new Float32Array(samples.length)
  for (let i = 0; i < envelope.length; i++)
    envelope[i] = Math.sqrt(I_lp[i] ** 2 + Q_lp[i] ** 2)

  // 4. Anti-alias + decimate 500 kHz → 50 kHz → 5 kHz
  const bwAA1 = Math.min(audioBw * 3, 22_000)
  const [b1, a1] = lpfCoeffs(bwAA1, fs)
  const dec1 = decimate(biquad(envelope, b1, a1), 10)       // 50 kHz

  const bwAA2 = Math.min(audioBw * 1.5, 2_000)
  const [b2, a2] = lpfCoeffs(bwAA2, fs / 10)
  const dec2 = decimate(biquad(dec1, b2, a2), 10)           // 5 kHz

  // 5. Resample 5 kHz → 44 100 Hz
  const audio44k = resampleLinear(dec2, fs / 100, 44100)

  // 6. Remove DC, normalise
  const cleaned = normalise(removeDC(audio44k))

  self.postMessage({ audio: cleaned, sampleRate: 44100 } satisfies DspResponse, {
    transfer: [cleaned.buffer],
  })
}
