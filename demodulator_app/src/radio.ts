import type { DspRequest, DspResponse } from './worker'

// ── Types ──────────────────────────────────────────────────────────
interface Station      { label: string; freq: number }
interface SpectrumData { freqsKHz: number[]; psdDb: number[]; stations: Station[] }

// ── Physical component ranges (logarithmic knobs) ──────────────────
// L: inductance µH
const L_MIN = 200,    L_MAX = 5_000,   L_INIT = 1_000
// C: capacitance pF
const C_MIN = 200,    C_MAX = 10_000,  C_INIT = 1_621
// R: resistance Ω — controls bandwidth = R/(4πL)
const R_MIN = 1,      R_MAX = 500,     R_INIT = 25

// Dial frequency bounds (Hz)
const F_MIN = 80_000
const F_MAX = 170_000

const KNOB_DEG = 270

// ── Component → frequency / bandwidth ─────────────────────────────
const toH  = (uH: number) => uH * 1e-6
const toF  = (pF: number) => pF * 1e-12
const getF0      = (L: number, C: number) => 1 / (2 * Math.PI * Math.sqrt(toH(L) * toF(C)))
const getAudioBw = (R: number, L: number) => Math.min(R / (4 * Math.PI * toH(L)), 4_500)

// ── Knob angle ↔ log-scale value ──────────────────────────────────
function valToAngle(v: number, min: number, max: number): number {
  const t = (Math.log10(v) - Math.log10(min)) / (Math.log10(max) - Math.log10(min))
  return (t - 0.5) * KNOB_DEG
}
function angleToVal(deg: number, min: number, max: number): number {
  const t = deg / KNOB_DEG + 0.5
  return Math.pow(10, Math.log10(min) + t * (Math.log10(max) - Math.log10(min)))
}

// ── Reusable rotary knob ───────────────────────────────────────────
class RotaryKnob {
  readonly el: HTMLElement
  private angle: number
  private dragging = false
  private startY   = 0
  private startAng = 0
  onChange: (angle: number) => void = () => {}

  constructor(el: HTMLElement, initAngle: number, private sensitivity = 0.75) {
    this.el = el; this.angle = initAngle
    el.style.transform = `rotate(${initAngle}deg)`
    el.addEventListener('mousedown',  e => this.down(e.clientY, e))
    window.addEventListener('mousemove',  e => this.move(e.clientY))
    window.addEventListener('mouseup',    () => { this.dragging = false })
    el.addEventListener('touchstart', e => this.down(e.touches[0].clientY, e), { passive: false })
    window.addEventListener('touchmove',  e => { if (this.dragging) this.move(e.touches[0].clientY) })
    window.addEventListener('touchend',   () => { this.dragging = false })
  }

  private down(y: number, e: Event) {
    this.dragging = true; this.startY = y; this.startAng = this.angle
    e.preventDefault()
  }
  private move(y: number) {
    if (!this.dragging) return
    this.set(this.startAng + (this.startY - y) * this.sensitivity)
  }
  set(deg: number) {
    const half = KNOB_DEG / 2
    this.angle = Math.max(-half, Math.min(half, deg))
    this.el.style.transform = `rotate(${this.angle}deg)`
    this.onChange(this.angle)
  }
  get(): number { return this.angle }
}

// ── Dial canvas — frequency scale only, no station markers ────────
function drawDial(canvas: HTMLCanvasElement, f0: number) {
  const dpr = devicePixelRatio || 1
  const W   = canvas.offsetWidth
  const H   = canvas.offsetHeight
  if (!W || !H) return

  const cw = Math.round(W * dpr), ch = Math.round(H * dpr)
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch }

  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, cw, ch)
  ctx.save(); ctx.scale(dpr, dpr)

  const fx = (f: number) => ((f - F_MIN) / (F_MAX - F_MIN)) * W

  // Minor ticks
  ctx.strokeStyle = 'rgba(80,50,10,.38)'; ctx.lineWidth = 1
  for (const k of [85, 95, 105, 115, 125, 135, 145, 155, 165]) {
    const x = fx(k * 1000)
    ctx.beginPath(); ctx.moveTo(x, H * .26); ctx.lineTo(x, H * .44); ctx.stroke()
  }

  // Major ticks + labels
  ctx.strokeStyle = 'rgba(70,42,8,.68)'; ctx.lineWidth = 1.5
  ctx.font = `${Math.round(H * .165)}px 'Cutive Mono', monospace`
  ctx.fillStyle = 'rgba(68,40,7,.72)'; ctx.textAlign = 'center'; ctx.textBaseline = 'top'
  for (const k of [80, 90, 100, 110, 120, 130, 140, 150, 160, 170]) {
    const x = fx(k * 1000)
    ctx.beginPath(); ctx.moveTo(x, H * .14); ctx.lineTo(x, H * .47); ctx.stroke()
    ctx.fillText(String(k), x, H * .55)
  }

  ctx.restore()
}

// ── WAV parser (PCM int16 or IEEE float32, 1 ch) ─────────────────
function parseWav(buf: ArrayBuffer): { samples: Float32Array; sampleRate: number } {
  const v = new DataView(buf)
  let pos = 12, audioFormat = 0, sampleRate = 0, bitsPerSample = 0, dataOffset = 0, dataBytes = 0
  while (pos < buf.byteLength - 8) {
    const id   = String.fromCharCode(v.getUint8(pos), v.getUint8(pos+1), v.getUint8(pos+2), v.getUint8(pos+3))
    const size = v.getUint32(pos + 4, true)
    pos += 8
    if (id === 'fmt ') {
      audioFormat   = v.getUint16(pos,      true)  // 1=PCM, 3=IEEE float
      sampleRate    = v.getUint32(pos + 4,  true)
      bitsPerSample = v.getUint16(pos + 14, true)
    } else if (id === 'data') { dataOffset = pos; dataBytes = size; break }
    pos += size
  }
  const sliced = buf.slice(dataOffset, dataOffset + dataBytes)
  if (audioFormat === 3 && bitsPerSample === 32) return { samples: new Float32Array(sliced), sampleRate }
  // PCM int16 → float32 [-1, 1]
  const pcm = new Int16Array(sliced)
  const out = new Float32Array(pcm.length)
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768.0
  return { samples: out, sampleRate }
}

// ── Main ───────────────────────────────────────────────────────────
async function main() {
  const knobLEl    = document.getElementById('knob-l')!
  const knobCEl    = document.getElementById('knob-c')!
  const knobREl    = document.getElementById('knob-r')!
  const dialCanvas = document.getElementById('dial') as HTMLCanvasElement
  const needleEl   = document.getElementById('needle')!
  const statusLine = document.getElementById('status-line')!

  // ── State ──────────────────────────────────────────────────────────
  let specData:      SpectrumData | null = null
  let rawSamples:    Float32Array | null = null
  let rawSampleRate  = 500_000
  let audioCtx:      AudioContext | null = null
  let audioBuffer:   AudioBuffer  | null = null
  let gainNode:      GainNode     | null = null
  let currentSource: AudioBufferSourceNode | null = null
  let playAnchorCtx  = 0    // audioCtx.currentTime at which signal t=0 was playing
  let debounce       = 0
  // Pending audio buffered before first user gesture unlocks AudioContext
  let pendingAudio:  { audio: Float32Array; sampleRate: number } | null = null

  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })

  // ── Three knobs: L, C, R ───────────────────────────────────────────
  const lKnob = new RotaryKnob(knobLEl, valToAngle(L_INIT, L_MIN, L_MAX), 0.65)
  const cKnob = new RotaryKnob(knobCEl, valToAngle(C_INIT, C_MIN, C_MAX))
  const rKnob = new RotaryKnob(knobREl, valToAngle(R_INIT, R_MIN, R_MAX), 0.65)

  const getL = () => angleToVal(lKnob.get(), L_MIN, L_MAX)
  const getC = () => angleToVal(cKnob.get(), C_MIN, C_MAX)
  const getR = () => angleToVal(rKnob.get(), R_MIN, R_MAX)

  // ── Needle & dial sync ─────────────────────────────────────────────
  function syncNeedle() {
    const f0  = getF0(getL(), getC())
    const pct = Math.max(0, Math.min(100, (f0 - F_MIN) / (F_MAX - F_MIN) * 100))
    needleEl.style.left = `${pct}%`
    drawDial(dialCanvas, f0)
  }

  // ── Playback ───────────────────────────────────────────────────────
  function startPlayback(resumeSecs: number) {
    if (!audioBuffer || !audioCtx || !gainNode) return
    currentSource?.stop(); currentSource = null
    if (audioCtx.state === 'suspended') audioCtx.resume()
    const src = audioCtx.createBufferSource()
    src.buffer = audioBuffer; src.loop = true; src.connect(gainNode)
    const off = resumeSecs % audioBuffer.duration
    src.start(0, off)
    currentSource  = src
    playAnchorCtx  = audioCtx.currentTime - off
  }

  // Called on first user gesture — creates AudioContext and plays pending audio
  function unlockAudio() {
    if (audioCtx) { if (audioCtx.state === 'suspended') audioCtx.resume(); return }
    audioCtx = new AudioContext()
    gainNode = audioCtx.createGain(); gainNode.gain.value = 1; gainNode.connect(audioCtx.destination)
    if (pendingAudio) {
      const { audio, sampleRate } = pendingAudio; pendingAudio = null
      const ab = audioCtx.createBuffer(1, audio.length, sampleRate)
      ab.copyToChannel(new Float32Array(audio.buffer as ArrayBuffer), 0)
      audioBuffer = ab
      startPlayback(0)
    }
  }
  document.addEventListener('mousedown',  unlockAudio, { once: true })
  document.addEventListener('touchstart', unlockAudio, { once: true })

  // ── DSP worker ─────────────────────────────────────────────────────
  worker.onmessage = (e: MessageEvent<DspResponse>) => {
    const { audio, sampleRate } = e.data
    if (!audioCtx) {
      // AudioContext not yet unlocked — hold audio until first user gesture
      pendingAudio = { audio, sampleRate }
    } else {
      if (!gainNode) { gainNode = audioCtx.createGain(); gainNode.gain.value = 1; gainNode.connect(audioCtx.destination) }
      const ab = audioCtx.createBuffer(1, audio.length, sampleRate)
      ab.copyToChannel(new Float32Array(audio.buffer as ArrayBuffer), 0)
      audioBuffer = ab
      const elapsed = playAnchorCtx > 0 ? audioCtx.currentTime - playAnchorCtx : 0
      startPlayback(Math.max(0, elapsed))
    }
    statusLine.textContent = `${(getF0(getL(), getC()) / 1000).toFixed(1)} kHz`
  }

  worker.onerror = e => { statusLine.textContent = `ERR ${e.message}` }

  function scheduleDemod() {
    clearTimeout(debounce)
    debounce = window.setTimeout(() => {
      if (!rawSamples) return
      const L = getL(), C = getC(), R = getR()
      const f0      = getF0(L, C)
      const audioBw = getAudioBw(R, L)
      if (f0 >= rawSampleRate / 2 - 1000 || f0 < 500) {
        statusLine.textContent = 'OUT OF RANGE'
        return
      }
      statusLine.textContent = 'PROCESSING…'
      const copy = new Float32Array(rawSamples)
      worker.postMessage(
        { samples: copy, sampleRate: rawSampleRate, f0, audioBw } satisfies DspRequest,
        { transfer: [copy.buffer] },
      )
    }, 340)
  }

  // All three knobs trigger needle sync + demod
  lKnob.onChange = () => { syncNeedle(); scheduleDemod() }
  cKnob.onChange = () => { syncNeedle(); scheduleDemod() }
  rKnob.onChange = () => { scheduleDemod() }

  new ResizeObserver(() => drawDial(dialCanvas, getF0(getL(), getC()))).observe(dialCanvas)

  // ── Boot ───────────────────────────────────────────────────────────
  statusLine.textContent = 'WARMING UP…'

  try {
    const r = await fetch(`${import.meta.env.BASE_URL}spectrum.json`)
    if (!r.ok) throw new Error('spectrum.json not found — run generate_am_spectrum.py first')
    specData = await r.json() as SpectrumData
    requestAnimationFrame(() => {
      needleEl.style.transition = 'left 0.07s linear'
      syncNeedle()
    })
  } catch (err) { statusLine.textContent = String(err); return }

  statusLine.textContent = 'LOADING…'
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}am_composite.wav`)
    if (!r.ok) throw new Error('am_composite.wav not found')
    const { samples, sampleRate } = parseWav(await r.arrayBuffer())
    rawSamples    = samples
    rawSampleRate = sampleRate
    statusLine.textContent = '—'
    scheduleDemod()
  } catch (err) { statusLine.textContent = String(err) }
}

main()
