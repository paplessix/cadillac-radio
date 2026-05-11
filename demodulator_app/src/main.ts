import type { DspRequest, DspResponse } from './worker'

// ── Types ──────────────────────────────────────────────────────────────────
interface Station { label: string; freq: number }
interface SpectrumData { freqsKHz: number[]; psdDb: number[]; stations: Station[] }

// ── WAV parser (PCM int16 or IEEE float32, 1 channel) ────────────────────
function parseWav(buf: ArrayBuffer): { samples: Float32Array; sampleRate: number } {
  const v = new DataView(buf)
  let pos = 12 // skip 'RIFF' + size + 'WAVE'
  let sampleRate = 0
  let audioFormat = 0
  let bitsPerSample = 0
  let dataOffset = 0
  let dataBytes = 0

  while (pos < buf.byteLength - 8) {
    const id   = String.fromCharCode(v.getUint8(pos), v.getUint8(pos+1), v.getUint8(pos+2), v.getUint8(pos+3))
    const size = v.getUint32(pos + 4, true)
    pos += 8
    if (id === 'fmt ') {
      audioFormat  = v.getUint16(pos,      true)  // 1=PCM, 3=IEEE float
      sampleRate   = v.getUint32(pos + 4,  true)
      bitsPerSample = v.getUint16(pos + 14, true)
    } else if (id === 'data') {
      dataOffset = pos
      dataBytes  = size
      break
    }
    pos += size
  }

  // buf.slice() copies into a new ArrayBuffer starting at offset 0 — always
  // 4-byte aligned, avoiding RangeError when scipy's float WAV has a fact
  // chunk that shifts the data payload to a non-multiple-of-4 offset.
  const sliced = buf.slice(dataOffset, dataOffset + dataBytes)

  if (audioFormat === 3 && bitsPerSample === 32) {
    // IEEE float32 — use directly
    return { samples: new Float32Array(sliced), sampleRate }
  }

  // PCM int16 — convert to float32 in [-1, 1]
  const pcm = new Int16Array(sliced)
  const out = new Float32Array(pcm.length)
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768.0
  return { samples: out, sampleRate }
}

// ── Analytical biquad BPF magnitude (for filter overlay on spectrum) ──────
// |H(f)|² = (w/Q)² / ((1-w²)² + (w/Q)²),  w = f/f0
function filterMagDb(freqsKHz: number[], f0Hz: number, Q: number): number[] {
  return freqsKHz.map(fkHz => {
    const w = (fkHz * 1e3) / f0Hz
    const num = (w / Q) ** 2
    const den = (1 - w * w) ** 2 + (w / Q) ** 2
    return 10 * Math.log10(num / den + 1e-12)
  })
}

// ── Canvas helpers ─────────────────────────────────────────────────────────
const STATION_COLORS = ['#38bdf8', '#fb923c', '#4ade80', '#f472b6',
                        '#a78bfa', '#facc15', '#34d399', '#f87171']

function drawSpectrum(
  canvas: HTMLCanvasElement,
  spec: SpectrumData,
  filterDb: number[],
  f0Hz: number,
  highlightStation: number | null,
) {
  const W = canvas.width
  const H = canvas.height
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, W, H)
  ctx.fillStyle = '#090d16'
  ctx.fillRect(0, 0, W, H)

  const { freqsKHz, psdDb, stations } = spec
  const fMin = freqsKHz[0], fMax = freqsKHz[freqsKHz.length - 1]
  const pMin = Math.min(...psdDb) - 5
  const pMax = Math.max(...psdDb) + 8

  const fx = (f: number) => ((f - fMin) / (fMax - fMin)) * W
  const fy = (p: number) => H - ((p - pMin) / (pMax - pMin)) * H

  // Grid lines
  ctx.strokeStyle = '#1f2937'
  ctx.lineWidth = 1
  for (let f = Math.ceil(fMin / 10) * 10; f < fMax; f += 10) {
    const x = Math.round(fx(f)) + 0.5
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
    ctx.fillStyle = '#374151'
    ctx.font = '10px JetBrains Mono'
    ctx.fillText(`${f}k`, x + 3, H - 4)
  }

  // PSD curve
  ctx.beginPath()
  ctx.strokeStyle = '#1d4ed8'
  ctx.lineWidth = 1.5
  for (let i = 0; i < freqsKHz.length; i++) {
    const x = fx(freqsKHz[i]), y = fy(psdDb[i])
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  }
  ctx.stroke()

  // PSD fill
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(fx(freqsKHz[0]), H)
  for (let i = 0; i < freqsKHz.length; i++) ctx.lineTo(fx(freqsKHz[i]), fy(psdDb[i]))
  ctx.lineTo(fx(freqsKHz[freqsKHz.length-1]), H)
  ctx.closePath()
  const grad = ctx.createLinearGradient(0, 0, 0, H)
  grad.addColorStop(0, 'rgba(56,189,248,0.18)')
  grad.addColorStop(1, 'rgba(56,189,248,0)')
  ctx.fillStyle = grad
  ctx.fill()
  ctx.restore()

  // Filter response overlay (peak aligned to top of psd)
  const filterPeak = Math.max(...filterDb)
  const psdMid = pMax - 8
  const shift  = psdMid - filterPeak

  ctx.beginPath()
  ctx.strokeStyle = 'rgba(248,113,113,0.9)'
  ctx.lineWidth = 2.5
  let first = true
  for (let i = 0; i < freqsKHz.length; i++) {
    const fd = filterDb[i]
    if (fd < -60) { first = true; continue }
    const x = fx(freqsKHz[i]), y = fy(fd + shift)
    first ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    first = false
  }
  ctx.stroke()

  // Station markers
  stations.forEach((st, idx) => {
    const x = Math.round(fx(st.freq / 1e3))
    const col = STATION_COLORS[idx % STATION_COLORS.length]
    const isHL = idx === highlightStation

    ctx.strokeStyle = col
    ctx.lineWidth = isHL ? 2 : 1
    ctx.setLineDash(isHL ? [] : [4, 4])
    ctx.globalAlpha = isHL ? 1 : 0.6
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
    ctx.setLineDash([])
    ctx.globalAlpha = 1

    ctx.fillStyle = col
    ctx.font = `${isHL ? 600 : 400} 10px JetBrains Mono`
    ctx.fillText(`${st.label} ${(st.freq/1e3).toFixed(0)}k`, x + 4, 14 + idx * 14)
  })

  // f0 marker (needle)
  const x0 = Math.round(fx(f0Hz / 1e3))
  ctx.strokeStyle = '#fbbf24'
  ctx.lineWidth = 1.5
  ctx.setLineDash([2, 3])
  ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, H); ctx.stroke()
  ctx.setLineDash([])
}

function drawWaveform(canvas: HTMLCanvasElement, samples: Float32Array | null) {
  const W = canvas.width, H = canvas.height
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#090d16'
  ctx.fillRect(0, 0, W, H)

  if (!samples || samples.length === 0) {
    ctx.fillStyle = '#1f2937'
    ctx.font = '11px JetBrains Mono'
    ctx.fillText('no audio', 12, H / 2 + 4)
    return
  }

  const step = Math.max(1, Math.floor(samples.length / W))
  ctx.beginPath()
  ctx.strokeStyle = '#818cf8'
  ctx.lineWidth = 1

  for (let x = 0; x < W; x++) {
    const i   = x * step
    const amp = samples[i] ?? 0
    const y   = (1 - amp) * H / 2
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  }
  ctx.stroke()

  // Centre line
  ctx.strokeStyle = '#1f2937'
  ctx.lineWidth = 0.5
  ctx.beginPath()
  ctx.moveTo(0, H/2); ctx.lineTo(W, H/2)
  ctx.stroke()
}

// ── Main ───────────────────────────────────────────────────────────────────
async function main() {
  const statusText   = document.getElementById('status-text')!
  const progressWrap = document.getElementById('progress-wrap') as HTMLElement
  const progressBar  = document.getElementById('progress-bar') as HTMLElement
  const specCanvas   = document.getElementById('spectrum-canvas') as HTMLCanvasElement
  const waveCanvas   = document.getElementById('wave-canvas') as HTMLCanvasElement
  const presets      = document.getElementById('station-presets')!
  const btnPlay      = document.getElementById('btn-play') as HTMLButtonElement
  const btnStop      = document.getElementById('btn-stop') as HTMLButtonElement
  const slL          = document.getElementById('sl-L') as HTMLInputElement
  const slC          = document.getElementById('sl-C') as HTMLInputElement
  const slR          = document.getElementById('sl-R') as HTMLInputElement
  const slVol        = document.getElementById('sl-vol') as HTMLInputElement
  const vL           = document.getElementById('v-L')!
  const vC           = document.getElementById('v-C')!
  const vR           = document.getElementById('v-R')!
  const rF0          = document.getElementById('r-f0')!
  const rQ           = document.getElementById('r-q')!
  const rBw          = document.getElementById('r-bw')!
  const chkLoop      = document.getElementById('chk-loop') as HTMLInputElement

  // Resize canvases to device pixels
  function fitCanvas(c: HTMLCanvasElement, h: number) {
    const dpr = devicePixelRatio || 1
    c.width  = c.offsetWidth  * dpr
    c.height = h * dpr
    c.style.height = `${h}px`
  }
  fitCanvas(specCanvas, 200)
  fitCanvas(waveCanvas, 100)
  window.addEventListener('resize', () => {
    fitCanvas(specCanvas, 200)
    fitCanvas(waveCanvas, 100)
    redraw()
  })

  // ── State ────────────────────────────────────────────────────────────────
  let specData: SpectrumData | null = null
  let rawSamples: Float32Array | null = null
  let rawSampleRate = 500_000
  let audioBuffer: AudioBuffer | null = null
  let audioCtx: AudioContext | null = null
  let currentSource: AudioBufferSourceNode | null = null
  let gainNode: GainNode | null = null
  let highlightIdx: number | null = null
  let debounceTimer = 0
  let playAnchorCtx = 0   // audioCtx.currentTime corresponding to signal t=0
  const OUT_FS   = 44100

  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })

  // ── RLC helpers ──────────────────────────────────────────────────────────
  const getRohm = () => Math.pow(10, parseFloat(slR.value))

  function getF0Q(): { f0: number; Q: number } {
    const L = parseFloat(slL.value) * 1e-6
    const C = parseFloat(slC.value) * 1e-12
    const R = getRohm()
    const f0 = 1 / (2 * Math.PI * Math.sqrt(L * C))
    const Q  = Math.sqrt(L / C) / R
    return { f0, Q }
  }

  function updateReadout() {
    const L = parseFloat(slL.value), C = parseFloat(slC.value), R = getRohm()
    vL.textContent = L.toFixed(1)
    vC.textContent = C.toFixed(1)
    vR.textContent = R < 1 ? R.toFixed(3) : R < 10 ? R.toFixed(2) : R.toFixed(1)
    const { f0, Q } = getF0Q()
    rF0.textContent = (f0 / 1e3).toFixed(2)
    rQ.textContent  = Q.toFixed(1)
    rBw.textContent = Math.round(f0 / Q).toString()
  }

  // ── Drawing ───────────────────────────────────────────────────────────────
  function redraw() {
    if (!specData) return
    const { f0, Q } = getF0Q()
    const fdb = filterMagDb(specData.freqsKHz, f0, Q)
    drawSpectrum(specCanvas, specData, fdb, f0, highlightIdx)
  }

  // ── Playback helpers ─────────────────────────────────────────────────────
  function startPlayback(offsetSecs = 0) {
    if (!audioBuffer || !audioCtx || !gainNode) return
    currentSource?.stop(); currentSource = null
    if (audioCtx.state === 'suspended') audioCtx.resume()
    const src = audioCtx.createBufferSource()
    src.buffer = audioBuffer
    src.loop   = chkLoop.checked
    src.connect(gainNode)
    const off = offsetSecs % audioBuffer.duration
    src.start(0, off)
    currentSource = src
    playAnchorCtx = audioCtx.currentTime - off
    src.onended = () => { currentSource = null }
  }

  // ── Worker response ───────────────────────────────────────────────────────
  worker.onmessage = (e: MessageEvent<DspResponse>) => {
    const { audio, sampleRate } = e.data
    drawWaveform(waveCanvas, audio)

    if (!audioCtx) audioCtx = new AudioContext()
    if (!gainNode) {
      gainNode = audioCtx.createGain()
      gainNode.gain.value = parseFloat(slVol.value)
      gainNode.connect(audioCtx.destination)
    }
    const ab = audioCtx.createBuffer(1, audio.length, sampleRate)
    ab.copyToChannel(new Float32Array(audio.buffer as ArrayBuffer), 0)
    audioBuffer = ab

    statusText.textContent = 'Ready — click Play to listen'
    btnPlay.disabled = false
    btnStop.disabled = false
  }

  worker.onerror = (e) => {
    statusText.textContent = `DSP error: ${e.message}`
  }

  // ── Trigger DSP ───────────────────────────────────────────────────────────
  function scheduleDemodulate() {
    clearTimeout(debounceTimer)
    debounceTimer = window.setTimeout(() => {
      if (!rawSamples) return
      const { f0, Q } = getF0Q()
      if (f0 > rawSampleRate / 2 - 1000 || f0 < 500) {
        statusText.textContent = 'f₀ out of range'
        return
      }
      // audioBw = half the RF bandwidth = f0/(2Q); cap at 4500 Hz for audio quality
      const audioBw = Math.min(f0 / (2 * Q), 4500)
      statusText.textContent = `Processing…  f₀=${(f0/1e3).toFixed(2)} kHz  Q=${Q.toFixed(1)}`
      btnPlay.disabled = true

      const copy = new Float32Array(rawSamples)
      worker.postMessage(
        { samples: copy, sampleRate: rawSampleRate, f0, audioBw } satisfies DspRequest,
        { transfer: [copy.buffer] }
      )
    }, 400)
  }

  // ── Slider events ─────────────────────────────────────────────────────────
  ;[slL, slC, slR].forEach(sl => {
    sl.addEventListener('input', () => {
      updateReadout()
      redraw()
      scheduleDemodulate()
    })
  })

  slVol.addEventListener('input', () => {
    if (gainNode) gainNode.gain.value = parseFloat(slVol.value)
  })

  // ── Playback ──────────────────────────────────────────────────────────────
  function stopPlayback() {
    currentSource?.stop()
    currentSource = null
  }

  btnPlay.addEventListener('click', () => startPlayback(0))
  btnStop.addEventListener('click', stopPlayback)

  // ── Station presets ───────────────────────────────────────────────────────
  function buildPresetButtons(stations: Station[]) {
    stations.forEach((st, idx) => {
      const col = STATION_COLORS[idx % STATION_COLORS.length]
      const btn = document.createElement('button')
      btn.className = 'preset-btn'
      btn.textContent = `${st.label}  ${(st.freq/1e3).toFixed(0)} kHz`
      btn.style.borderColor = col
      btn.style.color = col
      btn.addEventListener('click', () => {
        // Tune to this station: fix L=1mH, solve C
        const L_uH = 1000
        const L = L_uH * 1e-6
        const C_pF = 1e12 / (L * (2 * Math.PI * st.freq) ** 2)
        slL.value = String(L_uH)
        slC.value = String(Math.max(1, Math.min(5000, C_pF)).toFixed(1))
        // Mark active
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
        btn.classList.add('active')
        btn.style.background = col
        highlightIdx = idx
        updateReadout()
        redraw()
        scheduleDemodulate()
      })
      presets.appendChild(btn)
    })
  }

  // ── Click spectrum to tune ────────────────────────────────────────────────
  specCanvas.addEventListener('click', (e) => {
    if (!specData) return
    const rect  = specCanvas.getBoundingClientRect()
    const xFrac = (e.clientX - rect.left) / rect.width
    const fMin  = specData.freqsKHz[0], fMax = specData.freqsKHz[specData.freqsKHz.length-1]
    const fHz   = (fMin + xFrac * (fMax - fMin)) * 1e3
    // Tune C to hit that frequency (L fixed at slider value)
    const L = parseFloat(slL.value) * 1e-6
    const C_pF = 1e12 / (L * (2 * Math.PI * fHz) ** 2)
    slC.value = String(Math.max(1, Math.min(5000, C_pF)).toFixed(1))
    highlightIdx = null
    document.querySelectorAll('.preset-btn').forEach(b => {
      b.classList.remove('active');
      (b as HTMLButtonElement).style.background = 'transparent'
    })
    updateReadout()
    redraw()
    scheduleDemodulate()
  })

  // ── Load spectrum.json ────────────────────────────────────────────────────
  statusText.textContent = 'Loading spectrum…'
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}spectrum.json`)
    if (!r.ok) throw new Error(`spectrum.json not found (${r.status}) — run generate_am_spectrum.py first`)
    specData = await r.json() as SpectrumData
    buildPresetButtons(specData.stations)
    updateReadout()
    redraw()
  } catch (err) {
    statusText.textContent = String(err)
    return
  }

  // ── Load WAV ──────────────────────────────────────────────────────────────
  statusText.textContent = 'Downloading am_composite.wav…'
  progressWrap.hidden = false

  try {
    const r = await fetch(`${import.meta.env.BASE_URL}am_composite.wav`)
    if (!r.ok) throw new Error(`am_composite.wav not found (${r.status})`)
    const contentLength = parseInt(r.headers.get('content-length') ?? '0')
    const reader = r.body!.getReader()
    const chunks: Uint8Array[] = []
    let received = 0

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      received += value.length
      if (contentLength > 0) {
        progressBar.style.width = `${(received / contentLength * 100).toFixed(0)}%`
        statusText.textContent = `Downloading…  ${(received/1e6).toFixed(0)} / ${(contentLength/1e6).toFixed(0)} MB`
      }
    }

    // Concatenate
    const totalBuf = new Uint8Array(received)
    let off = 0
    for (const chunk of chunks) { totalBuf.set(chunk, off); off += chunk.length }

    progressWrap.hidden = true
    statusText.textContent = 'Parsing WAV…'
    const { samples, sampleRate } = parseWav(totalBuf.buffer)
    rawSamples    = samples
    rawSampleRate = sampleRate

    statusText.textContent = `Loaded ${(received/1e6).toFixed(0)} MB — ${(samples.length/sampleRate).toFixed(1)} s @ ${sampleRate/1e3} kHz`

    // Initial demodulate
    scheduleDemodulate()
  } catch (err) {
    progressWrap.hidden = true
    statusText.textContent = String(err)
  }
}

main()
