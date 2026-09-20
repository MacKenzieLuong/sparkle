export type Recording = { stop: () => void; cancel: () => void }
export async function startRecording(onComplete: (audio: Blob) => void, onError: (message: string) => void): Promise<Recording> {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone capture requires localhost or HTTPS in a supported browser.')
  const context = new AudioContext({ sampleRate: 16000 })
  let stream: MediaStream | undefined
  let node: AudioWorkletNode | undefined
  let source: MediaStreamAudioSourceNode | undefined
  let fallback: ReturnType<typeof setTimeout> | undefined
  let finished = false
  const chunks: Float32Array[] = []
  const cleanup = () => {
    clearTimeout(fallback)
    stream?.getTracks().forEach(track => track.stop())
    source?.disconnect(); node?.disconnect()
    void context.close().catch(() => {})
  }
  const cancel = () => { finished = true; cleanup() }
  const complete = () => {
    if (finished) return
    finished = true
    const rate = context.sampleRate
    cleanup()
    const total = Math.min(chunks.reduce((n, chunk) => n + chunk.length, 0), rate * 10)
    const data = new ArrayBuffer(44 + total * 2)
    const view = new DataView(data)
    const text = (offset: number, value: string) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)))
    text(0, 'RIFF'); view.setUint32(4, 36 + total * 2, true); text(8, 'WAVE'); text(12, 'fmt ')
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true)
    view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true)
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, total * 2, true)
    let index = 0
    for (const chunk of chunks) for (const sample of chunk) {
      if (index >= total) break
      const value = Math.max(-1, Math.min(1, sample))
      view.setInt16(44 + index++ * 2, value < 0 ? value * 32768 : value * 32767, true)
    }
    onComplete(new Blob([data], { type: 'audio/wav' }))
  }
  try {
    await context.resume()
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
    await context.audioWorklet.addModule('/pcm-recorder.js')
    node = new AudioWorkletNode(context, 'pcm-recorder')
    source = context.createMediaStreamSource(stream)
    node.port.onmessage = ({ data }) => {
      if (finished) return
      if (data.type === 'samples') chunks.push(data.chunk)
      if (data.type === 'done') complete()
    }
    node.onprocessorerror = () => { if (!finished) { cancel(); onError('Microphone recording failed. Please try again.') } }
    stream.getAudioTracks()[0].onended = () => { if (!finished) { cancel(); onError('Microphone disconnected. Command was not submitted.') } }
    source.connect(node); node.connect(context.destination)
    // Sample counting enforces 10 seconds even if page timers are throttled.
    fallback = setTimeout(() => { node?.port.postMessage('stop') }, 10000)
    return { stop: () => node?.port.postMessage('stop'), cancel }
  } catch (error) { cancel(); throw error }
}
