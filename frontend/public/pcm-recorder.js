class PcmRecorder extends AudioWorkletProcessor {
  constructor() {
    super()
    this.samples = 0
    this.recording = true
    this.port.onmessage = ({ data }) => { if (data === 'stop') this.finish() }
  }
  finish() {
    if (!this.recording) return
    this.recording = false
    this.port.postMessage({ type: 'done' })
  }
  process(inputs) {
    if (!this.recording) return false
    const input = inputs[0]?.[0]
    if (input) {
      const count = Math.min(input.length, sampleRate * 10 - this.samples)
      const chunk = input.slice(0, count)
      this.port.postMessage({ type: 'samples', chunk }, [chunk.buffer])
      this.samples += count
      if (this.samples >= sampleRate * 10) this.finish()
    }
    return this.recording
  }
}
registerProcessor('pcm-recorder', PcmRecorder)
