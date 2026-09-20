import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import vm from 'node:vm'

test('audio worklet cuts off at exactly ten seconds and sends done once', () => {
  let Recorder
  const context = {
    sampleRate: 16000,
    AudioWorkletProcessor: class { constructor() { this.messages = []; this.port = { postMessage: message => this.messages.push(message) } } },
    registerProcessor(name, Class) { Recorder = Class },
  }
  vm.runInNewContext(fs.readFileSync(new URL('../public/pcm-recorder.js', import.meta.url), 'utf8'), context)
  const recorder = new Recorder()
  for (let i = 0; i < 1300; i++) recorder.process([[new Float32Array(128)]])
  assert.equal(recorder.messages.filter(m => m.type === 'samples').reduce((n, m) => n + m.chunk.length, 0), 160000)
  assert.equal(recorder.messages.filter(m => m.type === 'done').length, 1)
  assert.equal(recorder.recording, false)
})
