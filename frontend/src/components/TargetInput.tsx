import { useState } from 'react'
import type { FormEvent } from 'react'
import { typedCommand } from '../services/typedCommand'
import type { VoiceResult } from '../types/api'

// Typing a target is the way in when the microphone is not available: over
// plain http at a LAN address the browser refuses getUserMedia, so the voice
// panel cannot be used at all. It is also the quicker way to retry a target
// while tuning, without saying the same sentence twenty times.
//
// A typed target is a navigate intent that skipped transcription, so it goes
// through the same accept() the voice panel uses rather than a second route to
// the robot — inheriting the queue, the duplicate check, the receipt handling
// and the logging, all of which would otherwise have to be repeated here.
export function TargetInput({ connected, onResult }: {
  connected: boolean; onResult: (result: VoiceResult) => Promise<void>
}) {
  const [target, setTarget] = useState('')
  const [feedback, setFeedback] = useState('')
  const [sending, setSending] = useState(false)
  const text = target.trim()

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!text || sending || !connected) return
    setSending(true)
    setFeedback('Queueing…')
    try {
      await onResult(typedCommand(text))
      setTarget('')
      setFeedback(`Queued → ${text}`)
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : 'Command was not queued.')
    } finally {
      setSending(false)
    }
  }

  return <section className="voice-panel target-panel" aria-labelledby="target-title">
    <div className="panel-heading">
      <h2 id="target-title"><span className="section-number">05</span> TYPED COMMAND</h2>
      <span>{sending ? 'SENDING…' : 'KEYBOARD'}</span>
    </div>
    <form className="target-body" onSubmit={event => { void submit(event) }}>
      <div className="voice-content">
        <label className="eyebrow" htmlFor="target-input">TARGET</label>
        <input
          id="target-input"
          value={target}
          placeholder="the red ball"
          onChange={event => setTarget(event.target.value)}
          disabled={!connected || sending}
          autoComplete="off"
          spellCheck="false"
        />
        <p className="voice-feedback" role="status">
          {!connected
            ? 'Connect to the robot to send a command.'
            : feedback || 'Type a target and press Enter. Works without a microphone.'}
        </p>
      </div>
      <button className="target-send" type="submit" disabled={!connected || sending || !text}>
        {sending ? 'SENDING…' : 'SEND'}
      </button>
    </form>
  </section>
}
