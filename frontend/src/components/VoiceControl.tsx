import { useEffect, useRef, useState } from 'react'
import { startRecording } from '../services/audioCapture'
import type { Recording } from '../services/audioCapture'
import { interpretAudio } from '../services/robotApi'
import type { Capabilities, VoiceResult } from '../types/api'

type Phase = 'idle' | 'requesting' | 'listening' | 'processing'
export function VoiceControl({ connected, capabilities, onResult }: {
  connected: boolean; capabilities: Capabilities | null; onResult: (result: VoiceResult) => Promise<void>
}) {
  const [phase, setPhase] = useState<Phase>('idle')
  const [transcript, setTranscript] = useState('')
  const [feedback, setFeedback] = useState('')
  const [remaining, setRemaining] = useState(10)
  const recording = useRef<Recording | null>(null)
  const token = useRef(0)
  const phaseRef = useRef<Phase>('idle')
  const changePhase = (next: Phase) => { phaseRef.current = next; setPhase(next) }
  useEffect(() => {
    const sessionToken = token
    if (!connected) { token.current++; recording.current?.cancel(); recording.current = null; phaseRef.current = 'idle' }
    return () => { sessionToken.current++; recording.current?.cancel(); recording.current = null; phaseRef.current = 'idle' }
  }, [connected])
  useEffect(() => {
    if (phase !== 'listening') return
    const started = Date.now()
    const interval = setInterval(() => setRemaining(Math.max(0, 10 - Math.floor((Date.now() - started) / 1000))), 200)
    return () => clearInterval(interval)
  }, [phase])
  async function submit(audio: Blob, current: number) {
    if (current !== token.current) return
    recording.current = null
    changePhase('processing')
    setFeedback('Processing command…')
    try {
      const result = await interpretAudio(audio, crypto.randomUUID())
      if (current !== token.current) return
      console.log('Voice interpretation', {
        requestId: result.requestId,
        transcript: result.transcript,
        intent: result.intent,
        target: result.target,
        reason: result.reason,
        simulated: result.simulated,
      })
      setTranscript(result.transcript)
      if (result.intent === 'reject') {
        setFeedback('The transcription is unclear, so this command cannot be accepted. Please try again.')
      } else {
        await onResult(result)
        if (current !== token.current) return
        setFeedback(result.intent === 'navigate' ? `Queued → ${result.target}` : `${result.intent === 'stop' ? 'Pause' : 'Resume'} command processed.`)
      }
    } catch (error) {
      if (current === token.current) setFeedback(error instanceof Error ? error.message : 'Audio processing failed. Please try again.')
    } finally { if (current === token.current) changePhase('idle') }
  }
  async function toggle() {
    if (phaseRef.current === 'listening') { recording.current?.stop(); changePhase('processing'); return }
    if (phaseRef.current !== 'idle' || !connected || capabilities?.voiceMode !== 'clip') return
    const current = ++token.current
    changePhase('requesting'); setTranscript(''); setFeedback('Allow microphone access to begin.'); setRemaining(10)
    try {
      const capture = await startRecording(audio => { void submit(audio, current) }, message => {
        if (current === token.current) { setFeedback(message); changePhase('idle') }
      })
      if (current !== token.current) { capture.cancel(); return }
      recording.current = capture
      setFeedback('Listening. Press again to submit; recording ends after 10 seconds.')
      changePhase('listening')
    } catch (error) {
      if (current === token.current) { setFeedback(error instanceof Error ? error.message : 'Microphone unavailable.'); changePhase('idle') }
    }
  }
  const shownPhase = connected ? phase : 'idle'
  const disabled = !connected || capabilities?.voiceMode !== 'clip' || shownPhase === 'processing' || shownPhase === 'requesting'
  return <section className="voice-panel" aria-labelledby="voice-title">
    <div className="panel-heading"><h2 id="voice-title"><span className="section-number">04</span> VOICE COMMAND</h2><span>{shownPhase === 'listening' ? `LISTENING / ${remaining}s` : shownPhase === 'processing' ? 'PROCESSING…' : 'MICROPHONE'}</span></div>
    <div className="voice-body">
      <button className={`mic-button ${shownPhase === 'listening' ? 'listening' : ''}`} onClick={() => { void toggle() }} disabled={disabled} aria-pressed={shownPhase === 'listening'} aria-label={shownPhase === 'listening' ? 'End recording and send command' : 'Start microphone'}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/></svg>
        <span>{shownPhase === 'processing' ? 'PROCESSING…' : shownPhase === 'requesting' ? 'OPENING MIC…' : shownPhase === 'listening' ? 'PRESS TO SEND' : 'PRESS TO SPEAK'}</span>
      </button>
      <div className="voice-content"><span className="eyebrow">TRANSCRIPT</span><p className="voice-transcript">{transcript || '--'}</p><p className="voice-feedback" role="status">{!connected ? 'Connect to the robot to send a command.' : capabilities?.voiceMode !== 'clip' ? 'Speech service is not configured on the backend.' : feedback || 'Ready. Record a command up to 10 seconds.'}</p></div>
    </div>
    {capabilities?.speechProvider === 'fake' && <div className="voice-foot">Test speech provider: returns a scripted command; it does not transcribe the recording.</div>}
  </section>
}
