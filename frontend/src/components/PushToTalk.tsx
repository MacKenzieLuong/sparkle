import { useState } from 'react'
import { config } from '../config/environment'
export function PushToTalk({ onCommand, onPause, onResume, connected, unavailable = false }: { unavailable?: boolean; onCommand: (target: string) => void; onPause: () => void; onResume: () => void; connected: boolean }) {
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState('Go to the green flag')
  const [feedback, setFeedback] = useState('Ready for your next command.')
  function pause() { onPause(); setListening(false); setFeedback('Navigation paused. Your queue is saved.') }
  function update(value: string) {
    setTranscript(value)
    if (listening && value.trim().toLowerCase() === config.stopWord) pause()
  }
  function toggle() {
    if (!listening) { setListening(true); setFeedback('Listening simulation. Edit the transcript, then press again.'); if (transcript.trim().toLowerCase() === config.stopWord) pause(); return }
    setListening(false)
    const text = transcript.trim().replace(/[.!?]+$/, '')
    if (text.toLowerCase() === config.stopWord) { pause(); return }
    if (text.toLowerCase() === 'resume') { onResume(); setFeedback('Navigation resumed.'); return }
    const match = text.match(/^(?:go|drive|navigate|find|head|move)\s+(?:to\s+)?(?:the\s+)?(.+)$/i)
    if (!match || /\b(?:not|don't|never|or|and|there|something|anything)\b/i.test(match[1])) { setFeedback('Command not understood. Try “go to the green flag”.'); return }
    const target = match[1].charAt(0).toUpperCase() + match[1].slice(1)
    onCommand(target); setFeedback(`Queued → ${target}`)
  }
  return <section className="voice-panel" aria-labelledby="voice-title"><div className="panel-heading"><h2 id="voice-title"><span className="section-number">04</span> VOICE COMMAND</h2><span>{unavailable ? 'MICROPHONE' : listening ? 'LISTENING / MOCK' : 'MICROPHONE / MOCK'}</span></div><div className="voice-body"><button className={'mic-button' + (listening ? ' listening' : '')} onClick={toggle} disabled={!connected} aria-pressed={listening} aria-label={unavailable ? 'Microphone unavailable' : listening ? 'Process mock voice command' : 'Start mock listening'}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/></svg><span>{listening ? 'PRESS TO SEND' : 'PRESS TO SPEAK'}</span></button><div className="voice-content"><label className="eyebrow" htmlFor="transcript">{unavailable ? 'TRANSCRIPT' : 'DEMO TRANSCRIPT'}</label><input id="transcript" value={unavailable ? '' : transcript} placeholder="--" onChange={event => update(event.target.value)} disabled={!connected} autoComplete="off" spellCheck="false" /><p className="voice-feedback" role="status">{unavailable ? 'Connect to the robot to send a command.' : connected ? feedback : 'Waiting for connection. Your queue is paused.'}</p></div></div>{!unavailable && <div className="voice-foot">Layout demo: edit the transcript to simulate speech. No microphone access.</div>}</section>
}
