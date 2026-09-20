import { randomId } from './randomId.ts'
import type { Capabilities, VoiceResult } from '../types/api'

// A typed command is an interpretation that skipped transcription, so it can go
// through the same accept() a spoken one does and inherit the queue, the
// duplicate check, the receipt handling and the logging.
//
// It has to cover stop and resume, not just navigation. `paused` is a latch:
// a backend restart or any dropped poll sets it, dispatch is gated on it, and
// only a resume intent clears it. Live mode has no resume control of its own,
// so if typing could produce navigate alone then on the setup where the
// microphone is unavailable -- plain http at a LAN address -- nothing could
// ever unpause the queue, and every target would sit in it silently.
//
// The words come from the backend's /capabilities rather than being fixed here,
// so typing matches whatever the spoken path accepts.
export function typedCommand(text: string, capabilities: Capabilities | null = null): VoiceResult {
  const word = text.trim().toLowerCase()
  const stop = (capabilities?.stopWord ?? 'stop').toLowerCase()
  const resume = (capabilities?.resumeWord ?? 'resume').toLowerCase()
  const intent = word === stop ? 'stop' : word === resume ? 'resume' : 'navigate'
  return {
    requestId: randomId(),
    transcript: text,
    intent,
    target: intent === 'navigate' ? text : null,
    reason: null,
    message: null,
    simulated: false,
  }
}
