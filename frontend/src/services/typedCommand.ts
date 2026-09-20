import { randomId } from './randomId.ts'
import type { VoiceResult } from '../types/api'

// A typed target is a navigate intent that skipped transcription, so it can go
// through the same accept() a spoken one does and inherit the queue, the
// duplicate check, the receipt handling and the logging.
//
// Shaped here rather than inline in the component so the same tests that cover
// the spoken path exercise it: accept() reads `intent` and `target`, and a
// drift in either would otherwise surface only in the browser.
export function typedCommand(target: string): VoiceResult {
  return {
    requestId: randomId(),
    transcript: target,
    intent: 'navigate',
    target,
    reason: null,
    message: null,
    simulated: false,
  }
}
