import type { Capabilities, Receipt, RobotApi, Status, VoiceResult } from '../types/api'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) { super(message); this.status = status }
}
async function request(path: string, options: RequestInit = {}, timeout = 4000): Promise<unknown> {
  const response = await fetch(path, { ...options, signal: AbortSignal.timeout(timeout) })
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = data?.detail
    if (detail?.code === 'speech_provider_error') {
      const status = typeof detail.upstreamStatus === 'number' ? detail.upstreamStatus : null
      const hint = status === 401 || status === 403 ? 'Check the provider key and its endpoint.'
        : status === 404 ? 'Check that this endpoint offers the selected model.'
          : status === 400 || status === 422 ? 'The provider rejected the audio request.'
            : status === 429 ? 'The provider rate limit was reached.' : 'Check the backend terminal for connectivity.'
      throw new ApiError(`Speech provider error${status ? ` (HTTP ${status})` : ''}. ${hint}`, response.status)
    }
    throw new ApiError(typeof detail === 'string' ? detail : detail?.code || `Request failed (${response.status})`, response.status)
  }
  return data
}
function object(data: unknown): Record<string, unknown> {
  if (!data || typeof data !== 'object') throw new Error('Invalid backend response')
  return data as Record<string, unknown>
}
function receipt(data: unknown): Receipt {
  const value = object(data)
  if (typeof value.running !== 'boolean' || typeof value.status !== 'string'
    || !(value.target === null || typeof value.target === 'string')
    || !(value.command_id === null || typeof value.command_id === 'string')
    || !(value.session_id === null || typeof value.session_id === 'string')) throw new Error('Invalid robot status')
  return value as Receipt
}
const post = (path: string, body: unknown) => request(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
})
export const robotApi: RobotApi = {
  async status() {
    const value = object(await request('/status'))
    receipt(value)
    if (typeof value.revision !== 'number' || typeof value.instance_id !== 'string' || typeof value.paused !== 'boolean') throw new Error('Backend protocol needs updating')
    return value as Status
  },
  async capabilities() {
    const value = object(await request('/capabilities'))
    if (!['clip', 'disabled'].includes(String(value.voiceMode))
      || !['fake', 'qwen', 'disabled'].includes(String(value.speechProvider))
      || typeof value.maxRecordingSeconds !== 'number' || value.maxRecordingSeconds <= 0
      || typeof value.stopWord !== 'string' || typeof value.resumeWord !== 'string') throw new Error('Invalid voice configuration')
    return value as Capabilities
  },
  heartbeat: sessionId => post('/heartbeat', { sessionId }),
  direct: command => post('/direct', command),
  stop: () => post('/stop', {}),
  resume: expectedRevision => post('/resume', { expectedRevision }),
  receipt: async id => receipt(await request(`/commands/${encodeURIComponent(id)}`)),
}
export async function interpretAudio(audio: Blob, requestId: string): Promise<VoiceResult> {
  const value = object(await request('/voice/interpret', {
    method: 'POST', headers: { 'Content-Type': 'audio/wav', 'X-Request-ID': requestId }, body: audio,
  }, 45000))
  if (value.requestId !== requestId || typeof value.transcript !== 'string'
    || !['navigate', 'stop', 'resume', 'reject'].includes(String(value.intent))
    || (value.intent === 'navigate' && (typeof value.target !== 'string' || !value.target.trim()))) {
    throw new Error('The transcription is unclear, so this command cannot be accepted. Please try again.')
  }
  return value as VoiceResult
}
