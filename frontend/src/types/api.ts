export type VoiceResult = {
  requestId: string
  transcript: string
  intent: 'navigate' | 'stop' | 'resume' | 'reject'
  target: string | null
  reason: string | null
  message: string | null
  simulated: boolean
}
export type Capabilities = {
  voiceMode: 'clip' | 'disabled'
  speechProvider: 'fake' | 'qwen' | 'disabled'
  maxRecordingSeconds: number
  stopWord: string
  resumeWord: string
}
export type Status = {
  running: boolean
  target: string | null
  status: string
  command_id: string | null
  session_id: string | null
  revision: number
  paused: boolean
  instance_id: string
}
export type Receipt = Pick<Status, 'running' | 'target' | 'status' | 'command_id' | 'session_id'>
export type Direct = { target: string; commandId: string; sessionId: string; expectedRevision: number; resume: boolean }
export interface RobotApi {
  status(): Promise<Status>
  capabilities(): Promise<Capabilities>
  heartbeat(sessionId: string): Promise<unknown>
  direct(command: Direct): Promise<unknown>
  stop(): Promise<unknown>
  resume(revision: number): Promise<unknown>
  receipt(id: string): Promise<Receipt>
}
