import type { Capabilities, RobotApi, Status, VoiceResult } from '../types/api'
import { randomId } from './randomId.ts'
import type { LogEntry, Task } from '../types/robot'

export type ConnectionView = {
  active: Task | null; queue: Task[]; logs: LogEntry[]; connected: boolean; paused: boolean
  ready: boolean; status: string; message: string; capabilities: Capabilities | null
}
const failedStates = new Set(['target_lost', 'error', 'connection_lost', 'stopped', 'halted', 'time_limit'])
export class RobotConnection {
  private api: RobotApi
  private sessionId = randomId()
  private snapshot: Status | null = null
  private view: ConnectionView = { active: null, queue: [], logs: [], connected: false, paused: false, ready: false, status: 'idle', message: '', capabilities: null }
  private listeners = new Set<() => void>()
  private timer: ReturnType<typeof setTimeout> | undefined
  private enabled = false
  private run = 0
  private polling: Promise<void> | null = null
  private sending = false
  private unknown = false
  private generation = 0
  private seenVoice = new Set<string>()
  constructor(api: RobotApi) { this.api = api }
  getSnapshot = () => this.view
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  private update(patch: Partial<ConnectionView>) { this.view = { ...this.view, ...patch }; this.listeners.forEach(listener => listener()) }
  private log(message: string) {
    this.update({ logs: [...this.view.logs, { id: (this.view.logs.at(-1)?.id ?? 0) + 1, time: new Date().toLocaleTimeString('en-GB', { hour12: false }), message }].slice(-200) })
  }
  start = () => { this.enabled = true; const run = ++this.run; void this.tick(run); return () => { this.enabled = false; this.run++; clearTimeout(this.timer) } }
  private async tick(run: number) {
    await this.refresh()
    if (this.enabled && run === this.run) this.timer = setTimeout(() => { void this.tick(run) }, 1000)
  }
  refresh = (): Promise<void> => {
    if (this.polling) return this.polling
    this.polling = this.poll().finally(() => { this.polling = null })
    return this.polling
  }
  private async poll() {
    try {
      await this.api.heartbeat(this.sessionId)
      const snapshot = await this.api.status()
      if (!this.enabled) return
      const previous = this.snapshot
      const restarted = previous && previous.instance_id !== snapshot.instance_id
      if (restarted) {
        this.generation++; this.unknown = false
        this.update({ paused: true, capabilities: null, message: 'Robot restarted. Queue paused; say resume to continue.' })
        this.log('Robot restarted · queue retained')
      }
      this.snapshot = snapshot
      if (!this.view.connected) this.log(this.view.ready ? 'Connection restored · awaiting resume' : 'Robot connected')
      this.update({ connected: true, ready: true, status: snapshot.status })
      if (!this.view.capabilities) {
        try { const capabilities = await this.api.capabilities(); if (this.enabled) this.update({ capabilities }) } catch { /* Status stays usable when voice is unavailable. */ }
      }
      if (!this.enabled) return
      if (!previous && snapshot.target && snapshot.running) {
        this.update({ active: { id: snapshot.command_id ?? randomId(), target: snapshot.target }, paused: snapshot.session_id !== this.sessionId })
      }
      if (!previous || previous.status !== snapshot.status || previous.command_id !== snapshot.command_id) this.log(`${snapshot.status}${snapshot.target ? ` → ${snapshot.target}` : ''}`)
      if (this.unknown && this.view.active) {
        try {
          const receipt = await this.api.receipt(String(this.view.active.id))
          if (!this.enabled) return
          this.unknown = false
          this.update({ message: 'Command receipt recovered. Queue remains paused; say resume to continue.' })
          if (!receipt.running && receipt.status === 'arrived') this.update({ active: null })
        } catch { /* An absent receipt is not proof that a delayed request cannot arrive. */ }
      }
      if (!this.sending && !this.unknown && this.view.active?.id === snapshot.command_id && !snapshot.running) {
        if (snapshot.status === 'arrived') {
          this.update({ active: null })
        } else if (failedStates.has(snapshot.status) || snapshot.status.startsWith('error:')) {
          this.update({ paused: true, message: `Navigation ${snapshot.status}. Queue paused; say resume to retry.` })
        }
      }
      if (snapshot.paused) this.update({ paused: true })
      if (!this.view.paused && !this.unknown && !this.sending && !snapshot.running && !this.view.active && this.view.queue.length) {
        await this.dispatchNext(false)
      }
    } catch {
      if (!this.enabled) return
      if (this.view.connected) { this.log('Connection lost · queue paused'); this.generation++ }
      this.update({ connected: false, paused: this.view.ready || this.view.paused, message: this.view.ready ? 'Lost connection. Queue paused.' : 'Waiting for the backend.' })
    }
  }
  private async send(task: Task, resume: boolean) {
    if (!this.snapshot || this.sending) return
    this.sending = true
    const generation = this.generation
    this.update({ active: task, message: 'Sending navigation command…' })
    try {
      await this.api.direct({ target: task.target, commandId: String(task.id), sessionId: this.sessionId, expectedRevision: this.snapshot.revision, resume })
      if (!this.enabled || generation !== this.generation) return
      this.update({ message: `Robot accepted → ${task.target}`, paused: false })
      this.log(`Command accepted → ${task.target}`)
    } catch (error) {
      if (!this.enabled || generation !== this.generation) return
      const status = error && typeof error === 'object' && 'status' in error ? Number(error.status) : 0
      this.unknown = !(status >= 400 && status < 500)
      this.update({ paused: true, message: this.unknown ? 'Outcome unknown. Refreshing robot status; command will not be resent.' : 'Robot rejected the command. Queue paused.' })
      this.log(this.view.message)
    } finally { this.sending = false }
  }
  private async dispatchNext(resume: boolean) {
    const task = this.view.queue[0]
    if (!task) return
    this.update({ queue: this.view.queue.slice(1) })
    await this.send(task, resume)
  }
  // Stop cancels: the queue is dropped and the robot is left ready for the
  // next target, rather than held paused until something resumes it.
  //
  // Clearing the backend's own pause is part of that, not an extra. `/stop`
  // pauses it server-side and `start()` then rejects a new target with
  // resume_required, so without the resume below a stop would silently make
  // the car refuse every command that followed.
  private async runStop() {
    this.generation++
    this.update({ queue: [], active: null, message: 'Stopping…' })
    try {
      await this.api.stop()
      this.unknown = false
    } catch {
      // The robot may still be driving. The queue is already cleared, but
      // pause as well so nothing is dispatched on top of a car that never
      // received the stop.
      this.update({ paused: true, message: 'Stop outcome unknown. Queue cleared; not dispatching until the robot answers.' })
      this.log(this.view.message)
      await this.refresh()
      return
    }
    try {
      const snapshot = await this.api.status()
      this.snapshot = snapshot
      if (snapshot.paused) await this.api.resume(snapshot.revision)
      this.update({ paused: false })
    } catch {
      // Left paused: the poll below reports it, and the resume control shows.
    }
    this.update({ message: 'Stopped. Queue cleared; ready for a new target.' })
    this.log('Stopped · queue cleared')
    await this.refresh()
  }

  // Deliberately not behind accept()'s connected check. That check reads a
  // poll up to a second old, and a stop control that refuses to even attempt
  // the request because of a stale belief about the network is the one
  // failure this button cannot have. A genuinely unreachable robot still
  // reports "outcome unknown" and leaves the queue paused.
  stopNow = () => this.runStop()

  async accept(result: VoiceResult) {
    if (!this.view.connected) throw new Error('Connection lost. Command was not queued.')
    if (this.seenVoice.has(result.requestId)) return
    this.seenVoice.add(result.requestId)
    if (this.seenVoice.size > 200) this.seenVoice.delete(this.seenVoice.values().next().value!)
    if (result.intent === 'reject') return
    if (result.intent === 'navigate' && result.target) {
      this.update({ queue: [...this.view.queue, { id: randomId(), target: result.target }] })
      this.log(`Queued → ${result.target}`)
      // Poll serially so each dispatch uses a freshly observed revision.
      await this.refresh()
    } else if (result.intent === 'stop') {
      await this.runStop()
    } else if (result.intent === 'resume') {
      await this.refresh()
      if (!this.view.connected || !this.snapshot) throw new Error('Waiting for robot connection.')
      if (this.unknown) throw new Error('Command outcome is still unknown. Say stop before trying to resume.')
      if (this.snapshot.running) {
        if (this.snapshot.session_id !== this.sessionId) throw new Error('Another session is active. Wait for it to stop.')
        this.update({ paused: false, message: 'Navigation resumed.' }); return
      }
      if (this.view.active) await this.send({ ...this.view.active, id: randomId() }, true)
      else if (this.view.queue.length) await this.dispatchNext(true)
      else { await this.api.resume(this.snapshot.revision); this.update({ paused: false, message: 'Queue is empty. Ready for a new target.' }) }
    }
  }
}
