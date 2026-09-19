import type { LogEntry, RobotStatus, Task } from '../types/robot'

export type MockState = {
  active: Task | null
  queue: Task[]
  status: RobotStatus
  paused: boolean
  connected: boolean
  logs: LogEntry[]
  nextId: number
}
export type MockAction =
  | { type: 'command'; target: string }
  | { type: 'pause' | 'resume' | 'complete' | 'connection' }

const time = () => new Date().toLocaleTimeString('en-GB', { hour12: false })
export function initialState(): MockState {
  return {
    active: { id: 1, target: 'Blue flag' },
    queue: [{ id: 2, target: 'Red ball' }, { id: 3, target: 'Yellow cone' }],
    status: 'moving', paused: false, connected: true, nextId: 4,
    logs: [
      { id: 1, time: time(), message: 'Mock robot connected' },
      { id: 2, time: time(), message: 'Navigation started → Blue flag' },
      { id: 3, time: time(), message: '2 targets added to queue' },
    ],
  }
}
function log(state: MockState, message: string): MockState {
  return { ...state, logs: [...state.logs, { id: (state.logs.at(-1)?.id ?? 0) + 1, time: time(), message }].slice(-200) }
}
export function mockReducer(state: MockState, action: MockAction): MockState {
  switch (action.type) {
    case 'command': {
      const task = { id: state.nextId, target: action.target }
      const next = { ...state, nextId: state.nextId + 1 }
      if (!state.active && !state.paused && state.connected) {
        return log({ ...next, active: task, status: 'moving' }, `Navigation started → ${task.target}`)
      }
      return log({ ...next, queue: [...state.queue, task] }, `Queued → ${task.target}`)
    }
    case 'pause':
      return log({ ...state, paused: true, status: 'stopped' }, 'Navigation paused · queue retained')
    case 'resume': {
      if (!state.connected) return state
      const active = state.active ?? state.queue[0] ?? null
      return log({ ...state, paused: false, active, queue: state.active ? state.queue : state.queue.slice(1), status: active ? 'moving' : 'idle' }, 'Navigation resumed')
    }
    case 'complete': {
      if (!state.active || state.paused || !state.connected) return state
      const next = state.queue[0] ?? null
      let result = log({ ...state, active: next, queue: state.queue.slice(1), status: next ? 'moving' : 'arrived' }, `Arrived → ${state.active.target}`)
      if (next) result = log(result, `Navigation started → ${next.target}`)
      return result
    }
    case 'connection':
      return log({ ...state, connected: !state.connected, paused: true, status: 'stopped' }, state.connected ? 'Connection lost · navigation paused' : 'Connection restored · awaiting resume')
  }
}
