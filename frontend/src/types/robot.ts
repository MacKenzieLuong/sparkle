export type Task = { id: number | string; target: string }
export type LogEntry = { id: number; time: string; message: string }
export type RobotStatus = 'idle' | 'moving' | 'arrived' | 'stopped'
