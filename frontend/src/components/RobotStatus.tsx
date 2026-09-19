import type { RobotStatus as Status } from '../types/robot'
export function RobotStatus({ status, target, paused, connected }: { status: Status; target: string | null; paused: boolean; connected: boolean }) {
  return <section className="status-strip" aria-label="Current robot status" aria-live="polite">
    <div><span className="eyebrow">ROBOT STATE</span><strong><span className={paused ? 'state-square hollow' : 'state-square'} />{!connected ? 'Disconnected' : paused ? 'Paused' : status === 'moving' ? 'Moving' : status === 'arrived' ? 'Arrived' : 'Idle'}</strong></div>
    <div><span className="eyebrow">CURRENT TARGET</span><strong>{target ?? 'No active target'}</strong></div>
    <div><span className="eyebrow">CONTROL</span><strong>Autonomous</strong></div>
  </section>
}
