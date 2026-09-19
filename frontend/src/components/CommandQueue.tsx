import type { Task } from '../types/robot'
export function CommandQueue({ active, queue, paused, unavailable = false }: { unavailable?: boolean; active: Task | null; queue: Task[]; paused: boolean }) {
  return <section className="queue-panel" aria-labelledby="queue-title">
    <div className="panel-heading"><h2 id="queue-title"><span className="section-number">02</span> COMMAND QUEUE</h2><span>{unavailable ? '--' : String(queue.length).padStart(2, '0')} WAITING</span></div>
    <div className="queue-active"><span className="eyebrow">{unavailable ? '--' : paused ? 'PAUSED' : active ? 'IN PROGRESS' : 'READY'}</span><div><span aria-hidden="true">↳</span><h3>{unavailable ? '--' : active?.target ?? 'Awaiting a target'}</h3></div><p>{unavailable ? 'Waiting for robot connection.' : paused ? 'Your place in the queue is saved.' : active ? 'Approaching the current target.' : 'Add a voice command to begin.'}</p></div>
    <ol className="queue-list">{queue.map((task, index) => <li key={task.id}><span className="queue-index">{String(index + 1).padStart(2, '0')}</span><span>{task.target}</span><span className="queue-arrow" aria-hidden="true">↗</span></li>)}</ol>
    {!queue.length && <p className="empty-queue">{unavailable ? '--' : 'No targets waiting.'}</p>}
    <div className="queue-note">Executed in order. New commands wait their turn.</div>
  </section>
}
