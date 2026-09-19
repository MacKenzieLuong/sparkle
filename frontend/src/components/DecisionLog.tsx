import { useEffect, useRef } from 'react'
import type { LogEntry } from '../types/robot'
export function DecisionLog({ entries, unavailable = false }: { unavailable?: boolean; entries: LogEntry[] }) {
  const list = useRef<HTMLOListElement>(null)
  useEffect(() => { if (list.current) list.current.scrollTop = list.current.scrollHeight }, [entries.length])
  return <section className="log-panel" aria-labelledby="log-title"><div className="panel-heading"><h2 id="log-title"><span className="section-number">03</span> ACTIVITY LOG</h2><span>LOCAL TIME</span></div><ol className="log-list" ref={list} role="log" aria-label="Observed robot activity" aria-live="polite">{unavailable && <li><span>--:--:--</span><span>Waiting for robot connection.</span></li>}{entries.map(entry => <li key={entry.id}><time>{entry.time}</time><span>{entry.message}</span></li>)}</ol><div className="log-foot">{unavailable ? '--' : entries.length} entries</div></section>
}
