import { useEffect, useState } from 'react'

// The one control that has to work when everything else is going wrong, so it
// lives in the header rather than in a panel that can be scrolled away, and it
// is bound to Escape as well.
//
// It is never disabled by connection state. The alternative is a stop that
// refuses to try because the last poll failed, which is precisely when someone
// is reaching for it. A robot that really is unreachable reports the outcome as
// unknown and leaves the queue paused, which is the safe end state either way.
//
// Stopping is not the same as arriving: the queue is retained and paused, so
// nothing dispatches afterwards until it is resumed.
export function StopButton({ onStop }: { onStop: () => Promise<void> }) {
  const [sending, setSending] = useState(false)

  async function run() {
    if (sending) return
    setSending(true)
    try {
      await onStop()
    } catch {
      // runStop reports the outcome through the connection's own message and
      // log; a throw here must not leave the button stuck in "STOPPING".
    } finally {
      setSending(false)
    }
  }

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key !== 'Escape') return
      event.preventDefault()
      void run()
    }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey) }
  })

  return <button
    className="stop-button"
    type="button"
    onClick={() => { if (!sending) void run() }}
    aria-label="Stop the robot now"
    title="Stop the robot (Esc)"
  >{sending ? 'STOPPING…' : 'STOP'}</button>
}
