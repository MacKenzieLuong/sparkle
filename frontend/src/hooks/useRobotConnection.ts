import { useEffect, useState, useSyncExternalStore } from 'react'
import { RobotConnection } from '../services/robotConnection'
import { robotApi } from '../services/robotApi'
export function useRobotConnection(enabled: boolean) {
  const [connection] = useState(() => new RobotConnection(robotApi))
  const robot = useSyncExternalStore(connection.subscribe, connection.getSnapshot)
  useEffect(() => { if (enabled) return connection.start() }, [enabled, connection])
  return { robot, accept: connection.accept.bind(connection), stop: connection.stopNow }
}
