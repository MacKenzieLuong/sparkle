import { CameraView } from './components/CameraView'
import { CommandQueue } from './components/CommandQueue'
import { DecisionLog } from './components/DecisionLog'
import { PushToTalk } from './components/PushToTalk'
import { useMockRobot } from './hooks/useMockRobot'
import { config } from './config/environment'
import './App.css'

function App() {
  const { robot: mockRobot, dispatch } = useMockRobot()
  const unavailable = !config.mockMode
  const robot = unavailable
    ? { ...mockRobot, active: null, queue: [], logs: [], connected: false, paused: false }
    : mockRobot
  return <div className="app-shell">
    <header className="app-header"><a className="wordmark" href="/" aria-label="Sparkle dashboard"><span className="brand-mark" aria-hidden="true">✳</span> SPARKLE<span className="wordmark-divider">/</span><span className="wordmark-sub">ROBOT CONSOLE</span></a><div className="header-status">{config.mockMode && <span className="outline-tag">MOCK MODE</span>}<span className={robot.connected ? 'connection-dot' : 'connection-dot offline'} /><span>{unavailable ? 'NOT CONFIGURED' : robot.connected ? 'CONNECTED' : 'DISCONNECTED'}</span></div></header>
    <main className="dashboard-main">
    <div className="dashboard-grid"><div className="left-column"><CameraView unavailable={unavailable} connected={robot.connected} target={robot.active?.target ?? null}/><PushToTalk unavailable={unavailable} connected={robot.connected} onCommand={target => dispatch({ type: 'command', target })} onPause={() => dispatch({ type: 'pause' })} onResume={() => dispatch({ type: 'resume' })}/></div><aside><CommandQueue unavailable={unavailable} active={robot.active} queue={robot.queue} paused={robot.paused}/><DecisionLog unavailable={unavailable} entries={robot.logs}/></aside></div>
    <footer className="app-footer">
      <span>SPARKLE / OPERATOR INTERFACE</span>
      {config.mockMode && <div className="demo-tools">
        <span>DEMO</span>
        {robot.paused && robot.connected && <button onClick={() => dispatch({ type: 'resume' })}>Resume demo ↗</button>}
        <button disabled={!robot.active || robot.paused || !robot.connected} onClick={() => dispatch({ type: 'complete' })}>
          {!robot.connected ? 'Complete target (disconnected)' : robot.paused ? 'Complete target (resume first)' : !robot.active ? 'No target to complete' : 'Complete target ↗'}
        </button>
        <button onClick={() => dispatch({ type: 'connection' })}>{robot.connected ? 'Simulate disconnect' : 'Restore connection'} ↗</button>
      </div>}
    </footer></main>
  </div>
}
export default App
