import { CameraView } from './components/CameraView'
import { CommandQueue } from './components/CommandQueue'
import { DecisionLog } from './components/DecisionLog'
import { PushToTalk } from './components/PushToTalk'
import { StopButton } from './components/StopButton'
import { TargetInput } from './components/TargetInput'
import { VoiceControl } from './components/VoiceControl'
import { useRobotConnection } from './hooks/useRobotConnection'
import { useMockRobot } from './hooks/useMockRobot'
import { config } from './config/environment'
import voidSparkleLogo from './assets/htn26 void sparkle.png'
import './App.css'

function App() {
  const { robot: mockRobot, dispatch } = useMockRobot()
  const live = useRobotConnection(!config.mockMode)
  const unavailable = !config.mockMode && !live.robot.ready
  const robot = config.mockMode ? mockRobot : live.robot
  return <div className="app-shell">
    <header className="app-header"><a className="wordmark" href="/" aria-label="void-sparkle dashboard"><img src={voidSparkleLogo} alt="void-sparkle" /></a><div className="header-status">{config.mockMode && <span className="outline-tag">MOCK MODE</span>}<span className={robot.connected ? 'connection-dot' : 'connection-dot offline'} /><span>{unavailable ? 'CONNECTING' : robot.connected ? 'CONNECTED' : 'DISCONNECTED'}</span>{!config.mockMode && <StopButton onStop={live.stop}/>}</div></header>
    <main className="dashboard-main">
    {!config.mockMode && live.robot.message && <div className="robot-message" role="status">{live.robot.message}</div>}
    <div className="dashboard-grid"><div className="left-column"><CameraView mock={config.mockMode} unavailable={unavailable} connected={robot.connected} target={robot.active?.target ?? null}/>{config.mockMode ? <PushToTalk unavailable={unavailable} connected={robot.connected} onCommand={target => dispatch({ type: 'command', target })} onPause={() => dispatch({ type: 'pause' })} onResume={() => dispatch({ type: 'resume' })}/> : <><VoiceControl key={String(robot.connected)} connected={robot.connected} capabilities={live.robot.capabilities} onResult={live.accept}/><TargetInput connected={robot.connected} paused={robot.paused} capabilities={live.robot.capabilities} onResult={live.accept}/></>}</div><aside><CommandQueue unavailable={unavailable} active={robot.active} queue={robot.queue} paused={robot.paused}/><DecisionLog unavailable={unavailable} entries={robot.logs}/></aside></div>
    {config.mockMode && <footer className="app-footer"><div className="demo-tools">
        <span>DEMO</span>
        {robot.paused && robot.connected && <button onClick={() => dispatch({ type: 'resume' })}>Resume demo ↗</button>}
        <button disabled={!robot.active || robot.paused || !robot.connected} onClick={() => dispatch({ type: 'complete' })}>
          {!robot.connected ? 'Complete target (disconnected)' : robot.paused ? 'Complete target (resume first)' : !robot.active ? 'No target to complete' : 'Complete target ↗'}
        </button>
        <button onClick={() => dispatch({ type: 'connection' })}>{robot.connected ? 'Simulate disconnect' : 'Restore connection'} ↗</button>
      </div></footer>}</main>
  </div>
}
export default App
