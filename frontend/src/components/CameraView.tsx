export function CameraView({ connected, target, unavailable = false }: { unavailable?: boolean; connected: boolean; target: string | null }) {
  return <section className="camera-panel" aria-labelledby="camera-title">
    <div className="panel-heading"><h2 id="camera-title"><span className="section-number">01</span> CAMERA VIEW</h2><span>CAM_01 / FRONT</span></div>
    <div className="camera-stage">
      {connected ? <>
        <div className="camera-caption"><span className="outline-tag">SIMULATED VIEW</span><span>640 × 480</span></div>
        <svg className="scene" viewBox="0 0 800 480" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Illustrated mock camera view of a flag in an empty room">
          <g stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke">
            <path d="M0 322H800 M220 0V250L0 440 M580 0V250L800 440 M220 250H580 M400 250V480 M220 250L120 480 M580 250L680 480 M126 332H674 M50 400H750" opacity=".2" />
            <path d="M475 313V154L535 173L475 193 M457 315H493 M400 213V227 M393 220H407" />
            <path d="M446 142H458 M446 142V154 M548 142H536 M548 142V154 M446 326V314 M446 326H458 M548 326H536 M548 326V314" />
            <path d="M28 60V40H48 M772 60V40H752 M28 420V440H48 M772 420V440H752" />
          </g>
        </svg>
        <div className="camera-bottom"><span>ILLUSTRATION / NO LIVE CAMERA</span><span>+ 00.000 / 00.000</span></div>
      </> : <div className="connection-empty"><span className="disconnect-symbol">×</span><h3>{unavailable ? 'Live connection is not configured' : 'Lost connection'}</h3><p>{unavailable ? 'Camera feed unavailable.' : 'Navigation paused. Waiting for the robot.'}</p></div>}
    </div>
    <div className="camera-foot"><span className="square" /><span>{unavailable ? 'Camera unavailable' : connected ? 'Mock camera ready' : 'Stream suspended'}</span><span className="camera-target">TARGET / {target ?? '—'}</span></div>
  </section>
}
