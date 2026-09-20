"""Exercise the real HTTP/proxy path, restricted to fake hardware/providers."""
import argparse
import io
import time
import uuid
import wave

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5173')
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=10) as client:
        def call(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        status = call('GET', '/status')
        capabilities = call('GET', '/capabilities')
        if not status['mock'] or status['driver'] is None or capabilities['speechProvider'] != 'fake':
            raise SystemExit('Smoke test requires fake vision, fake driver, and fake speech.')
        if status['running']:
            raise SystemExit('A task is already running. Finish it before running this test.')
        data = io.BytesIO()
        with wave.open(data, 'wb') as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(16000)
            recording.writeframes(b'\0\0' * 16000)
        result = call('POST', '/voice/interpret', content=data.getvalue(),
                      headers={'Content-Type': 'audio/wav', 'X-Request-ID': str(uuid.uuid4())})
        assert result['intent'] == 'navigate', 'Use a navigation FAKE_VOICE_TRANSCRIPT for this smoke test.'
        assert not call('GET', '/status')['running'], 'Interpretation must not start motors.'
        print('PASS: WAV upload through proxy -> scripted interpretation; no motion side effect.')
        command_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
        status = call('GET', '/status')
        call('POST', '/direct', json={'target': result['target'], 'commandId': command_id,
             'sessionId': session_id, 'expectedRevision': status['revision'], 'resume': True})
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                call('POST', '/heartbeat', json={'sessionId': session_id})
                status = call('GET', '/status')
                if not status['running']:
                    assert status['command_id'] == command_id and status['status'] == 'arrived', status
                    receipt = call('GET', '/commands/' + command_id)
                    assert receipt['status'] == 'arrived'
                    print('PASS: correlated navigation -> arrived -> command receipt; heartbeats maintained.')
                    break
                time.sleep(0.5)
            else:
                raise AssertionError('Navigation did not complete within 45 seconds.')
        finally:
            if call('GET', '/status')['running']:
                call('POST', '/stop')


if __name__ == '__main__':
    main()
