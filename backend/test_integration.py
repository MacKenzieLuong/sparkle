import asyncio
import io
import json
import threading
import wave
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from camera import FakeCamera
from drive import FakeDriver
from scenarios import FakeScene
from server import _build_app, ControlLoop
from speech import SpeechService, validate_interpretation, validate_wav
from vision import DetectedObject


def audio(seconds=1):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b'\0\0' * int(seconds * 16000))
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch):
    for key, value in {'MOCK': 'true', 'SPEECH_PROVIDER': 'fake', 'CAMERA': 'fake', 'DRIVER': 'fake', 'CONTROL_INTERVAL': '0.01'}.items():
        monkeypatch.setenv(key, value)
    with TestClient(_build_app()) as test_client:
        yield test_client


def test_audio_to_target_and_command_receipt(client):
    response = client.post('/voice/interpret', content=audio(), headers={'Content-Type': 'audio/wav', 'X-Request-ID': 'voice-1'})
    assert response.status_code == 200
    assert response.json()['intent'] == 'navigate'
    assert response.json()['target'] == 'blue flag'
    assert response.json()['simulated'] is True
    assert client.get('/status').json()['running'] is False  # Interpretation has no motor side effect.
    snapshot = client.get('/status').json()
    command = {'target': response.json()['target'], 'commandId': 'cmd-1', 'sessionId': 'test', 'expectedRevision': snapshot['revision']}
    assert client.post('/direct', json=command).status_code == 200
    assert client.post('/direct', json=command).json()['duplicate'] is True
    assert client.get('/commands/cmd-1').json()['target'] == 'blue flag'
    assert client.post('/direct', json={**command, 'commandId': 'cmd-2'}).status_code == 409
    client.post('/stop')
    assert client.get('/status').json()['running'] is False
    assert client.post('/direct', json={**command, 'commandId': 'cmd-late'}).status_code == 409


def test_upload_limits_and_format(client):
    headers = {'Content-Type': 'audio/wav', 'X-Request-ID': 'voice-2'}
    assert client.post('/voice/interpret', content=audio(10), headers=headers).status_code == 200
    assert client.post('/voice/interpret', content=audio(10.1), headers=headers).status_code == 413
    assert client.post('/voice/interpret', content=b'not wav', headers=headers).status_code == 422
    assert client.post('/voice/interpret', content=b'x' * 2_000_001, headers=headers).status_code == 413
    assert client.post('/voice/interpret', content=audio(), headers={**headers, 'Content-Type': 'audio/webm'}).status_code == 415


@pytest.mark.parametrize('transcript', ['go over there', "don't go to the blue flag", 'go to blue or red flag', 'go to red ball then blue flag'])
def test_ambiguous_commands_rejected(client, monkeypatch, transcript):
    monkeypatch.setenv('FAKE_VOICE_TRANSCRIPT', transcript)
    response = client.post('/voice/interpret', content=audio(), headers={'Content-Type': 'audio/wav', 'X-Request-ID': 'ambiguous'})
    assert response.json()['intent'] == 'reject'
    assert 'cannot be accepted' in response.json()['message']
    assert not client.get('/status').json()['running']


def test_custom_stop_word_and_bad_model_output():
    assert validate_interpretation('{"transcript":"freeze", "intent":"reject"}', 'freeze', 'continue').intent == 'stop'
    assert validate_interpretation('{"transcript":"go to blue flag", "intent":"stop"}', 'freeze', 'continue').intent == 'reject'
    assert validate_interpretation('garbage', 'stop', 'resume').intent == 'reject'


class BlockingVision:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()
    def start(self, target):
        pass
    def stop(self):
        pass
    def detect(self, target, frame):
        self.started.set()
        self.release.wait(2)
        self.done.set()
        return DetectedObject((350, 400, 650, 600), target)


def test_stop_and_watchdog_discard_late_inference():
    scene = FakeScene([(350, 400, 650, 600)])
    driver, vision = FakeDriver(), BlockingVision()
    loop = ControlLoop(FakeCamera(scene), vision, driver)
    loop.start('ball', 'one', 'operator', 0)
    assert vision.started.wait(1)
    loop._heartbeat -= 10
    loop.check_watchdog()
    assert loop.snapshot()['status'] == 'connection_lost'
    vision.release.set()
    assert vision.done.wait(1)
    # Lock barrier plus epoch assertion ensures a pending apply cannot pass.
    with loop._lock:
        assert loop._epoch == 2
        assert driver.last == (0, 0)
        assert not loop.state['running']


def test_provider_failure_is_sanitized(client):
    async def fail(data):
        raise RuntimeError('SECRET provider internals')
    client.app.state.speech.interpret = fail
    response = client.post('/voice/interpret', content=audio(), headers={'Content-Type': 'audio/wav', 'X-Request-ID': 'fail'})
    assert response.status_code == 503
    assert 'SECRET' not in response.text


def test_provider_http_status_is_visible_without_provider_message(client):
    class ProviderFailure(Exception):
        status_code = 404
        code = 'ModelNotFound'
    async def fail(data):
        raise ProviderFailure('SECRET provider internals')
    client.app.state.speech.interpret = fail
    response = client.post('/voice/interpret', content=audio(), headers={'Content-Type': 'audio/wav', 'X-Request-ID': 'fail-http'})
    assert response.status_code == 503
    assert response.json()['detail']['upstreamStatus'] == 404
    assert response.json()['detail']['upstreamCode'] == 'ModelNotFound'
    assert 'SECRET' not in response.text


def test_qwen_chat_completions_audio_shape(monkeypatch):
    monkeypatch.setenv('SPEECH_PROVIDER', 'qwen')
    monkeypatch.setenv('HUAWEI_VOICE_MODEL', 'qwen3.8-omni-flash')
    captured = {}
    class Stream:
        def __aiter__(self):
            async def chunks():
                for content in ['{"transcript":"go to red ball",', '"intent":"navigate","target":"red ball"}']:
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])
            return chunks()
        async def close(self):
            captured['closed'] = True
    async def create(**kwargs):
        captured.update(kwargs)
        return Stream()
    service = SpeechService()
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = asyncio.run(service.interpret(audio()))
    assert result.target == 'red ball'
    assert captured['model'] == 'qwen3.8-omni-flash'
    assert captured['stream'] is True
    media = captured['messages'][1]['content'][0]
    assert media['type'] == 'input_audio'
    assert media['input_audio']['format'] == 'wav'
    assert media['input_audio']['data'].startswith('data:;base64,')
    assert captured['closed']
