import pytest
import base64
from types import SimpleNamespace
from ur.agents.openai_agent import RealtimeAgent, RealtimeSession
from ur.config.voice_config import VoiceAgentConfig

@pytest.mark.anyio
async def test_text_buffering_and_display(monkeypatch):
    # Setup config and session
    config = VoiceAgentConfig()
    config.enable_hybrid_mode = True
    config.text_only_mode = False
    config.threaded_execution = False
    config.debug_mode = False
    agent = RealtimeAgent(name="test", instructions="instr", tools=[])
    session = RealtimeSession(agent, api_key="dummy", config=config)

    # Stub playback and print_message
    played = []
    printed = []
    async def fake_play(audio):
        played.append(audio)
    monkeypatch.setattr(session, '_play_audio_response', fake_play)
    def fake_print(msg, color=None): printed.append((msg, color))
    session.status.print_message = fake_print

    # Simulate streaming text deltas
    await session._process_server_event(SimpleNamespace(type="response.text.delta", delta="Hello"))
    await session._process_server_event(SimpleNamespace(type="response.text.delta", delta=" World"))
    await session._process_server_event(SimpleNamespace(type="response.text.done"))

    # Trigger response completed
    await session._handle_response_completed(SimpleNamespace(type="response.done", data={} ))

    # No audio should have played
    assert played == []
    # Combined text should be printed once
    assert printed == [("Hello World", "cyan")]
    # Buffers should be reset
    assert session.text_response_buffer == ""

@pytest.mark.anyio
async def test_audio_buffering_and_playback(monkeypatch):
    # Setup config and session
    config = VoiceAgentConfig()
    config.enable_hybrid_mode = True
    config.text_only_mode = False
    config.threaded_execution = False
    config.debug_mode = False
    agent = RealtimeAgent(name="test", instructions="instr", tools=[])
    session = RealtimeSession(agent, api_key="dummy", config=config)

    # Stub playback and print_message
    played = []
    printed = []
    async def fake_play(audio):
        played.append(audio)
    monkeypatch.setattr(session, '_play_audio_response', fake_play)
    session.status.print_message = lambda msg, color=None: printed.append((msg, color))

    # Prepare audio chunks
    chunk1 = b"\x01\x02"
    chunk2 = b"\x03\x04"
    b64_1 = base64.b64encode(chunk1).decode('utf-8')
    b64_2 = base64.b64encode(chunk2).decode('utf-8')

    # Simulate streaming audio deltas
    await session._process_server_event(SimpleNamespace(type="response.audio.delta", delta=b64_1))
    await session._process_server_event(SimpleNamespace(type="response.audio.delta", delta=b64_2))
    await session._process_server_event(SimpleNamespace(type="response.audio.done"))

    # Trigger response completed
    await session._handle_response_completed(SimpleNamespace(type="response.done", data={} ))

    # No text printed
    assert printed == []
    # Audio played equals concatenated chunks
    assert played == [chunk1 + chunk2]
    # Buffers should be reset
    assert session.audio_output_buffer == bytearray() 