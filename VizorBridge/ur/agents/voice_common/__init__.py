"""Shared voice agent components for VizorBridge.

This module contains common functionality used by both OpenAI and SmolAgent
voice implementations, including configuration, audio processing, status display,
and logging utilities.
"""

from ur.config.voice_config import VoiceAgentConfig, load_config_from_env
from .audio import AUDIO_CONFIG, SimpleVoiceProcessor, AudioLevel
from .status import VoiceAgentStatus

__all__ = [
    'VoiceAgentConfig',
    'load_config_from_env', 
    'AUDIO_CONFIG',
    'SimpleVoiceProcessor',
    'AudioLevel',
    'VoiceAgentStatus'
] 