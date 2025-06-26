"""Unified voice agent configuration management.

This module provides a common configuration system for both OpenAI and SmolAgent
voice implementations, handling environment variable loading and validation.

Environment Variables:
    Core Modes:
        TEXT_ONLY_MODE: Use text-only mode (true/false, default: false)
        ENABLE_TEXT_INPUT: Allow text input alongside voice (true/false, default: true)
    
    Audio Behavior:
        ENABLE_INTERRUPTION: Allow interrupting assistant speech (true/false, default: true)
        POST_SPEECH_DELAY: Delay after speaking to prevent feedback (float, default: 0.8)
        SILENCE_AFTER_SPEAKING: Silence period after assistant stops (float, default: 0.8)
        AUDIO_THRESHOLD: Minimum audio level for speech detection (0.0-1.0, default: 0.01)
        FEEDBACK_PREVENTION: Enable aggressive feedback prevention (true/false, default: false)
    
    Voice Activity Detection (VAD) - OpenAI Realtime API:
        VAD_THRESHOLD: VAD sensitivity threshold (0.0-1.0, higher = less sensitive, default: 0.4)
        VAD_SILENCE_DURATION_MS: Milliseconds of silence before processing speech (int, default: 500)
        VAD_PREFIX_PADDING_MS: Milliseconds of audio before speech detection (int, default: 200)
    
    Performance & Debugging:
        THREADED_TOOLS: Execute robot tools in background threads (true/false, default: true)
        DEBUG: Enable detailed debug output (true/false, default: false)
        ENABLE_LOGGING: Enable session logging (true/false, default: false)
    
    AI Model (SmolAgent):
        MODEL_PROVIDER: AI provider (openai/huggingface/litellm, default: openai)
        MODEL_ID: Model identifier (default: varies by provider)
        MAX_STEPS: Maximum steps for SmolAgent (int, default: 10)
        VERBOSE_AGENT: Verbose agent output (true/false, default: true)
        ENABLE_TTS: Enable text-to-speech (true/false, default: false)
        TTS_VOICE: OpenAI TTS voice (default: alloy)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from ur.config.system_config import SMOL_MODEL_ID

# Audio format constants
OPENAI_INPUT_AUDIO_FORMAT = "pcm16"
OPENAI_OUTPUT_AUDIO_FORMAT = "pcm16"


@dataclass
class VoiceAgentConfig:
    """Unified voice agent configuration supporting both OpenAI and SmolAgent modes."""
    
    # Core interaction modes
    text_only_mode: bool = False  # Set True to disable voice and use text only
    enable_text_input: bool = True  # Allow text input alongside voice
    
    # AI Model settings (primarily for SmolAgent)
    model_provider: str = "openai"  # openai, huggingface, litellm
    model_id: str = SMOL_MODEL_ID  # Model to use
    
    # Audio behavior
    enable_interruption: bool = True  # Allow interrupting assistant speech
    post_speech_delay: float = 0.8  # Extended delay after speaking to prevent feedback
    
    # Enhanced audio feedback prevention
    silence_after_speaking: float = 0.8  # Silence period after assistant stops speaking
    audio_threshold: float = 0.005  # Minimum audio level to consider as speech (0.0-1.0)
    feedback_prevention_enabled: bool = False  # Enable aggressive feedback prevention
    
    # Voice Activity Detection (VAD) settings for OpenAI Realtime API
    vad_threshold: float = 0.4  # VAD sensitivity threshold (0.0-1.0, higher = less sensitive)
    vad_silence_duration_ms: int = 500  # Milliseconds of silence before considering speech finished
    vad_prefix_padding_ms: int = 200  # Milliseconds of audio before speech detection
    
    # Performance & debugging
    threaded_execution: bool = True  # Execute robot tools in background threads (unified name)
    debug_mode: bool = True  # Enable detailed debug output
    
    # SmolAgent specific settings
    max_steps: int = 10  # Max steps for SmolAgent
    verbose_agent: bool = True  # Verbose output from agent
    
    # TTS settings
    tts_voice: str = "alloy"  # OpenAI TTS voice
    enable_tts: bool = False  # Enable text-to-speech
    
    # Session logging
    enable_logging: bool = False  # Enable session logging
    log_directory: str = "logs/sessions"  # Directory for session logs
    
    # OpenAI Realtime specific settings
    openai_voice: str = "alloy"
    openai_temperature: float = 0.7
    commentary_temperature: float = 0.3
    commentary_max_tokens: int = 50
    
    @property
    def threaded_tools(self) -> bool:
        """Alias for threaded_execution for OpenAI compatibility."""
        return self.threaded_execution
    
    @threaded_tools.setter 
    def threaded_tools(self, value: bool):
        """Alias setter for threaded_execution."""
        self.threaded_execution = value
    
    @classmethod
    def from_environment(cls) -> 'VoiceAgentConfig':
        """Load configuration from environment variables."""
        return cls(
            # Core modes
            text_only_mode=_env_bool("TEXT_ONLY_MODE", False),
            enable_text_input=_env_bool("ENABLE_TEXT_INPUT", True),
            
            # AI Model settings
            model_provider=os.getenv("MODEL_PROVIDER", "openai"),
            model_id=os.getenv("MODEL_ID", SMOL_MODEL_ID),
            
            # Audio behavior
            enable_interruption=_env_bool("ENABLE_INTERRUPTION", True),
            post_speech_delay=_env_float("POST_SPEECH_DELAY", 0.8),
            silence_after_speaking=_env_float("SILENCE_AFTER_SPEAKING", 0.8),
            audio_threshold=_env_float("AUDIO_THRESHOLD", 0.005),
            feedback_prevention_enabled=_env_bool("FEEDBACK_PREVENTION", False),
            
            # VAD settings
            vad_threshold=_env_float("VAD_THRESHOLD", 0.4),
            vad_silence_duration_ms=_env_int("VAD_SILENCE_DURATION_MS", 500),
            vad_prefix_padding_ms=_env_int("VAD_PREFIX_PADDING_MS", 200),
            
            # Performance
            threaded_execution=_env_bool("THREADED_TOOLS", True) or _env_bool("THREADED_CODE_EXECUTION", True),
            debug_mode=_env_bool("DEBUG", False),
            
            # SmolAgent specific
            max_steps=_env_int("MAX_STEPS", 10),
            verbose_agent=_env_bool("VERBOSE_AGENT", True),
            
            # TTS
            tts_voice=os.getenv("TTS_VOICE", "alloy"),
            enable_tts=_env_bool("ENABLE_TTS", False),
            
            # Logging
            enable_logging=_env_bool("ENABLE_LOGGING", False),
            log_directory=os.getenv("LOG_DIRECTORY", "logs/sessions"),
            
            # OpenAI Realtime specific
            openai_voice=os.getenv("OPENAI_VOICE", "alloy"),
            openai_temperature=_env_float("OPENAI_TEMPERATURE", 0.7),
            commentary_temperature=_env_float("COMMENTARY_TEMPERATURE", 0.3),
            commentary_max_tokens=_env_int("COMMENTARY_MAX_TOKENS", 50)
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if self.model_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider")
        elif self.model_provider == "huggingface" and not os.getenv("HF_TOKEN"):
            raise ValueError("HF_TOKEN not set for HuggingFace provider")
        
        if self.audio_threshold < 0.0 or self.audio_threshold > 1.0:
            raise ValueError("audio_threshold must be between 0.0 and 1.0")
        
        if self.post_speech_delay < 0.0:
            raise ValueError("post_speech_delay must be non-negative")


def _env_bool(key: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    value = os.getenv(key, "").lower()
    if value in ["true", "1", "yes", "on"]:
        return True
    elif value in ["false", "0", "no", "off"]:
        return False
    else:
        return default


def _env_float(key: str, default: float) -> float:
    """Parse float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """Parse integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def load_env() -> None:
    """Load environment variables from .env files.
    
    Searches for .env files in the following order:
    1. Same directory as the calling script with .env extension
    2. Project root directory (.env)
    """
    # Try current script directory first
    import inspect
    caller_frame = inspect.currentframe().f_back
    if caller_frame and caller_frame.f_globals.get('__file__'):
        script_path = Path(caller_frame.f_globals['__file__'])
        env_path = script_path.with_suffix('.env')
        if env_path.exists():
            _load_env_file(env_path)
            return
    
    # Fall back to project root
    project_root = Path(__file__).resolve().parents[3]  # Go up from this file
    env_path = project_root / ".env"
    if env_path.exists():
        _load_env_file(env_path)


def _load_env_file(env_path: Path) -> None:
    """Load environment variables from a specific file."""
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    except Exception as e:
        print(f"Warning: Failed to load {env_path}: {e}")


def load_config_from_env() -> VoiceAgentConfig:
    """Load and return configuration from environment with optional debug output."""
    load_env()
    config = VoiceAgentConfig.from_environment()
    
    if config.debug_mode:
        print(f"🔧 Voice Agent Configuration:")
        print(f"  Text-only mode: {config.text_only_mode}")
        print(f"  Text input enabled: {config.enable_text_input}")
        print(f"  Model provider: {config.model_provider}")
        print(f"  Model ID: {config.model_id}")
        print(f"  Audio interruption: {config.enable_interruption}")
        print(f"  Feedback prevention: {config.feedback_prevention_enabled}")
        print(f"  Silence period: {config.silence_after_speaking}s")
        print(f"  Audio threshold: {config.audio_threshold}")
        print(f"  Post-speech delay: {config.post_speech_delay}s")
        print(f"  VAD threshold: {config.vad_threshold}")
        print(f"  VAD silence duration: {config.vad_silence_duration_ms}ms")
        print(f"  VAD prefix padding: {config.vad_prefix_padding_ms}ms")
        print(f"  Threaded execution: {config.threaded_execution}")
        print(f"  Debug mode: {config.debug_mode}")
        print(f"  Session logging: {config.enable_logging}")
        if config.model_provider != "openai":
            print(f"  Max steps: {config.max_steps}")
            print(f"  Verbose agent: {config.verbose_agent}")
            print(f"  TTS enabled: {config.enable_tts}")
    
    return config


# Global config instance - can be imported and used directly
CONFIG: Optional[VoiceAgentConfig] = None


def get_config() -> VoiceAgentConfig:
    """Get the global configuration instance, loading it if necessary."""
    global CONFIG
    if CONFIG is None:
        CONFIG = load_config_from_env()
    return CONFIG


def update_config(**kwargs) -> None:
    """Update the global configuration with new values."""
    global CONFIG
    if CONFIG is None:
        CONFIG = load_config_from_env()
    
    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}") 