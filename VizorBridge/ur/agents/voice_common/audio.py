"""Shared audio processing utilities for voice agents.

This module provides common audio configuration, processing, and utilities
used by both OpenAI and SmolAgent voice implementations.
"""

import asyncio
import base64
import os
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠ Audio libraries not available. Install with: uv add sounddevice numpy")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ OpenAI not available. Install with: uv add openai")


# Shared audio configuration
AUDIO_CONFIG = {
    "samplerate": 24000,
    "channels": 1,
    "format": "pcm16",
    "blocksize": 1024,
    "duration": 0.3,  # Max recording duration in seconds
    "dtype": "int16",
}


@dataclass
class AudioLevel:
    """Audio level measurement utilities."""
    
    @staticmethod
    def calculate_level(audio_data: np.ndarray) -> float:
        """Calculate the audio level (RMS) from audio data."""
        if len(audio_data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_data ** 2)))
    
    @staticmethod
    def calculate_peak_level(audio_data: np.ndarray) -> float:
        """Calculate the peak audio level from audio data."""
        if len(audio_data) == 0:
            return 0.0
        return float(np.max(np.abs(audio_data)))
    
    @staticmethod
    def is_above_threshold(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if audio level is above the specified threshold."""
        level = AudioLevel.calculate_peak_level(audio_data)
        return level > threshold


class SimpleVoiceProcessor:
    """Simple voice processor for audio recording, transcription, and TTS.
    
    This class provides a unified interface for audio processing that can be
    used by both OpenAI and SmolAgent implementations.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.openai_client = None
        self._setup_openai()
        
        # Audio control state
        self.current_recording = None
        self.is_recording = False
        
    def _setup_openai(self):
        """Setup OpenAI client for STT and TTS."""
        if not OPENAI_AVAILABLE:
            return
            
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
    
    def check_audio_availability(self) -> bool:
        """Check if audio processing is available."""
        return AUDIO_AVAILABLE
    
    def record_audio(self, duration: float = None) -> Optional[bytes]:
        """Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds (uses config default if None)
            
        Returns:
            Audio data as bytes in WAV format, or None if recording failed
        """
        if not AUDIO_AVAILABLE:
            return None
            
        if duration is None:
            duration = AUDIO_CONFIG["duration"]
            
        try:
            print(f"🎤 Recording for {duration} seconds... (speak now)")
            
            self.is_recording = True
            
            # Record audio
            recording = sd.rec(
                int(duration * AUDIO_CONFIG["samplerate"]),
                samplerate=AUDIO_CONFIG["samplerate"],
                channels=AUDIO_CONFIG["channels"],
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            self.is_recording = False
            
            # Convert to wav bytes
            return self._convert_to_wav_bytes(recording)
                
        except Exception as e:
            self.is_recording = False
            print(f"❌ Audio recording error: {e}")
            return None
    
    async def record_audio_async(self, duration: float = None) -> Optional[bytes]:
        """Async version of audio recording."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.record_audio, duration)
    
    def _convert_to_wav_bytes(self, recording: np.ndarray) -> Optional[bytes]:
        """Convert numpy audio array to WAV format bytes."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Convert to int16 for wav format
                audio_int16 = (recording * 32767).astype(np.int16)
                
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(AUDIO_CONFIG["channels"])
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(AUDIO_CONFIG["samplerate"])
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Read back as bytes
                with open(temp_file.name, 'rb') as f:
                    audio_bytes = f.read()
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
                return audio_bytes
                
        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return None
    
    def transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper.
        
        Args:
            audio_bytes: Audio data in WAV format
            
        Returns:
            Transcribed text or None if transcription failed
        """
        if not self.openai_client or not audio_bytes:
            return None
            
        try:
            # Save audio to temp file for OpenAI API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                # Transcribe using OpenAI Whisper
                with open(temp_file.name, 'rb') as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
                return transcript.strip() if transcript else None
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    async def transcribe_audio_async(self, audio_bytes: bytes) -> Optional[str]:
        """Async version of audio transcription."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_audio, audio_bytes)
    
    def text_to_speech(self, text: str, voice: str = None) -> Optional[bytes]:
        """Convert text to speech using OpenAI TTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (uses config default if None)
            
        Returns:
            Audio data in MP3 format or None if TTS failed
        """
        if not self.openai_client or not text:
            return None
            
        if voice is None:
            voice = getattr(self.config, 'tts_voice', 'alloy') if self.config else 'alloy'
            
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            return response.content
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None
    
    async def text_to_speech_async(self, text: str, voice: str = None) -> Optional[bytes]:
        """Async version of text to speech."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.text_to_speech, text, voice)
    
    def play_audio(self, audio_bytes: bytes, audio_format: str = "mp3"):
        """Play audio bytes through speakers.
        
        Args:
            audio_bytes: Audio data to play
            audio_format: Format of the audio data ("mp3", "wav", etc.)
        """
        if not AUDIO_AVAILABLE or not audio_bytes:
            return
            
        try:
            # Save to temp file and play
            suffix = f".{audio_format}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                if audio_format == "wav":
                    # For WAV files, we can play directly with sounddevice
                    self._play_wav_file(temp_file.name)
                else:
                    # For other formats (like MP3), just indicate playback
                    print("🔊 Playing audio response...")
                
                # Clean up
                os.unlink(temp_file.name)
                
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
    
    def _play_wav_file(self, wav_path: str):
        """Play a WAV file using sounddevice."""
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                samplerate = wav_file.getframerate()
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to float and normalize
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # Play audio
                sd.play(audio_float, samplerate)
                sd.wait()  # Wait for playback to complete
                
        except Exception as e:
            print(f"❌ WAV playback error: {e}")
    
    async def play_audio_async(self, audio_bytes: bytes, audio_format: str = "mp3"):
        """Async version of audio playback."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.play_audio, audio_bytes, audio_format)
    
    def stop_recording(self):
        """Stop any ongoing recording."""
        if self.is_recording:
            try:
                sd.stop()
                self.is_recording = False
            except:
                pass


def create_audio_callback(audio_queue: asyncio.Queue, config, 
                         is_speaking_callback=None, debug_mode: bool = False):
    """Create an audio input callback for streaming applications.
    
    Args:
        audio_queue: Queue to put audio chunks into
        config: Configuration object with audio settings
        is_speaking_callback: Function to check if assistant is currently speaking
        debug_mode: Whether to print debug information
        
    Returns:
        Audio callback function suitable for sounddevice.InputStream
    """
    
    def audio_callback(indata, frames, time_info, status):
        try:
            if status:
                if debug_mode:
                    print(f"Audio input status: {status}")
            
            should_record = True
            audio_level = AudioLevel.calculate_peak_level(indata)
            
            # Check if assistant is speaking to prevent feedback
            if is_speaking_callback and is_speaking_callback():
                should_record = False
                if debug_mode:
                    print("🔇 Microphone blocked: Assistant is speaking")
            
            # Apply audio level threshold
            elif audio_level < config.audio_threshold:
                should_record = False
                if debug_mode and audio_level > 0.001:
                    print(f"🔇 Audio too quiet: level={audio_level:.4f} < threshold={config.audio_threshold}")
            
            if should_record:
                # Convert to int16 PCM format
                audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                try:
                    audio_queue.put_nowait(audio_int16.tobytes())
                    if debug_mode:
                        print(f"🎤 Recording audio: level={audio_level:.4f}")
                except:
                    # Queue might be full, skip this chunk
                    if debug_mode:
                        print(f"⚠ Audio queue full, skipping chunk")
                    pass
            elif debug_mode and audio_level > config.audio_threshold:
                print(f"🔇 Audio blocked but above threshold: level={audio_level:.4f}")
                
        except Exception as e:
            if debug_mode:
                print(f"❌ Audio callback error: {e}")
                
    return audio_callback


def encode_audio_for_openai(audio_bytes: bytes) -> str:
    """Encode audio bytes to base64 for OpenAI Realtime API."""
    return base64.b64encode(audio_bytes).decode('utf-8')


def decode_audio_from_openai(base64_audio: str) -> bytes:
    """Decode base64 audio from OpenAI Realtime API."""
    return base64.b64decode(base64_audio)


def convert_audio_to_numpy(audio_bytes: bytes) -> np.ndarray:
    """Convert audio bytes to numpy array for processing."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def convert_numpy_to_audio(audio_array: np.ndarray) -> bytes:
    """Convert numpy audio array back to bytes."""
    return audio_array.astype(np.int16).tobytes()


def play_audio(audio_data: bytes, device: Optional[int] = None):
    """
    Plays audio data using sounddevice in a separate thread.
    
    Args:
        audio_data: Raw audio bytes to play.
        device: The output device ID to use.
    """
    if not AUDIO_AVAILABLE:
        print("Cannot play audio, sounddevice library not available.")
        return

    try:
        # Create a stream for playback
        stream = sd.RawOutputStream(
            samplerate=AUDIO_CONFIG["samplerate"],
            blocksize=AUDIO_CONFIG["blocksize"],
            channels=AUDIO_CONFIG["channels"],
            dtype=AUDIO_CONFIG["dtype"],
            device=device
        )
        with stream:
            stream.write(audio_data)
    except Exception as e:
        print(f"❌ Audio playback error: {e}") 