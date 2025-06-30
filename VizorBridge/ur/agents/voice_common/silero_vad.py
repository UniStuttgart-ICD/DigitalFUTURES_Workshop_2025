"""Silero VAD integration for voice activity detection."""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

try:
    import silero_vad
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SileroVADProcessor:
    """Silero VAD processor for real-time speech detection."""
    
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250, 
                 min_silence_duration_ms: int = 500, window_size_samples: int = 512):
        if not SILERO_AVAILABLE:
            raise ImportError("silero-vad is required. Install with: pip install silero-vad")
            
        self.model = silero_vad.load_silero_vad()
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        
        # State tracking
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.sample_rate = 16000  # Silero expects 16kHz
        self.current_time_ms = 0.0
        
        # Audio buffering for fixed-size processing
        self.audio_buffer = np.array([], dtype=np.int16)
        
        logger.info(f"Silero VAD initialized: threshold={threshold}, min_speech={min_speech_duration_ms}ms, min_silence={min_silence_duration_ms}ms")
        
    def process_audio_chunk(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """Process audio chunk and return VAD results.
        
        Args:
            audio_data: Int16 audio data
            sample_rate: Sample rate of input audio
            
        Returns:
            dict with keys: 'speech_detected', 'speech_ended', 'should_flush'
        """
        # Resample to 16kHz if needed (Silero requirement)
        if sample_rate != self.sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, self.sample_rate)
        
        # Add new audio to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        
        # Initialize result
        result = {
            'speech_detected': False,
            'speech_ended': False, 
            'should_flush': False,
            'speech_probability': 0.0
        }
        
        # Process audio in fixed-size windows that Silero expects
        while len(self.audio_buffer) >= self.window_size_samples:
            # Extract exactly window_size_samples for Silero processing
            window = self.audio_buffer[:self.window_size_samples]
            self.audio_buffer = self.audio_buffer[self.window_size_samples:]
            
            # Convert to float32 tensor for Silero (normalize int16 to [-1, 1])
            audio_tensor = torch.from_numpy(window.astype(np.float32) / 32768.0)
            
            try:
                # Get speech probability from Silero VAD
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
                result['speech_probability'] = speech_prob
                
                # Update current time based on window length
                window_duration_ms = len(window) / self.sample_rate * 1000
                self.current_time_ms += window_duration_ms
                
                # State machine logic
                if speech_prob > self.threshold:
                    # Speech detected
                    if not self.is_speech_active:
                        self.is_speech_active = True
                        self.speech_start_time = self.current_time_ms
                        self.silence_start_time = None
                        result['speech_detected'] = True
                        logger.debug(f"Speech started at {self.current_time_ms:.1f}ms (prob={speech_prob:.3f})")
                    
                else:
                    # Silence detected
                    if self.is_speech_active:
                        if self.silence_start_time is None:
                            self.silence_start_time = self.current_time_ms
                        
                        silence_duration = self.current_time_ms - self.silence_start_time
                        if silence_duration >= self.min_silence_duration_ms:
                            # Speech ended after sufficient silence
                            speech_duration = self.silence_start_time - self.speech_start_time
                            
                            # Only trigger speech end if we had enough speech
                            if speech_duration >= self.min_speech_duration_ms:
                                self.is_speech_active = False
                                result['speech_ended'] = True
                                result['should_flush'] = True
                                logger.debug(f"Speech ended at {self.current_time_ms:.1f}ms (duration={speech_duration:.1f}ms, silence={silence_duration:.1f}ms)")
                            else:
                                # Too short, consider it noise - reset to silence state
                                self.is_speech_active = False
                                logger.debug(f"Speech too short ({speech_duration:.1f}ms), ignoring")
                            
                            self.speech_start_time = None
                            self.silence_start_time = None
                            break  # Exit window processing loop early on speech end
                            
            except Exception as e:
                logger.error(f"Silero VAD processing error: {e}")
                # Continue with default result on error
                break
        
        return result
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling (basic linear interpolation)."""
        if orig_sr == target_sr:
            return audio
        
        # Basic linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        
        if target_length == 0:
            return np.array([], dtype=np.int16)
        
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)
    
    def reset(self):
        """Reset VAD state."""
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.current_time_ms = 0.0
        self.audio_buffer = np.array([], dtype=np.int16)
        logger.debug("VAD state reset") 