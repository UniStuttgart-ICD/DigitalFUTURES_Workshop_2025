"""Robot voice agent using SmolAgents CodeAgent framework.

This implementation uses HuggingFace's smolagents CodeAgent for robot control
through natural language voice interaction. Unlike function calling approaches,
SmolAgents generates Python code to execute robot operations.

Features:
- SmolAgents CodeAgent for intelligent code generation
- Simple voice interaction with audio recording
- Safety validation with wake-word requirements
- Code-first approach to robot control
- Multiple model provider support (OpenAI, HuggingFace, Anthropic)

Run:
    python demo_smol_agents_voice.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
import warnings
import threading
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum

# SmolAgents imports
try:
    from smolagents import CodeAgent, OpenAIServerModel, LiteLLMModel, InferenceClientModel
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    print("⚠ SmolAgents not available. Install with: uv add smolagents")

# Allow running directly from the samples folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ur.tools import register_tools_for_smolagents
from ur.agents.shared_prompts import get_system_prompt
from ur.agents.voice_common import (
    VoiceAgentConfig, load_config_from_env, 
    AUDIO_CONFIG, SimpleVoiceProcessor,
    VoiceAgentStatus
)

# Global config instance - now using shared configuration
CONFIG = load_config_from_env()

class EventType(Enum):
    """Event types for the voice agent session."""
    SESSION_STARTED = "session_started"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    TRANSCRIPTION_COMPLETED = "transcription_completed"
    CODE_GENERATION_STARTED = "code_generation_started"
    CODE_EXECUTION_STARTED = "code_execution_started"
    CODE_EXECUTION_COMPLETED = "code_execution_completed"
    RESPONSE_COMPLETED = "response_completed"
    AUDIO_INTERRUPTED = "audio_interrupted"
    ERROR = "error"

# Use shared status class
SmolAgentStatus = VoiceAgentStatus

# SimpleVoiceProcessor is now imported from shared module

class SmolAgentVoiceSession:
    """Voice session using SmolAgents CodeAgent for robot control."""
    
    def __init__(self, config: Optional[VoiceAgentConfig] = None):
        self.config = config or CONFIG
        self.status = SmolAgentStatus("SmolAgent")
        self.agent: Optional[CodeAgent] = None
        self.voice_processor = SimpleVoiceProcessor()
        
        # Session state
        self.is_connected = False
        self.conversation_active = False
        
        # Set up immediate response callback for robot tools
        set_immediate_response_callback(self._send_immediate_response)
        
    async def connect(self):
        """Initialize the SmolAgent."""
        if not SMOLAGENTS_AVAILABLE:
            raise RuntimeError("SmolAgents not available. Install with: uv add smolagents")
        
        self.status.update_status("Initializing SmolAgent...")
        
        # Initialize the SmolAgent model
        await self._setup_smolagent()
        
        self.is_connected = True
        self.status.update_status("Connected and ready!")
        self.status.print_success("SmolAgent voice session ready")
        
    async def _setup_smolagent(self):
        """Setup the SmolAgents CodeAgent."""
        try:
            # Choose model based on provider
            if self.config.model_provider == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY not set")
                model = OpenAIServerModel(
                    model_id=self.config.model_id,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            elif self.config.model_provider == "litellm":
                model = LiteLLMModel(
                    model_id=self.config.model_id,
                    api_key=os.getenv("OPENAI_API_KEY")  # Or other provider key
                )
            else:  # huggingface
                if not os.getenv("HF_TOKEN"):
                    raise ValueError("HF_TOKEN not set for HuggingFace models")
                model = InferenceClientModel(
                    model_id=self.config.model_id,
                    token=os.getenv("HF_TOKEN")
                )

            # Get tools from the new registry system
            robot_tools = register_tools_for_smolagents()

            # Initialize the CodeAgent
            self.agent = CodeAgent(
                tools=robot_tools,
                model=model,
                max_steps=self.config.max_steps,
                verbosity_level=2 if self.config.verbose_agent else 0
            )

            self.status.print_success("SmolAgent CodeAgent initialized")
            
        except Exception as e:
            self.status.print_error(f"Failed to setup SmolAgent: {e}")
            raise
    
    def _send_immediate_response(self, response_text: str):
        """Send immediate response for robot feedback."""
        self.status.print_message(f"🤖 Robot: {response_text}", "green")
    
    async def process_voice_command(self, user_text: str) -> str:
        """Process voice command through SmolAgent."""
        try:
            self.status.update_status(f"Processing: {user_text[:50]}...")
            self.status.print_user_message(user_text)
            
            # Generate enhanced prompt for robot control
            enhanced_prompt = self._create_robot_prompt(user_text)
            
            # Set code generation state
            self.status.set_generating_code(True)
            self.status.update_status("Generating Python code...")
            
            if self.config.threaded_execution:
                # Run agent in thread to prevent blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(executor, self.agent.run, enhanced_prompt)
            else:
                # Run synchronously
                result = self.agent.run(enhanced_prompt)
            
            self.status.set_generating_code(False)
            self.status.set_executing_code(False)
            
            # Extract the final response
            response = str(result) if result else "I encountered an issue processing your request."
            
            self.status.print_assistant_message(response)
            self.status.update_status("Response ready")
            
            return response
            
        except Exception as e:
            self.status.set_generating_code(False)
            self.status.set_executing_code(False)
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.status.print_error(f"Processing error: {e}")
            return error_msg
    
    def _create_robot_prompt(self, user_text: str) -> str:
        """Create an enhanced prompt for robot operations."""
        try:
            from ur.tools.movement_tools import get_robot_state
            robot_state = get_robot_state()
            robot_status = "connected" if isinstance(robot_state, dict) and robot_state.get("connected") else "disconnected"
        except:
            robot_status = "unknown"
        
        prompt = f"""You are UR-HAL-9000, a sophisticated AI assistant controlling a Universal Robot using code generation.

Current robot status: {robot_status}
User request: {user_text}

IMPORTANT SAFETY RULES:
1. ALL robot movements require the wake word "timbra" in the user's request
2. If no wake word is found, politely remind the user and refuse to move the robot  
3. Always validate safety before any movement
4. Generate Python code using the available robot tools
5. Provide clear feedback about what you're doing

Available robot tools (use these as Python functions):
- move_home(wake_phrase="timbra"): Move robot to home position
- get_robot_state(): Get current robot pose and joint positions  
- move_relative_xyz(dx_m, dy_m, dz_m, wake_phrase="timbra"): Move robot relatively in meters
- move_to_absolute_position(x_m, y_m, z_m, wake_phrase="timbra"): Move to absolute position
- move_to_supply_station(distance, wake_phrase="timbra"): Move to predefined supply stations
- control_gripper(action, wake_phrase="timbra"): Open or close gripper (action="open" or "close")
- get_supply_element(element_length, wake_phrase="timbra"): Pick up elements from supply station
- place_element_at_position(x, y, z, wake_phrase="timbra"): Place elements at specific positions
- release_element(wake_phrase="timbra"): Release currently held element
- stop_robot(wake_phrase="timbra"): Emergency stop

Generate Python code to fulfill the user's request. Be conversational and explain what you're doing.
"""
        return prompt
    
    async def start_session(self):
        """Start the voice session and begin processing."""
        if not self.is_connected:
            await self.connect()
        
        try:
            # Show welcome info
            self.status.show_welcome_panel()
            
            # Start animation
            await asyncio.sleep(2)
            await self.status.start_animation()
            
            self.conversation_active = True
            self.status.update_status("Ready for voice or text input!")
            
            # Start input processing
            await self._input_loop()
                
        except KeyboardInterrupt:
            self.status.print_message("\n👋 Shutting down gracefully...", "yellow")
        except Exception as e:
            self.status.print_error(f"Session error: {e}")
        finally:
            await self.stop_session()
    
    async def _input_loop(self):
        """Main input loop for voice and text commands."""
        loop = asyncio.get_event_loop()
        
        while self.conversation_active:
            try:
                # Show input options
                if RICH_AVAILABLE and self.status.console:
                    self.status.console.print("\n💬 Commands: [cyan]'r'[/cyan] to record voice, [cyan]'t'[/cyan] to type, [cyan]'q'[/cyan] to quit")
                    choice = await loop.run_in_executor(None, input, "Choose input method (r/t/q): ")
                else:
                    print("\n💬 Commands: 'r' to record voice, 't' to type, 'q' to quit")
                    choice = await loop.run_in_executor(None, input, "Choose input method (r/t/q): ")
                
                if choice.lower() == 'q':
                    break
                elif choice.lower() == 'r':
                    await self._handle_voice_input()
                elif choice.lower() == 't':
                    await self._handle_text_input()
                else:
                    self.status.print_message("Invalid choice. Use 'r', 't', or 'q'.", "yellow")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.status.print_error(f"Input loop error: {e}")
    
    async def _handle_voice_input(self):
        """Handle voice input recording and processing."""
        try:
            # Record audio
            self.status.set_listening(True)
            self.status.update_status("Recording audio...")
            
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                None, self.voice_processor.record_audio, AUDIO_CONFIG["duration"]
            )
            
            self.status.set_listening(False)
            
            if not audio_bytes:
                self.status.print_error("Failed to record audio")
                return
            
            # Transcribe audio
            self.status.update_status("Transcribing audio...")
            transcript = await loop.run_in_executor(
                None, self.voice_processor.transcribe_audio, audio_bytes
            )
            
            if not transcript:
                self.status.print_error("Failed to transcribe audio")
                return
            
            # Process command
            response = await self.process_voice_command(transcript)
            
            # Text-to-speech response (optional)
            if CONFIG.enable_tts:
                self.status.set_speaking(True)
                self.status.update_status("Generating speech...")
                
                audio_response = await loop.run_in_executor(
                    None, self.voice_processor.text_to_speech, response
                )
                
                if audio_response:
                    await loop.run_in_executor(
                        None, self.voice_processor.play_audio, audio_response
                    )
                
                self.status.set_speaking(False)
            
            self.status.update_status("Ready for next interaction!")
            
        except Exception as e:
            self.status.set_listening(False)
            self.status.set_speaking(False)
            self.status.print_error(f"Voice input error: {e}")
    
    async def _handle_text_input(self):
        """Handle text input."""
        try:
            loop = asyncio.get_event_loop()
            
            if RICH_AVAILABLE and self.status.console:
                text_input = await loop.run_in_executor(
                    None, self.status.console.input, "\n💬 Type your command: "
                )
            else:
                text_input = await loop.run_in_executor(
                    None, input, "\n💬 Type your command: "
                )
            
            if text_input.strip():
                await self.process_voice_command(text_input.strip())
                
        except Exception as e:
            self.status.print_error(f"Text input error: {e}")
    
    async def stop_session(self):
        """Stop the session and clean up resources."""
        self.conversation_active = False
        await self.status.stop_animation()
        cleanup_robot()
        self.status.print_message("SmolAgent session stopped.", "yellow")

# _load_env is now in shared config module

async def main() -> None:
    """Main function to run the SmolAgent voice demo."""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Load configuration from environment
    global CONFIG
    CONFIG = load_config_from_env()
    
    # Check dependencies
    missing_deps = []
    if not SMOLAGENTS_AVAILABLE:
        missing_deps.append("smolagents")
    
    # Check audio availability using shared module
    from ur.agents.voice_common.audio import AUDIO_AVAILABLE, OPENAI_AVAILABLE
    if not AUDIO_AVAILABLE:
        missing_deps.append("sounddevice numpy")
    if not OPENAI_AVAILABLE:
        missing_deps.append("openai")
    
    # Check Rich
    from ur.agents.voice_common.status import RICH_AVAILABLE
    if not RICH_AVAILABLE:
        missing_deps.append("rich")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: uv add {' '.join(missing_deps)}")
        return

    # Check required environment variables
    required_vars = []
    if CONFIG.model_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        required_vars.append("OPENAI_API_KEY")
    elif CONFIG.model_provider == "huggingface" and not os.getenv("HF_TOKEN"):
        required_vars.append("HF_TOKEN")
    
    if required_vars:
        print(f"❌ Missing environment variables: {', '.join(required_vars)}")
        print("Please set the required environment variables in your .env file")
        return
        
    # Create and start session
    session = SmolAgentVoiceSession(CONFIG)
    await session.start_session()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise 