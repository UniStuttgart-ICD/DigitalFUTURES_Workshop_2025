"""
Enhanced SmolAgent Voice Agent Implementation
===========================================

A voice agent that uses HuggingFace's smolagents CodeAgent framework 
for robot control. This implementation uses the shared voice_common module
for consistent behavior across all voice agents.

Features:
- Uses smolagents CodeAgent for intelligent code generation
- Shared voice configuration and status management
- Enhanced audio processing and feedback prevention
- Session logging and metrics tracking
- Safety validation with wake-word requirements
- Conversational context management
- Error handling and recovery

Usage:
    agent = SmolVoiceAgent(bridge_ref, ui_ref, model_id="gpt-4o-mini")
    await agent.start()

Author: ICD  
Version: 2.0.0 - Integrated with voice_common
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


def _timestamp() -> str:
    """Generate a timestamp string for print statements."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

# SmolAgents imports
try:
    from smolagents import CodeAgent, OpenAIServerModel, LiteLLMModel, InferenceClientModel
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    print(f"[{_timestamp()}] ⚠ SmolAgents not available. Install with: uv add smolagents")

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))  # For ur module (VizorBridge root)

from ur.agents.base_agent import BaseVoiceAgent
from ur.config.voice_config import VoiceAgentConfig, load_config_from_env
from ur.ui.console import VoiceAgentStatus
from ur.agents.voice_common.audio import SimpleVoiceProcessor
from ur.tools import register_tools_for_smolagents
from ur.config.system_prompts import get_system_prompt
from ur.config.system_config import SMOL_MODEL_ID, SMOL_PROVIDER
from ur.config.robot_config import ROBOT_AGENT_NAME


class EventType(Enum):
    """Event types for the SmolAgent voice session."""
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


# Global variable for immediate response callback
_immediate_response_callback = None

def set_immediate_response_callback(callback):
    """Set the immediate response callback for robot feedback."""
    global _immediate_response_callback
    _immediate_response_callback = callback

def send_immediate_response(response_text: str):
    """Send immediate response using the global callback."""
    if _immediate_response_callback:
        _immediate_response_callback(response_text)


class SmolAgentVoiceSession:
    """Voice session using SmolAgents CodeAgent for robot control."""
    
    def __init__(self, bridge_ref=None, config: Optional[VoiceAgentConfig] = None):
        self.config = config or load_config_from_env()
        self.bridge = bridge_ref
        self.status = VoiceAgentStatus("SmolAgent")
        self.agent: Optional[CodeAgent] = None
        self.voice_processor = SimpleVoiceProcessor()
        
        # Session state
        self.is_connected = False
        self.should_stop = False  # Add shutdown flag for clean exit
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
        
        # Get base system prompt
        try:
            system_prompt = get_system_prompt("code")
        except:
            system_prompt = f"You are {ROBOT_AGENT_NAME}, a sophisticated AI assistant controlling a Universal Robot using code generation."
        
        prompt = f"""{system_prompt}

Current robot status: {robot_status}

User request: "{user_text}"

IMPORTANT INSTRUCTIONS:
1. Generate Python code that uses the available robot tools to fulfill the user's request
2. ALL robot movements require the wake phrase "mave" to be mentioned by the user
3. If no wake phrase is present, politely ask the user to include it for safety
4. Use send_immediate_response() to provide real-time feedback during long operations
5. Always check robot state before attempting movements
6. Handle errors gracefully and provide helpful feedback

Available tools are already imported and ready to use. Write clean, safe code."""
        
        return prompt

    async def start_session(self):
        """Start the voice session and begin processing commands."""
        if not self.is_connected:
            await self.connect()
            
        # Start enhanced live status display
        self.status.start_live_display()
        
        # Different instructions based on mode
        if self.config.text_only_mode:
            self.status.update_status("Text mode ready - type your messages!")
            self.status.print_message("💬 Text-only mode enabled. Type messages and press Enter.", "cyan")
        elif self.config.enable_text_input:
            self.status.update_status("Hybrid mode ready - speak OR type!")
            self.status.print_message("🎤💬 Hybrid mode: You can speak OR type messages (press Enter to send).", "cyan")
        else:
            self.status.update_status("Voice input ready - start speaking!")
        
        # Wait a moment for user to read the info, then collapse to compact mode
        await asyncio.sleep(3)
        self.status.collapse_to_compact()
        
        try:
            await self._input_loop()
        except KeyboardInterrupt:
            self.status.print_message("\n👋 Shutting down gracefully...", "yellow")
        finally:
            self.status.stop_live_display()
            
    async def _input_loop(self):
        """Main input processing loop."""
        input_tasks = []
        
        # Add text input if enabled
        if self.config.enable_text_input or self.config.text_only_mode:
            text_task = asyncio.create_task(self._handle_text_input())
            input_tasks.append(text_task)
        
        # Add voice input if not text-only mode
        if not self.config.text_only_mode and self.voice_processor.check_audio_availability():
            voice_task = asyncio.create_task(self._handle_voice_input())
            input_tasks.append(voice_task)
        elif self.config.text_only_mode:
            self.status.print_message("🔇 Voice input disabled (text-only mode)", "yellow")
        
        # Wait for any input task to complete
        if input_tasks:
            await asyncio.gather(*input_tasks, return_exceptions=True)

    async def _handle_voice_input(self):
        """Handle voice input processing."""
        try:
            self.status.print_message("🎤 Voice input ready! Speak your commands...", "green")
            
            while self.conversation_active:
                try:
                    # Record audio
                    self.status.update_status("🎤 Press Enter when ready to record...")
                    input("Press Enter to start recording...")
                    
                    self.status.update_status("🎤 Recording... (5 seconds)")
                    audio_bytes = await self.voice_processor.record_audio_async(duration=5.0)
                    
                    if not audio_bytes:
                        self.status.print_message("❌ Failed to record audio", "red")
                        continue
                    
                    # Transcribe audio
                    self.status.update_status("🧠 Transcribing audio...")
                    transcript = self.voice_processor.transcribe_audio(audio_bytes)
                    
                    if not transcript:
                        self.status.print_message("❌ Failed to transcribe audio", "red")
                        continue
                    
                    # Special handling: 'quit'/'exit' stops voice input, 'stop' triggers end_fabrication
                    lower = transcript.lower()
                    if lower in ['quit', 'exit']:
                        self.status.print_message("👋 Voice input stopped", "yellow")
                        break
                    if lower in ['stop', 'stop fabrication', 'end fabrication']:
                        self.status.print_message("👋 Stop command received, ending fabrication...", "yellow")
                        # Trigger end_fabrication command via bridge
                        from ur.config.system_config import COMMAND_END
                        if self.bridge:
                            self.bridge.process_command({'data': COMMAND_END})
                        break
                    
                    # Process the voice command
                    response = await self.process_voice_command(transcript)
                    
                    # Generate TTS response if enabled
                    if self.config.enable_tts:
                        self.status.update_status("🔊 Generating speech response...")
                        tts_audio = await self.voice_processor.text_to_speech_async(
                            response, voice=self.config.tts_voice
                        )
                        if tts_audio:
                            await self.voice_processor.play_audio_async(tts_audio)
                        
                except KeyboardInterrupt:
                    self.status.print_message("\n👋 Voice input interrupted", "yellow")
                    break
                except Exception as e:
                    self.status.print_error(f"Voice processing error: {e}")
                    continue
                    
        except Exception as e:
            self.status.print_error(f"Voice input handler error: {e}")

    async def _handle_text_input(self):
        """Handle text input processing."""
        import sys
        
        try:
            self.status.print_message("💬 Text input ready! Type your commands...", "green")
            self.status.print_message("💡 Commands: 'quit' to exit, Ctrl+C to stop completely", "cyan")
            
            loop = asyncio.get_event_loop()
            
            while self.conversation_active:
                try:
                    print(f"\n[{_timestamp()}] 💬 Type command: ", end="", flush=True)
                    text_input = await loop.run_in_executor(
                        None, 
                        lambda: sys.stdin.readline().strip()
                    )
                    
                    if not text_input:
                        continue
                        
                    if text_input.lower() in ['quit', 'exit', 'q']:
                        self.status.print_message("👋 Text input stopped", "yellow")
                        break
                    
                    # Process the text command
                    response = await self.process_voice_command(text_input)
                    
                except KeyboardInterrupt:
                    self.status.print_message("\n👋 Text input stopped", "yellow")
                    break
                except Exception as e:
                    self.status.print_error(f"Text input error: {e}")
                    continue
                    
        except Exception as e:
            self.status.print_error(f"Text input handler error: {e}")

    async def stop_session(self):
        """Stop the voice session and cleanup."""
        self.conversation_active = False
        self.is_connected = False
        self.status.stop_live_display()
        self.status.print_success("SmolAgent session stopped")


class SmolVoiceAgent(BaseVoiceAgent):
    """
    Enhanced SmolAgent voice agent with shared voice_common components.
    """

    def __init__(self, bridge_ref, ui_ref, model_id: str = SMOL_MODEL_ID, provider: str = SMOL_PROVIDER, config: Optional[VoiceAgentConfig] = None):
        super().__init__(bridge_ref, ui_ref)
        self.config = config or load_config_from_env()
        
        # Update config with provided model settings
        self.config.model_id = model_id
        self.config.model_provider = provider
        
        self.session = None

    async def start(self):
        """Starts the enhanced SmolAgent voice session."""
        # Use the shared status system
        self.ui = VoiceAgentStatus("SmolAgent")
        self.ui.start_live_display()
        self.ui.update_status("Creating SmolAgent session...")

        # Create enhanced session
        self.session = SmolAgentVoiceSession(self.bridge, self.config)
        self.session.conversation_active = True

        try:
            await self.session.start_session()
        except Exception as e:
            self.ui.print_error(f"Failed to start SmolAgent session: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stops the session and cleans up resources."""
        self.should_stop = True  # Set flag first to stop all loops
        if self.session:
            await self.session.stop_session()
        if self.ui:
            self.ui.stop_live_display()
            self.ui.print_message("SmolAgent session stopped.", "yellow")

    def stop_sync(self):
        """Synchronous stop method that can be called from any thread."""
        self.should_stop = True  # Set flag first to stop all loops
        
        # Stop session synchronously if possible
        if self.session:
            self.session.should_stop = True
            self.session.is_connected = False
            
        # Stop UI display with complete shutdown
        if self.ui:
            try:
                if hasattr(self.ui, 'force_shutdown'):
                    self.ui.force_shutdown()  # Use force shutdown method
                elif hasattr(self.ui, 'complete_shutdown'):
                    self.ui.complete_shutdown()  # Use aggressive shutdown method
                else:
                    # Fallback to regular stop method
                    self.ui.stop_live_display()
                self.ui.print_message("SmolAgent session stopped.", "yellow")
            except Exception as e:
                print(f"[{_timestamp()}] ⚠️ UI shutdown error: {e}")
                # Force basic stop
                if hasattr(self.ui, 'is_stopped'):
                    self.ui.is_stopped = True

    async def handle_task_event(self, event_type: str, event_data: dict):
        """Handle task events from the bridge using system prompts for dynamic LLM commentary."""
        
        if not self.session or not self.session.agent:
            return
        
        try:
            from ur.config.system_prompts import build_commentary_prompt
            
            # Generate appropriate commentary using system prompts
            if event_type == 'task_received':
                # Add NEW TASK decorator to indicate this is from task manager
                task_context = f"[NEW TASK FROM MANAGER] Task: {event_data.get('task_name', 'unknown')}"
                commentary = await self._prompt_smol_for_task_announcement(task_context)
                
            elif event_type == 'command_received':
                prompt = build_commentary_prompt(
                    context_type='command',
                    command=event_data.get('command')
                )
                commentary = await self._prompt_smol_for_commentary(prompt)
                
            elif event_type == 'robot_action':
                prompt = build_commentary_prompt(
                    context_type='task_execute',
                    task_name=event_data.get('task_name'),
                    action_type=event_data.get('action_type')
                )
                commentary = await self._prompt_smol_for_commentary(prompt)
                
            else:
                # Fallback for other event types
                commentary = f"Processing {event_type}..."
            
            # Send the generated commentary using immediate response
            if commentary and commentary.strip():
                send_immediate_response(commentary)
                if self.config.debug_mode:
                    print(f"[{_timestamp()}] 🎭 [SMOL LLM COMMENTARY] {event_type}: {commentary}")
                    
        except Exception as e:
            if self.config.debug_mode:
                print(f"❌ [SMOL COMMENTARY ERROR] {event_type}: {e}")

    async def generate_fabrication_message(self, message_type: str) -> str:
        """Generate welcome/goodbye message for fabrication start/end."""
        if message_type == "start":
            prompt = "Generate a brief, friendly welcome message for starting a collaborative fabrication session with a human. Be encouraging and ready to help. Keep it under 20 words."
            default = "Hello! I'm ready to help with your fabrication project. Let's build something amazing together!"
        else:  # end
            prompt = "Generate a brief, appreciative goodbye message for ending a collaborative fabrication session. Thank the human for working together. Keep it under 20 words."
            default = "Great work! It was a pleasure collaborating with you on this fabrication project. Until next time!"
        
        try:
            # Use the existing SmolAgent to generate the message
            response = await self._prompt_smol_for_commentary(prompt)
            return response.strip() if response else default
        except Exception as e:
            if self.config.debug_mode:
                print(f"⚠️ [SMOL AGENT] Error generating fabrication message: {e}")
            return default

    async def _prompt_smol_for_commentary(self, full_prompt: str) -> str:
        """Generate commentary using SmolAgent's LLM with complete system prompt."""
        try:
            if not self.session or not self.session.agent:
                return "Executing task..."
            
            # Parse the prompt to extract just the user query part
            parts = full_prompt.split('\n\nTask received:', 1)
            if len(parts) == 2:
                user_query = 'Task received:' + parts[1]
            else:
                parts = full_prompt.split('\n\nCommand received:', 1)
                if len(parts) == 2:
                    user_query = 'Command received:' + parts[1]
                else:
                    parts = full_prompt.split('\n\nCurrent task:', 1)
                    if len(parts) == 2:
                        user_query = 'Current task:' + parts[1]
                    else:
                        user_query = "Generate appropriate robot commentary."
            
            # Use SmolAgent to generate commentary with the full system context
            # Create a focused prompt for commentary generation
            commentary_prompt = f"""
            {full_prompt}
            
            IMPORTANT: Generate ONLY a brief, natural robot announcement (under 15 words).
            Do not include any code, explanations, or instructions.
            Speak as the robot in first person.
            """
            
            # Process through SmolAgent
            response = await self.session.process_voice_command(commentary_prompt)
            
            # Extract just the commentary from the response (remove any code or extra text)
            if response:
                lines = response.split('\n')
                # Find the most likely commentary line (short, natural language)
                for line in lines:
                    line = line.strip()
                    if (line and 
                        len(line.split()) <= 15 and 
                        not line.startswith('```') and 
                        not line.startswith('#') and
                        not '=' in line and
                        not 'import' in line.lower()):
                        return line
                
                # Fallback: return first non-empty line
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('```'):
                        return line[:100]  # Limit length
            
            return "Executing task..."
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"❌ [SMOL LLM ERROR] {e}")
            return "Executing task..."  # Fallback

    def _generate_task_received_commentary(self, event_data: dict) -> str:
        """Generate commentary when a new task is received."""
        task_name = event_data.get('task_name', 'Unknown')
        if 'pickup' in task_name.lower():
            # Extract element size from task name (e.g., "pickup-element-40cm" -> "40cm")
            element_size = "40cm"  # Default
            if "40cm" in task_name: element_size = "40cm"
            elif "50cm" in task_name: element_size = "50cm" 
            elif "30cm" in task_name: element_size = "30cm"
            return f"Please place the {element_size} beam on the supply station."
        return f"Starting {task_name} task."

    def _generate_task_starting_commentary(self, event_data: dict) -> str:
        """Generate commentary when task execution begins."""
        task_name = event_data.get('task_name', 'Unknown')
        if 'pickup' in task_name.lower():
            return "Moving to pick up beam."
        elif 'place' in task_name.lower():
            return "Moving to assembly position."
        elif 'home' in task_name.lower():
            return "Beam secured. I'm returning to the home position."
        else:
            return f"Starting {task_name}."

    def _generate_robot_action_commentary(self, event_data: dict) -> str:
        """Generate commentary for robot actions."""
        action = event_data.get('action', '')
        task_name = event_data.get('task_name', '')
        
        # Match the PlantUML script announcements
        if action == 'gripper_opened':
            if 'place' in task_name.lower():
                return "Beam secured. I'm returning to the home position."
            else:
                return ""  # Don't announce gripper opening during pickup
        elif action == 'moving_to_pickup':
            return ""  # Already announced in task_starting
        elif action == 'element_secured':
            return ""  # Don't announce - let task completion handle it
        elif action == 'moving_to_assembly':
            return ""  # Already announced in task_starting
        elif action == 'returning_home':
            return ""  # Already announced in task_starting
        
        return ""  # Most robot actions don't need separate announcements

    def _generate_human_request_commentary(self, event_data: dict) -> str:
        """Generate commentary for human action requests."""
        action = event_data.get('action', '')
        task_name = event_data.get('task_name', '')
        
        if action == 'place_element_on_supply_station':
            # This is handled in task_received for pickup tasks
            return ""
        elif action == 'secure_screws':
            return "Beam in position. I'll hold steady while you secure the screws."
        else:
            return f"I need your assistance: {action}"

    def _generate_task_completed_commentary(self, event_data: dict) -> str:
        """Generate commentary when task is completed."""
        task_name = event_data.get('task_name', 'Unknown')
        success = event_data.get('success', False)
        
        if success:
            if 'pickup' in task_name.lower():
                return ""  # Don't announce pickup completion - robot continues to next task
            elif 'place' in task_name.lower():
                return ""  # Don't announce place completion - robot announces return home
            elif 'home' in task_name.lower():
                return ""  # Home completion is silent - ready for next beam
            else:
                return f"Task {task_name} completed successfully."
        else:
            return f"Encountered an issue with {task_name}. Let me try again."

    async def _prompt_smol_for_task_announcement(self, task_context: str) -> str:
        """Generate commentary for task announcements."""
        try:
            if not self.session or not self.session.agent:
                return "Executing task..."
            
            # Create a focused prompt for task announcement generation
            task_prompt = f"""
            {task_context}
            
            IMPORTANT: Generate ONLY a brief, natural robot announcement (under 15 words).
            Do not include any code, explanations, or instructions.
            Speak as the robot in first person.
            """
            
            # Process through SmolAgent
            response = await self.session.process_voice_command(task_prompt)
            
            # Extract just the commentary from the response (remove any code or extra text)
            if response:
                lines = response.split('\n')
                # Find the most likely commentary line (short, natural language)
                for line in lines:
                    line = line.strip()
                    if (line and 
                        len(line.split()) <= 15 and 
                        not line.startswith('```') and 
                        not line.startswith('#') and
                        not '=' in line and
                        not 'import' in line.lower()):
                        return line
                
                # Fallback: return first non-empty line
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('```'):
                        return line[:100]  # Limit length
            
            return "Executing task..."
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"❌ [SMOL LLM ERROR] {e}")
            return "Executing task..."  # Fallback


def create_smol_agent(bridge_ref, ui_ref, model_id: str = SMOL_MODEL_ID, provider: str = SMOL_PROVIDER, config: Optional[VoiceAgentConfig] = None) -> SmolVoiceAgent:
    """Create a SmolAgent voice agent with enhanced features."""
    return SmolVoiceAgent(bridge_ref, ui_ref, model_id, provider, config)


# --- Demo and Testing ---
if __name__ == "__main__":
    # Mock UI for testing
    class MockUI:
        def start_live_display(self): pass
        def stop_live_display(self): pass
        def update_status(self, status): print(f"Status: {status}")
        def print_error(self, message): print(f"Error: {message}")
        def print_message(self, message, style=None): print(f"Message: {message}")

    async def test_smol_agent():
        """Test the SmolAgent implementation."""
        config = load_config_from_env()
        config.text_only_mode = True  # For testing
        config.debug_mode = True
        
        agent = SmolVoiceAgent(None, MockUI(), config=config)
        await agent.start()

    try:
        asyncio.run(test_smol_agent())
    except KeyboardInterrupt:
        print("\n👋 Test stopped") 