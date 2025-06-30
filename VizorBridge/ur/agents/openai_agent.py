"""Enhanced OpenAI Realtime API voice agent with advanced features.

This module provides a production-ready OpenAI Realtime API implementation integrated
from the samples with comprehensive session management, logging, and audio handling.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
import ssl
import sys
import threading
import warnings
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _timestamp() -> str:
    """Generate a timestamp string for print statements."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

import websockets

# Add project root to path to allow imports from other modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ur.agents.base_agent import BaseVoiceAgent
from ur.config.voice_config import VoiceAgentConfig, load_config_from_env
from ur.agents.voice_common.status import VoiceAgentStatus
from ur.agents.voice_common.audio import create_audio_callback, encode_audio_for_openai, AUDIO_CONFIG, play_audio
from ur.tools import register_tools_for_openai
from ur.config.system_prompts import get_system_prompt, build_commentary_prompt
from ur.config.system_config import (
    OPENAI_MODEL_ID,
    OPENAI_TRANSCRIPTION_MODEL,
    TOOL_EXECUTOR_MAX_WORKERS,
    SESSION_ID_PREFIX,
    COMMAND_END
)
from ur.config.voice_config import (
    OPENAI_INPUT_AUDIO_FORMAT,
    OPENAI_OUTPUT_AUDIO_FORMAT
)
from ur.config.robot_config import (
    ROBOT_AGENT_NAME,
    ROBOT_AGENT_UI_NAME,
    ROBOT_TOOL_THREAD_PREFIX
)
from ur.config.system_prompts import OPENAI_VOICE_INSTRUCTIONS
from ur.agents.voice_common.silero_vad import SileroVADProcessor

# Optional audio imports
try:
    import openai
except ImportError:
    raise ImportError("openai is required for voice agent")

# Audio and numeric dependencies (optional)
try:
    import numpy as np
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Rich imports for enhanced console display
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class EventType(Enum):
    """Event types for the realtime voice agent session."""
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    CONVERSATION_STARTED = "conversation_started"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    TRANSCRIPTION_COMPLETED = "transcription_completed"
    RESPONSE_STARTED = "response_started"
    RESPONSE_COMPLETED = "response_completed"
    AUDIO_INTERRUPTED = "audio_interrupted"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    ERROR = "error"


@dataclass
class ToolDefinition:
    """Tool definition following OpenAI Agents patterns."""
    name: str
    description: str
    parameters: Dict[str, Any]
    execute: Callable
    needs_approval: bool = False


@dataclass
class RealtimeEvent:
    """Event data structure for session events."""
    type: EventType
    data: Dict[str, Any]
    timestamp: float


class RealtimeAgent:
    """Realtime voice agent following OpenAI Agents SDK patterns."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[ToolDefinition]] = None,
        voice: str = None,
        temperature: float = None,
        model: str = OPENAI_MODEL_ID,
        config: Optional[VoiceAgentConfig] = None
    ):
        config = config or load_config_from_env()
        
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.voice = voice or config.openai_voice
        self.temperature = temperature or config.openai_temperature
        self.model = model
        
        # Internal state
        self._tool_map = {tool.name: tool for tool in self.tools}
        self.config = config
        
    def add_tool(self, tool: ToolDefinition):
        """Add a tool to the agent."""
        self.tools.append(tool)
        self._tool_map[tool.name] = tool
        
    def get_session_config(self) -> Dict[str, Any]:
        """Get session configuration for the Realtime API."""
        cfg = {
            "modalities": ["audio", "text"] if self.config.enable_hybrid_mode else ["audio"],
            "instructions": self.instructions,
            "voice": self.voice,
            "speed": self.config.openai_voice_speed,
            "input_audio_format": OPENAI_INPUT_AUDIO_FORMAT,
            "output_audio_format": OPENAI_OUTPUT_AUDIO_FORMAT,
            "tools": [self._convert_tool_to_openai_spec(tool) for tool in self.tools],
            "temperature": self.temperature,
            "tool_choice": "auto"
        }
        # Configure turn detection: server-side VAD if enabled, otherwise defer to client VAD
        if self.config.openai_use_server_vad:
            cfg["turn_detection"] = {
                "type": "server_vad",
                "create_response": True,
                "interrupt_response": True,
                "silence_duration_ms": self.config.vad_silence_duration_ms,
                "prefix_padding_ms": self.config.vad_prefix_padding_ms,
            }
            # Force the model to invoke tools for applicable operations
        return cfg
        
    def _convert_tool_to_openai_spec(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Convert tool definition to OpenAI API specification."""
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }


class RealtimeSession:
    """Enhanced session manager with comprehensive features from samples."""
    
    def __init__(
        self,
        agent: RealtimeAgent,
        api_key: str,
        bridge_ref=None,
        config: Optional[VoiceAgentConfig] = None
    ):
        self.agent = agent
        self.api_key = api_key
        self.bridge = bridge_ref
        self.config = config or load_config_from_env()
        
        # Console status manager
        self.status = VoiceAgentStatus(agent.name, agent)
        
        # Session logging
        self.session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # WebSocket connection client will be created once under the session's event loop
        self.client = None
        self.connection = None
        self.connection_manager = None  # Initialize connection manager
        
        # Session state
        self.is_connected = False
        self.should_stop = False  # Add shutdown flag for clean exit
        self.is_speaking = False
        self.conversation_history: List[Dict[str, Any]] = []
        self.audio_output_buffer = bytearray()
        self.audio_queue: Optional[asyncio.Queue] = None
        
        # Text interaction state
        self.text_response_buffer = ""
        self.is_receiving_text = False
        
        # Audio transcript buffer
        self.audio_transcript_buffer = ""
        
        # Audio control for interruption
        self.current_audio_stream = None
        self.audio_interrupt_event = threading.Event()
        
        # Enhanced audio feedback prevention
        self._audio_silence_start_time = None
        self._last_audio_output_time = None
        self._microphone_muted = False
        
        # Thread pool for tool execution
        self.tool_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=TOOL_EXECUTOR_MAX_WORKERS,
            thread_name_prefix=ROBOT_TOOL_THREAD_PREFIX
        )
        
        # Event handlers
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        
        # Initialize event handlers
        self._setup_default_handlers()
        
        # Add tool call tracking to the session state
        # Tool call deduplication tracking
        self.processed_tool_calls = set()  # Track processed call_ids to prevent duplicates
        self._active_response = False  # Track if we have an active response to prevent conflicts
        
        # Audio buffer tracking for preventing empty flushes
        self._audio_chunks_buffered = 0
    
    def on(self, event_type: EventType, handler: Callable):
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
    def _emit_event(self, event_type: EventType, data: Optional[Dict[str, Any]] = None):
        """Emit an event to all registered handlers."""
        if event_type in self._event_handlers:
            event = RealtimeEvent(event_type, data or {}, asyncio.get_event_loop().time())
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    print(f"[{_timestamp()}] ❌ Event handler error: {e}")
    
    def _setup_default_handlers(self):
        """Setup default event handlers."""
        self.on(EventType.SESSION_CREATED, self._handle_session_created)
        self.on(EventType.SPEECH_STARTED, self._handle_speech_started)
        self.on(EventType.SPEECH_ENDED, self._handle_speech_ended)
        self.on(EventType.TRANSCRIPTION_COMPLETED, self._handle_transcription)
        self.on(EventType.RESPONSE_STARTED, self._handle_response_started)
        self.on(EventType.RESPONSE_COMPLETED, self._handle_response_completed)
        self.on(EventType.TOOL_CALL_REQUESTED, self._handle_tool_call)
        self.on(EventType.AUDIO_INTERRUPTED, self._handle_audio_interrupted)
        self.on(EventType.ERROR, self._handle_error)

    def _mute_microphone(self):
        """Mute the microphone to prevent feedback."""
        if self.config.feedback_prevention_enabled:
            self._microphone_muted = True
            if self.config.debug_mode:
                self.status.print_message("🎤 Microphone muted", "dim")

    def _unmute_microphone(self):
        """Unmute the microphone."""
        if self.config.feedback_prevention_enabled:
            self._microphone_muted = False
            if self.config.debug_mode:
                self.status.print_message("🎤 Microphone unmuted", "dim")

    async def connect(self):
        """Connect to OpenAI Realtime API using the official SDK."""
        self.status.update_status("Connecting to OpenAI Realtime API...")
        # Instantiate the AsyncOpenAI client under the current event loop
        import openai as _openai
        self.client = _openai.AsyncOpenAI(api_key=self.api_key)
        # Prepare the realtime connection manager
        self.connection_manager = self.client.beta.realtime.connect(model=self.agent.model)
        
        self.status.print_success("Ready to connect to OpenAI Realtime API")

    async def start_session_with_connection(self):
        """Start the session using the connection manager."""
        if not self.connection_manager:
            raise RuntimeError("Must call connect() first")

        try:
            async with self.connection_manager as connection:
                self.connection = connection
                self.is_connected = True

                self.status.print_success("Connected to OpenAI Realtime API")

                # Configure the session with the agent settings
                await self._configure_session()

                # Start the live status display
                self.status.start_live_display()

                # Inform the user based on mode
                if self.config.text_only_mode:
                    self.status.update_status("Text mode ready - type your messages!")
                    self.status.print_message("💬 Text-only mode enabled. Type messages and press Enter.", "cyan")
                elif self.config.enable_hybrid_mode:
                    self.status.update_status("Hybrid mode ready - speak OR type!")
                    self.status.print_message("🎤💬 Hybrid mode: You can speak OR type messages (press Enter to send).", "cyan")
                else:
                    self.status.update_status("Voice input ready - start speaking!")

                # Give user a moment to read, then switch to compact display
                await asyncio.sleep(3)
                self.status.collapse_to_compact()

                # Start concurrent tasks: event handling + input streams
                event_task = asyncio.create_task(self._handle_server_events())
                tasks = [event_task]

                if self.config.enable_hybrid_mode:
                    # Hybrid: both text & audio streams
                    tasks.append(asyncio.create_task(self._text_input_stream_impl()))
                    if AUDIO_AVAILABLE:
                        tasks.append(asyncio.create_task(self._audio_input_stream_impl()))
                    else:
                        self.status.print_message("🔇 Audio input not available", "yellow")
                elif self.config.text_only_mode:
                    # Text-only
                    tasks.append(asyncio.create_task(self._text_input_stream_impl()))
                else:
                    # Audio-only
                    if AUDIO_AVAILABLE:
                        tasks.append(asyncio.create_task(self._audio_input_stream_impl()))
                    else:
                        self.status.print_message("🔇 Voice input disabled (text-only mode)", "yellow")

                # Await both event loop and input streams
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            self.status.print_error(f"Session error: {e}")
            raise
        finally:
            self.is_connected = False
            self.connection = None

    async def _configure_session(self):
        """Configure the session with agent settings."""
        if not self.connection:
            return
            
        self.status.update_status("Configuring session...")
        
        # Use the session.update method as shown in the Python examples
        try:
            # Use agent helper to build session config
            session_config = self.agent.get_session_config()
            await self.connection.session.update(session=session_config)
            
            self.status.update_status("Session configured - ready to chat!")
            self.status.print_success("Session configured with agent settings")
            
        except Exception as e:
            self.status.print_error(f"Failed to configure session: {e}")
            raise

    async def _send_event(self, event: Dict[str, Any]):
        """Send an event using the SDK connection."""
        if self.connection and self.is_connected:
            try:
                await self.connection.send(event)
            except Exception as e:
                if self.config.debug_mode:
                    self.status.print_message(f"Failed to send event: {e}", "dim")

    async def send_message(self, text: str):
        """Send a text message to the assistant."""
        if not self.is_connected or not self.connection:
            self.status.print_error("Cannot send message - not connected")
            return
            
        try:
            # Use SDK helpers to send the user message and trigger response
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                }
            )
            await self.connection.response.create()
        except Exception as e:
            self.status.print_error(f"Failed to send message: {e}")

    async def _handle_server_events(self):
        """Handle incoming events from the Realtime API using the SDK."""
        if not self.connection:
            return
            
        try:
            async for event in self.connection:
                # Check if we should stop processing events
                if self.should_stop:
                    self.status.print_message("🛑 Stopping event processing due to should_stop flag", "yellow")
                    break
                await self._process_server_event(event)
        except Exception as e:
            if "ConnectionClosed" in str(e):
                self.status.print_message("🔌 Connection closed.", "yellow")
            else:
                self.status.print_error(f"Connection error: {e}")
            self.is_connected = False
            
    async def _process_server_event(self, event: Any):
        """Process individual server events from the SDK."""
        event_type = event.type
        
        # Debug output for important events if enabled
        if self.config.debug_mode:
            if "function" in event_type or "tool" in event_type:
                print(f"[{_timestamp()}] 🚨 FUNCTION/TOOL EVENT: {event_type} - {event}")
            elif event_type == "response.audio.delta":
                print(f"[{_timestamp()}] 🚨 AUDIO DELTA: {event_type} (audio data: {len(str(getattr(event, 'delta', ''))) if hasattr(event, 'delta') else 'N/A'} bytes)")
            elif "response" in event_type:
                print(f"[{_timestamp()}] 🚨 RESPONSE EVENT: {event_type}")
            elif "conversation" in event_type:
                print(f"[{_timestamp()}] 🚨 CONVERSATION EVENT: {event_type} - {event}")
            else:
                print(f"[{_timestamp()}] 🔍 DEBUG EVENT: {event_type}")
        
        try:
            if event_type == "session.created":
                self._emit_event(EventType.SESSION_CREATED, event.model_dump())
                
            elif event_type == "session.updated": 
                self._emit_event(EventType.SESSION_UPDATED, event.model_dump())
                
            elif event_type == "conversation.created":
                self._emit_event(EventType.CONVERSATION_STARTED, event.model_dump())
                
            elif event_type == "input_audio_buffer.speech_started":
                self._emit_event(EventType.SPEECH_STARTED, event.model_dump())
                if self.config.debug_mode:
                    self.status.print_message("🎤 OpenAI detected speech started!", "green")
                
            elif event_type == "input_audio_buffer.speech_stopped":
                self._emit_event(EventType.SPEECH_ENDED, event.model_dump())
                if self.config.debug_mode:
                    self.status.print_message("🛑 OpenAI detected speech stopped - processing...", "yellow")
                
            elif event_type == "conversation.item.input_audio_transcription.completed":
                self._emit_event(EventType.TRANSCRIPTION_COMPLETED, event.model_dump())
                if self.config.debug_mode and hasattr(event, 'transcript'):
                    transcript = getattr(event, 'transcript', '')
                    self.status.print_message(f"📝 Transcription: '{transcript}'", "cyan")
                
            elif event_type == "response.created":
                self._emit_event(EventType.RESPONSE_STARTED, event.model_dump())
                
            elif event_type == "response.content_part.added":
                # Handle content part added - check if it's audio
                if hasattr(event, 'part') and getattr(event.part, 'type', None) == "audio":
                    self.audio_output_buffer = bytearray()
                    
            elif event_type == "response.audio.delta":
                if hasattr(event, 'delta'):
                    audio_data = base64.b64decode(event.delta)
                self.audio_output_buffer.extend(audio_data)
                
            elif event_type == "response.audio.done":
                # buffered for later playback
                pass
                
            elif event_type == "response.text.delta":
                text_delta = getattr(event, 'delta', "")
                self.text_response_buffer += text_delta
                self.is_receiving_text = True
                
            elif event_type == "response.text.done":
                # completed text streaming; will display in _handle_response_completed
                self.is_receiving_text = False
                
            elif event_type == "response.audio_transcript.delta":
                transcript_delta = getattr(event, 'delta', "")
                self.audio_transcript_buffer += transcript_delta
                
            elif event_type == "response.audio_transcript.done":
                if self.audio_transcript_buffer:
                    self.status.print_assistant_message(self.audio_transcript_buffer)
                    self.audio_transcript_buffer = ""
                    
            # Function call detection for Python Realtime API
            elif event_type == "response.output_item.done":
                if hasattr(event, 'item') and hasattr(event.item, 'type') and event.item.type == "function_call":
                    if self.config.debug_mode:
                        print(f"[{_timestamp()}] 🚨 FUNCTION CALL DETECTED: {event}")
                    # Extract function call data properly from the Python SDK event
                    item_data = event.item.model_dump()
                    self._emit_event(EventType.TOOL_CALL_REQUESTED, {
                        "name": item_data.get("name"),
                        "arguments": item_data.get("arguments", "{}"),
                        "call_id": item_data.get("call_id") or item_data.get("id"),
                        "source_event": "response.output_item.done"
                    })
                elif self.config.debug_mode and hasattr(event, 'item'):
                    print(f"[{_timestamp()}] 🔍 Response item completed: {event.item.type if hasattr(event.item, 'type') else 'unknown'}")
                
            elif event_type == "response.done":
                self._emit_event(EventType.RESPONSE_COMPLETED, event.model_dump())
                
            elif event_type == "error":
                error_details = getattr(event, 'error', None)
                if error_details:
                    error_code = getattr(error_details, 'code', '')
                    error_message = getattr(error_details, 'message', 'Unknown error')
                
                # Don't show cancellation errors (these are expected during interruption)
                if error_code == "response_cancel_not_active":
                    if self.config.debug_mode:
                            self.status.print_message(f"Debug: {error_message}", "dim")
                else:
                        self.status.print_error(f"OpenAI API Error: {error_message}")
                    
                        self._emit_event(EventType.ERROR, event.model_dump())
                
            elif event_type == "task_starting":
                # Task execution starting: announce movement
                commentary = self._generate_task_starting_commentary(event.model_dump())
            elif event_type == "command_received":
                # Handle command received event
                prompt = build_commentary_prompt(
                    context_type='command',
                    command=event.model_dump().get('command')
                )
                commentary = await self._prompt_llm_for_commentary(prompt)
                
            else:
                # Optional logging for unknown event types
                if self.config.debug_mode:
                    print(f"[{_timestamp()}] 🔍 Unknown event type: {event_type}")
                
        except Exception as e:
            print(f"[{_timestamp()}] ❌ Error processing event {event_type}: {e}")
            if self.config.debug_mode:
                import traceback
                traceback.print_exc()
                # Don't re-raise - continue processing other events

    async def close(self):
        """Close the session and cleanup resources."""
        self.should_stop = True  # Signal all loops to stop
        self.is_connected = False
        if self.connection:
            await self.connection.close()
        self.tool_executor.shutdown(wait=False)
        if self.status:
            self.status.stop_live_display()
        # Safely close the OpenAI HTTP client to finish background tasks without loop-closed errors
        import asyncio
        loop = asyncio.get_running_loop()
        prev_handler = loop.get_exception_handler()
        # Suppress exception handler for closed-loop warnings
        loop.set_exception_handler(lambda l, ctx: None)
        try:
            await self.client.aclose()
        except Exception:
            try:
                await self.client.close()
            except Exception:
                pass
        finally:
            # Restore previous exception handler
            loop.set_exception_handler(prev_handler)

    # Event handlers
    async def _handle_session_created(self, event: RealtimeEvent):
        """Handle session creation."""
        self.status.update_status("Session created successfully!")
        
    async def _handle_speech_started(self, event: RealtimeEvent):
        """Handle speech detection."""
        self.status.set_listening(True)
        self.status.update_status("Listening to your speech...")
        
        # IMMEDIATE interruption when user speaks
        if self.is_speaking or self.current_audio_stream:
            await self.interrupt_audio()
            
            try:
                await self._send_event({"type": "response.cancel"})
                await self._send_event({"type": "input_audio_buffer.clear"})
            except Exception as e:
                if self.config.debug_mode:
                    self.status.print_message(f"Response cancellation: {e}", "dim")
            
            self.is_speaking = False
            self.status.set_speaking(False)
            self.status.print_message("🛑 Interrupted assistant - listening to you", "yellow")
    
    async def _handle_speech_ended(self, event: RealtimeEvent):
        """Handle speech end."""
        self.status.set_listening(False)
        self.status.set_processing(True)
        self.status.update_status("Processing your speech...")
        
    async def _handle_transcription(self, event: RealtimeEvent):
        """Handle transcription completion."""
        transcript = event.data.get("transcript", "")
        self.status.print_user_message(transcript)
        
        # Special handling: if user says stop or stop fabrication, trigger end_fabrication
        lower = transcript.lower().strip()
        if lower in ['stop', 'stop fabrication', 'end fabrication']:
            self.status.print_message("👋 Stop command received, ending fabrication...", "yellow")
            if self.bridge:
                self.bridge.process_command({'data': COMMAND_END})
            return
        
    async def _handle_response_started(self, event: RealtimeEvent):
        """Handle response start."""
        self.status.set_processing(False)
        self.status.update_status("Assistant is responding...")
        
    async def _handle_response_completed(self, event: RealtimeEvent):
        """Handle response completion."""
        # Log session event
        self._log_event("response_completed", {"content": self.text_response_buffer})
        
        # On completion, play audio if available, otherwise display text
        if self.audio_output_buffer and not self.config.text_only_mode and AUDIO_AVAILABLE:
            await self._play_audio_response(bytes(self.audio_output_buffer))
        elif self.text_response_buffer:
            self.status.print_message(self.text_response_buffer, "cyan")

        # Reset buffers after processing
        self.audio_output_buffer.clear()
        self.text_response_buffer = ""
        self.is_receiving_text = False
        
        # Reset response tracking to allow new responses
        self._active_response = False
        
        # After response, go back to ready state
        self.status.set_processing(False)
        self.status.update_status("Ready for next interaction!")
        if not self.is_speaking:
            self.status.set_listening(True)
        
    async def _handle_audio_interrupted(self, event: RealtimeEvent):
        """Handle audio interrupted event."""
        self.status.print_message("🎤 Audio interrupted by user", "yellow")
            
    async def _handle_error(self, event: RealtimeEvent):
        """Handle error events."""
        if self.config.debug_mode:
            error_data = event.data.get("error", {})
            self.status.print_message(f"⚠️ Error event: {error_data.get('message', 'Unknown error')}", "dim")

    async def _handle_tool_call(self, event: RealtimeEvent):
        """Handle tool call execution with deduplication and improved parsing."""
        if self.config.debug_mode:
            print(f"[{_timestamp()}] 🚨 TOOL CALL HANDLER TRIGGERED! Event: {event}")
            print(f"[{_timestamp()}] 🔧 Tool call event data: {event.data}")
        
        try:
            event_data = event.data
            
            if self.config.debug_mode:
                print(f"[{_timestamp()}] 🔧 Processing event data: {event_data}")
                print(f"[{_timestamp()}] 🔧 Available keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'Not a dict'}")
            
            # Extract function call information - use improved direct extraction
            function_name = event_data.get("name")
            arguments_str = event_data.get("arguments", "{}")
            call_id = event_data.get("call_id")
            
            if self.config.debug_mode:
                print(f"[{_timestamp()}] 🔧 Extracted: name={function_name}, args={arguments_str}, call_id={call_id}")
            
            if not function_name or not call_id:
                if self.config.debug_mode:
                    print(f"[{_timestamp()}] ⚠️ Skipping invalid function call - missing name ({function_name}) or call_id ({call_id})")
                return
                
            # Deduplication check
            if call_id in self.processed_tool_calls:
                if self.config.debug_mode:
                    print(f"[{_timestamp()}] 🔄 Skipping duplicate tool call: {call_id}")
                return
            
            # Mark as processed to prevent duplicates
            self.processed_tool_calls.add(call_id)
            
            tool = self.agent._tool_map.get(function_name)
            if not tool:
                self.status.print_error(f"Tool '{function_name}' not found in tool map")
                error_msg = f"Tool '{function_name}' not available"
                await self._send_tool_error(call_id, error_msg)
                return
            
            try:
                # Parse arguments
                args = json.loads(arguments_str) if arguments_str else {}
                if self.config.debug_mode:
                    print(f"[{_timestamp()}] 🤖 Executing: {function_name}({args})")
                
                # Execute the tool
                result = await self._execute_tool(tool, args)
                if self.config.debug_mode:
                    print(f"[{_timestamp()}] ✅ Tool execution result: {result}")
                
                # Send function result using SDK helpers
                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result)
                    }
                )
                await self.connection.response.create()
                
            except json.JSONDecodeError as e:
                self.status.print_error(f"Failed to parse function arguments: {e}")
                await self._send_tool_error(call_id, f"Invalid arguments format: {e}")
                
            except Exception as e:
                self.status.print_error(f"Tool execution error: {e}")
                if self.config.debug_mode:
                    import traceback
                    traceback.print_exc()
                    
                await self._send_tool_error(call_id, str(e))
                
        except Exception as e:
            self.status.print_error(f"Fatal error in tool call handler: {e}")
            if self.config.debug_mode:
                import traceback
                traceback.print_exc()

    async def _execute_tool(self, tool: ToolDefinition, args: Dict[str, Any]) -> Any:
        """Execute a tool function (threaded for non-blocking robot operations)."""
        self.status.set_tool_executing(tool.name)
        
        try:
            if self.config.threaded_execution:
                # Execute in thread pool to prevent blocking conversation
                self.status.print_tool_message(f"Executing {tool.name} in background thread...")
                if asyncio.iscoroutinefunction(tool.execute):
                    result = await tool.execute(**args)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.tool_executor, 
                                                       lambda: tool.execute(**args))
            else:
                # Original synchronous execution
                self.status.print_tool_message(f"Executing {tool.name}...")
                if asyncio.iscoroutinefunction(tool.execute):
                    result = await tool.execute(**args)
                else:
                    result = tool.execute(**args)
            
            self.status.print_tool_message(f"Tool {tool.name} completed successfully ✅")
            return result
            
        finally:
            self.status.set_tool_executing(None)

    async def _send_tool_error(self, call_id: str, error_message: str):
        """Send tool execution error back to the API."""
        try:
            if self.config.debug_mode:
                print(f"[{_timestamp()}] ⚠️ Sending tool error for call_id {call_id}: {error_message}")
                
            # Send error output using SDK helpers
            await self.connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"error": error_message})
                }
            )
            await self.connection.response.create()
            
        except Exception as e:
            self.status.print_error(f"Failed to send tool error: {e}")

    async def interrupt_audio(self):
        """Interrupt any ongoing audio playback."""
        if self.is_speaking:
            self.audio_interrupt_event.set()
            if self.config.debug_mode:
                self.status.print_message("Audio interrupt event set", "dim")

    async def flush_audio_buffer(self):
        """Manually flush the audio buffer to the server."""
        if self.is_connected and self.connection:
            # Only flush if we have buffered some audio chunks
            if self._audio_chunks_buffered < 5:  # Need at least 5 chunks (~100ms) before flushing
                if self.config.debug_mode:
                    self.status.print_message(f"🚫 Skipping flush: only {self._audio_chunks_buffered} chunks buffered", "dim")
                return
                
            try:
                await self._send_event({"type": "input_audio_buffer.commit"})
                self.status.print_message(f"🎤 Manually triggered send to server ({self._audio_chunks_buffered} chunks).", "yellow")
                self._audio_chunks_buffered = 0  # Reset counter after flush
            except Exception as e:
                self.status.print_error(f"Failed to flush audio buffer: {e}")

    async def _play_audio_response(self, audio_data: bytes):
        """Play the audio response from the assistant."""
        if not audio_data or not AUDIO_AVAILABLE:
            return

        try:
            self.is_speaking = True
            self.status.set_listening(False)
            self.status.set_speaking(True)
            self.status.update_status("Assistant speaking...")
            
            self._mute_microphone()

            # Convert raw bytes to numpy array for sounddevice
            # OpenAI sends PCM16 data, so we need to convert it properly
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize for sounddevice
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Reshape for correct channels if needed
            if AUDIO_CONFIG['channels'] == 1 and len(audio_float.shape) == 1:
                # Mono audio - keep as is
                pass
            elif AUDIO_CONFIG['channels'] == 2 and len(audio_float.shape) == 1:
                # Convert mono to stereo by duplicating
                audio_float = np.column_stack([audio_float, audio_float])
            
            # Get selected output device from config if available
            output_device = getattr(self.config, 'selected_output_device', None)
            
            # Play back the received PCM audio_data directly
            import sounddevice as _sd
            # Play normalized float32 audio buffer
            _sd.play(audio_float, AUDIO_CONFIG['samplerate'], device=output_device)
            _sd.wait()

        except Exception as e:
            print(f"[{_timestamp()}] ❌ Audio playback failed: {e}")
            if self.config.debug_mode:
                import traceback
                traceback.print_exc()
        finally:
            self.is_speaking = False
            self.status.set_speaking(False)
            self._unmute_microphone()
            self.status.update_status("Ready for next interaction!")
            
            # After speaking, go back to listening
            if not self.config.text_only_mode:
                self.status.set_listening(True)

    async def _text_input_stream_impl(self):
        """Handle keyboard text input alongside voice input."""
        import sys
        
        loop = asyncio.get_event_loop()
        
        try:
            self.status.print_message("💬 Text input ready! Type your message and press Enter...", "green")
            self.status.print_message("💡 Press [Enter] to manually send audio. Type 'quit' to exit, or Ctrl+C to stop.", "cyan")
            self.status.update_status("ready_to_accept_tasks")
            
            while self.is_connected and not self.should_stop:
                try:
                    print(f"\n[{_timestamp()}] 💬 Type message (or press Enter to send audio): ", end="", flush=True)
                    text_input = await loop.run_in_executor(
                        None, 
                        lambda: sys.stdin.readline().strip()
                    )
                except KeyboardInterrupt:
                    self.status.print_message("\n👋 Text input stopped", "yellow")
                    break
                except Exception as e:
                    self.status.print_error(f"Text input error: {e}")
                    continue

                if not text_input:
                    await self.flush_audio_buffer()
                    continue
                        
                if text_input.lower() in ['quit', 'exit', 'q']:
                    self.status.print_message("\n👋 Text input stopped", "yellow")
                    break
                        
                # Show user's message
                self.status.print_user_message(text_input)
                
                # Send text message to the assistant
                await self.send_message(text_input)
                
                # Brief pause to see the response
                await asyncio.sleep(0.5)
                    
        except Exception as e:
            self.status.print_error(f"Text input stream error: {e}")
        finally:
            self.status.update_status("Ready for next interaction!")

    async def _audio_input_stream_impl(self):
        """Stream audio input to the Realtime API."""
        if not AUDIO_AVAILABLE:
            self.status.print_error("Audio not available for voice input")
            return
            
        self.status.update_status("Voice input active - start speaking!")
        
        # Setup Silero VAD processor for accurate speech detection
        vad_processor = SileroVADProcessor(
            threshold=self.config.silero_vad_threshold,
            min_speech_duration_ms=self.config.silero_min_speech_duration_ms,
            min_silence_duration_ms=self.config.silero_min_silence_duration_ms,
        )
        
        try:
            # Setup audio queue
            audio_queue = asyncio.Queue()

            # Use shared audio callback from voice_common.audio
            audio_callback = create_audio_callback(
                audio_queue,
                self.config,
                is_speaking_callback=lambda: self.is_speaking,
                debug_mode=self.config.debug_mode
            )
            # Get selected input device from config if available
            input_device = getattr(self.config, 'selected_input_device', None)
            
            stream = sd.InputStream(
                callback=audio_callback,
                samplerate=AUDIO_CONFIG["samplerate"],
                channels=AUDIO_CONFIG["channels"],
                blocksize=AUDIO_CONFIG["blocksize"],
                dtype=np.float32,
                device=input_device
            )

            with stream:
                chunks_sent = 0
                while self.is_connected and not self.should_stop:
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)

                        # Process with Silero VAD - convert bytes to int16 numpy array
                        audio_array = np.frombuffer(chunk, dtype=np.int16)
                        
                        # Process with Silero VAD
                        vad_result = vad_processor.process_audio_chunk(
                            audio_array, 
                            AUDIO_CONFIG["samplerate"]
                        )
                        
                        # Handle VAD events with detailed logging
                        if vad_result['speech_detected']:
                            if self.config.debug_mode:
                                self.status.print_message(f"🎤 Speech started (prob={vad_result['speech_probability']:.3f})", "green")
                            
                        if vad_result['speech_ended']:
                            if self.config.debug_mode:
                                self.status.print_message(f"🔈 Speech ended (prob={vad_result['speech_probability']:.3f}), requesting flush...", "yellow")
                            # Trigger the same behavior as before
                            self._trigger_speech_ended()
                        
                        # Log high probability silence that might be triggering false positives
                        elif self.config.debug_mode and vad_result['speech_probability'] > 0.3:
                            if chunks_sent % 50 == 0:  # Don't spam, log occasionally
                                self.status.print_message(f"🔇 Silence but prob={vad_result['speech_probability']:.3f}", "dim")

                        # Use the SDK's method to append audio
                        if self.connection:
                            # Audio chunk is already in bytes format from the callback
                            await self.connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("utf-8"))
                            self._audio_chunks_buffered += 1  # Track audio chunks for flush guard

                        chunks_sent += 1

                        if self.config.debug_mode and chunks_sent % 100 == 0:
                            self.status.print_message(f"🎤 Sent {chunks_sent} audio chunks", "dim")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if self.config.debug_mode:
                            self.status.print_error(f"Audio stream error: {e}")
                        continue

        except Exception as e:
            self.status.print_error(f"Failed to start audio input: {e}")
        finally:
            self.status.update_status("Audio input stopped")

    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Basic session event logger (placeholder)."""
        if self.config and getattr(self.config, "session_logging", False):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {"time": timestamp, "event": event_type, "details": details}
            # For now, just print to console or you could write to a file/db
            print(f"[{_timestamp()}] 📝 LOG {log_entry}")

    def _trigger_speech_ended(self):
        """Handle speech end detection - replaces the old _on_speech_paused callback."""
        # Immediately switch to processing state for UI
        self.status.set_listening(False)
        self.status.set_processing(True)
        self.status.update_status("Processing your speech...")
        
        # Flush buffered audio to server
        asyncio.get_event_loop().create_task(self.flush_audio_buffer())

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

def _generate_dynamic_tool_list(function_map: Dict[str, Callable]) -> dict:
    """Generate tool list dynamically from actual available tools."""
    available_tools = []
    
    for name, func in function_map.items():
        # Extract info from the actual tool function
        description = getattr(func, 'description', 'No description available')
        # Extract parameter info to check if wake_phrase is required
        requires_wake = "wake_phrase" in str(getattr(func, 'inputs', {}))
        
        tool_info = {
            "name": name,
            "description": description,
            "usage": f"Use {name} for robot operations"
        }
        
        if requires_wake:
            tool_info["requires"] = "wake_phrase with 'mave'"
        
        available_tools.append(tool_info)
    
    return {
        "status": "success",
        "available_tools": available_tools,
        "total_tools": len(available_tools),
        "safety_note": "Most movement tools require the wake word 'mave' for safety"
    }


def create_robot_tools() -> List[ToolDefinition]:
    """Create robot tool definitions using auto-discovery from smolagents registry."""
    tools = []
    
    try:
        # Get tools and function map from registry (auto-discovery)
        tool_specs, function_map = register_tools_for_openai()
        
        # Convert registry specs to ToolDefinition objects
        for spec in tool_specs:
            func_info = spec["function"]
            tool = ToolDefinition(
                name=func_info["name"],
                description=func_info["description"], 
                parameters=func_info["parameters"],
                execute=function_map[func_info["name"]]
            )
            tools.append(tool)
            
        # Add the meta-tool for listing (not a robot tool, but useful for users)
        list_tools_tool = ToolDefinition(
            name="list_available_tools",
            description="List all available robot tools and their descriptions",
            parameters={"type": "object", "properties": {}, "required": []},
            execute=lambda: _generate_dynamic_tool_list(function_map)
        )
        tools.append(list_tools_tool)
        
    except Exception as e:
        print(f"[{_timestamp()}] ❌ ERROR with auto-discovery: {e}")
        return []
        
    return tools

def create_robot_agent(config: Optional[VoiceAgentConfig] = None) -> RealtimeAgent:
    """Create a robot voice agent using OpenAI Agents patterns."""
    if not config:
        config = load_config_from_env()
    
    try:
        system_prompt = get_system_prompt("voice")
    except Exception as e:
        if config.debug_mode:
            print(f"[{_timestamp()}] ❌ ERROR getting system prompt: {e}")
        system_prompt = "You are a robot control assistant."
    
    
    # Create tools
    try:
        robot_tools = create_robot_tools()
        if config.debug_mode:
            print(f"[{_timestamp()}] 🚨 CREATED {len(robot_tools)} ROBOT TOOLS")
    except Exception as e:
        if config.debug_mode:
            print(f"[{_timestamp()}] ❌ ERROR creating robot tools: {e}")
        robot_tools = []
    
    # Create agent with robot tools (use OPENAI_MODEL_ID from .env or fallback to default)
    openai_model = OPENAI_MODEL_ID
    agent = RealtimeAgent(
        name=ROBOT_AGENT_NAME,
        instructions=f"{system_prompt} + {OPENAI_VOICE_INSTRUCTIONS}",
        tools=robot_tools,
        voice=config.openai_voice,
        temperature=config.openai_temperature,
        model=openai_model,
        config=config
    )
    
    return agent


class OpenAIVoiceAgent(BaseVoiceAgent):
    """
    Enhanced OpenAI voice agent with comprehensive features from samples.
    """

    def __init__(self, bridge_ref, ui_ref, api_key: str, config: Optional[VoiceAgentConfig] = None):
        super().__init__(bridge_ref, ui_ref)
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        self.config = config or load_config_from_env()
        self.session = None
        self.should_stop = False

    async def start(self):
        """Starts the enhanced OpenAI voice agent session, including agent-driven startup and goodbye speech."""
        # UI for session is managed by the realtime session; skip agent UI startup

        # Create agent and session
        agent = create_robot_agent(self.config)
        self.session = RealtimeSession(agent, self.api_key, self.bridge, self.config)

        try:
            # Connect to Realtime API
            await self.session.connect()

            # Agent-driven welcome message
            start_msg = await self.generate_fabrication_message("start")
            if start_msg:
                await self.session.send_message(start_msg)

            # Enter main streaming loop
            await self.session.start_session_with_connection()

        except Exception as e:
            self.ui.print_error(f"Failed to start OpenAI session: {e}")
        finally:
            # Agent-driven goodbye message
            try:
                end_msg = await self.generate_fabrication_message("end")
                if end_msg and self.session and self.session.is_connected:
                    await self.session.send_message(end_msg)
            except Exception:
                pass
            # Clean up
            await self.stop()

    async def stop(self):
        """Stops the session and cleans up resources."""
        self.should_stop = True  # Set flag first to stop all loops
        if self.session:
            await self.session.close()
        if self.ui:
            self.ui.stop_live_display()
            self.ui.print_message("OpenAI session stopped.", "yellow")

    def stop_sync(self):
        """Synchronous stop method that can be called from any thread."""
        self.should_stop = True  # Set flag first to stop all loops
        
        # Stop session synchronously if possible
        if self.session:
            self.session.should_stop = True
            self.session.is_connected = False
        # Ensure AsyncOpenAI HTTP client closes its underlying HTTPX client
        if hasattr(self, 'session') and self.session and hasattr(self.session, 'client'):
            try:
                import asyncio
                asyncio.run(self.session.client.aclose())
            except Exception:
                pass
        
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
                self.ui.print_message("OpenAI session stopped.", "yellow")
            except Exception as e:
                print(f"[{_timestamp()}] ⚠️ UI shutdown error: {e}")
                # Force basic stop
                if hasattr(self.ui, 'is_stopped'):
                    self.ui.is_stopped = True

    async def start_session(self):
        """Start the realtime session with enhanced features."""
        if not self.session:
            # Create enhanced agent and session
            agent = create_robot_agent(self.config)
            self.session = RealtimeSession(agent, self.api_key, self.bridge, self.config)
            
        await self.session.connect()
        
        try:
            # Start the session using the new connection manager pattern
            await self.session.start_session_with_connection()
                
        except KeyboardInterrupt:
            self.session.status.print_message("\n👋 Shutting down gracefully...", "yellow")
        finally:
            if self.session and not self.should_stop:
                try:
                    await self.session.close()
                except Exception as e:
                    # Ignore connection errors during shutdown
                    if self.config.debug_mode:
                        print(f"[{_timestamp()}] Warning: Session close error during shutdown: {e}")

    def _text_input_stream(self):
        """Handle keyboard text input alongside voice input - method needs to be defined in parent class."""
        # This will need to be moved or referenced from OpenAIVoiceAgent
        pass

    def _audio_input_stream(self):
        """Stream audio input to the Realtime API - method needs to be defined in parent class."""
        # This will need to be moved or referenced from OpenAIVoiceAgent
        pass

    async def handle_task_event(self, event_type: str, event_data: dict):
        """Handle task events from the bridge using system prompts for dynamic LLM commentary."""
        
        if not self.session:
            return
        
        try:
            from ur.config.system_prompts import build_commentary_prompt
            
            # Generate appropriate commentary using system prompts or custom directives
            if event_type == 'task_received':
                # Use the standardized commentary prompt for task announcements
                prompt = build_commentary_prompt(
                    context_type='task_execute',
                    task_name=event_data.get('task_name')
                )
                commentary = await self._prompt_llm_for_commentary(prompt)
            elif event_type == 'task_starting':
                # Task execution starting: announce movement
                commentary = self._generate_task_starting_commentary(event_data)
            elif event_type == 'command_received':
                prompt = build_commentary_prompt(
                    context_type='command',
                    command=event_data.get('command')
                )
                commentary = await self._prompt_llm_for_commentary(prompt)
            elif event_type == 'robot_action':
                prompt = build_commentary_prompt(
                    context_type='task_execute',
                    task_name=event_data.get('task_name'),
                    action_type=event_data.get('action_type')
                )
                commentary = await self._prompt_llm_for_commentary(prompt)
            else:
                # Fallback for other event types
                commentary = f"Processing {event_type}..."
            
            # Send the generated commentary
            if commentary and commentary.strip():
                await self.speak_commentary(commentary)

        except Exception as e:
            if self.config.debug_mode:
                print(f"[{_timestamp()}] ❌ [COMMENTARY ERROR] {event_type}: {e}")

    async def speak_commentary(self, text: str):
        """Generates and speaks commentary without sending it as a user message."""
        # Skip if commentary TTS is disabled or no session/context
        if not text or not self.session or not self.config.enable_commentary_tts:
            return

        if self.config.debug_mode:
            print(f"[{_timestamp()}] 🎤 [ASSISTANT SPEAKING] {text}")

        try:
            # Mute microphone and update UI status for commentary TTS
            # Block audio input by marking session as speaking
            self.session.is_speaking = True
            self.session.status.set_listening(False)
            self.session.status.set_speaking(True)

            # Interrupt any current response
            await self.session.interrupt_audio()

            # Get OpenAI client
            client = getattr(self.session, 'client', None)
            if client is None:
                import openai as _openai
                client = _openai.AsyncOpenAI(api_key=self.api_key)

            # Synchronously fetch the full WAV bytes and play them
            resp = await client.audio.speech.create(
                model="tts-1",
                voice=self.session.agent.voice,
                input=text,
                response_format='wav',
                speed=self.config.openai_voice_speed,
                instructions=self.session.agent.instructions,
            )
            audio_bytes = resp.content
            # Decode WAV bytes and play via sounddevice
            import io, wave
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio_array = np.frombuffer(frames, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                import sounddevice as _sd
                _sd.play(audio_float, wav_file.getframerate())
                _sd.wait()

        except Exception as e:
            if self.config.debug_mode:
                print(f"[{_timestamp()}] ❌ [TTS ERROR] Failed to speak commentary: {e}")
        finally:
            # Unmute microphone and revert UI status after commentary
            self.session.is_speaking = False
            self.session.status.set_speaking(False)
            if not self.config.text_only_mode:
                self.session.status.set_listening(True)

    async def generate_fabrication_message(self, message_type: str) -> str:
        """Generate welcome/goodbye message for fabrication start/end."""
        if message_type == "start":
            prompt = "Generate a concise, friendly welcome: 'Mave here—ready to assist with fabrication tasks. How can I help you today?' Keep under 20 words."
            default = "Mave here—ready to assist with fabrication tasks. How can I help you today?"
        else:  # end
            prompt = "Generate a concise, appreciative goodbye: 'Fabrication complete. Thank you for collaborating—until next time!' Keep under 20 words."
            default = "Fabrication complete. Thank you for collaborating—until next time!"
        
        try:
            # Use the existing LLM to generate the message
            response = await self._prompt_llm_for_commentary(prompt)
            return response.strip() if response else default
        except Exception as e:
            if self.config.debug_mode:
                print(f"[{_timestamp()}] ⚠️ [AGENT] Error generating fabrication message: {e}")
            return default

    async def _prompt_llm_for_commentary(self, full_prompt: str) -> str:
        """Generate commentary using the OpenAI API with complete system prompt."""
        try:
            # Parse the prompt to separate system and user parts
            parts = full_prompt.split('\n\nTask received:', 1)
            if len(parts) == 2:
                system_prompt = parts[0]
                user_prompt = 'Task received:' + parts[1]
            else:
                parts = full_prompt.split('\n\nCommand received:', 1)
                if len(parts) == 2:
                    system_prompt = parts[0]
                    user_prompt = 'Command received:' + parts[1]
                else:
                    parts = full_prompt.split('\n\nCurrent task:', 1)
                    if len(parts) == 2:
                        system_prompt = parts[0]
                        user_prompt = 'Current task:' + parts[1]
                    else:
                        system_prompt = full_prompt
                        user_prompt = "Generate appropriate commentary."
            
            # Use existing AsyncOpenAI client from session, or fallback to new AsyncOpenAI
            client = getattr(self.session, 'client', None)
            if client is None:
                import openai as _openai
                client = _openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.commentary_max_tokens,
                temperature=self.config.commentary_temperature
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"[{_timestamp()}] ❌ [LLM ERROR] {e}")
            return f"Executing task..."  # Fallback

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

    async def _prompt_llm_for_task_announcement(self, task_context: str) -> str:
        """Generate commentary for task announcements."""
        try:
            from ur.config.system_prompts import get_system_prompt
            
            # Get the system prompt with fabrication context
            system_prompt = get_system_prompt(mode="fabrication", robot_connected=True)
            
            # Use existing AsyncOpenAI client from session, or fallback to new AsyncOpenAI
            client = getattr(self.session, 'client', None)
            if client is None:
                import openai as _openai
                client = _openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_context}
                ],
                max_tokens=self.config.commentary_max_tokens,
                temperature=self.config.commentary_temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.config.debug_mode:
                print(f"[{_timestamp()}] ❌ [LLM ERROR] {e}")
            return f"Starting {task_context}..."  # Fallback
