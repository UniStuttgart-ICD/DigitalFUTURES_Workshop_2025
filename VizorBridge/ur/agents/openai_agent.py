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

import websockets

# Add project root to path to allow imports from other modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ur.agents.base_agent import BaseVoiceAgent
from ur.config.voice_config import VoiceAgentConfig, load_config_from_env
from ur.agents.voice_common.status import VoiceAgentStatus
from ur.agents.voice_common.audio import create_audio_callback, encode_audio_for_openai, AUDIO_CONFIG
from ur.tools import register_tools_for_openai
from ur.agents.shared_prompts import get_system_prompt
from ur.config.system_config import (
    OPENAI_MODEL_ID,
    OPENAI_TRANSCRIPTION_MODEL,
    TOOL_EXECUTOR_MAX_WORKERS,
    SESSION_ID_PREFIX
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
from ur.config.system_prompts import build_enhanced_instructions

# Optional audio imports
try:
    import openai
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
        
    def add_tool(self, tool: ToolDefinition):
        """Add a tool to the agent."""
        self.tools.append(tool)
        self._tool_map[tool.name] = tool
        
    def get_session_config(self) -> Dict[str, Any]:
        """Get session configuration for the Realtime API."""
        return {
            "modalities": ["audio", "text"],
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": OPENAI_INPUT_AUDIO_FORMAT,
            "output_audio_format": OPENAI_OUTPUT_AUDIO_FORMAT,
            "input_audio_transcription": {
                "model": OPENAI_TRANSCRIPTION_MODEL
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.4,  # Default VAD threshold (configurable via VAD_THRESHOLD env var)
                "prefix_padding_ms": 200,  # Default padding (configurable via VAD_PREFIX_PADDING_MS env var)
                "silence_duration_ms": 500,  # Default silence duration (configurable via VAD_SILENCE_DURATION_MS env var)
                "create_response": True
            },
            "tools": [self._convert_tool_to_openai_spec(tool) for tool in self.tools],
            "temperature": self.temperature
        }
        
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
        
        # WebSocket connection
        self.client = openai.AsyncOpenAI(api_key=api_key)
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
                    print(f"❌ Event handler error: {e}")
    
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

    def _convert_tool_to_openai_spec(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Convert tool definition to OpenAI API specification."""
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }

    async def connect(self):
        """Connect to OpenAI Realtime API using the official SDK."""
        self.status.update_status("Connecting to OpenAI Realtime API...")
        
        # Store the connection manager for later use in start_session
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
                elif self.config.enable_text_input:
                    self.status.update_status("Hybrid mode ready - speak OR type!")
                    self.status.print_message("🎤💬 Hybrid mode: You can speak OR type messages (press Enter to send).", "cyan")
                else:
                    self.status.update_status("Voice input ready - start speaking!")

                # Give user a moment to read, then switch to compact display
                await asyncio.sleep(3)
                self.status.collapse_to_compact()

                # Start concurrent tasks: event handling + input streams
                event_task = asyncio.create_task(self._handle_server_events())
                input_tasks = []

                if self.config.enable_text_input or self.config.text_only_mode:
                    input_tasks.append(asyncio.create_task(self._text_input_stream_impl()))

                if not self.config.text_only_mode and AUDIO_AVAILABLE:
                    input_tasks.append(asyncio.create_task(self._audio_input_stream_impl()))
                elif self.config.text_only_mode:
                    self.status.print_message("🔇 Voice input disabled (text-only mode)", "yellow")

                tasks_to_wait = [event_task] + input_tasks
                if tasks_to_wait:
                    await asyncio.gather(*tasks_to_wait, return_exceptions=True)

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
            await self.connection.session.update(
                session={
                    "instructions": self.agent.instructions,
                    "voice": self.agent.voice,
                    "input_audio_format": OPENAI_INPUT_AUDIO_FORMAT,
                    "output_audio_format": OPENAI_OUTPUT_AUDIO_FORMAT,
                    "input_audio_transcription": {
                        "model": OPENAI_TRANSCRIPTION_MODEL
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self.config.vad_threshold,  # Configurable VAD sensitivity
                        "prefix_padding_ms": self.config.vad_prefix_padding_ms,  # Configurable padding
                        "silence_duration_ms": self.config.vad_silence_duration_ms,  # Configurable silence duration
                        "create_response": True
                    },
                    "tools": [self._convert_tool_to_openai_spec(tool) for tool in self.agent.tools],
                    "temperature": self.agent.temperature
                }
            )
            
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
            # Use the send method to create a conversation item and trigger response
            await self.connection.send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                        }
            })
            
            # Trigger response
            await self.connection.send({"type": "response.create"})
            
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
                print(f"🚨 FUNCTION/TOOL EVENT: {event_type} - {event}")
            elif event_type == "response.audio.delta":
                print(f"🚨 AUDIO DELTA: {event_type} (audio data: {len(str(getattr(event, 'delta', ''))) if hasattr(event, 'delta') else 'N/A'} bytes)")
            elif "response" in event_type:
                print(f"🚨 RESPONSE EVENT: {event_type}")
            elif "conversation" in event_type:
                print(f"🚨 CONVERSATION EVENT: {event_type} - {event}")
            else:
                print(f"🔍 DEBUG EVENT: {event_type}")
        
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
                if self.audio_output_buffer:
                    await self._play_audio_response(bytes(self.audio_output_buffer))
                    self.audio_output_buffer = bytearray()
                    
            elif event_type == "response.text.delta":
                text_delta = getattr(event, 'delta', "")
                self.text_response_buffer += text_delta
                self.is_receiving_text = True
                
            elif event_type == "response.text.done":
                if self.text_response_buffer:
                    self.status.print_message(f"🤖 Assistant: {self.text_response_buffer}", "blue bold")
                    self.text_response_buffer = ""
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
                        print(f"🚨 FUNCTION CALL DETECTED: {event}")
                    # Extract function call data properly from the Python SDK event
                    item_data = event.item.model_dump()
                    self._emit_event(EventType.TOOL_CALL_REQUESTED, {
                        "name": item_data.get("name"),
                        "arguments": item_data.get("arguments", "{}"),
                        "call_id": item_data.get("call_id") or item_data.get("id"),
                        "source_event": "response.output_item.done"
                    })
                elif self.config.debug_mode and hasattr(event, 'item'):
                    print(f"🔍 Response item completed: {event.item.type if hasattr(event.item, 'type') else 'unknown'}")
                
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
                
            else:
                # Optional logging for unknown event types
                if self.config.debug_mode:
                    print(f"🔍 Unknown event type: {event_type}")
                
        except Exception as e:
            print(f"❌ Error processing event {event_type}: {e}")
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
        
    async def _handle_response_started(self, event: RealtimeEvent):
        """Handle response start."""
        self.status.set_processing(False)
        self.status.update_status("Assistant is responding...")
        
    async def _handle_response_completed(self, event: RealtimeEvent):
        """Handle response completion."""
        # Log session event
        self._log_event("response_completed", {"content": self.text_response_buffer})
        
        # Play audio if available
        if self.audio_output_buffer and not self.config.text_only_mode and AUDIO_AVAILABLE:
            await self._play_audio_response(bytes(self.audio_output_buffer))

        # Reset buffers after processing
        self.audio_output_buffer = bytearray()
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
            print(f"🚨 TOOL CALL HANDLER TRIGGERED! Event: {event}")
            print(f"🔧 Tool call event data: {event.data}")
        
        try:
            event_data = event.data
            
            if self.config.debug_mode:
                print(f"🔧 Processing event data: {event_data}")
                print(f"🔧 Available keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'Not a dict'}")
            
            # Extract function call information - use improved direct extraction
            function_name = event_data.get("name")
            arguments_str = event_data.get("arguments", "{}")
            call_id = event_data.get("call_id")
            
            if self.config.debug_mode:
                print(f"🔧 Extracted: name={function_name}, args={arguments_str}, call_id={call_id}")
            
            if not function_name or not call_id:
                if self.config.debug_mode:
                    print(f"⚠️ Skipping invalid function call - missing name ({function_name}) or call_id ({call_id})")
                return
                
            # Deduplication check
            if call_id in self.processed_tool_calls:
                if self.config.debug_mode:
                    print(f"🔄 Skipping duplicate tool call: {call_id}")
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
                    print(f"🤖 Executing: {function_name}({args})")
                
                # Execute the tool
                result = await self._execute_tool(tool, args)
                if self.config.debug_mode:
                    print(f"✅ Tool execution result: {result}")
                
                # Send function result back using the correct Python SDK method
                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result)
                    }
                )
                
                # Trigger response after successful tool execution
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
                print(f"⚠️ Sending tool error for call_id {call_id}: {error_message}")
                
            # Use the correct Python SDK method to create a function call output
            await self.connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"error": error_message})
                }
            )
            
            # Trigger response after sending error
            await self.connection.response.create()
            
        except Exception as e:
            self.status.print_error(f"Failed to send tool error: {e}")

    async def interrupt_audio(self):
        """Interrupt any ongoing audio playback."""
        if self.is_speaking:
            self.audio_interrupt_event.set()
            if self.config.debug_mode:
                self.status.print_message("Audio interrupt event set", "dim")

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
            
            # Play the audio
            sd.play(audio_float, samplerate=AUDIO_CONFIG['samplerate'], device=output_device)
            sd.wait()  # Wait for playback to complete
            
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
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
                print("🎤 Started listening...")
                self.status.set_listening(True)

    async def _text_input_stream_impl(self):
        """Handle keyboard text input alongside voice input."""
        import sys
        
        loop = asyncio.get_event_loop()
        
        try:
            self.status.print_message("💬 Text input ready! Type your message and press Enter...", "green")
            self.status.print_message("💡 Commands: 'quit' to exit text mode, Ctrl+C to stop completely", "cyan")
            
            while self.is_connected and not self.should_stop:
                try:
                    print("\n💬 Type message: ", end="", flush=True)
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
                    continue
                        
                if text_input.lower() in ['quit', 'exit', 'q']:
                    self.status.print_message("👋 Text input stopped", "yellow")
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

                        # Use the SDK's method to append audio
                        if self.connection:
                            # Audio chunk is already in bytes format from the callback
                            await self.connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("utf-8"))

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
            print(f"📝 LOG {log_entry}")

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
            tool_info["requires"] = "wake_phrase with 'timbra'"
        
        available_tools.append(tool_info)
    
    return {
        "status": "success",
        "available_tools": available_tools,
        "total_tools": len(available_tools),
        "safety_note": "Most movement tools require the wake word 'timbra' for safety"
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
        print(f"❌ ERROR with auto-discovery: {e}")
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
            print(f"❌ ERROR getting system prompt: {e}")
        system_prompt = "You are a robot control assistant."
    
    enhanced_instructions = build_enhanced_instructions(system_prompt)
    
    # Create tools
    try:
        robot_tools = create_robot_tools()
        if config.debug_mode:
            print(f"🚨 CREATED {len(robot_tools)} ROBOT TOOLS")
    except Exception as e:
        if config.debug_mode:
            print(f"❌ ERROR creating robot tools: {e}")
        robot_tools = []
    
    # Create agent with robot tools (use OPENAI_MODEL_ID from .env or fallback to default)
    openai_model = OPENAI_MODEL_ID
    agent = RealtimeAgent(
        name=ROBOT_AGENT_NAME,
        instructions=enhanced_instructions,
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
        """Starts the enhanced OpenAI voice agent session."""
        # Use the shared status system
        self.ui = VoiceAgentStatus(ROBOT_AGENT_UI_NAME)
        self.ui.start_live_display()
        self.ui.update_status("Creating agent specification...")

        # Create enhanced agent
        agent = create_robot_agent(self.config)
        self.session = RealtimeSession(agent, self.api_key, self.bridge, self.config)

        try:
            await self.session.connect()
            await self.start_session()
        except Exception as e:
            self.ui.print_error(f"Failed to start OpenAI session: {e}")
        finally:
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
                if hasattr(self.ui, 'complete_shutdown'):
                    self.ui.complete_shutdown()  # Use aggressive shutdown method
                else:
                    # Fallback to regular stop method
                    self.ui.stop_live_display()
                self.ui.print_message("OpenAI session stopped.", "yellow")
            except Exception as e:
                print(f"⚠️ UI shutdown error: {e}")
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
                        print(f"Warning: Session close error during shutdown: {e}")

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
            
            # Generate appropriate commentary using system prompts
            if event_type == 'task_received':
                prompt = build_commentary_prompt(
                    context_type='task_execute',
                    task_name=event_data.get('task_name')
                )
                commentary = await self._prompt_llm_for_commentary(prompt)
                
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
                await self.session.send_message(commentary)
                if self.config.debug_mode:
                    print(f"🎭 [LLM COMMENTARY] {event_type}: {commentary}")
                    
        except Exception as e:
            if self.config.debug_mode:
                print(f"❌ [COMMENTARY ERROR] {event_type}: {e}")

    async def generate_fabrication_message(self, message_type: str) -> str:
        """Generate welcome/goodbye message for fabrication start/end."""
        if message_type == "start":
            prompt = "Generate a brief, friendly welcome message for starting a collaborative fabrication session with a human. Be encouraging and ready to help. Keep it under 20 words."
            default = "Hello! I'm ready to help with your fabrication project. Let's build something amazing together!"
        else:  # end
            prompt = "Generate a brief, appreciative goodbye message for ending a collaborative fabrication session. Thank the human for working together. Keep it under 20 words."
            default = "Great work! It was a pleasure collaborating with you on this fabrication project. Until next time!"
        
        try:
            # Use the existing LLM to generate the message
            response = await self._prompt_llm_for_commentary(prompt)
            return response.strip() if response else default
        except Exception as e:
            if self.config.debug_mode:
                print(f"⚠️ [AGENT] Error generating fabrication message: {e}")
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
            
            # Use existing AsyncOpenAI client from the active session for commentary generation
            client = self.session.client
            
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
                print(f"❌ [LLM ERROR] {e}")
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
