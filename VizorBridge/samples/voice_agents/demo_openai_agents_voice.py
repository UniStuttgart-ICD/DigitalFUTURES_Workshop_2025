"""Robot voice agent using OpenAI Agents Voice SDK patterns.

This implementation follows the OpenAI Agents Voice SDK architecture
adapted for Python, providing:

- Simplified agent configuration with RealtimeAgent pattern
- Tool integration with automatic execution
- Session management with event handling
- Voice activity detection and interruption handling
- Audio streaming with built-in controls
- Enhanced audio feedback prevention

Configuration:
The agent supports these key settings (set via environment variables):
- TEXT_ONLY_MODE: Use text-only mode (true/false)
- ENABLE_TEXT_INPUT: Allow text input alongside voice (true/false)
- ENABLE_INTERRUPTION: Allow interrupting assistant speech (true/false)
- FEEDBACK_PREVENTION: Enable enhanced feedback prevention (true/false)
- SILENCE_AFTER_SPEAKING: Silence period after assistant speech in seconds (0.8)
- AUDIO_THRESHOLD: Minimum audio level threshold (0.005)
- POST_SPEECH_DELAY: Delay after speaking in seconds (0.8)
- THREADED_TOOLS: Execute robot tools in background threads (true/false)
- DEBUG: Enable detailed debug output (true/false)
- ENABLE_LOGGING: Enable session logging (true/false)

Run examples:
    # Normal mode with feedback prevention
    python demo_openai_agents_voice.py
    
    # Debug mode to see audio filtering
    DEBUG=true python demo_openai_agents_voice.py
    
    # Disable feedback prevention if having issues
    FEEDBACK_PREVENTION=false python demo_openai_agents_voice.py
    
    # Lower audio threshold for more sensitive microphone
    AUDIO_THRESHOLD=0.001 python demo_openai_agents_voice.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import base64
import warnings
import concurrent.futures
import threading
import time
import uuid
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

import websockets
import ssl

# Allow running directly from the samples folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ur.tools import register_tools_for_openai  # noqa: E402
from ur.agents.shared_prompts import get_system_prompt  # noqa: E402
from ur.agents.voice_common import (
    VoiceAgentConfig, load_config_from_env,
    AUDIO_CONFIG, SimpleVoiceProcessor,
    VoiceAgentStatus
)

# Global config instance - now using shared configuration
CONFIG = load_config_from_env()

class EventType(Enum):
    """Event types for the voice agent session."""
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
    TOOL_APPROVAL_REQUESTED = "tool_approval_requested"
    GUARDRAIL_TRIPPED = "guardrail_tripped"
    ERROR = "error"

# VoiceAgentStatus is now imported from shared module

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

@dataclass
class ConversationItem:
    """Single conversation item for logging."""
    timestamp: datetime
    type: str  # 'user_speech', 'assistant_speech', 'text_input', 'text_output', 'tool_call'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolCallLog:
    """Tool call execution log."""
    timestamp: datetime
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class SessionMetrics:
    """Session performance metrics."""
    total_duration_seconds: float = 0.0
    speech_interactions: int = 0
    text_interactions: int = 0
    tool_calls_total: int = 0
    tool_calls_successful: int = 0
    tool_calls_failed: int = 0
    interruptions: int = 0
    errors: int = 0
    average_response_time_ms: float = 0.0
    robot_connection_time_seconds: float = 0.0

@dataclass
class SessionLog:
    """Complete session log data structure."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    agent_name: str = ""
    model: str = ""
    voice: str = ""
    
    # Conversation data
    conversation_items: List[ConversationItem] = field(default_factory=list)
    tool_calls: List[ToolCallLog] = field(default_factory=list)
    
    # Session metrics
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    
    # Configuration snapshots
    voice_config: Dict[str, Any] = field(default_factory=dict)
    audio_config: Dict[str, Any] = field(default_factory=dict)
    
    # Technical data
    robot_connection_status: bool = False
    api_response_times: List[float] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # File paths
    log_file_path: Optional[str] = None
    
    # Metadata
    user_id: Optional[str] = None
    session_notes: str = ""

class SessionLogger:
    """Session logger for comprehensive conversation and audio logging."""
    
    def __init__(self, session_id: str, agent: RealtimeAgent, log_directory: Optional[str] = None):
        self.session_id = session_id
        self.agent = agent
        
        # Setup log directory
        if log_directory:
            self.log_directory = Path(log_directory)
        else:
            self.log_directory = Path(CONFIG.log_directory)
            
        # Create directory structure with date organization
        self.date_dir = self.log_directory / datetime.now().strftime("%Y-%m-%d")
        self.date_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session log
        self.session_log = SessionLog(
            session_id=session_id,
            start_time=datetime.now(),
            agent_name=agent.name,
            model=agent.model,
            voice=agent.voice,
            voice_config=CONFIG.__dict__.copy(),
            audio_config=AUDIO_CONFIG.copy()
        )
        
        # Simplified - no audio recording to reduce complexity
        self.audio_recording_enabled = False
            
        # Performance tracking
        self._response_start_times = {}
        self._tool_call_start_times = {}
        
        # Log file path
        self.log_file_path = self.date_dir / f"{session_id}.json"
        self.session_log.log_file_path = str(self.log_file_path)
        
        print(f"📝 Session logger initialized: {session_id}")
            
    def start_audio_recording(self):
        """Simplified - no audio recording."""
        pass
            
    def stop_audio_recording(self):
        """Simplified - no audio recording."""
        pass
            
    def record_audio_chunk(self, audio_data: bytes, source: str = "unknown"):
        """Simplified - no audio recording."""
        pass
            
    def log_conversation_item(self, item_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a conversation item."""
        if not CONFIG.enable_logging and item_type in ["user_speech", "assistant_speech"]:
            return
            
        item = ConversationItem(
            timestamp=datetime.now(),
            type=item_type,
            content=content,
            metadata=metadata or {}
        )
        
        self.session_log.conversation_items.append(item)
        
        # Update metrics
        if item_type == "user_speech":
            self.session_log.metrics.speech_interactions += 1
        elif item_type in ["text_input", "text_output"]:
            self.session_log.metrics.text_interactions += 1
            
        if CONFIG.debug_mode:
            print(f"📝 Logged conversation item: {item_type} - {content[:50]}...")
            
    def log_tool_call_start(self, tool_name: str, arguments: Dict[str, Any], call_id: str):
        """Log the start of a tool call."""
        self._tool_call_start_times[call_id] = time.time()
        
    def log_tool_call_completion(self, tool_name: str, arguments: Dict[str, Any], result: Any, 
                               call_id: str, success: bool = True, error_message: Optional[str] = None):
        """Log tool call completion."""
        if not CONFIG.enable_logging:
            return
            
        start_time = self._tool_call_start_times.get(call_id, time.time())
        execution_time_ms = (time.time() - start_time) * 1000
        
        tool_log = ToolCallLog(
            timestamp=datetime.now(),
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message
        )
        
        self.session_log.tool_calls.append(tool_log)
        
        # Update metrics
        self.session_log.metrics.tool_calls_total += 1
        if success:
            self.session_log.metrics.tool_calls_successful += 1
        else:
            self.session_log.metrics.tool_calls_failed += 1
            
        # Clean up tracking
        if call_id in self._tool_call_start_times:
            del self._tool_call_start_times[call_id]
            
        if CONFIG.debug_mode:
            print(f"📝 Logged tool call: {tool_name} - {execution_time_ms:.1f}ms - {'✅' if success else '❌'}")
            
    def log_event(self, event_type: EventType, event_data: Dict[str, Any]):
        """Log session events."""
        if event_type == EventType.AUDIO_INTERRUPTED:
            self.session_log.metrics.interruptions += 1
        elif event_type == EventType.ERROR:
            self.session_log.metrics.errors += 1
            self.session_log.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error_type": event_data.get("error_type", "unknown"),
                "error_message": event_data.get("error_message", ""),
                "error_code": event_data.get("error_code", "")
            })
            
    def log_response_time(self, response_time_ms: float):
        """Log API response time."""
        self.session_log.api_response_times.append(response_time_ms)
        
        # Update average
        if self.session_log.api_response_times:
            self.session_log.metrics.average_response_time_ms = sum(self.session_log.api_response_times) / len(self.session_log.api_response_times)
            
    def update_robot_connection_status(self, connected: bool):
        """Update robot connection status."""
        self.session_log.robot_connection_status = connected
        
    def finalize_and_save(self):
        """Finalize session log and save to file."""
        # Finalize session data
        self.session_log.end_time = datetime.now()
        
        if self.session_log.start_time:
            duration = self.session_log.end_time - self.session_log.start_time
            self.session_log.metrics.total_duration_seconds = duration.total_seconds()
            
        # Save main log file
        self._save_log_file()
        
        # Print summary
        self._print_session_summary()
        
        return self.log_file_path
        
    def _save_log_file(self):
        """Save session log to JSON file."""
        try:
            log_data = {
                "session_id": self.session_log.session_id,
                "metadata": {
                    "start_time": self.session_log.start_time.isoformat(),
                    "end_time": self.session_log.end_time.isoformat() if self.session_log.end_time else None,
                    "duration_seconds": self.session_log.metrics.total_duration_seconds,
                    "agent_name": self.session_log.agent_name,
                    "model": self.session_log.model,
                    "voice": self.session_log.voice,
                    "voice_config": self.session_log.voice_config,
                    "audio_config": self.session_log.audio_config,
                    "robot_connected": self.session_log.robot_connection_status,
                    "log_file_path": self.session_log.log_file_path
                },
                "conversation": [
                    {
                        "timestamp": item.timestamp.isoformat(),
                        "type": item.type,
                        "content": item.content,
                        "metadata": item.metadata
                    }
                    for item in self.session_log.conversation_items
                ],
                "tool_calls": [
                    {
                        "timestamp": call.timestamp.isoformat(),
                        "tool_name": call.tool_name,
                        "arguments": call.arguments,
                        "result": call.result,
                        "execution_time_ms": call.execution_time_ms,
                        "success": call.success,
                        "error_message": call.error_message
                    }
                    for call in self.session_log.tool_calls
                ],
                "metrics": {
                    "total_duration_seconds": self.session_log.metrics.total_duration_seconds,
                    "speech_interactions": self.session_log.metrics.speech_interactions,
                    "text_interactions": self.session_log.metrics.text_interactions,
                    "tool_calls_total": self.session_log.metrics.tool_calls_total,
                    "tool_calls_successful": self.session_log.metrics.tool_calls_successful,
                    "tool_calls_failed": self.session_log.metrics.tool_calls_failed,
                    "interruptions": self.session_log.metrics.interruptions,
                    "errors": self.session_log.metrics.errors,
                    "average_response_time_ms": self.session_log.metrics.average_response_time_ms
                },
                "errors": self.session_log.errors,
                "api_response_times": self.session_log.api_response_times
            }
            
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
            file_size_kb = self.log_file_path.stat().st_size / 1024
            print(f"💾 Session log saved: {self.log_file_path} ({file_size_kb:.1f} KB)")
            
        except Exception as e:
            print(f"❌ Failed to save session log: {e}")
            
    def _print_session_summary(self):
        """Print session summary."""
        if not RICH_AVAILABLE:
            self._print_simple_summary()
            return
            
        from rich.table import Table
        from rich.panel import Panel
        
        # Create summary table
        table = Table(title="Session Summary")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Session ID", self.session_log.session_id)
        table.add_row("Duration", f"{self.session_log.metrics.total_duration_seconds:.1f}s")
        table.add_row("Speech Interactions", str(self.session_log.metrics.speech_interactions))
        table.add_row("Text Interactions", str(self.session_log.metrics.text_interactions))
        table.add_row("Tool Calls", f"{self.session_log.metrics.tool_calls_successful}/{self.session_log.metrics.tool_calls_total}")
        table.add_row("Interruptions", str(self.session_log.metrics.interruptions))
        table.add_row("Errors", str(self.session_log.metrics.errors))
        table.add_row("Avg Response Time", f"{self.session_log.metrics.average_response_time_ms:.1f}ms")
        
        # Show files
        if self.session_log.log_file_path:
            table.add_row("Log File", str(self.session_log.log_file_path))
            
        console = Console() if RICH_AVAILABLE else None
        if console:
            console.print(Panel(table, title="📊 Session Complete", border_style="green"))
            
    def _print_simple_summary(self):
        """Print simple text summary when Rich is not available."""
        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print("="*50)
        print(f"Session ID: {self.session_log.session_id}")
        print(f"Duration: {self.session_log.metrics.total_duration_seconds:.1f}s")
        print(f"Speech Interactions: {self.session_log.metrics.speech_interactions}")
        print(f"Text Interactions: {self.session_log.metrics.text_interactions}")
        print(f"Tool Calls: {self.session_log.metrics.tool_calls_successful}/{self.session_log.metrics.tool_calls_total}")
        print(f"Interruptions: {self.session_log.metrics.interruptions}")
        print(f"Errors: {self.session_log.metrics.errors}")
        print(f"Average Response Time: {self.session_log.metrics.average_response_time_ms:.1f}ms")
        if self.session_log.log_file_path:
            print(f"Log File: {self.session_log.log_file_path}")
        print("="*50)

class RealtimeAgent:
    """Realtime voice agent following OpenAI Agents SDK patterns."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[ToolDefinition]] = None,
        voice: str = "alloy",
        temperature: float = 0.7,
        model: str = "gpt-4o-mini-realtime-preview-2024-12-17"
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.voice = voice
        self.temperature = temperature
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
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.3,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 800,
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
    """Session manager following OpenAI Agents SDK patterns."""
    
    def __init__(
        self,
        agent: RealtimeAgent,
        api_key: str,
        output_guardrails: Optional[List[Callable]] = None
    ):
        self.agent = agent
        self.api_key = api_key
        self.output_guardrails = output_guardrails or []
        
        # Console status manager
        self.status = VoiceAgentStatus(agent)
        
        # Session logging
        self.session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.logger = None
        if CONFIG.enable_logging:
            self.logger = SessionLogger(self.session_id, agent)
        
        # WebSocket connection
        self.ws = None
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Session state
        self.is_connected = False
        self.is_speaking = False
        self.conversation_history: List[Dict[str, Any]] = []
        self.audio_output_buffer = bytearray()
        self.audio_queue: Optional[asyncio.Queue] = None
        
        # Text interaction state
        self.text_response_buffer = ""
        self.is_receiving_text = False
        self.is_typing = False  # Flag to pause live display during typing
        
        # Audio transcript buffer for showing what the assistant said
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
            max_workers=2, thread_name_prefix="robot_tool"
        )
        
        # Event handlers
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        
        # Initialize event handlers
        self._setup_default_handlers()
        
        # Setup logging event handlers
        if self.logger:
            self._setup_logging_handlers()
        
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
        
    def _setup_logging_handlers(self):
        """Setup logging event handlers."""
        if not self.logger:
            return
            
        # Log all events for comprehensive session tracking
        self.on(EventType.SPEECH_STARTED, self._log_speech_started)
        self.on(EventType.TRANSCRIPTION_COMPLETED, self._log_user_speech)
        self.on(EventType.RESPONSE_COMPLETED, self._log_assistant_response)
        self.on(EventType.TOOL_CALL_REQUESTED, self._log_tool_call_start)
        self.on(EventType.AUDIO_INTERRUPTED, self._log_interruption)
        self.on(EventType.ERROR, self._log_error_event)
        
    async def connect(self):
        """Connect to OpenAI Realtime API."""
        url = f"wss://api.openai.com/v1/realtime?model={self.agent.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        self.status.update_status("Connecting to OpenAI Realtime API...")
        
        # Show connection progress
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("🔗 Connecting to OpenAI Realtime API...", total=None)
                self.ws = await websockets.connect(
                    url, 
                    additional_headers=headers, 
                    ssl=self.ssl_context,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10    # Wait 10 seconds for pong response
                )
                progress.update(task, description="✅ Connected to OpenAI Realtime API")
        else:
            self.status.print_message("🔗 Connecting to OpenAI Realtime API...")
            self.ws = await websockets.connect(
                url, 
                additional_headers=headers, 
                ssl=self.ssl_context,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10    # Wait 10 seconds for pong response
            )
        
        self.is_connected = True
        self.status.update_status("Connected and ready!")
        self.status.print_success("Connected to OpenAI Realtime API")
        
        # Configure session
        await self._configure_session()
        
    async def _configure_session(self):
        """Configure the session with agent settings."""
        self.status.update_status("Configuring session...")
        
        agent_config = self.agent.get_session_config()
        session_config = {
            "type": "session.update",
            "session": agent_config
        }
        
        # Debug output if enabled
        if CONFIG.debug_mode:
            print(f"🚨 AGENT CONFIG - Number of tools: {len(agent_config.get('tools', []))}")
            print(f"🚨 AGENT CONFIG - Tools: {[t.get('name') for t in agent_config.get('tools', [])]}")
            print(f"🚨 FULL SESSION CONFIG: {json.dumps(session_config, indent=2)}")
        
        await self._send_event(session_config)
        self.status.update_status("Session configured - ready to chat!")
        self.status.print_success("Session configured with agent settings")
        
    async def _send_event(self, event: Dict[str, Any]):
        """Send an event to the Realtime API."""
        if self.ws and self.is_connected:
            await self.ws.send(json.dumps(event))
            
    async def send_message(self, text: str):
        """Send a text message to the agent."""
        message_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        }
        await self._send_event(message_event)
        await self._send_event({"type": "response.create"})
        
    async def interrupt(self):
        """Manually interrupt the current response."""
        # Immediately stop local audio first
        await self.interrupt_audio()
        
        # Cancel the response on the server side
        await self._send_event({"type": "response.cancel"})
        
        # Clear any pending audio in the output buffer
        await self._send_event({"type": "input_audio_buffer.clear"})
        
        # Emit event for logging
        self._emit_event(EventType.AUDIO_INTERRUPTED)
        
        # Reset all states immediately
        self.is_speaking = False
        self.status.set_speaking(False)
        self.status.set_processing(False)
        self.status.update_status("Interrupted - ready to listen!")
        
    async def start_session(self):
        """Start the session and begin processing events."""
        if not self.is_connected:
            await self.connect()
            
        # Start session logging
        if self.logger:
            self.logger.start_audio_recording()
            # Update robot connection status for logging
            if hasattr(self.status, 'robot_connected'):
                self.logger.update_robot_connection_status(self.status.robot_connected)
            
        # Start event handling task
        event_task = asyncio.create_task(self._handle_server_events())
        
        try:
            # Start enhanced live status display (includes welcome info)
            self.status.start_live_display()
            
            # Different instructions based on mode
            if CONFIG.text_only_mode:
                self.status.update_status("Text mode ready - type your messages!")
                self.status.print_message("💬 Text-only mode enabled. Type messages and press Enter.", "cyan")
            elif CONFIG.enable_text_input:
                self.status.update_status("Hybrid mode ready - speak OR type!")
                self.status.print_message("🎤💬 Hybrid mode: You can speak OR type messages (press Enter to send).", "cyan")
            else:
                self.status.update_status("Voice input ready - start speaking!")
            
            # Wait a moment for user to read the info, then collapse to compact mode
            await asyncio.sleep(3)
            self.status.collapse_to_compact()
            
            # Start input tasks based on configuration
            input_tasks = []
            
            # Add text input if enabled
            if CONFIG.enable_text_input or CONFIG.text_only_mode:
                text_task = asyncio.create_task(self._text_input_stream())
                input_tasks.append(text_task)
            
            # Add audio input if not text-only mode
            if not CONFIG.text_only_mode and AUDIO_AVAILABLE:
                audio_task = asyncio.create_task(self._audio_input_stream())
                input_tasks.append(audio_task)
            elif CONFIG.text_only_mode:
                self.status.print_message("🔇 Voice input disabled (text-only mode)", "yellow")
            
            # Wait for any input task to complete (or be interrupted)
            if input_tasks:
                await asyncio.gather(*input_tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            self.status.print_message("\n👋 Shutting down gracefully...", "yellow")
        except Exception as e:
            self.status.print_error(f"Session error: {e}")
            if self.logger:
                self.logger.log_event(EventType.ERROR, {
                    "error_type": "session_fatal_error",
                    "error_message": str(e)
                })
        finally:
            event_task.cancel()
            # Stop any ongoing audio
            if self.is_speaking:
                await self.interrupt_audio()
            # Stop live display
            self.status.stop_live_display()
            # Shutdown thread pool
            self.tool_executor.shutdown(wait=False)
            if self.ws:
                await self.ws.close()
                
            # Finalize and save session log
            if self.logger:
                try:
                    log_file = self.logger.finalize_and_save()
                    self.status.print_success(f"Session log saved: {log_file}")
                except Exception as e:
                    self.status.print_error(f"Failed to save session log: {e}")
                
    async def _handle_server_events(self):
        """Handle incoming events from the Realtime API."""
        if not self.ws:
            return
            
        try:
            async for message in self.ws:
                try:
                    event = json.loads(message)
                    await self._process_server_event(event)
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse server message: {e}")
                    print(f"Raw message: {message}")
                except Exception as e:
                    print(f"❌ Error processing server event: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue processing other events
                    
        except websockets.ConnectionClosed:
            print("🔌 WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            print(f"❌ Error handling server events: {e}")
            import traceback
            traceback.print_exc()
            
    async def _process_server_event(self, event: Dict[str, Any]):
        """Process individual server events."""
        event_type = event.get("type")
        
        # Helper function to create clean debug output
        def get_clean_event_data(event_data):
            """Create a clean version of event data without massive fields."""
            clean_event = event_data.copy()
            
            # Truncate large data fields
            if "delta" in clean_event and len(str(clean_event["delta"])) > 100:
                clean_event["delta"] = f"<audio_data_{len(str(clean_event['delta']))}bytes>"
            
            if "audio" in clean_event and len(str(clean_event["audio"])) > 100:
                clean_event["audio"] = f"<audio_data_{len(str(clean_event['audio']))}bytes>"
                
            # Truncate response objects with large usage data
            if "response" in clean_event and isinstance(clean_event["response"], dict):
                if "usage" in clean_event["response"] and clean_event["response"]["usage"] is not None:
                    # Keep usage but make it more readable
                    usage = clean_event["response"]["usage"]
                    clean_event["response"]["usage"] = f"tokens: {usage.get('total_tokens', 'unknown')}"
                    
            return clean_event
        
        # Debug output for important events if enabled
        if CONFIG.debug_mode:
            if "function" in event_type or "tool" in event_type:
                # Always show function/tool events fully since they're important
                print(f"🚨 IMPORTANT EVENT: {event_type} - {event}")
            elif "response" in event_type:
                # Show response events but with cleaned data
                if event_type == "response.audio.delta":
                    # Just show that audio delta occurred
                    print(f"🚨 AUDIO DELTA: {event_type} (audio data: {len(str(event.get('delta', '')))} bytes)")
                elif event_type == "response.done":
                    # Show response done but clean up the massive usage data
                    clean_event = get_clean_event_data(event)
                    print(f"🚨 IMPORTANT EVENT: {event_type} - {clean_event}")
                else:
                    # Other response events with cleaned data
                    clean_event = get_clean_event_data(event)
                    print(f"🚨 IMPORTANT EVENT: {event_type} - {clean_event}")
        
        # Optional debug logging
        if CONFIG.debug_mode:
            # Show more detail for function-related events
            if "function" in event_type or "tool" in event_type:
                print(f"📨 Server event: {event_type} - {event}")
            elif event_type == "response.audio.delta":
                # Just show that audio delta occurred
                print(f"📨 Server event: {event_type} (audio data: {len(str(event.get('delta', '')))} bytes)")
            else:
                print(f"📨 Server event: {event_type}")
        
        try:
            if event_type == "session.created":
                self._emit_event(EventType.SESSION_CREATED, event)
                
            elif event_type == "session.updated": 
                self._emit_event(EventType.SESSION_UPDATED, event)
                
            elif event_type == "conversation.created":
                self._emit_event(EventType.CONVERSATION_STARTED, event)
                
            elif event_type == "input_audio_buffer.speech_started":
                self._emit_event(EventType.SPEECH_STARTED, event)
                
            elif event_type == "input_audio_buffer.speech_stopped":
                self._emit_event(EventType.SPEECH_ENDED, event)
                
            elif event_type == "conversation.item.input_audio_transcription.completed":
                self._emit_event(EventType.TRANSCRIPTION_COMPLETED, event)
                
            elif event_type == "response.created":
                self._emit_event(EventType.RESPONSE_STARTED, event)
                
            elif event_type == "response.content_part.added":
                content_part = event.get("part", {})
                if content_part.get("type") == "audio":
                    self.audio_output_buffer = bytearray()
                    
            elif event_type == "response.audio.delta":
                audio_data = base64.b64decode(event["delta"])
                self.audio_output_buffer.extend(audio_data)
                
                # Record audio for session logging
                if self.logger:
                    self.logger.record_audio_chunk(audio_data, "assistant_output")
                
            elif event_type == "response.audio.done":
                if self.audio_output_buffer:
                    await self._play_audio_response(bytes(self.audio_output_buffer))
                    self.audio_output_buffer = bytearray()
                    
            elif event_type == "response.text.delta":
                text_delta = event.get("delta", "")
                self.text_response_buffer += text_delta
                self.is_receiving_text = True
                
            elif event_type == "response.text.done":
                if self.text_response_buffer:
                    self.status.print_message(f"🤖 Assistant: {self.text_response_buffer}", "blue bold")
                    
                    # Log text response
                    if self.logger:
                        self.logger.log_conversation_item("text_output", self.text_response_buffer)
                        
                    self.text_response_buffer = ""
                self.is_receiving_text = False
                    
            elif event_type == "response.audio_transcript.delta":
                # Handle incremental audio transcript updates
                transcript_delta = event.get("delta", "")
                self.audio_transcript_buffer += transcript_delta
                
            elif event_type == "response.audio_transcript.done":
                # Display the complete audio transcript
                if CONFIG.debug_mode:
                    print(f"🚨 AUDIO TRANSCRIPT DONE: '{self.audio_transcript_buffer}'")
                if self.audio_transcript_buffer:
                    self.status.print_assistant_message(self.audio_transcript_buffer)
                    self.audio_transcript_buffer = ""
                elif CONFIG.debug_mode:
                    print("🚨 NO AUDIO TRANSCRIPT BUFFER!")
                    
            elif event_type == "response.function_call_arguments.done":
                self._emit_event(EventType.TOOL_CALL_REQUESTED, event)
                
            elif event_type == "response.function_call_arguments.delta":
                # Handle function call arguments being built up
                if CONFIG.debug_mode:
                    print(f"🔧 Function call arguments delta: {event}")
                    
            elif event_type.startswith("response.output_item") and "function_call" in event_type:
                # Handle function calls from output items
                if CONFIG.debug_mode:
                    print(f"🔧 Function call from output item: {event_type} - {event}")
                self._emit_event(EventType.TOOL_CALL_REQUESTED, event)
                
            elif event_type == "response.done":
                self._emit_event(EventType.RESPONSE_COMPLETED, event)
                
            elif event_type == "error":
                error_details = event.get("error", {})
                error_code = error_details.get("code", "")
                
                # Don't show cancellation errors (these are expected during interruption)
                if error_code == "response_cancel_not_active":
                    if CONFIG.debug_mode:
                        self.status.print_message(f"Debug: {error_details.get('message', 'Response cancellation')}", "dim")
                else:
                    self.status.print_error(f"OpenAI API Error: {error_details.get('message', 'Unknown error')}")
                    
                self._emit_event(EventType.ERROR, event)
                
            else:
                # Optional logging for unknown event types
                if CONFIG.debug_mode:
                    print(f"🔍 Unknown event type: {event_type}")
                
        except Exception as e:
            print(f"❌ Error processing event {event_type}: {e}")
            import traceback
            traceback.print_exc()
            # Don't re-raise - continue processing other events
        
    # Event handlers
    async def _handle_session_created(self, event: RealtimeEvent):
        """Handle session creation."""
        self.status.update_status("Session created successfully!")
        
    async def _handle_speech_started(self, event: RealtimeEvent):
        """Handle speech detection."""
        self.status.set_listening(True)
        self.status.update_status("Listening to your speech...")
        
        # IMMEDIATE interruption when user speaks - no delays
        if self.is_speaking or self.current_audio_stream:
            # Stop local audio immediately
            await self.interrupt_audio()
            
            # Cancel server response immediately  
            try:
                await self._send_event({"type": "response.cancel"})
                await self._send_event({"type": "input_audio_buffer.clear"})
            except Exception as e:
                if CONFIG.debug_mode:
                    self.status.print_message(f"Response cancellation: {e}", "dim")
            
            # Clear our state immediately
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
        # This event is mainly for logging, status updates happen elsewhere
        pass
        
    async def _handle_audio_interrupted(self, event: RealtimeEvent):
        """Handle audio interruption."""
        if CONFIG.debug_mode:
            self.status.print_message("🛑 Audio interruption logged", "dim")
            
    async def _handle_error(self, event: RealtimeEvent):
        """Handle error events."""
        if CONFIG.debug_mode:
            error_data = event.data.get("error", {})
            self.status.print_message(f"⚠️ Error event: {error_data.get('message', 'Unknown error')}", "dim")
    
    # Logging event handlers
    async def _log_speech_started(self, event: RealtimeEvent):
        """Log speech start event."""
        if self.logger:
            self.logger.log_conversation_item("speech_started", "", {"event_data": event.data})
            
    async def _log_user_speech(self, event: RealtimeEvent):
        """Log user speech transcription."""
        if self.logger:
            transcript = event.data.get("transcript", "")
            confidence = event.data.get("confidence", 0.0)
            metadata = {
                "confidence": confidence,
                "event_data": event.data
            }
            self.logger.log_conversation_item("user_speech", transcript, metadata)
            
    async def _log_assistant_response(self, event: RealtimeEvent):
        """Log assistant response."""
        if self.logger and self.audio_transcript_buffer:
            self.logger.log_conversation_item("assistant_speech", self.audio_transcript_buffer, {
                "event_data": event.data
            })
            
    async def _log_tool_call_start(self, event: RealtimeEvent):
        """Log tool call start for timing."""
        if self.logger:
            function_name = event.data.get("name")
            call_id = event.data.get("call_id")
            arguments_str = event.data.get("arguments", "{}")
            
            if function_name and call_id:
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                    self.logger.log_tool_call_start(function_name, arguments, call_id)
                except json.JSONDecodeError:
                    pass
                    
    async def _log_interruption(self, event: RealtimeEvent):
        """Log audio interruption."""
        if self.logger:
            self.logger.log_event(EventType.AUDIO_INTERRUPTED, event.data)
            
    async def _log_error_event(self, event: RealtimeEvent):
        """Log error events."""
        if self.logger:
            error_data = event.data.get("error", {}) if isinstance(event.data, dict) else {}
            self.logger.log_event(EventType.ERROR, {
                "error_type": "api_error",
                "error_message": error_data.get("message", str(event.data)),
                "error_code": error_data.get("code", "")
            })
        
    async def _handle_tool_call(self, event: RealtimeEvent):
        """Handle tool call execution."""
        # Debug output if enabled
        if CONFIG.debug_mode:
            print(f"🚨 TOOL CALL HANDLER TRIGGERED! Event data: {event.data}")
        
        if CONFIG.debug_mode:
            print(f"🔧 Tool call event received: {event.data}")
        
        try:
            # The event structure from OpenAI Realtime API for function calls
            # comes directly in the event data, not nested under response.output
            event_data = event.data
            
            # Extract function call information
            function_name = event_data.get("name")
            arguments_str = event_data.get("arguments", "{}")
            call_id = event_data.get("call_id")
            
            if CONFIG.debug_mode:
                print(f"🔧 Function call detected: {function_name}, args: {arguments_str}, call_id: {call_id}")
            
            if not function_name or not call_id:
                self.status.print_error("Invalid function call - missing name or call_id")
                return
            
            tool = self.agent._tool_map.get(function_name)
            if not tool:
                self.status.print_error(f"Tool '{function_name}' not found in tool map")
                error_msg = f"Tool '{function_name}' not available"
                await self._send_tool_error(call_id, error_msg)
                return
            
            try:
                # Parse arguments
                args = json.loads(arguments_str) if arguments_str else {}
                if CONFIG.debug_mode:
                    print(f"🤖 Executing: {function_name}({args})")
                
                # Check if tool needs approval
                if tool.needs_approval:
                    approval_data = {
                        "tool_name": function_name,
                        "arguments": args,
                        "call_id": call_id
                    }
                    self._emit_event(EventType.TOOL_APPROVAL_REQUESTED, approval_data)
                    return
                
                # Execute the tool
                result = await self._execute_tool(tool, args)
                if CONFIG.debug_mode:
                    print(f"✅ Tool execution result: {result}")
                
                # Log successful tool call completion
                if self.logger:
                    self.logger.log_tool_call_completion(
                        function_name, args, result, call_id, success=True
                    )
                
                # Send function result back
                function_result = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result)
                    }
                }
                await self._send_event(function_result)
                
                # Trigger a new response 
                await self._send_event({"type": "response.create"})
                
            except json.JSONDecodeError as e:
                self.status.print_error(f"Failed to parse function arguments: {e}")
                await self._send_tool_error(call_id, f"Invalid arguments format: {e}")
                
            except Exception as e:
                self.status.print_error(f"Tool execution error: {e}")
                if CONFIG.debug_mode:
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
                    
                # Log failed tool call
                if self.logger:
                    self.logger.log_tool_call_completion(
                        function_name, args, None, call_id, success=False, error_message=str(e)
                    )
                    
                await self._send_tool_error(call_id, str(e))
                
        except Exception as e:
            self.status.print_error(f"Fatal error in tool call handler: {e}")
            if CONFIG.debug_mode:
                import traceback
                traceback.print_exc()
            # Don't re-raise - just log and continue to prevent session crash
        
    async def _execute_tool(self, tool: ToolDefinition, args: Dict[str, Any]) -> Any:
        """Execute a tool function (threaded for non-blocking robot operations)."""
        self.status.set_tool_executing(tool.name)
        
        try:
            if CONFIG.threaded_execution:
                # Execute in thread pool to prevent blocking conversation
                self.status.print_tool_message(f"Executing {tool.name} in background thread...")
                if asyncio.iscoroutinefunction(tool.execute):
                    # For async functions, run in current loop
                    result = await tool.execute(**args)
                else:
                    # For sync functions, run in thread pool
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
            if CONFIG.debug_mode:
                print(f"⚠️ Sending tool error for call_id {call_id}: {error_message}")
                
            error_result = {
                "type": "conversation.item.create", 
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"error": error_message})
                }
            }
            await self._send_event(error_result)
            
            # Also trigger a response to continue the conversation
            await self._send_event({"type": "response.create"})
            
            if CONFIG.debug_mode:
                print(f"✅ Tool error sent successfully for call_id {call_id}")
            
        except Exception as e:
            self.status.print_error(f"Failed to send tool error: {e}")
            if CONFIG.debug_mode:
                import traceback
                traceback.print_exc()
        
    async def _play_audio_response(self, audio_data: bytes):
        """Play audio response from the assistant (non-blocking with interruption support)."""
        if not AUDIO_AVAILABLE:
            return
            
        # Check if we should even start playing (might be interrupted already)
        if self.audio_interrupt_event.is_set():
            return
            
        try:
            self.is_speaking = True
            self.audio_interrupt_event.clear()
            
            # Convert bytes to numpy array for sounddevice
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            self.status.set_speaking(True)
            self.status.update_status("Playing audio response...")
            
            # Check again before starting playback
            if self.audio_interrupt_event.is_set():
                return
            
            # Non-blocking audio playback with immediate interruption check
            self.current_audio_stream = sd.play(audio_float, AUDIO_CONFIG["samplerate"])
            
            # Wait for completion or interruption with faster checking
            await self._wait_for_audio_completion()
            
        except Exception as e:
            if CONFIG.debug_mode:
                print(f"❌ Audio playback error: {e}")
        finally:
            # Always cleanup, even if interrupted
            self.is_speaking = False
            self.status.set_speaking(False)
            if self.current_audio_stream:
                try:
                    self.current_audio_stream.stop()
                    self.current_audio_stream.close()
                except:
                    pass
            self.current_audio_stream = None
            
            # Mark the time we finished speaking for feedback prevention
            self._last_audio_output_time = time.time()
            
            # Enhanced post-speech silence period if feedback prevention is enabled
            if not self.audio_interrupt_event.is_set() and CONFIG.feedback_prevention_enabled:
                silence_duration = CONFIG.silence_after_speaking
                self._recently_speaking = True
                self._microphone_muted = True
                
                self.status.update_status(f"Silence period ({silence_duration}s) to prevent audio feedback...")
                
                # Start silence timer
                self._audio_silence_start_time = time.time()
                await asyncio.sleep(silence_duration)
                
                # End silence period
                self._microphone_muted = False
                self._audio_silence_start_time = None
                if hasattr(self, '_recently_speaking'):
                    delattr(self, '_recently_speaking')
                    
                if CONFIG.debug_mode:
                    self.status.print_message("🔇 Microphone feedback prevention period ended", "dim")
            elif not self.audio_interrupt_event.is_set() and CONFIG.post_speech_delay > 0:
                # Fallback to original shorter delay if enhanced prevention is disabled
                self._recently_speaking = True
                self.status.update_status(f"Waiting {CONFIG.post_speech_delay}s to prevent audio feedback...")
                await asyncio.sleep(CONFIG.post_speech_delay)
                if hasattr(self, '_recently_speaking'):
                    delattr(self, '_recently_speaking')
            
            if not self.audio_interrupt_event.is_set():
                self.status.update_status("Ready for next interaction!")
                    
    async def _wait_for_audio_completion(self):
        """Wait for audio to complete or be interrupted."""
        if not self.current_audio_stream:
            return
            
        # Much faster interrupt checking - every 5ms instead of 10ms
        while self.current_audio_stream and self.current_audio_stream.active:
            if self.audio_interrupt_event.is_set():
                try:
                    self.current_audio_stream.stop()
                    self.current_audio_stream.close()
                except:
                    pass
                if CONFIG.debug_mode:
                    print("🛑 Audio interrupted for safety")
                break
            await asyncio.sleep(0.005)  # Check every 5ms for faster response
        
    async def interrupt_audio(self):
        """Interrupt current audio playback for safety."""
        if CONFIG.enable_interruption:
            # Set interrupt flag first
            self.audio_interrupt_event.set()
            
            # Force stop any active audio stream immediately
            if self.current_audio_stream:
                try:
                    self.current_audio_stream.stop()
                    self.current_audio_stream.close()  # Force close the stream
                except Exception as e:
                    if CONFIG.debug_mode:
                        print(f"Audio stop error: {e}")
                        
            # Clear ALL audio buffers aggressively
            self.audio_output_buffer = bytearray()
            
            # Update status immediately
            self.is_speaking = False
            self.status.set_speaking(False)
            self.status.update_status("Interrupted - ready to listen!")
            
            # Clear the current stream reference
            self.current_audio_stream = None
            
    async def _text_input_stream(self):
        """Handle keyboard text input alongside voice input."""
        import sys
        
        loop = asyncio.get_event_loop()
        
        try:
            self.status.print_message("💬 Text input ready! Type your message and press Enter...", "green")
            self.status.print_message("💡 Commands: 'quit' to exit text mode, Ctrl+C to stop completely", "cyan")
            
            while self.is_connected:
                live_was_active = self.status.live is not None and self.status.live.is_started
                if live_was_active and self.status.live:
                    self.status.live.stop()

                text_input = ""
                try:
                    # Get input from the user
                    if RICH_AVAILABLE and self.status.console:
                        prompt = Text("\n💬 Type message: ", style="cyan", end="")
                        text_input = await loop.run_in_executor(
                            None, 
                            lambda: self.status.console.input(prompt) if self.status.console else ""
                        )
                    else:
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
                finally:
                    if live_was_active and self.status.live:
                        self.status.live.start(refresh=True)

                if not text_input:
                    continue
                        
                if text_input.lower() in ['quit', 'exit', 'q']:
                    self.status.print_message("👋 Text input stopped", "yellow")
                    break
                        
                # Show user's message using status print.
                # This will now appear correctly above the live status line.
                self.status.print_user_message(text_input)
                
                # Log text input
                if self.logger:
                    self.logger.log_conversation_item("text_input", text_input)
                
                # Send text message to the assistant
                await self.send_message(text_input)
                
                # Brief pause to see the response
                await asyncio.sleep(0.5)
                    
        except Exception as e:
            self.status.print_error(f"Text input stream error: {e}")
        finally:
            # Clean up on exit
            if self.status.live and not self.status.live.is_started:
                self.status.live.start(refresh=True)
            self.status.update_status("Ready for next interaction!")

    async def _audio_input_stream(self):
        """Stream audio input to the Realtime API."""
        if not AUDIO_AVAILABLE:
            self.status.print_error("Audio not available for voice input")
            return
            
        self.status.update_status("Voice input active - start speaking!")
        
        try:
            # Setup audio queue
            audio_queue = asyncio.Queue()
            self.audio_queue = audio_queue
            
            def audio_callback(indata, frames, time_info, status):
                try:
                    if status:
                        print(f"Audio input status: {status}")
                    
                    # Simplified audio processing with minimal filtering for troubleshooting
                    should_record = True
                    
                    # Calculate audio level first
                    audio_level = np.max(np.abs(indata))
                    
                    # Only block during assistant speech (essential for preventing feedback)
                    if self.is_speaking:
                        should_record = False
                        if CONFIG.debug_mode:
                            print("🔇 Microphone blocked: Assistant is speaking")
                    # Apply enhanced filtering only if explicitly enabled
                    elif CONFIG.feedback_prevention_enabled:
                        if self._microphone_muted:
                            should_record = False
                            if CONFIG.debug_mode:
                                print("🔇 Microphone blocked: Silence period active")
                        elif hasattr(self, '_recently_speaking'):
                            should_record = False
                            if CONFIG.debug_mode:
                                print("🔇 Microphone blocked: Recently speaking")
                    
                    # Add silence filtering - only send if audio level is above threshold
                    if should_record and audio_level < CONFIG.audio_threshold:
                        should_record = False
                        if CONFIG.debug_mode and audio_level > 0.001:  # Only show if there's some audio
                            print(f"🔇 Audio too quiet: level={audio_level:.4f} < threshold={CONFIG.audio_threshold}")
                    
                    if should_record:
                        # Convert to int16 PCM format
                        audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                        audio_queue.put_nowait(audio_int16.tobytes())
                        
                        # Show audio activity for debugging (only if debug mode is on)
                        if CONFIG.debug_mode:
                            print(f"🎤 Recording audio: level={audio_level:.4f}")
                    elif CONFIG.debug_mode and audio_level > CONFIG.audio_threshold:
                        print(f"🔇 Audio blocked but above threshold: level={audio_level:.4f}")
                        
                except Exception as e:
                    print(f"❌ Audio callback error: {e}")
                    
            # Start audio stream  
            stream = sd.InputStream(
                callback=audio_callback,
                samplerate=AUDIO_CONFIG["samplerate"],
                channels=AUDIO_CONFIG["channels"],
                blocksize=AUDIO_CONFIG["blocksize"],
                dtype=np.float32
            )
            
            with stream:
                chunks_sent = 0
                while self.is_connected:
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        
                        # Record audio for session logging
                        if self.logger:
                            self.logger.record_audio_chunk(chunk, "user_input")
                        
                        # Send audio chunk to Realtime API
                        base64_chunk = base64.b64encode(chunk).decode('utf-8')
                        await self._send_event({
                            "type": "input_audio_buffer.append",
                            "audio": base64_chunk
                        })
                        
                        chunks_sent += 1
                        
                        # Debug: Show audio being sent to OpenAI
                        if CONFIG.debug_mode:
                            print(f"📡 Sent {len(chunk)} bytes to OpenAI (total: {chunks_sent})")
                        
                    except asyncio.TimeoutError:
                        continue
                    except KeyboardInterrupt:
                        print("\n👋 Voice input stopped")
                        break
                    except Exception as e:
                        # Log error but don't add delay - continue processing
                        if hasattr(e, '__traceback__'):
                            print(f"❌ Audio processing error: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
            import traceback
            traceback.print_exc()

def create_robot_tools() -> List[ToolDefinition]:
    """Create robot tool definitions following OpenAI Agents patterns."""
    tools = []
    
    try:
        if CONFIG.debug_mode:
            print("🚨 ENTERING create_robot_tools()")
            print("🚨 Getting tools from registry...")
        
        # Get tools and function map from the new registry
        tool_specs, function_map = register_tools_for_openai()
        
        if CONFIG.debug_mode:
            print(f"🚨 Found {len(function_map)} functions in registry")
            print(f"🚨 Function map keys: {list(function_map.keys())}")
    except Exception as e:
        if CONFIG.debug_mode:
            print(f"🚨 ERROR accessing registry: {e}")
        return []
    
    # Move relative tool
    move_relative_tool = ToolDefinition(
        name="move_relative_xyz",
        description="Move robot relative to current position in X, Y, Z directions in meters",
        parameters={
            "type": "object",
            "properties": {
                "dx_m": {"type": "number", "description": "Distance to move in X direction (meters)"},
                "dy_m": {"type": "number", "description": "Distance to move in Y direction (meters)"},
                "dz_m": {"type": "number", "description": "Distance to move in Z direction (meters)"},
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["dx_m", "dy_m", "dz_m", "wake_phrase"]
        },
        execute=function_map["move_relative_xyz"]
    )
    tools.append(move_relative_tool)
    
    # Move absolute tool
    move_absolute_tool = ToolDefinition(
        name="move_to_absolute_position",
        description="Move robot to absolute position in Cartesian space (coordinates in meters)",
        parameters={
            "type": "object",
            "properties": {
                "x_m": {"type": "number", "description": "Target X position in meters"},
                "y_m": {"type": "number", "description": "Target Y position in meters"},
                "z_m": {"type": "number", "description": "Target Z position in meters"},
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["x_m", "y_m", "z_m", "wake_phrase"]
        },
        execute=function_map["move_to_absolute_position"]
    )
    tools.append(move_absolute_tool)
    
    # Home tool
    home_tool = ToolDefinition(
        name="move_home",
        description="Move robot to home position",
        parameters={
            "type": "object",
            "properties": {
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["wake_phrase"]
        },
        execute=function_map["move_home"]
    )
    tools.append(home_tool)
    
    # Robot state tool
    state_tool = ToolDefinition(
        name="get_robot_state",
        description="Get current robot position and status",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        execute=function_map["get_robot_state"]
    )
    tools.append(state_tool)
    
    # Gripper tool
    gripper_tool = ToolDefinition(
        name="control_gripper",
        description="Control robot gripper - open or close",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "close"], "description": "Gripper action"},
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["action", "wake_phrase"]
        },
        execute=function_map["control_gripper"]
    )
    tools.append(gripper_tool)
    
    # Supply station tool
    supply_station_tool = ToolDefinition(
        name="move_to_supply_station",
        description="Move robot to a supply station position",
        parameters={
            "type": "object",
            "properties": {
                "distance": {"type": "string", "description": "Supply station location key (e.g., '10cm', '20cm', '30cm', '40cm', '50cm', '60cm')"},
                "approach": {"type": "boolean", "description": "Whether to move to approach height before final position (default: true)"},
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["distance", "wake_phrase"]
        },
        execute=function_map["move_to_supply_station"]
    )
    tools.append(supply_station_tool)
    
    # Get supply element tool
    get_element_tool = ToolDefinition(
        name="get_supply_element",
        description="Retrieve element from supply station",
        parameters={
            "type": "object",
            "properties": {
                "element_length": {"type": "string", "description": "Length of element (e.g., '40cm')"},
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["element_length", "wake_phrase"]
        },
        execute=function_map["get_supply_element"]
    )
    tools.append(get_element_tool)
    
    # Place element tool
    place_element_tool = ToolDefinition(
        name="place_element_at_position",
        description="Place held element at specified coordinates",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate in meters"},
                "y": {"type": "number", "description": "Y coordinate in meters"},
                "z": {"type": "number", "description": "Z coordinate in meters"},
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["x", "y", "z", "wake_phrase"]
        },
        execute=function_map["place_element_at_position"]
    )
    tools.append(place_element_tool)
    
    # Release element tool
    release_tool = ToolDefinition(
        name="release_element",
        description="Release currently held element and return to safe position",
        parameters={
            "type": "object",
            "properties": {
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["wake_phrase"]
        },
        execute=function_map["release_element"]
    )
    tools.append(release_tool)
    
    # Stop robot tool
    stop_tool = ToolDefinition(
        name="stop_robot",
        description="Emergency stop - immediately halt all robot movements",
        parameters={
            "type": "object",
            "properties": {
                "wake_phrase": {"type": "string", "description": "Safety wake phrase containing 'timbra'"}
            },
            "required": ["wake_phrase"]
        },
        execute=function_map["stop_robot"]
    )
    tools.append(stop_tool)
    
    if CONFIG.debug_mode:
        print(f"🚨 create_robot_tools() COMPLETE - returning {len(tools)} tools")
    return tools

def create_robot_agent() -> RealtimeAgent:
    """Create a robot voice agent using OpenAI Agents patterns."""
    if CONFIG.debug_mode:
        print("🚨 STARTING create_robot_agent()")
    
    try:
        system_prompt = get_system_prompt("voice")
        if CONFIG.debug_mode:
            print("🚨 Got system prompt")
    except Exception as e:
        if CONFIG.debug_mode:
            print(f"🚨 ERROR getting system prompt: {e}")
        system_prompt = "You are a robot control assistant."
    
    enhanced_instructions = (
        f"{system_prompt}\n\n"
        "VOICE INTERACTION SPECIFIC:\n"
        "- Always respond with enthusiasm about robot operations!\n"
        "- You speak naturally and execute robot commands safely and efficiently\n"
        "- Keep responses very short, nothing verbose but engaging for voice interaction\n"
        "- Acknowledge tool execution progress during longer operations"
    )
    
    # Create tools and show debug info
    try:
        if CONFIG.debug_mode:
            print("🚨 About to call create_robot_tools()")
        robot_tools = create_robot_tools()
        if CONFIG.debug_mode:
            print(f"🚨 CREATED {len(robot_tools)} ROBOT TOOLS:")
            for tool in robot_tools:
                print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        if CONFIG.debug_mode:
            print(f"🚨 ERROR creating robot tools: {e}")
            import traceback
            traceback.print_exc()
        robot_tools = []
    
    # Create agent with robot tools
    try:
        agent = RealtimeAgent(
            name="UR-HAL-9000",
            instructions=enhanced_instructions,
            tools=robot_tools,
            voice="alloy",
            temperature=0.7
        )
        if CONFIG.debug_mode:
            print(f"🚨 AGENT CREATED WITH {len(agent.tools)} TOOLS")
    except Exception as e:
        if CONFIG.debug_mode:
            print(f"🚨 ERROR creating agent: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    return agent

def get_voice_config() -> Dict[str, Any]:
    """Get current voice interaction configuration."""
    return CONFIG.__dict__.copy()

def configure_voice_agent(
    text_only_mode: Optional[bool] = None,
    enable_text_input: Optional[bool] = None,
    enable_interruption: Optional[bool] = None,
    post_speech_delay: Optional[float] = None,
    silence_after_speaking: Optional[float] = None,
    audio_threshold: Optional[float] = None,
    feedback_prevention_enabled: Optional[bool] = None,
    threaded_tools: Optional[bool] = None,
    debug_mode: Optional[bool] = None,
    enable_logging: Optional[bool] = None
):
    """Configure voice agent settings at runtime.
    
    Args:
        text_only_mode: Disable voice and use text only
        enable_text_input: Allow text input alongside voice
        enable_interruption: Allow interrupting assistant speech
        post_speech_delay: Delay after speaking to prevent feedback (seconds)
        silence_after_speaking: Extended silence period after assistant speech (seconds)
        audio_threshold: Minimum audio level to consider as speech (0.0-1.0)
        feedback_prevention_enabled: Enable aggressive feedback prevention
        threaded_tools: Execute robot tools in background threads
        debug_mode: Enable detailed debug output
        enable_logging: Enable session logging
    """
    global CONFIG
    if text_only_mode is not None:
        CONFIG.text_only_mode = text_only_mode
    if enable_text_input is not None:
        CONFIG.enable_text_input = enable_text_input
    if enable_interruption is not None:
        CONFIG.enable_interruption = enable_interruption
    if post_speech_delay is not None:
        CONFIG.post_speech_delay = post_speech_delay
    if silence_after_speaking is not None:
        CONFIG.silence_after_speaking = silence_after_speaking
    if audio_threshold is not None:
        CONFIG.audio_threshold = audio_threshold
    if feedback_prevention_enabled is not None:
        CONFIG.feedback_prevention_enabled = feedback_prevention_enabled
    if threaded_tools is not None:
        CONFIG.threaded_execution = threaded_tools
    if debug_mode is not None:
        CONFIG.debug_mode = debug_mode
    if enable_logging is not None:
        CONFIG.enable_logging = enable_logging
    
    print(f"📝 Voice agent configuration updated: {CONFIG}")

async def main() -> None:
    """Main function to run the OpenAI Agents voice robot agent."""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Load configuration from environment
    global CONFIG
    CONFIG = load_config_from_env()
    
    # Show current voice config for debugging
    if CONFIG.debug_mode:
        print(f"🔧 Voice Config: {CONFIG}")
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key.startswith("sk-"):
        print("❌ OPENAI_API_KEY not configured")
        print("Please set your OpenAI API key in environment variables or .env file")
        return

    # Check audio availability using shared module
    from ur.agents.voice_common.audio import AUDIO_AVAILABLE
    if not AUDIO_AVAILABLE:
        print("❌ Audio libraries not available")
        print("Install with: uv add sounddevice numpy")
        return
        
    # Create agent and session
    agent = create_robot_agent()
    session = RealtimeSession(agent, api_key)
    
    # Add custom event handlers if needed
    session.on(EventType.TOOL_APPROVAL_REQUESTED, lambda event: print(f"🔒 Tool approval requested: {event.data}"))
    session.on(EventType.ERROR, lambda event: print(f"❌ Error: {event.data}"))
    
    # Start the session
    await session.start_session()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise 