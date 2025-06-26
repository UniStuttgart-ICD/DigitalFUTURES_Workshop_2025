"""Voice Session Manager

This module provides session management capabilities for voice agents,
including session creation, lifecycle management, and cleanup.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ur.agents.voice_common import (
    VoiceAgentConfig, load_config_from_env,
    VoiceAgentStatus
)
from ur.agents.base_agent import BaseVoiceAgent
from ur.config.system_config import SMOL_MODEL_ID, SMOL_PROVIDER


class SessionType(Enum):
    """Types of voice sessions."""
    OPENAI_REALTIME = "openai_realtime"
    SMOLAGENT_CODE = "smolagent_code"
    HYBRID = "hybrid"


class SessionState(Enum):
    """Session lifecycle states."""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SessionInfo:
    """Session information and metadata."""
    session_id: str
    session_type: SessionType
    created_at: datetime
    config: VoiceAgentConfig
    bridge_ref: Any = None
    agent_instance: Optional[BaseVoiceAgent] = None
    state: SessionState = SessionState.INITIALIZING
    error_message: Optional[str] = None
    
    @property
    def uptime_seconds(self) -> float:
        """Get session uptime in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class VoiceSessionManager:
    """Manages voice agent sessions with comprehensive lifecycle support."""
    
    def __init__(self, config: Optional[VoiceAgentConfig] = None):
        self.config = config or load_config_from_env()
        self.sessions: Dict[str, SessionInfo] = {}
        self.active_session: Optional[str] = None
        self.status = VoiceAgentStatus("SessionManager")
        
    def create_session(
        self, 
        session_type: Union[SessionType, str], 
        bridge_ref=None,
        **kwargs
    ) -> str:
        """Create a new voice session.
        
        Args:
            session_type: Type of session to create
            bridge_ref: Bridge reference for robot control
            **kwargs: Additional configuration options
            
        Returns:
            Session ID
        """
        if isinstance(session_type, str):
            session_type = SessionType(session_type)
            
        session_id = f"{session_type.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Create session config with overrides
        session_config = VoiceAgentConfig(**{
            **self.config.__dict__,
            **kwargs
        })
        
        session_info = SessionInfo(
            session_id=session_id,
            session_type=session_type,
            created_at=datetime.now(),
            config=session_config,
            bridge_ref=bridge_ref
        )
        
        self.sessions[session_id] = session_info
        
        self.status.print_success(f"Created {session_type.value} session: {session_id}")
        return session_id
    
    async def start_session(self, session_id: str) -> bool:
        """Start a voice session.
        
        Args:
            session_id: ID of the session to start
            
        Returns:
            True if session started successfully
        """
        if session_id not in self.sessions:
            self.status.print_error(f"Session not found: {session_id}")
            return False
            
        session = self.sessions[session_id]
        session.state = SessionState.CONNECTING
        
        try:
            self.status.update_status(f"Starting {session.session_type.value} session...")
            
            # Create agent instance based on session type
            session.agent_instance = await self._create_agent_instance(session)
            
            if not session.agent_instance:
                raise RuntimeError(f"Failed to create {session.session_type.value} agent")
            
            session.state = SessionState.CONNECTED
            
            # Set as active session
            self.active_session = session_id
            
            # Start the agent
            session.state = SessionState.ACTIVE
            await session.agent_instance.start()
            
            self.status.print_success(f"Session {session_id} started successfully")
            return True
            
        except Exception as e:
            session.state = SessionState.ERROR
            session.error_message = str(e)
            self.status.print_error(f"Failed to start session {session_id}: {e}")
            return False
    
    async def _create_agent_instance(self, session: SessionInfo) -> Optional[BaseVoiceAgent]:
        """Create agent instance based on session type."""
        if session.session_type == SessionType.OPENAI_REALTIME:
            from ur.agents.openai_agent import OpenAIVoiceAgent
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set for OpenAI session")
                
            return OpenAIVoiceAgent(
                bridge_ref=session.bridge_ref,
                ui_ref=self.status,
                api_key=api_key,
                config=session.config
            )
            
        elif session.session_type == SessionType.SMOLAGENT_CODE:
            # Try enhanced SmolAgent first
            try:
                from ur.agents.enhanced_smol_agent import EnhancedSmolVoiceAgent
                return EnhancedSmolVoiceAgent(
                    bridge_ref=session.bridge_ref,
                    ui_ref=self.status,
                    model_id=session.config.model_id,
                    provider=session.config.model_provider,
                    config=session.config
                )
            except ImportError:
                # Fallback to original SmolAgent
                from ur.agents.smol_agent import SmolVoiceAgent
                return SmolVoiceAgent(
                    bridge_ref=session.bridge_ref,
                    ui_ref=self.status,
                    model_id=session.config.model_id,
                    provider=session.config.model_provider
                )
                
        else:
            raise ValueError(f"Unsupported session type: {session.session_type}")
    
    async def stop_session(self, session_id: str) -> bool:
        """Stop a voice session.
        
        Args:
            session_id: ID of the session to stop
            
        Returns:
            True if session stopped successfully
        """
        if session_id not in self.sessions:
            self.status.print_error(f"Session not found: {session_id}")
            return False
            
        session = self.sessions[session_id]
        session.state = SessionState.STOPPING
        
        try:
            if session.agent_instance:
                await session.agent_instance.stop()
                
            session.state = SessionState.STOPPED
            
            # Clear active session if this was it
            if self.active_session == session_id:
                self.active_session = None
                
            self.status.print_success(f"Session {session_id} stopped successfully")
            return True
            
        except Exception as e:
            session.state = SessionState.ERROR
            session.error_message = str(e)
            self.status.print_error(f"Failed to stop session {session_id}: {e}")
            return False
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a voice session."""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        if session.state == SessionState.ACTIVE:
            session.state = SessionState.PAUSED
            # Implementation depends on agent capabilities
            return True
        return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused voice session."""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        if session.state == SessionState.PAUSED:
            session.state = SessionState.ACTIVE
            # Implementation depends on agent capabilities
            return True
        return False
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> Dict[str, SessionInfo]:
        """List all sessions."""
        return self.sessions.copy()
    
    def get_active_session(self) -> Optional[SessionInfo]:
        """Get the currently active session."""
        if self.active_session:
            return self.sessions.get(self.active_session)
        return None
    
    async def cleanup_sessions(self) -> None:
        """Cleanup all sessions."""
        self.status.print_message("Cleaning up all sessions...", "yellow")
        
        for session_id in list(self.sessions.keys()):
            await self.stop_session(session_id)
            
        self.sessions.clear()
        self.active_session = None
        
        self.status.print_success("All sessions cleaned up")
    
    def print_session_summary(self) -> None:
        """Print a summary of all sessions."""
        if not self.sessions:
            self.status.print_message("No sessions found", "dim")
            return
            
        self.status.print_message("\n" + "="*50, "cyan")
        self.status.print_message("📊 VOICE SESSION SUMMARY", "cyan bold")
        self.status.print_message("="*50, "cyan")
        
        for session_id, session in self.sessions.items():
            active_marker = "🟢" if session_id == self.active_session else "⚪"
            state_color = {
                SessionState.ACTIVE: "green",
                SessionState.CONNECTED: "blue", 
                SessionState.PAUSED: "yellow",
                SessionState.ERROR: "red",
                SessionState.STOPPED: "dim"
            }.get(session.state, "white")
            
            self.status.print_message(
                f"{active_marker} {session_id} ({session.session_type.value})", 
                "white"
            )
            self.status.print_message(
                f"    State: {session.state.value} | Uptime: {session.uptime_seconds:.1f}s", 
                state_color
            )
            
            if session.error_message:
                self.status.print_message(f"    Error: {session.error_message}", "red")
                
        self.status.print_message("="*50, "cyan")


def create_session_manager(config: Optional[VoiceAgentConfig] = None) -> VoiceSessionManager:
    """Create a voice session manager with optional configuration."""
    return VoiceSessionManager(config)


# Demo and testing
if __name__ == "__main__":
    async def demo_session_manager():
        """Demo the session manager functionality."""
        print("🚀 Voice Session Manager Demo")
        
        # Create session manager
        manager = create_session_manager()
        
        # Create sessions
        openai_session = manager.create_session(
            SessionType.OPENAI_REALTIME,
            text_only_mode=True,
            debug_mode=True
        )
        
        smol_session = manager.create_session(
            SessionType.SMOLAGENT_CODE,
            model_id=SMOL_MODEL_ID,
            model_provider=SMOL_PROVIDER
        )
        
        # Show summary
        manager.print_session_summary()
        
        print("\n✅ Demo completed!")
        
        # Cleanup
        await manager.cleanup_sessions()
    
    try:
        asyncio.run(demo_session_manager())
    except KeyboardInterrupt:
        print("\n Demo interrupted") 