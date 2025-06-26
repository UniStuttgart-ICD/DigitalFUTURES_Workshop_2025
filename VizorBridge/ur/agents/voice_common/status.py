"""Shared status display utilities for voice agents.

This module provides Rich-based animated status displays and console output
formatting used by both OpenAI and SmolAgent voice implementations.
"""

import asyncio
import time
from typing import Optional, List, Any
from enum import Enum

from ur.core.connection import get_robot

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠ Rich library not available. Install with: uv add rich")
    rprint = print


class AgentState(Enum):
    """Agent state enumeration for consistent status tracking."""
    INITIALIZING = "initializing"
    READY = "ready"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    GENERATING_CODE = "generating_code"
    EXECUTING_CODE = "executing_code"
    TOOL_EXECUTING = "tool_executing"
    ERROR = "error"


class VoiceAgentStatus:
    """Unified console status manager with animated displays for voice agent states.
    
    This class provides a consistent interface for status display across both
    OpenAI and SmolAgent implementations, including Rich animations and progress.
    """
    
    def __init__(self, agent_name: str = "VoiceAgent", agent=None):
        self.console = Console() if RICH_AVAILABLE else None
        self.agent_name = agent_name
        self.agent = agent
        self.current_status = "Initializing..."
        self.current_state = AgentState.INITIALIZING
        
        # State flags
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.is_generating_code = False
        self.is_executing_code = False
        self.tool_executing: Optional[str] = None
        self.show_initial_info = True
        
        # Robot connection status
        self.robot_connected = False
        self._last_robot_check = 0
        
        # Fabrication state tracking
        self.fabrication_active = False
        
        # Live display
        self.live: Optional[Live] = None
        self._animation_active = False
        self._animation_frame = 0
        self._status_displayed = False
        
        # Animation sequences
        self._listening_frames = ["🎤 ●", "🎤 ●●", "🎤 ●●●", "🎤 ●●", "🎤 ●"]
        self._speaking_frames = ["🔊 ♪", "🔊 ♫", "🔊 ♪♫", "🔊 ♫♪", "🔊 ♪"]
        self._processing_frames = ["🧠 ⚡", "🧠 💭", "🧠 ⚡⚡", "🧠 💭💭", "🧠 ⚡"]
        self._generating_frames = ["🧠 ⚡", "🧠 💭", "🧠 ⚡⚡", "🧠 💭💭", "🧠 ⚡"]
        self._executing_frames = ["🔧 ⚙️", "🔧 ⚙️⚙️", "🔧 ⚙️⚙️⚙️", "🔧 ⚙️⚙️", "🔧 ⚙️"]
        self._tool_frames = ["🔧 ⚙️", "🔧 ⚙️⚙️", "🔧 ⚙️⚙️⚙️", "🔧 ⚙️⚙️", "🔧 ⚙️"]
    
    def check_robot_connection(self):
        """Check robot connection status directly without using a tool."""
        current_time = time.time()
        
        # Only check every 5 seconds to avoid spam
        if current_time - self._last_robot_check < 5:
            return
            
        self._last_robot_check = current_time
        
        try:
            robot = get_robot()
            self.robot_connected = robot.is_connected
        except Exception:
            self.robot_connected = False
    
    async def start_animation(self):
        """Start the status animation using Rich Live display."""
        if not RICH_AVAILABLE or not self.console:
            return
            
        try:
            self.check_robot_connection()
            
            self.live = Live(
                self._generate_status_renderable(),
                console=self.console,
                refresh_per_second=4,
            )
            self.live.start()
            
            self._animation_active = True
            asyncio.create_task(self._animation_loop())
            
        except Exception as e:
            print(f"Animation start error: {e}")
    
    async def stop_animation(self):
        """Stop the status animation."""
        self._animation_active = False
        if self.live:
            self.live.stop()
    
    async def _animation_loop(self):
        """Main animation loop that updates the live status display."""
        try:
            while self._animation_active:
                self._animation_frame = (self._animation_frame + 1) % 5
                
                # Check robot connection periodically
                self.check_robot_connection()
                
                if self.live:
                    self.live.update(self._generate_status_renderable())
                
                await asyncio.sleep(0.25)  # Refresh rate for animation
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Animation loop error: {e}")
    
    def _generate_status_renderable(self) -> Text:
        """Generate the rich Text object for the status line."""
        if not RICH_AVAILABLE:
            return Text("")
            
        status_items = []
        
        # API connection status (styled based on fabrication state)
        if hasattr(self.agent, 'model'):
            model_name = self.agent.model if hasattr(self.agent, 'model') else "Unknown"
            if "openai" in model_name.lower() or "gpt" in model_name.lower():
                if self.fabrication_active:
                    status_items.append("● OpenAI")
                else:
                    status_items.append("[green]● OpenAI[/green]")
            else:
                if self.fabrication_active:
                    status_items.append("● SmolAgent")
                else:
                    status_items.append("[green]● SmolAgent[/green]")
        else:
            if self.fabrication_active:
                status_items.append("● Agent")
            else:
                status_items.append("[green]● Agent[/green]")
        
        # Robot connection status (styled based on fabrication state)
        if self.robot_connected:
            if self.fabrication_active:
                status_items.append("● Robot")
            else:
                status_items.append("[green]● Robot[/green]")
        else:
            if self.fabrication_active:
                status_items.append("● Robot")  # Shows as white when fabrication active
            else:
                status_items.append("[red]● Robot[/red]")
        
        # Animated state indicators (styled based on fabrication state)
        if self.is_listening:
            if self.fabrication_active:
                animated_indicator = f"{self._listening_frames[self._animation_frame]}"
            else:
                animated_indicator = f"[green]{self._listening_frames[self._animation_frame]}[/green]"
            status_items.append(animated_indicator)
        elif self.is_speaking:
            if self.fabrication_active:
                animated_indicator = f"{self._speaking_frames[self._animation_frame]}"
            else:
                animated_indicator = f"[blue]{self._speaking_frames[self._animation_frame]}[/blue]"
            status_items.append(animated_indicator)
        elif self.is_generating_code:
            if self.fabrication_active:
                animated_indicator = f"{self._generating_frames[self._animation_frame]} Generating code"
            else:
                animated_indicator = f"[yellow]{self._generating_frames[self._animation_frame]} Generating code[/yellow]"
            status_items.append(animated_indicator)
        elif self.is_executing_code:
            if self.fabrication_active:
                animated_indicator = f"{self._executing_frames[self._animation_frame]} Executing code"
            else:
                animated_indicator = f"[magenta]{self._executing_frames[self._animation_frame]} Executing code[/magenta]"
            status_items.append(animated_indicator)
        elif self.is_processing:
            if self.fabrication_active:
                animated_indicator = f"{self._processing_frames[self._animation_frame]}"
            else:
                animated_indicator = f"[yellow]{self._processing_frames[self._animation_frame]}[/yellow]"
            status_items.append(animated_indicator)
        elif self.tool_executing:
            if self.fabrication_active:
                animated_indicator = f"{self._tool_frames[self._animation_frame]} {self.tool_executing}"
            else:
                animated_indicator = f"[magenta]{self._tool_frames[self._animation_frame]} {self.tool_executing}[/magenta]"
            status_items.append(animated_indicator)
        else:
            if self.fabrication_active:
                status_items.append("🎤 Ready")
            else:
                status_items.append("[dim]🎤 Ready[/dim]")
        
        # Current status (styled based on fabrication state)
        if self.fabrication_active:
            status_items.append(f"{self.current_status}")
        else:
            status_items.append(f"[cyan]{self.current_status}[/cyan]")
        
        # Show model and tool info (style based on fabrication state)
        if self.agent:
            if hasattr(self.agent, 'tools') and isinstance(self.agent.tools, list):
                if self.fabrication_active:
                    status_items.append(f"{len(self.agent.tools)} tools")
                else:
                    status_items.append(f"[dim]{len(self.agent.tools)} tools[/dim]")
            
            if hasattr(self.agent, 'model'):
                model_display = self.agent.model.split("/")[-1] if "/" in self.agent.model else self.agent.model
                if "gpt-4o-mini" in model_display.lower():
                    model_display = "GPT-4o-mini"
                if self.fabrication_active:
                    status_items.append(f"{model_display}")
                else:
                    status_items.append(f"[dim]{model_display}[/dim]")
        
        status_content = " │ ".join(status_items)
        
        # Change entire status line color based on fabrication state
        if self.fabrication_active:
            # White/bright text when fabrication is active
            return Text.from_markup(f"[bright_white]🤖 {self.agent_name}: {status_content}[/bright_white]")
        else:
            # Dim/gray text when fabrication is inactive
            return Text.from_markup(f"[dim]🤖 {self.agent_name}: {status_content}[/dim]")
    
    def show_welcome_panel(self, config=None):
        """Show the initial welcome panel with agent and configuration info."""
        if not RICH_AVAILABLE or not self.console:
            return
            
        self.check_robot_connection()
        
        content_lines = []
        
        # Agent info header
        content_lines.append(f"[bold green]🤖 {self.agent_name} Voice Agent[/bold green]")
        content_lines.append("")
        
        # Connection status
        content_lines.append("🔗 [bold]Connection Status:[/bold]")
        
        if hasattr(self.agent, 'model') and self.agent.model:
            if "openai" in self.agent.model.lower() or "gpt" in self.agent.model.lower():
                content_lines.append("   🌐 OpenAI API: [green]✅ Connected[/green]")
            else:
                content_lines.append(f"   🧠 SmolAgent: [green]✅ Connected[/green]")
        else:
            content_lines.append("   🤖 Agent: [green]✅ Connected[/green]")
            
        if self.robot_connected:
            content_lines.append("   🤖 UR Robot: [green]✅ Connected[/green]")
        else:
            content_lines.append("   🤖 UR Robot: [red]❌ Disconnected[/red]")
        content_lines.append("")
        
        # Quick start info
        content_lines.append("🎤 [cyan]Speak naturally - I'll respond in real-time![/cyan]")
        content_lines.append("💡 [yellow]Try saying: 'timbra move the robot forward'[/yellow]")
        content_lines.append("🛑 [red]Press Ctrl+C to stop[/red]")
        content_lines.append("")
        
        # Agent details
        if hasattr(self.agent, 'model') and self.agent.model:
            model_display = self.agent.model.split("/")[-1] if "/" in self.agent.model else self.agent.model
            if "gpt-4o-mini" in model_display.lower():
                model_display = "GPT-4o-mini"
            content_lines.append(f"🧠 [bold]Model:[/bold] [blue]{model_display}[/blue]")
        
        if hasattr(self.agent, 'tools') and isinstance(self.agent.tools, list):
            content_lines.append(f"🔧 [bold]Tools Available:[/bold] [magenta]{len(self.agent.tools)} robot commands[/magenta]")
        
        if hasattr(self.agent, 'voice'):
            content_lines.append(f"🎙️ [bold]Voice:[/bold] [green]{self.agent.voice.title()}[/green]")
        
        content_lines.append("")
        
        # Configuration info
        if config:
            content_lines.append("⚙️ [bold]Configuration:[/bold]")
            content_lines.append(f"   🔊 Audio interruption: {'[green]✅ ON[/green]' if getattr(config, 'enable_interruption', True) else '[red]❌ OFF[/red]'}")
            content_lines.append(f"   🔇 Feedback prevention: {'[green]✅ ON[/green]' if getattr(config, 'feedback_prevention_enabled', False) else '[red]❌ OFF[/red]'}")
            content_lines.append(f"   ⏱️ Silence period: [blue]{getattr(config, 'silence_after_speaking', 0.8)}s[/blue]")
            content_lines.append(f"   🎚️ Audio threshold: [blue]{getattr(config, 'audio_threshold', 0.01)}[/blue]")
            content_lines.append(f"   🔧 Threaded execution: {'[green]✅ ON[/green]' if getattr(config, 'threaded_execution', True) else '[red]❌ OFF[/red]'}")
        
        content = "\n".join(content_lines)
        
        panel = Panel(
            content,
            title=f"[bold blue]🤖 {self.agent_name}[/bold blue]",
            border_style="blue"
        )
        
        self.console.print(panel)
        self._status_displayed = True
    
    def start_live_display(self):
        """Start the status display."""
        if not RICH_AVAILABLE:
            return
        
        if self.show_initial_info and not self._status_displayed:
            self.show_welcome_panel()
    
    def stop_live_display(self):
        """Stop the status display."""
        if self._animation_active:
            asyncio.create_task(self.stop_animation())
    
    def collapse_to_compact(self):
        """Switch to compact status mode with animations."""
        self.show_initial_info = False
        if RICH_AVAILABLE and self.console:
            self.console.print("\n" + "─" * 80, style="dim")
            asyncio.create_task(self.start_animation())
    
    # State management methods
    def update_status(self, status: str):
        """Update the general status message."""
        self.current_status = status
    
    def set_state(self, state: AgentState):
        """Set the current agent state."""
        self.current_state = state
        # Update individual flags based on state
        self.is_listening = (state == AgentState.LISTENING)
        self.is_speaking = (state == AgentState.SPEAKING)
        self.is_processing = (state == AgentState.PROCESSING)
        self.is_generating_code = (state == AgentState.GENERATING_CODE)
        self.is_executing_code = (state == AgentState.EXECUTING_CODE)
    
    def set_listening(self, listening: bool):
        """Update listening state."""
        was_listening = self.is_listening
        self.is_listening = listening
        if listening and not was_listening and not self.show_initial_info:
            self.print_message("[green]🎤 Started listening...[/green]")
    
    def set_speaking(self, speaking: bool):
        """Update speaking state."""
        was_speaking = self.is_speaking
        self.is_speaking = speaking
        if speaking and not was_speaking and not self.show_initial_info:
            self.print_message("[blue]🔊 Assistant speaking...[/blue]")
    
    def set_processing(self, processing: bool):
        """Update processing state."""
        was_processing = self.is_processing
        self.is_processing = processing
        if processing and not was_processing and not self.show_initial_info:
            self.print_message("[yellow]🧠 Processing speech...[/yellow]")
    
    def set_generating_code(self, generating: bool):
        """Update code generation state."""
        was_generating = self.is_generating_code
        self.is_generating_code = generating
        if generating and not was_generating and not self.show_initial_info:
            self.print_message("[yellow]🧠 Generating Python code...[/yellow]")
    
    def set_executing_code(self, executing: bool):
        """Update code execution state."""
        was_executing = self.is_executing_code
        self.is_executing_code = executing
        if executing and not was_executing and not self.show_initial_info:
            self.print_message("[magenta]🔧 Executing code...[/magenta]")
    
    def set_tool_executing(self, tool_name: Optional[str]):
        """Update tool execution state."""
        was_executing = self.tool_executing
        self.tool_executing = tool_name
        if tool_name and not was_executing and not self.show_initial_info:
            self.print_message(f"[magenta]🔧 Executing {tool_name}...[/magenta]")
    
    def set_fabrication_active(self, active: bool):
        """Update fabrication state - changes entire UI color scheme."""
        self.fabrication_active = active
        if active:
            self.print_message("🟢 [white]Fabrication session STARTED - UI active (white)[/white]", "green bold")
        else:
            self.print_message("⚫ [black]Fabrication session ENDED - UI idle (black)[/black]", "red bold")
    
    # Message printing methods
    def print_message(self, message: str, style: str = "white"):
        """Print a message with optional Rich styling."""
        if RICH_AVAILABLE and self.console:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def print_user_message(self, message: str):
        """Print user speech with special formatting."""
        self.print_message(f"\n👤 You said: {message}", "blue bold")
    
    def print_assistant_message(self, message: str):
        """Print assistant response with special formatting."""
        self.print_message(f"\n🤖 Assistant: {message}", "green bold")
    
    def print_error(self, message: str):
        """Print error message with formatting."""
        self.print_message(f"❌ {message}", "red bold")
    
    def print_success(self, message: str):
        """Print success message with formatting."""
        self.print_message(f"✅ {message}", "green bold")
    
    def print_tool_message(self, message: str):
        """Print tool execution message with special formatting."""
        self.print_message(f"🔧 {message}", "magenta bold")
    
    def print_code_message(self, message: str):
        """Print code generation/execution message with special formatting."""
        self.print_message(f"🧠 {message}", "yellow bold")
    
    def show_connection_progress(self):
        """Show connection progress animation."""
        if not RICH_AVAILABLE:
            print("🔗 Connecting to services...")
            return None, None
            
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )
        task = progress.add_task("🔗 Connecting to services...", total=None)
        return progress, task


def create_status_manager(agent_name: str, agent=None, config=None) -> VoiceAgentStatus:
    """Factory function to create a status manager with consistent configuration.
    
    Args:
        agent_name: Name of the agent for display
        agent: Agent instance (optional)
        config: Configuration object (optional)
        
    Returns:
        Configured VoiceAgentStatus instance
    """
    status = VoiceAgentStatus(agent_name, agent)
    
    # Store config reference for welcome panel
    if config:
        status._config = config
    
    return status 