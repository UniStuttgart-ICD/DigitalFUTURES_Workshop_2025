from __future__ import annotations

import threading
import time
from typing import Optional

# Rich console library for animations and better UI
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to regular print if rich is not available
    print("⚠️ Rich library not available for enhanced UI. Install with: pip install rich")


class VoiceAgentStatus:
    """
    Console status manager with animations for voice agent states.
    This class is a UI component and does not contain any agent logic.
    """

    def __init__(self, agent_name: str = "Agent"):
        if not RICH_AVAILABLE:
            self.console = None
            return

        self.console = Console()
        self.live_display = None
        self.agent_name = agent_name
        self.current_status = "Initializing..."
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.tool_executing = None
        self.show_initial_info = True
        self.is_stopped = False  # Flag to prevent updates after stop
        
        # Fabrication state tracking for UI color changes
        self.fabrication_active = False

    def start_live_display(self):
        """Start the live updating display."""
        if not self.console:
            print("UI Initializing...")
            return

        if self.live_display:
            self.stop_live_display()

        self.live_display = Live(
            self._create_status_panel(),
            console=self.console,
            refresh_per_second=4,
            auto_refresh=True,
            vertical_overflow="visible"
        )
        self.live_display.start()

    def stop_live_display(self):
        """Stop the live display immediately and completely."""
        self.is_stopped = True  # Prevent any further updates
        
        if self.live_display:
            try:
                # Stop the live display immediately
                self.live_display.stop()
                
                # Clear any pending updates by disabling auto-refresh
                self.live_display.auto_refresh = False
                
                # Force the console to finish any pending operations
                if self.console:
                    try:
                        self.console.file.flush()
                        # Clear the console's internal state
                        if hasattr(self.console.file, 'close'):
                            # Don't actually close stdout/stderr, just flush
                            pass
                    except Exception:
                        pass
                    
                # Wait a brief moment for threads to finish
                time.sleep(0.1)
                    
            except Exception as e:
                # Ignore errors during shutdown to prevent blocking
                if hasattr(self, 'console') and self.console and not self.is_stopped:
                    try:
                        self.console.print(f"[dim]UI stop warning: {e}[/dim]")
                    except:
                        pass
            finally:
                # Always clear the reference
                self.live_display = None
                
        # Additional cleanup - clear console reference if needed
        # Note: Don't actually clear self.console as it might be used for final messages

    def _create_status_panel(self):
        """Create an enhanced status panel for the agent."""
        if not self.console:
            return self.current_status

        if self.show_initial_info:
            # Display a more detailed welcome panel at the start
            content_lines = [
                f"[bold green]🤖 {self.agent_name} Voice Agent[/bold green]",
                "",
                "🎤 [cyan]Speak naturally to control the robot.[/cyan]",
                "🛑 [red]Press Ctrl+C to stop.[/red]",
                ""
            ]
        else:
            # Compact mode for ongoing status
            content_lines = []

        # Always show current status line
        status_items = ["[green]●[/green] Connected"]
        if self.is_listening:
            status_items.append("[green]🎤 Listening ●●●[/green]")
        elif self.is_speaking:
            status_items.append("[blue]🔊 Speaking ♪♫♪[/blue]")
        elif self.is_processing:
            status_items.append("[yellow]🧠 Processing ⚡[/yellow]")
        elif self.tool_executing:
            status_items.append(f"[magenta]🔧 {self.tool_executing} ⚙️[/magenta]")
        else:
            status_items.append("[dim]🎤 Ready[/dim]")

        status_items.append(f"[cyan]{self.current_status}[/cyan]")
        
        # Apply fabrication state coloring to the entire status line
        status_line = " │ ".join(status_items)
        if self.fabrication_active:
            # White/bright text when fabrication is active
            colored_status_line = f"[white]{status_line}[/white]"
        else:
            # Black/dim text when fabrication is inactive
            colored_status_line = f"[black]{status_line}[/black]"
        
        content_lines.append(colored_status_line)
        content = "\n".join(content_lines)

        height = None if self.show_initial_info else 3
        return Panel(content, title=f"[bold blue]🤖 {self.agent_name}[/bold blue]", border_style="blue", height=height)

    def collapse_to_compact(self):
        """Switch to the compact display mode after startup."""
        self.show_initial_info = False
        self._refresh_display()

    def update_status(self, status: str):
        """Update the general status message."""
        if not self.is_stopped:
            self.current_status = status
            self._refresh_display()

    def set_listening(self, listening: bool):
        """Update listening state."""
        if not self.is_stopped:
            self.is_listening = listening
            self._refresh_display()

    def set_speaking(self, speaking: bool):
        """Update speaking state."""
        if not self.is_stopped:
            self.is_speaking = speaking
            self._refresh_display()

    def set_processing(self, processing: bool):
        """Update processing state."""
        if not self.is_stopped:
            self.is_processing = processing
            self._refresh_display()

    def set_tool_executing(self, tool_name: Optional[str]):
        """Update tool execution state."""
        if not self.is_stopped:
            self.tool_executing = tool_name
            self._refresh_display()

    def set_fabrication_active(self, active: bool):
        """Update fabrication state - changes entire UI color scheme."""
        if not self.is_stopped:
            self.fabrication_active = active
            if active:
                self.print_message("🟢 [white]Fabrication session STARTED - UI active (white)[/white]", "green bold")
            else:
                self.print_message("⚫ [black]Fabrication session ENDED - UI idle (black)[/black]", "red bold")
            self._refresh_display()

    def _refresh_display(self):
        """Refresh the live display with the current panel."""
        if not self.is_stopped and self.live_display:
            try:
                self.live_display.update(self._create_status_panel())
            except Exception:
                # Ignore errors during refresh to prevent blocking
                pass

    def print_message(self, message: str, style: str = "white"):
        """Print a message below the live display without disrupting it."""
        if not self.is_stopped and self.console:
            try:
                self.console.print(message, style=style)
            except Exception:
                # Ignore errors if console is not available
                pass
        elif self.is_stopped:
            # If stopped, fall back to basic print to avoid hanging
            try:
                print(message)
            except Exception:
                pass

    def print_user_message(self, message: str):
        """Print user speech with special formatting."""
        self.print_message(f"👤 You said: {message}", "green bold")

    def print_robot_response(self, message: str):
        """Print the robot's text response with special formatting."""
        self.print_message(f"🤖 Assistant: {message}", "blue bold")

    def print_error(self, message: str):
        """Print an error message with formatting."""
        self.print_message(f"❌ {message}", "red bold")

    def print_success(self, message: str):
        """Print a success message with formatting."""
        self.print_message(f"✅ {message}", "green bold")

    def print_tool_message(self, message: str):
        """Print a tool execution message with special formatting."""
        self.print_message(f"🔧 {message}", "magenta bold")

    def show_connection_progress(self, service_name: str):
        """Show a connection progress animation."""
        if not self.console:
            print(f"🔗 Connecting to {service_name}...")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self.console
        ) as progress:
            task = progress.add_task(f"🔗 Connecting to {service_name}...", total=None)
            return progress, task

    def complete_shutdown(self):
        """Complete shutdown of all UI components including background threads."""
        self.is_stopped = True
        
        # Stop live display first
        self.stop_live_display()
        
        # Clear all state
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.tool_executing = None
        
        # Force cleanup of Rich console internals
        if self.console:
            try:
                # Flush and clear any pending operations
                self.console.file.flush()
                
                # Clear Rich's internal thread pools if accessible
                if hasattr(self.console, '_thread_pool'):
                    try:
                        self.console._thread_pool.shutdown(wait=False)
                    except:
                        pass
                        
                # Clear console reference
                self.console = None
                
            except Exception:
                # Ignore any errors during final cleanup
                pass
        
        # Wait longer for any remaining threads to finish
        time.sleep(0.2)

    def force_shutdown(self):
        """Force complete shutdown of all UI components."""
        self.is_stopped = True
        
        # Stop live display immediately
        if self.live_display:
            try:
                self.live_display.stop()
            except Exception:
                pass
            finally:
                self.live_display = None
        
        # Clear all references
        self.console = None
        
        # Clear all state
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.tool_executing = None 