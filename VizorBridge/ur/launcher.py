import asyncio
import os
import sys
import threading
from pathlib import Path

# Add project root to sys.path to allow finding other modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import roslibpy
from ur.bridge import URBridge
from ur.ui.console import VoiceAgentStatus
from ur.agents.base_agent import BaseVoiceAgent
from ur.agents.openai_agent import OpenAIVoiceAgent
from ur.config.system_config import (
    AGENT_TYPE,
    VOICE_ENABLED,
    SMOL_MODEL_ID,
    SMOL_PROVIDER,
    ROS_HOST,
    ROS_PORT,
)
from ur.config.robot_config import ROBOT_NAME, ROBOT_IP
from ur.config.voice_config import load_config_from_env


class URVoiceSystem:
    """
    Enhanced launcher class that encapsulates the setup and management of the
    URBridge and its associated voice agent with comprehensive features.
    """

    def __init__(self, client: roslibpy.Ros):
        """
        Initializes the UR voice system in idle mode.
        Bridge connects but voice agent waits for START_FABRICATION command.

        Args:
            client: An active roslibpy.Ros client instance.
        """
        self.client = client
        
        # Load voice configuration
        self.voice_config = load_config_from_env()
        
        # --- Configuration ---
        self.robot_name = ROBOT_NAME
        self.robot_ip = ROBOT_IP
        self.agent_type = AGENT_TYPE
        self.voice_enabled = VOICE_ENABLED
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # --- Initialization ---
        self.ui = VoiceAgentStatus(agent_name=f"UR-Voice-{self.agent_type}")
        self.bridge = self._initialize_bridge()
        
        # Voice agent starts as None - will be created on START_FABRICATION
        self.agent = None
        self.agent_thread = None
        
        # Shutdown flag for main.py to check
        self.shutdown_requested = False
        
        # Register launcher with bridge for agent creation
        self.bridge.set_launcher_reference(self)
        
        # Register components with cleanup manager for coordinated shutdown
        self._register_with_cleanup_manager()
        
        # Display idle status
        self.ui.print_message("🛌 System initialized and waiting for START_FABRICATION command...", style="cyan bold")

    def _load_env(self):
        # .env loading removed; configuration is handled in config modules
        pass

    def _initialize_bridge(self) -> URBridge:
        """Initializes and connects the URBridge."""
        self.ui.update_status(f"Initializing robot bridge for '{self.robot_name}'...")
        bridge = URBridge(name=self.robot_name, client=self.client, robot_ip=self.robot_ip)
        if self.robot_ip:
            if bridge.connect():
                self.ui.print_success(f"Bridge connected to robot at {self.robot_ip}")
            else:
                self.ui.print_error(f"Failed to connect to robot at {self.robot_ip}. Running in simulation.")
        else:
            self.ui.print_message("No ROBOT_IP provided. Bridge running in simulation mode.", style="yellow")
        
        self.ui.print_message("📡 Bridge connected - ready to receive START_FABRICATION command", style="green")
        return bridge

    def _initialize_agent(self) -> BaseVoiceAgent | None:
        """Selects and initializes the appropriate voice agent with enhanced features."""
        if not self.voice_enabled:
            self.ui.print_message("Voice agent is disabled (VOICE_ENABLED=false).", style="dim")
            return None

        self.ui.update_status(f"Initializing '{self.agent_type}' voice agent...")
        
        if self.agent_type == "openai":
            if not self.openai_api_key:
                self.ui.print_error("AGENT_TYPE is 'openai' but OPENAI_API_KEY is not set.")
                return None
            return OpenAIVoiceAgent(
                bridge_ref=self.bridge, 
                ui_ref=self.ui, 
                api_key=self.openai_api_key,
                config=self.voice_config
            )
        elif self.agent_type in ("smol", "smolagents"):
            model_id = SMOL_MODEL_ID
            provider = SMOL_PROVIDER
            from ur.agents.smol_agent import SmolVoiceAgent
            self.ui.print_message(f"Using SmolAgent with model: {model_id} (provider: {provider})", style="blue")
            return SmolVoiceAgent(
                bridge_ref=self.bridge,
                ui_ref=self.ui,
                model_id=model_id,
                provider=provider,
                config=self.voice_config
            )
        else:
            self.ui.print_error(f"Unknown AGENT_TYPE: '{self.agent_type}'. Supported types: openai, smol, smolagents")
            return None

    def _start_agent_thread(self):
        """Starts the agent's main loop in a separate, non-blocking thread."""
        async def run_agent_async():
            await self.agent.start()

        self.agent_thread = threading.Thread(
            target=lambda: asyncio.run(run_agent_async()),
            daemon=True,
            name=f"{self.agent_type}-AgentThread"
        )
        self.agent_thread.start()
        self.ui.print_success(f"'{self.agent_type}' voice agent is running with enhanced features.")

    def _register_with_cleanup_manager(self):
        """Register components with the cleanup manager for coordinated shutdown."""
        try:
            from ur.core.cleanup_manager import register_for_cleanup
            from ur.core.connection import get_robot
            
            # Get robot connection for registration
            robot_connection = None
            try:
                robot_connection = get_robot()
            except Exception:
                pass  # Robot connection might not be available
            
            # Register current components
            register_for_cleanup(
                component_name="voice_system",
                voice_agent=self.agent,  # Will be None initially, updated when agent is created
                bridge=self.bridge,
                robot_connection=robot_connection,
                ui=self.ui
            )
            
            self.ui.print_message("📋 Components registered with cleanup manager", style="dim")
            
        except Exception as e:
            self.ui.print_message(f"⚠️ Could not register with cleanup manager: {e}", style="yellow")

    def create_agent(self) -> BaseVoiceAgent | None:
        """Create and return a voice agent instance. Called by bridge on START_FABRICATION."""
        if not self.voice_enabled:
            self.ui.print_message("Voice agent is disabled (VOICE_ENABLED=false).", style="dim")
            return None
            
        if self.agent:
            self.ui.print_message("Voice agent already exists.", style="yellow")
            return self.agent

        self.ui.update_status(f"Creating '{self.agent_type}' voice agent...")
        
        # Create new agent instance
        self.agent = self._initialize_agent()
        
        # Register with bridge for task notifications
        if self.agent and self.bridge:
            self.bridge.set_agent_reference(self.agent)
        
        # Re-register with cleanup manager now that we have an agent
        self._register_with_cleanup_manager()
            
        return self.agent

    def print_configuration_summary(self):
        """Print a summary of the current configuration."""
        self.ui.print_message("\n" + "="*60, "cyan")
        self.ui.print_message("🚀 UR VOICE SYSTEM CONFIGURATION", "cyan bold")
        self.ui.print_message("="*60, "cyan")
        
        # System info
        self.ui.print_message(f"📡 Robot: {self.robot_name} @ {self.robot_ip or 'simulation'}", "white")
        self.ui.print_message(f"🤖 Agent Type: {self.agent_type}", "white")
        self.ui.print_message(f"🔊 Voice Enabled: {self.voice_enabled}", "white")
        
        # Voice configuration
        if self.voice_config:
            self.ui.print_message("\n🎤 VOICE CONFIGURATION:", "yellow")
            self.ui.print_message(f"  Text-only mode: {self.voice_config.text_only_mode}", "white")
            self.ui.print_message(f"  Text input enabled: {self.voice_config.enable_text_input}", "white")
            self.ui.print_message(f"  Audio interruption: {self.voice_config.enable_interruption}", "white")
            self.ui.print_message(f"  Feedback prevention: {self.voice_config.feedback_prevention_enabled}", "white")
            self.ui.print_message(f"  Threaded execution: {self.voice_config.threaded_execution}", "white")
            self.ui.print_message(f"  Debug mode: {self.voice_config.debug_mode}", "white")
            self.ui.print_message(f"  Session logging: {self.voice_config.enable_logging}", "white")
            
        self.ui.print_message("="*60, "cyan")
        
        # Show environment setup instructions
        self.ui.print_message("\n💡 CONFIGURATION TIPS:", "yellow")
        self.ui.print_message("  Set TEXT_ONLY_MODE=true for text-only interaction", "dim")
        self.ui.print_message("  Set DEBUG=true for detailed debug output", "dim")
        self.ui.print_message("  Set ENABLE_LOGGING=true for session logging", "dim")
        self.ui.print_message("  Set FEEDBACK_PREVENTION=true for enhanced audio filtering", "dim")
        self.ui.print_message("="*60, "cyan")

    def reset_for_restart(self):
        """Reset the system state to allow for restart after shutdown."""
        print("🔄 [SYSTEM] Resetting system for restart...")
        
        # Reset shutdown flag
        self.shutdown_requested = False
        
        # Clean up existing agent properly
        if self.agent:
            try:
                # Stop the agent cleanly if it has a stop method
                if hasattr(self.agent, 'stop_sync'):
                    self.agent.stop_sync()
                elif hasattr(self.agent, 'should_stop'):
                    self.agent.should_stop = True
                    
                # Clear session connections
                if hasattr(self.agent, 'session') and self.agent.session:
                    self.agent.session.is_connected = False
                    if hasattr(self.agent.session, 'should_stop'):
                        self.agent.session.should_stop = True
                        
            except Exception as e:
                print(f"⚠️ [SYSTEM] Agent cleanup warning: {e}")
        
        # Reset agent reference
        self.agent = None
        self.agent_thread = None
        
        # Reset fabrication state
        if self.bridge:
            self.bridge.fabrication_active = False
            self.bridge.voice_agent = None
            self.bridge.voice_agent_thread = None
            # Reset agent reference to clear old agent notifications
            if hasattr(self.bridge, 'agent_ref'):
                self.bridge.agent_ref = None
            
        # Clear any existing cleanup tasks - but don't run full cleanup
        try:
            from ur.core.cleanup_manager import reset_cleanup_manager
            reset_cleanup_manager()
        except Exception as e:
            print(f"⚠️ [SYSTEM] Cleanup manager reset warning: {e}")
            
        print("✅ [SYSTEM] Reset complete - ready for new fabrication session")

    def cleanup(self):
        """Shuts down the UI and bridge connections gracefully using cleanup manager."""
        self.ui.print_message("\nShutting down UR Voice System...", "yellow")
        
        # Set shutdown flag so main.py knows to restart process
        self.shutdown_requested = True
        
        # Use the centralized cleanup manager for coordinated shutdown
        try:
            import asyncio
            from ur.core.cleanup_manager import cleanup_system

            # Handle different event loop contexts
            try:
                # Check if we're in an existing event loop
                loop = asyncio.get_running_loop()
                # If we're in a running loop, run cleanup in a separate thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, cleanup_system())
                    future.result(timeout=15)
            except RuntimeError:
                # No running loop, run cleanup directly
                asyncio.run(cleanup_system())

            self.ui.print_message("✅ UR Voice System shutdown complete.", "green")
        except Exception as e:
            self.ui.print_message("⚠️ UR Voice System shutdown completed with warnings.", "yellow")
            if getattr(self.voice_config, 'debug_mode', False):
                print(f"Debug: Cleanup error: {e}")
        
        # Clear references
        self.bridge = None
        self.voice_agent = None


def create_enhanced_voice_system(client: roslibpy.Ros, show_config: bool = True) -> URVoiceSystem:
    """Create an enhanced UR voice system with comprehensive features.
    
    Args:
        client: ROS client instance
        show_config: Whether to show configuration summary
        
    Returns:
        URVoiceSystem instance
    """
    system = URVoiceSystem(client)
    
    if show_config:
        system.print_configuration_summary()
    
    return system


def main():
    """Main entry point for running the enhanced UR voice system standalone."""
    import signal
    
    # Load configuration
    config = load_config_from_env()
    
    print("🚀 Starting Enhanced UR Voice System...")
    print(f"💡 Voice Config: {config}")
    
    # Create ROS client
    client = roslibpy.Ros(host=ROS_HOST, port=ROS_PORT)
    
    try:
        # Connect to ROS
        client.run_forever()
        
        # Create enhanced voice system
        system = create_enhanced_voice_system(client, show_config=True)
        
        # Handle shutdown gracefully
        def signal_handler(sig, frame):
            print("\n🛑 Received shutdown signal...")
            system.cleanup()
            client.terminate()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Keep the system running
        print("\n✅ Enhanced UR Voice System is running!")
        print("💬 Press Ctrl+C to shutdown")
        
        while True:
            try:
                asyncio.sleep(1)
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        print(f"❌ Failed to start enhanced voice system: {e}")
        sys.exit(1)
    finally:
        if 'system' in locals():
            system.cleanup()
        if client:
            client.terminate()


if __name__ == "__main__":
    main() 