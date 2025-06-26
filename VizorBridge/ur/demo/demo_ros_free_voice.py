"""
ROS-Free Robot Voice Agent Demo
==============================

A clean, simple demo using the core UR library with direct RTDE connection.
No ROS dependencies - just voice conversation and robot control.

This demo showcases:
- Direct RTDE robot connection (no ROS bridge)
- OpenAI Realtime API voice interaction
- Integrated robot tool execution
- Session logging and status display

Configuration via environment variables:
- OPENAI_API_KEY: Your OpenAI API key (required)
- UR_ROBOT_IP: Robot IP address (default: 192.168.56.101)
- TEXT_ONLY_MODE: Use text-only mode (true/false, default: false)
- ENABLE_TEXT_INPUT: Allow text input alongside voice (true/false, default: true)
- DEBUG: Enable debug output (true/false, default: false)
- ENABLE_LOGGING: Enable session logging (true/false, default: true)

Usage:
    python demo_ros_free_voice.py
    
    # Text-only mode
    TEXT_ONLY_MODE=true python demo_ros_free_voice.py
    
    # Debug mode
    DEBUG=true python demo_ros_free_voice.py
"""

import asyncio
import os
import signal
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ur.core.connection import URConnection, get_robot, cleanup_global_robot
from ur.agents.openai_agent import OpenAIVoiceAgent
from ur.agents.voice_common import VoiceAgentStatus, load_config_from_env


class SimpleRobotBridge:
    """Minimal robot bridge for ROS-free operation."""
    
    def __init__(self, robot_ip: str = None):
        self.robot_ip = robot_ip or os.getenv("UR_ROBOT_IP", "192.168.56.101")
        # Use global robot connection to avoid duplicate RTDE connections
        os.environ["UR_ROBOT_IP"] = self.robot_ip  # Ensure global uses correct IP
        self.robot_connection = get_robot()
        self.is_connected = self.robot_connection.is_connected
        
    def get_current_state(self):
        """Get current robot state."""
        if not self.robot_connection:
            return None
            
        return {
            "tcp_pose": self.robot_connection.get_tcp_pose(),
            "joints": self.robot_connection.get_joints(),
            "connected": self.robot_connection.is_connected
        }
    
    def stop(self):
        """Clean shutdown."""
        # Don't cleanup directly since this is a shared global connection
        # The global cleanup will be handled in the main cleanup
        pass
    
    def emergency_stop(self):
        """Emergency stop robot movements."""
        try:
            if self.robot_connection and self.robot_connection.is_connected:
                if hasattr(self.robot_connection, 'rtde_c') and self.robot_connection.rtde_c:
                    print("🛑 Emergency stopping robot movements...")
                    self.robot_connection.rtde_c.stopL()  # Stop linear movements
                    self.robot_connection.rtde_c.stopJ()  # Stop joint movements
                    print("✅ Robot movements stopped")
                else:
                    print("⚠️ Robot control interface not available")
            else:
                print("⚠️ Robot not connected - no movements to stop")
        except Exception as e:
            print(f"⚠️ Error stopping robot: {e}")


class ROSFreeVoiceSystem:
    """ROS-free voice system using core UR library."""
    
    def __init__(self):
        self._load_environment()
        self.bridge = None
        self.agent = None
        self.ui = None
        self._shutdown_requested = False
        self._setup_signal_handlers()
        
    def _load_environment(self):
        """Load environment variables and configuration."""
        # Load .env file if available
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
        except ImportError:
            pass
            
        # Validate required configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.robot_ip = os.getenv("UR_ROBOT_IP", "192.168.56.101")
        
        
        self.voice_config = load_config_from_env()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n🚨 Received signal {signum} (Ctrl+C) - initiating emergency shutdown...")
            self._shutdown_requested = True
            
            # Emergency stop robot immediately
            if self.bridge:
                self.bridge.emergency_stop()
            
            # Force exit if called multiple times
            if hasattr(self, '_shutdown_count'):
                self._shutdown_count += 1
                if self._shutdown_count > 2:
                    print("🚨 Force exit requested!")
                    os._exit(1)
            else:
                self._shutdown_count = 1
        
        # Handle Ctrl+C (SIGINT) and termination (SIGTERM)
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):  # Not available on Windows
            signal.signal(signal.SIGTERM, signal_handler)
        
    def initialize(self):
        """Initialize the voice system components."""
        print("🚀 Initializing ROS-Free Robot Voice System...")
        
        # Initialize UI
        self.ui = VoiceAgentStatus("ROS-Free-Robot-Agent")
        
        # Initialize robot bridge
        self.ui.update_status("Connecting to robot...")
        self.bridge = SimpleRobotBridge(self.robot_ip)
        
        if self.bridge.is_connected:
            self.ui.print_success(f"✅ Robot connected to {self.robot_ip}")
        else:
            self.ui.print_message(f"⚠️ Robot not connected - running in simulation mode", "yellow")
            
        # Initialize voice agent
        self.ui.update_status("Initializing OpenAI voice agent...")
        
        if self.voice_config.debug_mode:
            print(f"🔧 Debug mode enabled - config: {self.voice_config}")
        
        self.agent = OpenAIVoiceAgent(
            bridge_ref=self.bridge,
            ui_ref=self.ui,
            api_key=self.api_key,
            config=self.voice_config
        )
        
        self.ui.print_success("🎤 Voice agent initialized")
        
    def print_configuration(self):
        """Print system configuration."""
        print("\n" + "="*60)
        print("🤖 ROS-FREE ROBOT VOICE SYSTEM")
        print("="*60)
        print(f"🔧 Robot IP: {self.robot_ip}")
        print(f"🔌 Robot Connected: {self.bridge.is_connected if self.bridge else 'Unknown'}")
        print(f"🤖 Voice Agent: OpenAI Realtime API")
        print(f"💬 Text-only Mode: {self.voice_config.text_only_mode}")
        print(f"🎤 Text Input Enabled: {self.voice_config.enable_text_input}")
        print(f"🛑 Audio Interruption: {self.voice_config.enable_interruption}")
        print(f"🔇 Feedback Prevention: {self.voice_config.feedback_prevention_enabled}")
        print(f"🧵 Threaded Tools: {self.voice_config.threaded_execution}")
        print(f"🐛 Debug Mode: {self.voice_config.debug_mode}")
        print(f"📝 Session Logging: {self.voice_config.enable_logging}")
        print("="*60)
        
        print("\n💡 USAGE TIPS:")
        print("  • Speak naturally to control the robot")
        print("  • Use 'timbra' wake word for robot commands")
        print("  • Try: 'Move home timbra' or 'Get robot status'")
        if self.voice_config.enable_text_input:
            print("  • Type messages if voice input is not working")
        print("  • Press Ctrl+C to exit")
        print("="*60)
        
    async def start(self):
        """Start the voice system."""
        try:
            self.print_configuration()
            
            print("\n🎤 Starting voice interaction...")
            print("✨ Say something to begin! (or type if text input is enabled)")
            print("⚠️ Press Ctrl+C to safely stop and disconnect robot")
            
            # Start the voice agent
            await self.agent.start()
            
        except KeyboardInterrupt:
            print("\n👋 Keyboard interrupt received - shutting down gracefully...")
            self._shutdown_requested = True
        except Exception as e:
            print(f"❌ System error: {e}")
            # Emergency stop on any error
            if self.bridge:
                self.bridge.emergency_stop()
            raise
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Clean shutdown of all components."""
        print("\n🧹 Cleaning up...")
        
        # Emergency stop robot first (most critical)
        if self.bridge:
            try:
                print("🛑 Ensuring robot is stopped...")
                self.bridge.emergency_stop()
            except Exception as e:
                print(f"⚠️ Robot emergency stop warning: {e}")
        
        # Stop voice agent
        if self.agent:
            try:
                print("🔇 Stopping voice agent...")
                await self.agent.stop()
            except Exception as e:
                print(f"⚠️ Agent cleanup warning: {e}")
                
        # Stop bridge
        if self.bridge:
            try:
                print("🔌 Stopping robot bridge...")
                self.bridge.stop()
            except Exception as e:
                print(f"⚠️ Bridge cleanup warning: {e}")
        
        # Clean up global robot connection (includes RTDE disconnect)
        try:
            print("🤖 Cleaning up robot connection...")
            cleanup_global_robot()
        except Exception as e:
            print(f"⚠️ Robot cleanup warning: {e}")
                
        # Stop UI last
        if self.ui:
            try:
                print("🖥️ Stopping UI...")
                self.ui.stop_live_display()
            except Exception as e:
                print(f"⚠️ UI cleanup warning: {e}")
                
        print("✅ Cleanup complete - robot safely disconnected")


def main():
    """Main entry point for the ROS-free robot voice demo."""
    print("🤖 ROS-Free Robot Voice Agent Demo")
    print("==================================")
    
    system = None
    try:
        # Create and initialize system
        system = ROSFreeVoiceSystem()
        system.initialize()
        
        # Start the voice interaction
        asyncio.run(system.start())
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Make sure to set OPENAI_API_KEY in your environment")
        print("   You can create a .env file in the project root with:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user - ensuring robot safety...")
        if system and system.bridge:
            try:
                system.bridge.emergency_stop()
                cleanup_global_robot()
            except Exception as e:
                print(f"⚠️ Final cleanup warning: {e}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        
        # Emergency robot safety
        if system and system.bridge:
            try:
                print("🚨 Emergency robot stop due to fatal error...")
                system.bridge.emergency_stop()
                cleanup_global_robot()
            except Exception as cleanup_error:
                print(f"⚠️ Emergency cleanup error: {cleanup_error}")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 