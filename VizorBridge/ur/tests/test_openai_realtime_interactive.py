"""Test OpenAI Realtime API agent for pure conversation without robot functionality.

This test validates the OpenAI agent's conversation capabilities,
event handling, and session management without any robot-specific tools.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Rich imports for better console output
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the OpenAI agent components
from ur.agents.openai_agent import (
    RealtimeAgent, 
    RealtimeSession, 
    ToolDefinition, 
    EventType,
    RealtimeEvent
)
from ur.config.voice_config import VoiceAgentConfig, get_config

console = Console()

# Test constants
TEST_API_KEY = "test_api_key_for_mocking"


def list_audio_devices():
    """List available audio input and output devices."""
    try:
        import sounddevice as sd
        
        devices = sd.query_devices()
        input_devices = []
        output_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device))
            if device['max_output_channels'] > 0:
                output_devices.append((i, device))
        
        return input_devices, output_devices
    except ImportError:
        console.print("[red]❌[/red] sounddevice not available")
        return [], []


def display_audio_devices(input_devices, output_devices):
    """Display available audio devices in a nice table format."""
    
    # Input devices table
    if input_devices:
        input_table = Table(title="🎤 Available Microphones", show_header=True, header_style="bold cyan")
        input_table.add_column("ID", style="dim", width=6)
        input_table.add_column("Device Name", style="cyan")
        input_table.add_column("Channels", style="green", width=8)
        input_table.add_column("Sample Rate", style="yellow", width=12)
        
        for device_id, device in input_devices:
            channels = str(device['max_input_channels'])
            sample_rate = f"{device['default_samplerate']:.0f} Hz"
            input_table.add_row(str(device_id), device['name'], channels, sample_rate)
        
        console.print(input_table)
    else:
        console.print("[red]❌ No input devices found[/red]")
    
    console.print()
    
    # Output devices table
    if output_devices:
        output_table = Table(title="🔊 Available Speakers", show_header=True, header_style="bold cyan")
        output_table.add_column("ID", style="dim", width=6)
        output_table.add_column("Device Name", style="cyan")
        output_table.add_column("Channels", style="green", width=8)
        output_table.add_column("Sample Rate", style="yellow", width=12)
        
        for device_id, device in output_devices:
            channels = str(device['max_output_channels'])
            sample_rate = f"{device['default_samplerate']:.0f} Hz"
            output_table.add_row(str(device_id), device['name'], channels, sample_rate)
        
        console.print(output_table)
    else:
        console.print("[red]❌ No output devices found[/red]")


def select_audio_devices():
    """Let user select audio input and output devices."""
    console.print(Panel("[bold cyan]🎧 Audio Device Setup[/bold cyan]", expand=False))
    
    try:
        import sounddevice as sd
        
        # Get current default devices
        try:
            default_input = sd.default.device[0] if sd.default.device[0] is not None else 'auto'
            default_output = sd.default.device[1] if sd.default.device[1] is not None else 'auto'
        except:
            default_input = 'auto'
            default_output = 'auto'
        
        console.print(f"[dim]Current defaults: Input={default_input}, Output={default_output}[/dim]")
        console.print()
        
        # List available devices
        input_devices, output_devices = list_audio_devices()
        
        if not input_devices and not output_devices:
            console.print("[yellow]⚠[/yellow] No audio devices detected. Using system defaults.")
            return None, None
        
        display_audio_devices(input_devices, output_devices)
        console.print()
        
        # Select input device
        selected_input = None
        if input_devices:
            console.print("[cyan]Select microphone (input device):[/cyan]")
            console.print("Press Enter for default, or enter device ID:")
            
            try:
                choice = input("🎤 Microphone ID: ").strip()
                if choice:
                    device_id = int(choice)
                    # Validate choice
                    valid_ids = [dev[0] for dev in input_devices]
                    if device_id in valid_ids:
                        selected_input = device_id
                        device_name = next(dev[1]['name'] for dev in input_devices if dev[0] == device_id)
                        console.print(f"[green]✓[/green] Selected microphone: {device_name}")
                    else:
                        console.print(f"[red]❌[/red] Invalid ID {device_id}. Using default.")
                else:
                    console.print("[yellow]Using default microphone[/yellow]")
            except (ValueError, KeyboardInterrupt):
                console.print("[yellow]Using default microphone[/yellow]")
        
        # Select output device
        selected_output = None
        if output_devices:
            console.print("\n[cyan]Select speakers (output device):[/cyan]")
            console.print("Press Enter for default, or enter device ID:")
            
            try:
                choice = input("🔊 Speaker ID: ").strip()
                if choice:
                    device_id = int(choice)
                    # Validate choice
                    valid_ids = [dev[0] for dev in output_devices]
                    if device_id in valid_ids:
                        selected_output = device_id
                        device_name = next(dev[1]['name'] for dev in output_devices if dev[0] == device_id)
                        console.print(f"[green]✓[/green] Selected speakers: {device_name}")
                    else:
                        console.print(f"[red]❌[/red] Invalid ID {device_id}. Using default.")
                else:
                    console.print("[yellow]Using default speakers[/yellow]")
            except (ValueError, KeyboardInterrupt):
                console.print("[yellow]Using default speakers[/yellow]")
        
        console.print()
        return selected_input, selected_output
        
    except ImportError:
        console.print("[red]❌[/red] sounddevice not available for device selection")
        return None, None


def test_audio_devices(input_device=None, output_device=None):
    """Test the selected audio devices."""
    console.print("[cyan]🧪 Testing audio devices...[/cyan]")
    
    try:
        import sounddevice as sd
        import numpy as np
        import time
        
        # Test microphone
        if input_device is not None:
            sd.default.device[0] = input_device
        
        console.print("🎤 Testing microphone (2 seconds)...")
        try:
            recording = sd.rec(int(2 * 24000), samplerate=24000, channels=1, dtype=np.float32, device=input_device)
            sd.wait()
            
            # Check if we got audio
            audio_level = np.max(np.abs(recording))
            if audio_level > 0.001:
                console.print(f"[green]✓[/green] Microphone working! Peak level: {audio_level:.3f}")
            else:
                console.print(f"[yellow]⚠[/yellow] Microphone very quiet. Peak level: {audio_level:.3f}")
        except Exception as e:
            console.print(f"[red]❌[/red] Microphone test failed: {e}")
        
        # Test speakers with a simple tone
        if output_device is not None:
            sd.default.device[1] = output_device
            
        console.print("🔊 Testing speakers (1 second tone)...")
        try:
            # Generate a simple 440Hz tone
            duration = 1
            sample_rate = 24000
            t = np.linspace(0, duration, int(sample_rate * duration))
            tone = 0.1 * np.sin(2 * np.pi * 440 * t)  # Quiet 440Hz tone
            
            sd.play(tone, samplerate=sample_rate, device=output_device)
            sd.wait()
            console.print("[green]✓[/green] Speaker test complete")
        except Exception as e:
            console.print(f"[red]❌[/red] Speaker test failed: {e}")
            
        console.print()
        
    except ImportError:
        console.print("[red]❌[/red] sounddevice not available for testing")


def create_simple_tools() -> List[ToolDefinition]:
    """Create simple test tools without robot functionality."""
    
    def echo_message(message: str) -> str:
        """Simple echo tool for testing."""
        return f"Echo: {message}"
    
    def get_time() -> str:
        """Get current time as string."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers for testing."""
        return a + b
    
    def calculate_multiply(a: float, b: float) -> float:
        """Multiply two numbers for testing."""
        return a * b
    
    def get_weather_info(location: str = "unknown") -> str:
        """Mock weather function for testing."""
        import random
        conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy"]
        temp = random.randint(15, 30)
        condition = random.choice(conditions)
        return f"Weather in {location}: {condition}, {temp}°C"
    
    tools = [
        ToolDefinition(
            name="echo_message",
            description="Echo back a message",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo back"}
                },
                "required": ["message"]
            },
            execute=echo_message
        ),
        ToolDefinition(
            name="get_time",
            description="Get current time",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            execute=get_time
        ),
        ToolDefinition(
            name="add_numbers",
            description="Add two numbers together",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            execute=add_numbers
        ),
        ToolDefinition(
            name="calculate_multiply",
            description="Multiply two numbers together",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            execute=calculate_multiply
        ),
        ToolDefinition(
            name="get_weather_info",
            description="Get weather information for a location (mock data)",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location to get weather for"}
                },
                "required": ["location"]
            },
            execute=get_weather_info
        )
    ]
    
    return tools


def create_test_agent() -> RealtimeAgent:
    """Create a test agent with simple conversation tools."""
    instructions = (
        "You are a helpful, friendly AI assistant for testing conversation capabilities. "
        "You have access to several simple tools that you can use to help users: "
        "- echo_message: to repeat back messages "
        "- get_time: to tell the current time "
        "- add_numbers: to add two numbers together "
        "- calculate_multiply: to multiply two numbers "
        "- get_weather_info: to provide mock weather information for any location "
        "\n"
        "You should be conversational, helpful, and engaging. "
        "Use tools when appropriate to answer user questions. "
        "Keep responses natural and friendly. "
        "This is a test environment, so feel free to be creative and helpful!"
    )
    
    tools = create_simple_tools()
    
    # Create a pure conversation agent without robot dependencies
    agent = RealtimeAgent(
        name="ConversationTestAgent",
        instructions=instructions,
        tools=tools,
        voice="alloy",
        temperature=0.8,  # Slightly more creative
        model="gpt-4o-mini-realtime-preview-2024-12-17"
    )
    
    return agent


async def interactive_conversation():
    """Start an interactive conversation with the OpenAI agent."""
    console.print(Panel("[bold cyan]🤖 Interactive OpenAI Conversation[/bold cyan]", expand=False))
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]✗[/red] OPENAI_API_KEY not found in environment variables.")
        console.print("[yellow]💡[/yellow] Please set your OpenAI API key:")
        console.print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Audio device setup for voice modes
    selected_input_device = None
    selected_output_device = None
    
    console.print("\n[cyan]First, let's set up your audio devices:[/cyan]")
    console.print("Would you like to select specific microphone/speakers? (y/N)")
    
    try:
        setup_audio = input("Setup audio devices? ").strip().lower()
        if setup_audio in ['y', 'yes']:
            selected_input_device, selected_output_device = select_audio_devices()
            
            # Test the selected devices
            console.print("\n[cyan]Would you like to test the audio devices? (Y/n)[/cyan]")
            test_choice = input("Test devices? ").strip().lower()
            if test_choice not in ['n', 'no']:
                test_audio_devices(selected_input_device, selected_output_device)
        else:
            console.print("[yellow]Using system default audio devices[/yellow]")
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Using system default audio devices[/yellow]")
    
    # Ask user for interaction mode
    console.print("\n[cyan]Choose interaction mode:[/cyan]")
    console.print("1. 🎤 Voice conversation (speak and listen)")
    console.print("2. 💬 Text conversation (type messages)")
    console.print("3. 🎤💬 Hybrid mode (both voice and text)")
    
    try:
        mode_choice = input("\nEnter your choice (1/2/3): ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]👋[/yellow] Goodbye!")
        return
    
    # Get base config with debug settings
    base_config = get_config()
    
    # Configure based on user choice
    if mode_choice == "1":
        # Voice-only mode
        config = VoiceAgentConfig(
            text_only_mode=False,
            debug_mode=True,  # Enable debug mode to see what's happening
            enable_text_input=False,
            feedback_prevention_enabled=True,
            threaded_execution=False
        )
        console.print("[green]🎤[/green] Voice mode selected!")
        console.print("[dim]Speak naturally - the agent will respond with voice[/dim]")
    elif mode_choice == "3":
        # Hybrid mode
        config = VoiceAgentConfig(
            text_only_mode=False,
            debug_mode=True,  # Enable debug mode to see what's happening
            enable_text_input=True,
            feedback_prevention_enabled=True,
            threaded_execution=False
        )
        console.print("[green]🎤💬[/green] Hybrid mode selected!")
        console.print("[dim]You can both speak AND type messages[/dim]")
    else:
        # Text-only mode (default)
        config = VoiceAgentConfig(
            text_only_mode=True,
            debug_mode=True,  # Enable debug mode to see what's happening
            enable_text_input=True,
            feedback_prevention_enabled=True,
            threaded_execution=False
        )
        console.print("[green]💬[/green] Text mode selected!")
        console.print("[dim]Type your messages and press Enter[/dim]")
    
    try:
        # Create agent and session
        agent = create_test_agent()
        
        # Store selected audio devices in config for later use
        if hasattr(config, 'selected_input_device'):
            config.selected_input_device = selected_input_device
            config.selected_output_device = selected_output_device
        else:
            # Add as custom attributes
            config.selected_input_device = selected_input_device
            config.selected_output_device = selected_output_device
        
        # Create session WITHOUT robot bridge to avoid robot connections
        session = RealtimeSession(agent, api_key, bridge_ref=None, config=config)
        
        console.print("[green]✓[/green] Agent created successfully!")
        console.print("[blue]Available tools:[/blue] echo_message, get_time, add_numbers, calculate_multiply, get_weather_info")
        
        # Show selected audio devices
        if selected_input_device is not None or selected_output_device is not None:
            console.print("[cyan]🎧 Audio devices configured:[/cyan]")
            if selected_input_device is not None:
                console.print(f"   🎤 Microphone: Device {selected_input_device}")
            if selected_output_device is not None:
                console.print(f"   🔊 Speakers: Device {selected_output_device}")
        
        if config.text_only_mode:
            console.print("[cyan]💬[/cyan] Start typing! Type 'quit', 'exit', or 'bye' to end.")
        elif config.enable_text_input:
            console.print("[cyan]🎤💬[/cyan] Start speaking OR typing! Say/type 'quit' to end.")
        else:
            console.print("[cyan]🎤[/cyan] Start speaking! Say 'quit' or press Ctrl+C to end.")
            
        console.print("[dim]Example: 'What time is it?', 'Add 5 and 3', 'What's the weather in Paris?'[/dim]")
        console.print("[yellow]💡 Voice Tip: Pause for 0.5 seconds after speaking for auto-response[/yellow]")
        if config.enable_text_input:
            console.print("[yellow]🔄 Manual Trigger: Type 'trigger' if voice response doesn't auto-start[/yellow]")
        console.print("-" * 60)
        
        # Connect to OpenAI
        console.print("[yellow]Connecting to OpenAI Realtime API...[/yellow]")
        await session.connect()
        
        # Start the session with proper error handling
        session_task = None
        try:
            session_task = asyncio.create_task(session.start_session_with_connection())
            
            # Give the session a moment to initialize
            await asyncio.sleep(3)
            
            if not session.is_connected:
                console.print("[red]❌[/red] Failed to establish connection to OpenAI")
                return
                
            console.print("[green]✅[/green] Connected! Ready to chat.")
            
            if not config.text_only_mode:
                console.print("[cyan]🎤[/cyan] Voice input is active - start speaking!")
                console.print("[dim]The agent will respond with voice automatically[/dim]")
            
            # For voice-only mode, just wait for the session to handle everything
            if not config.enable_text_input:
                console.print("\n[yellow]Press Ctrl+C to end the conversation[/yellow]")
                try:
                    # Keep the session running for voice input
                    while session.is_connected:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    console.print("\n[yellow]👋[/yellow] Conversation ended by user.")
            else:
                # Interactive conversation loop with text input for hybrid/text modes
                conversation_active = True
                while conversation_active and session.is_connected:
                    try:
                        # Use asyncio for input to avoid blocking
                        loop = asyncio.get_event_loop()
                        
                        # Prompt for input
                        if config.text_only_mode:
                            console.print("\n[bold cyan]💬 You:[/bold cyan] ", end="")
                        else:
                            console.print("\n[bold cyan]💬 Type (or speak):[/bold cyan] ", end="")
                        
                        # Get user input with timeout
                        try:
                            user_input = await asyncio.wait_for(
                                loop.run_in_executor(None, input),
                                timeout=60.0  # 60 second timeout
                            )
                            user_input = user_input.strip()
                        except asyncio.TimeoutError:
                            console.print("\n[yellow]⏰[/yellow] Input timeout. Type something to continue...")
                            continue
                        except EOFError:
                            console.print("\n[yellow]👋[/yellow] Input stream closed. Ending conversation.")
                            break
                        
                        if not user_input:
                            continue
                            
                        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye', 'stop']:
                            console.print("[yellow]👋[/yellow] Goodbye!")
                            conversation_active = False
                            break
                        
                        if user_input.lower() == 'trigger':
                            # Manual trigger for voice response
                            console.print("[cyan]🔄[/cyan] Manually triggering response from current audio...")
                            try:
                                await session.connection.send({"type": "response.create"})
                                console.print("[green]✓[/green] Response triggered!")
                            except Exception as e:
                                console.print(f"[red]❌[/red] Failed to trigger response: {e}")
                            continue
                        
                        # Send message to the agent
                        console.print("[dim]Sending message to assistant...[/dim]")
                        await session.send_message(user_input)
                        
                        # Give time for response to be processed
                        await asyncio.sleep(2)
                        
                    except KeyboardInterrupt:
                        console.print("\n[yellow]👋[/yellow] Conversation interrupted by user.")
                        conversation_active = False
                        break
                    except Exception as e:
                        console.print(f"[red]❌[/red] Error during conversation: {e}")
                        if "EOF" in str(e) or "input" in str(e).lower():
                            console.print("[yellow]Input stream issue. Try again or type 'quit' to exit.[/yellow]")
                            continue
                        else:
                            break
            
        except Exception as e:
            console.print(f"[red]❌[/red] Session error: {e}")
        finally:
            # Clean up properly
            console.print("\n[yellow]Cleaning up...[/yellow]")
            
            if session_task and not session_task.done():
                session_task.cancel()
                try:
                    await asyncio.wait_for(session_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            
            # Close session safely
            try:
                if session and session.is_connected:
                    await session.close()
            except Exception as e:
                console.print(f"[dim]Cleanup warning: {e}[/dim]")
                
        console.print("[green]✅[/green] Conversation ended successfully!")
        
    except Exception as e:
        console.print(f"[red]❌[/red] Failed to start conversation: {e}")
        import traceback
        if config.debug_mode:
            traceback.print_exc()


async def test_openai_conversation():
    """Test basic OpenAI conversation functionality."""
    console.print("[cyan]Testing OpenAI conversation functionality...[/cyan]")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]⚠[/yellow] OPENAI_API_KEY not found. Skipping integration test.")
        return False
    
    # Get base config with debug settings
    base_config = get_config()
    
    try:
        config = VoiceAgentConfig(
            text_only_mode=True,
            debug_mode=base_config.debug_mode,
            enable_text_input=True,
            feedback_prevention_enabled=False,
            threaded_execution=False
        )
        
        agent = create_test_agent()
        session = RealtimeSession(agent, api_key, bridge_ref=None, config=config)
        
        console.print("[green]✓[/green] Agent and session created successfully")
        
        # Test session configuration
        session_config = agent.get_session_config()
        console.print(f"[blue]Session config keys:[/blue] {list(session_config.keys())}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] OpenAI conversation test failed: {e}")
        import traceback
        if base_config.debug_mode:
            traceback.print_exc()
        return False


async def test_tool_functionality():
    """Test tool functionality without OpenAI."""
    console.print("[cyan]Testing tool functionality...[/cyan]")
    
    try:
        tools = create_simple_tools()
        
        # Test echo tool
        echo_result = tools[0].execute(message="Test message")
        console.print(f"[green]✓[/green] Echo tool: {echo_result}")
        
        # Test add numbers tool
        add_result = tools[2].execute(a=10, b=5)
        console.print(f"[green]✓[/green] Add numbers tool: {add_result}")
        
        # Test multiply tool
        multiply_result = tools[3].execute(a=4, b=7)
        console.print(f"[green]✓[/green] Multiply tool: {multiply_result}")
        
        # Test get time tool
        time_result = tools[1].execute()
        console.print(f"[green]✓[/green] Get time tool: {time_result}")
        
        # Test weather tool
        weather_result = tools[4].execute(location="New York")
        console.print(f"[green]✓[/green] Weather tool: {weather_result}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] Tool functionality test failed: {e}")
        return False


async def test_agent_creation():
    """Test agent creation and configuration."""
    console.print("[cyan]Testing agent creation...[/cyan]")
    
    try:
        agent = create_test_agent()
        
        # Test basic properties
        assert agent.name == "ConversationTestAgent"
        assert "helpful, friendly AI assistant" in agent.instructions
        assert len(agent.tools) == 5  # Updated for new tools
        assert agent.voice == "alloy"
        assert agent.temperature == 0.8
        
        # Test tool mapping
        expected_tools = ["echo_message", "get_time", "add_numbers", "calculate_multiply", "get_weather_info"]
        for tool_name in expected_tools:
            assert tool_name in agent._tool_map
        
        # Test session config
        config = agent.get_session_config()
        assert config["modalities"] == ["audio", "text"]
        assert "helpful, friendly AI assistant" in config["instructions"]
        assert config["voice"] == "alloy"
        assert len(config["tools"]) == 5  # Updated for new tools
        assert "turn_detection" in config
        
        console.print("[green]✓[/green] Agent creation tests passed")
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] Agent creation test failed: {e}")
        return False


async def test_session_creation():
    """Test session creation without connecting to OpenAI."""
    console.print("[cyan]Testing session creation...[/cyan]")
    
    # Get base config with debug settings
    base_config = get_config()
    
    try:
        config = VoiceAgentConfig(
            text_only_mode=True,
            debug_mode=base_config.debug_mode,
            enable_text_input=True,
            feedback_prevention_enabled=False,
            threaded_execution=False
        )
        
        agent = create_test_agent()
        session = RealtimeSession(agent, TEST_API_KEY, bridge_ref=None, config=config)
        
        # Test session properties
        assert session.agent == agent
        assert session.api_key == TEST_API_KEY
        assert session.session_id is not None
        assert session.is_connected == False
        
        console.print("[green]✓[/green] Session creation tests passed")
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] Session creation test failed: {e}")
        return False


async def test_tool_execution():
    """Test tool execution without OpenAI connection."""
    console.print("[cyan]Testing tool execution...[/cyan]")
    
    # Get base config with debug settings
    base_config = get_config()
    
    try:
        config = VoiceAgentConfig(
            text_only_mode=True,
            debug_mode=base_config.debug_mode,
            enable_text_input=True,
            feedback_prevention_enabled=False,
            threaded_execution=False
        )
        
        agent = create_test_agent()
        session = RealtimeSession(agent, TEST_API_KEY, bridge_ref=None, config=config)
        
        # Test echo tool
        echo_tool = agent._tool_map["echo_message"]
        result = await session._execute_tool(echo_tool, {"message": "Hello World"})
        assert result == "Echo: Hello World"
        
        # Test add numbers tool
        add_tool = agent._tool_map["add_numbers"]
        result = await session._execute_tool(add_tool, {"a": 5, "b": 3})
        assert result == 8
        
        # Test multiply tool
        multiply_tool = agent._tool_map["calculate_multiply"]
        result = await session._execute_tool(multiply_tool, {"a": 4, "b": 6})
        assert result == 24
        
        # Test get time tool
        time_tool = agent._tool_map["get_time"]
        result = await session._execute_tool(time_tool, {})
        assert isinstance(result, str)
        assert "T" in result  # ISO format should contain 'T'
        
        # Test weather tool
        weather_tool = agent._tool_map["get_weather_info"]
        result = await session._execute_tool(weather_tool, {"location": "Tokyo"})
        assert isinstance(result, str)
        assert "Tokyo" in result
        
        console.print("[green]✓[/green] Tool execution tests passed")
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] Tool execution test failed: {e}")
        return False


async def run_conversation_tests():
    """Run all conversation tests."""
    console.print(Panel("[bold cyan]OpenAI Agent Conversation Test Suite[/bold cyan]", expand=False))
    
    tests = [
        ("Agent Creation Test", test_agent_creation()),
        ("Session Creation Test", test_session_creation()),
        ("Tool Functionality Test", test_tool_functionality()),
        ("Tool Execution Test", test_tool_execution()),
        ("OpenAI Conversation Test", test_openai_conversation())
    ]
    
    results = []
    for test_name, test_coro in tests:
        console.print(f"\n[bold yellow]--- {test_name} ---[/bold yellow]")
        result = await test_coro
        results.append((test_name, result))
    
    # Show results
    console.print(Panel("[bold cyan]Test Results[/bold cyan]", expand=False))
    for test_name, result in results:
        status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        console.print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    overall_status = "[green]ALL TESTS PASSED[/green]" if all_passed else "[red]SOME TESTS FAILED[/red]"
    console.print(f"\n[bold]Overall: {overall_status}[/bold]")


def main():
    """Main function with menu options."""
    console.print(Panel("[bold cyan]🤖 OpenAI Conversation Test & Chat[/bold cyan]", expand=False))
    console.print("[cyan]Choose an option:[/cyan]")
    console.print("1. 💬 Interactive Conversation (with voice option)")
    console.print("2. 🧪 Run Test Suite")
    console.print("3. 🔧 Quick Tool Test")
    console.print("q. Quit")
    
    choice = input("\nEnter your choice (1/2/3/q): ").strip().lower()
    
    if choice == "1":
        asyncio.run(interactive_conversation())
    elif choice == "2":
        asyncio.run(run_conversation_tests())
    elif choice == "3":
        asyncio.run(test_tool_functionality())
    elif choice == "q":
        console.print("[yellow]👋[/yellow] Goodbye!")
    else:
        console.print("[red]Invalid choice. Please try again.[/red]")
        main()


if __name__ == "__main__":
    main()
