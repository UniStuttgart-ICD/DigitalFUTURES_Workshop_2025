"""Test OpenAI Realtime API agent for pure conversation without robot functionality.

This test suite validates the OpenAI agent's conversation capabilities,
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
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the OpenAI agent components
from ur.agents.openai_agent import (
    RealtimeAgent, 
    RealtimeSession, 
    ToolDefinition, 
    EventType,
    RealtimeEvent,
    OpenAIVoiceAgent
)
from ur.config.voice_config import VoiceAgentConfig, load_config_from_env

console = Console()

# Mock API key for testing
TEST_API_KEY = "sk-test-dummy-key-for-testing"


@dataclass
class MockEvent:
    """Mock event for testing event processing."""
    type: str
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
    
    def model_dump(self):
        return {"type": self.type, **self.data}


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
        )
    ]
    
    return tools


def create_test_agent() -> RealtimeAgent:
    """Create a test agent with simple conversation tools."""
    instructions = (
        "You are a helpful assistant for testing conversation capabilities. "
        "You can echo messages, get the current time, and add numbers. "
        "Keep responses friendly and concise for testing purposes."
    )
    
    tools = create_simple_tools()
    
    agent = RealtimeAgent(
        name="TestAgent",
        instructions=instructions,
        tools=tools,
        voice="alloy",
        temperature=0.7,
        model="gpt-4o-mini-realtime-preview-2024-12-17"
    )
    
    return agent


class TestRealtimeAgent(unittest.TestCase):
    """Test RealtimeAgent creation and configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = create_test_agent()
    
    def test_agent_creation(self):
        """Test agent is created with correct properties."""
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertIn("helpful assistant", self.agent.instructions)
        self.assertEqual(len(self.agent.tools), 3)
        self.assertEqual(self.agent.voice, "alloy")
        self.assertEqual(self.agent.temperature, 0.7)
    
    def test_tool_mapping(self):
        """Test tools are correctly mapped."""
        expected_tools = ["echo_message", "get_time", "add_numbers"]
        for tool_name in expected_tools:
            self.assertIn(tool_name, self.agent._tool_map)
    
    def test_session_config(self):
        """Test session configuration generation."""
        config = self.agent.get_session_config()
        
        self.assertEqual(config["modalities"], ["audio", "text"])
        self.assertIn("helpful assistant", config["instructions"])
        self.assertEqual(config["voice"], "alloy")
        self.assertEqual(len(config["tools"]), 3)
        self.assertIn("turn_detection", config)
    
    def test_tool_conversion(self):
        """Test tool conversion to OpenAI specification."""
        openai_spec = self.agent._convert_tool_to_openai_spec(self.agent.tools[0])
        
        self.assertEqual(openai_spec["type"], "function")
        self.assertEqual(openai_spec["name"], "echo_message")
        self.assertIn("description", openai_spec)
        self.assertIn("parameters", openai_spec)


class TestRealtimeSession(unittest.TestCase):
    """Test RealtimeSession functionality with mocked OpenAI client."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = create_test_agent()
        self.config = VoiceAgentConfig(
            text_only_mode=True,
            debug_mode=True,
            enable_text_input=True,
            feedback_prevention_enabled=False,
            threaded_execution=False
        )
        
    @patch('ur.agents.openai_agent.openai.AsyncOpenAI')
    def test_session_creation(self, mock_openai):
        """Test session creation with mocked OpenAI client."""
        session = RealtimeSession(
            self.agent, 
            TEST_API_KEY, 
            bridge_ref=None, 
            config=self.config
        )
        
        self.assertEqual(session.agent, self.agent)
        self.assertEqual(session.api_key, TEST_API_KEY)
        self.assertIsNotNone(session.session_id)
        self.assertFalse(session.is_connected)
    
    @patch('ur.agents.openai_agent.openai.AsyncOpenAI')
    def test_event_handling(self, mock_openai):
        """Test event emission and handling."""
        session = RealtimeSession(
            self.agent, 
            TEST_API_KEY, 
            bridge_ref=None, 
            config=self.config
        )
        
        # Test event handler registration
        events_received = []
        
        def test_handler(event: RealtimeEvent):
            events_received.append(event)
        
        session.on(EventType.SESSION_CREATED, test_handler)
        
        # Emit test event
        session._emit_event(EventType.SESSION_CREATED, {"test": "data"})
        
        # Verify event was received
        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0].type, EventType.SESSION_CREATED)
        self.assertEqual(events_received[0].data["test"], "data")


class TestConversationFlow(unittest.TestCase):
    """Test conversation flow and tool execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = create_test_agent()
        self.config = VoiceAgentConfig(
            text_only_mode=True,
            debug_mode=True,
            enable_text_input=True,
            feedback_prevention_enabled=False,
            threaded_execution=False
        )
    
    async def test_tool_execution(self):
        """Test tool execution without OpenAI connection."""
        session = RealtimeSession(
            self.agent, 
            TEST_API_KEY, 
            bridge_ref=None, 
            config=self.config
        )
        
        # Test echo tool
        echo_tool = self.agent._tool_map["echo_message"]
        result = await session._execute_tool(echo_tool, {"message": "Hello World"})
        self.assertEqual(result, "Echo: Hello World")
        
        # Test add numbers tool
        add_tool = self.agent._tool_map["add_numbers"]
        result = await session._execute_tool(add_tool, {"a": 5, "b": 3})
        self.assertEqual(result, 8)
        
        # Test get time tool
        time_tool = self.agent._tool_map["get_time"]
        result = await session._execute_tool(time_tool, {})
        self.assertIsInstance(result, str)
        self.assertIn("T", result)  # ISO format should contain 'T'
    
    async def test_event_processing(self):
        """Test processing of various event types."""
        session = RealtimeSession(
            self.agent, 
            TEST_API_KEY, 
            bridge_ref=None, 
            config=self.config
        )
        
        # Test session created event
        event = MockEvent("session.created", {"session": {"id": "test"}})
        await session._process_server_event(event)
        
        # Test speech started event
        event = MockEvent("input_audio_buffer.speech_started")
        await session._process_server_event(event)
        self.assertTrue(session.status.is_listening)
        
        # Test speech ended event
        event = MockEvent("input_audio_buffer.speech_stopped")
        await session._process_server_event(event)
        self.assertFalse(session.status.is_listening)


async def test_openai_conversation():
    """Test basic OpenAI conversation functionality (integration test)."""
    console.print("[cyan]Testing OpenAI conversation functionality...[/cyan]")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]⚠[/yellow] OPENAI_API_KEY not found. Skipping integration test.")
        return False
    
    try:
        config = VoiceAgentConfig(
            text_only_mode=True,
            debug_mode=False,
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
        
        # Test get time tool
        time_result = tools[1].execute()
        console.print(f"[green]✓[/green] Get time tool: {time_result}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] Tool functionality test failed: {e}")
        return False


async def run_conversation_tests():
    """Run all conversation tests."""
    console.print(Panel("[bold cyan]OpenAI Agent Conversation Test Suite[/bold cyan]", expand=False))
    
    tests = [
        ("Tool Functionality Test", test_tool_functionality()),
        ("OpenAI Conversation Test", test_openai_conversation())
    ]
    
    results = []
    for test_name, test_coro in tests:
        console.print(f"\n[bold yellow]--- {test_name} ---[/bold yellow]")
        result = await test_coro
        results.append((test_name, result))
    
    # Run unit tests
    console.print(f"\n[bold yellow]--- Unit Tests ---[/bold yellow]")
    unittest_result = True
    try:
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        suite.addTests(loader.loadTestsFromTestCase(TestRealtimeAgent))
        suite.addTests(loader.loadTestsFromTestCase(TestRealtimeSession))
        suite.addTests(loader.loadTestsFromTestCase(TestConversationFlow))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        test_result = runner.run(suite)
        unittest_result = test_result.wasSuccessful()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Unit tests failed: {e}")
        unittest_result = False
    
    results.append(("Unit Tests", unittest_result))
    
    # Show results
    console.print(Panel("[bold cyan]Test Results[/bold cyan]", expand=False))
    for test_name, result in results:
        status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        console.print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    overall_status = "[green]ALL TESTS PASSED[/green]" if all_passed else "[red]SOME TESTS FAILED[/red]"
    console.print(f"\n[bold]Overall: {overall_status}[/bold]")


if __name__ == "__main__":
    # Run async tests
    asyncio.run(run_conversation_tests()) 