"""
UR Robot Control Module
======================

Modular robot control system for Universal Robots with LLM integration.

This module provides:
- Core robot connection and control (ur.core)
- Automatic tool discovery for LLM agents (ur.tools) 
- Configuration management (ur.config)
- Control primitives (ur.control)
- Examples and tests (ur.examples, ur.tests)
"""

# Core functionality
from .core import URConnection
# from .bridge import URBridge  # Bridge has dependencies, skip for now
from .config import ROBOT_IP, ROBOT_MOVE_SPEED, WAKEWORD, HOME_POSITION

# Tool registry for LLM agents
from .tools import get_tool_registry, register_tools_for_openai, register_tools_for_smolagents

__version__ = "1.0.0"

__all__ = [
    # Core classes
    'URConnection',
    # 'URBridge',
    
    # Configuration
    'ROBOT_IP',
    'ROBOT_MOVE_SPEED', 
    'WAKEWORD',
    'HOME_POSITION',
    
    # Tool registry
    'get_tool_registry',
    'register_tools_for_openai',
    'register_tools_for_smolagents'
]
