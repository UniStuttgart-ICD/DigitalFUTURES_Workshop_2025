"""
UR Tools Module
==============

LLM-compatible tool functions with automatic discovery and registration.
"""

from .registry import (
    get_tool_registry,
    register_tools_for_openai, 
    register_tools_for_smolagents
)

# Auto-discover tools on import
_registry = get_tool_registry()

__all__ = [
    'get_tool_registry',
    'register_tools_for_openai',
    'register_tools_for_smolagents'
] 