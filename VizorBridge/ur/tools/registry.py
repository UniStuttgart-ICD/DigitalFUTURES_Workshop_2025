"""
Automatic Tool Discovery and Registry
====================================

Automatically discovers and registers all @tool decorated functions
from tool modules for use with LLM agents.
"""

import importlib
import inspect
import pkgutil
from typing import Dict, List, Any, Callable
from pathlib import Path

class ToolRegistry:
    """Automatic discovery and registration of LLM tools."""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.openai_specs: List[Dict[str, Any]] = []
        self.smolagents_tools: List[Any] = []
        
    def discover_tools(self, package_name: str = "ur.tools"):
        """Automatically discover all @tool decorated functions."""
        try:
            package = importlib.import_module(package_name)
            package_path = Path(package.__file__).parent
            
            # Find all Python modules in the tools package
            for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
                if module_name.startswith('_') or module_name == 'registry':  # Skip private modules and self
                    continue
                    
                try:
                    module = importlib.import_module(f"{package_name}.{module_name}")
                    self._extract_tools_from_module(module)
                except ImportError as e:
                    print(f"Warning: Could not import {module_name}: {e}")
        except ImportError:
            print(f"Warning: Could not import package {package_name}")
    
    def _extract_tools_from_module(self, module):
        """Extract @tool decorated functions from a module."""
        for name, obj in inspect.getmembers(module):
            # Only register objects decorated as tools (must have 'name' and 'inputs' attributes)
            if hasattr(obj, 'name') and hasattr(obj, 'inputs'):
                # Use the description attribute as the docstring if available
                if hasattr(obj, 'description') and isinstance(obj.description, str):
                    obj.__doc__ = obj.description
                # Register the tool
                self.tools[obj.name] = obj
                self.smolagents_tools.append(obj)
                # Always generate an OpenAI spec if inputs exist (description optional)
                openai_spec = self._generate_openai_spec(obj)
                self.openai_specs.append(openai_spec)
    
    def _generate_openai_spec(self, tool_func) -> Dict[str, Any]:
        """Generate OpenAI function calling spec from smolagents tool."""
        # Determine underlying function or fallback to tool object for signature
        base_func = getattr(tool_func, '__wrapped__', tool_func)
        sig = inspect.signature(base_func)
        required_params = []
        
        for param_name, param in sig.parameters.items():
            if param.default == param.empty:
                required_params.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": tool_func.name,
                "description": tool_func.description,
                "parameters": {
                    "type": "object",
                    "properties": tool_func.inputs,
                    "required": required_params
                }
            }
        }
    
    def get_tools_for_agent(self, agent_type: str) -> List[Any]:
        """Get tools formatted for specific agent type."""
        if agent_type == "openai":
            return self.openai_specs
        elif agent_type == "smolagents":
            # Return unique tool objects same as registry.tools
            return list(self.tools.values())
        else:
            return list(self.tools.values())
    
    def get_function_map(self) -> Dict[str, Callable]:
        """Get mapping of function names to callable functions."""
        return self.tools.copy()
    
    def refresh(self):
        """Refresh the tool registry by re-discovering tools."""
        self.tools.clear()
        self.openai_specs.clear()
        self.smolagents_tools.clear()
        self.discover_tools()

# Global registry instance
_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _registry.discover_tools()
    return _registry

def register_tools_for_openai() -> tuple[List[Dict], Dict[str, Callable]]:
    """Get tools and function map for OpenAI agents."""
    registry = get_tool_registry()
    return registry.get_tools_for_agent("openai"), registry.get_function_map()

def register_tools_for_smolagents() -> List[Any]:
    """Get tools for smolagents."""
    registry = get_tool_registry()
    return registry.get_tools_for_agent("smolagents")

def refresh_tools():
    """Refresh all discovered tools (useful for development)."""
    global _registry
    if _registry:
        _registry.refresh()
    else:
        get_tool_registry() 