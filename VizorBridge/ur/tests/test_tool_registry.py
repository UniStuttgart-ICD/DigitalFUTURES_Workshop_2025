"""
Test Tool Registry
=================

Test script to verify automatic tool discovery and registration.
"""

import sys, os
# Add project root to path so `import ur` works when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ur.tools.registry import get_tool_registry, register_tools_for_openai, register_tools_for_smolagents
from ur.utils import validate_wakeword, send_immediate_response

def test_tool_discovery():
    """Test that tools are automatically discovered."""
    registry = get_tool_registry()
    
    print("🔍 Discovered Tools:")
    print(f"   Total tools found: {len(registry.tools)}")
    
    for tool_name, tool_func in registry.tools.items():
        print(f"   - {tool_name}: {tool_func.__doc__.split('.')[0] if tool_func.__doc__ else 'No description'}")
    
    print(f"\n📊 OpenAI function specs: {len(registry.openai_specs)}")
    print(f"📊 Smolagents tools: {len(registry.smolagents_tools)}")

def test_utils():
    """Test shared utility functions in ur.utils."""
    print("\n🛠 Testing ur.utils:")
    print(f"  validate_wakeword('mave test'): {validate_wakeword('mave test')}")
    print(f"  validate_wakeword('no'): {validate_wakeword('no')}")
    print("  send_immediate_response: ", end='')
    send_immediate_response("test command")

def test_openai_integration():
    """Test OpenAI function calling integration."""
    print("\n🤖 Testing OpenAI Integration:")
    tools, function_map = register_tools_for_openai()
    
    print(f"   Function specs: {len(tools)}")
    print(f"   Function map: {len(function_map)}")
    
    if tools:
        print("   Sample function spec:")
        sample_tool = tools[0]
        print(f"   - Name: {sample_tool['function']['name']}")
        print(f"   - Description: {sample_tool['function']['description'][:100]}...")

def test_smolagents_integration():
    """Test smolagents integration."""
    print("\n🔧 Testing SmoLAgents Integration:")
    tools = register_tools_for_smolagents()
    
    print(f"   Tools: {len(tools)}")
    
    if tools:
        print("   Sample tool:")
        sample_tool = tools[0]
        # Name attribute should exist
        name = getattr(sample_tool, 'name', repr(sample_tool))
        print(f"   - Name: {name}")
        # Description may not be present
        desc = getattr(sample_tool, 'description', None)
        if isinstance(desc, str) and desc:
            trimmed = desc[:100] + ('...' if len(desc) > 100 else '')
            print(f"   - Description: {trimmed}")
        else:
            print("   - Description: No description")

def test_counts_consistency():
    """Verify that all tool counts are consistent."""
    registry = get_tool_registry()
    openai_tools, _ = register_tools_for_openai()
    smol_tools = register_tools_for_smolagents()
    total = len(registry.tools)
    openai_count = len(openai_tools)
    smol_count = len(smol_tools)
    assert total == openai_count == smol_count, (
        f"Tool counts mismatch: registry={total}, openai={openai_count}, smol={smol_count}"
    )
    print(f"\n🔢 Tool counts consistent: {total} each")

if __name__ == "__main__":
    print("🧪 Testing Automatic Tool Registry System")
    print("=" * 50)
    
    try:
        test_tool_discovery()
        test_openai_integration()  
        test_smolagents_integration()
        test_counts_consistency()
        test_utils()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 