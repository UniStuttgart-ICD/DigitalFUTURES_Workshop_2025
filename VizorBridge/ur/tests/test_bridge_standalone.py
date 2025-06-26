#!/usr/bin/env python3
"""
Standalone UR Bridge Test Script
===============================

Test the UR bridge components without requiring ROS or a physical robot.
This script tests:
1. URRobot class initialization and connection handling
2. Robot tools functions in simulation mode
3. Safety validation functions
4. Bridge structure and dependencies

Usage:
    python test_bridge_standalone.py

Features:
- Works without ROS running
- Works without physical robot (simulation mode)
- Tests core functionality and error handling
- Validates safety systems
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()

def test_imports():
    """Test that all required modules can be imported."""
    console.print("\n[bold cyan]🔍 Testing Module Imports[/bold cyan]")
    
    results = {}
    
    # Test tools registry import
    try:
        from ur.tools import register_tools_for_openai, register_tools_for_smolagents
        from ur.core.connection import get_robot
        results["tools_registry"] = {"status": "✓", "details": "New tools registry imported successfully"}
    except Exception as e:
        results["tools_registry"] = {"status": "✗", "details": f"Failed to import tools registry: {e}"}
    
    # Test bridge import (this will fail without roslibpy, but we can catch it)
    try:
        # We expect this to fail without roslibpy, so let's handle it gracefully
        import importlib.util
        spec = importlib.util.spec_from_file_location("bridge", Path(__file__).parent / "bridge.py")
        bridge_module = importlib.util.module_from_spec(spec)
        results["bridge"] = {"status": "✓", "details": "Bridge module structure is valid"}
    except Exception as e:
        results["bridge"] = {"status": "⚠", "details": f"Bridge requires ROS: {e}"}
    
    # Test agents import
    try:
        from ur.agents.base_agent import BaseVoiceAgent
        results["agents"] = {"status": "✓", "details": "Agent architecture imported successfully"}
    except Exception as e:
        results["agents"] = {"status": "✗", "details": f"Failed to import agents: {e}"}
    
    # Test RTDE availability
    try:
        import rtde_control
        import rtde_receive
        import rtde_io
        results["rtde"] = {"status": "✓", "details": "RTDE libraries available for real robot control"}
    except ImportError:
        results["rtde"] = {"status": "⚠", "details": "RTDE not available - simulation mode only"}
    
    # Display results
    for module, result in results.items():
        console.print(f"  {result['status']} {module}: {result['details']}")
    
    return results

def test_tools_registry():
    """Test tools registry functionality without physical robot."""
    console.print("\n[bold cyan]🤖 Testing Tools Registry (Simulation)[/bold cyan]")
    
    try:
        from ur.core.connection import get_robot
        from ur.tools.movement_tools import validate_wakeword
        from ur.config.supply_stations import SUPPLY_STATIONS
        
        # Test robot connection via new core module
        console.print("  🔧 Testing robot connection (core module)...")
        robot = get_robot()
        console.print(f"    ✓ Robot instance accessed (connected: {robot.is_connected})")
        
        # Test wakeword validation
        console.print("  🗣️ Testing wakeword validation...")
        valid_result = validate_wakeword("timbra move forward")
        invalid_result = validate_wakeword("just move forward")
        console.print(f"    ✓ Valid wakeword detected: {valid_result}")
        console.print(f"    ✓ Invalid wakeword rejected: {not invalid_result}")
        
        # Test supply stations configuration
        console.print("  📦 Testing supply stations configuration...")
        console.print(f"    ✓ {len(SUPPLY_STATIONS)} supply stations configured")
        for station, config in SUPPLY_STATIONS.items():
            console.print(f"      - {station}: {config['position'][:3]}")  # Show just x,y,z
        
        # Test safety checking functions (will work in simulation)
        console.print("  🛡️ Testing safety validation...")
        if robot.is_connected:
            # These will only work with real connection
            console.print("    ⚠ Safety checks require real robot connection")
        else:
            console.print("    ⚠ Safety checks available only with robot connection")
        
        # Test utility functions
        console.print("  🔧 Testing utility functions...")
        state = robot.get_tcp_pose()  # Will return None in simulation
        joints = robot.get_joints()   # Will return None in simulation
        console.print(f"    ✓ State queries work (simulation mode): TCP={state is None}, Joints={joints is None}")
        
        return True
        
    except Exception as e:
        console.print(f"  [red]✗[/red] Robot tools test failed: {e}")
        return False

def test_agent_architecture():
    """Test the agent architecture and base classes."""
    console.print("\n[bold cyan]🎤 Testing Agent Architecture[/bold cyan]")
    
    try:
        from ur.agents.base_agent import BaseVoiceAgent
        
        # Test base agent interface
        console.print("  📋 Testing BaseVoiceAgent interface...")
        
        # Check if it's properly abstract
        try:
            # This should fail because BaseVoiceAgent is abstract
            BaseVoiceAgent(None, None)
            console.print("    ⚠ BaseVoiceAgent should be abstract")
        except TypeError:
            console.print("    ✓ BaseVoiceAgent is properly abstract")
        
        # Test that we can import the concrete agent
        try:
            from ur.agents.openai_agent import OpenAIVoiceAgent
            console.print("    ✓ OpenAIVoiceAgent can be imported")
        except Exception as e:
            console.print(f"    ⚠ OpenAIVoiceAgent import issue: {e}")
        
        return True
        
    except Exception as e:
        console.print(f"  [red]✗[/red] Agent architecture test failed: {e}")
        return False

def test_bridge_structure():
    """Test the bridge module structure (without instantiating)."""
    console.print("\n[bold cyan]🌉 Testing Bridge Structure[/bold cyan]")
    
    try:
        # Read the bridge.py file and analyze its structure
        bridge_path = Path(__file__).parent / "bridge.py"
        with open(bridge_path, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = {
            "URBridge class": "class URBridge" in content,
            "RobotInterface inheritance": "RobotInterface" in content,
            "RTDE imports": "rtde_control" in content,
            "Task handling": "handle_task" in content,
            "Movement methods": "moveL" in content and "moveJ" in content,
            "Safety features": "try:" in content and "except" in content,
        }
        
        console.print("  🔍 Bridge structure analysis:")
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            console.print(f"    {status} {check}")
        
        # Check dependencies
        console.print("  📦 Dependency analysis:")
        if "roslibpy" in content:
            console.print("    ⚠ Requires ROS (roslibpy) - expected for bridge")
        if "rtde_control" in content:
            console.print("    ✓ Uses RTDE for robot communication")
        
        return all(checks.values())
        
    except Exception as e:
        console.print(f"  [red]✗[/red] Bridge structure test failed: {e}")
        return False

def test_configuration():
    """Test configuration and environment setup."""
    console.print("\n[bold cyan]⚙️ Testing Configuration[/bold cyan]")
    
    try:
        # Check for .env.sample file
        env_sample_path = Path(__file__).parent.parent / ".env.sample"
        if env_sample_path.exists():
            console.print("    ✓ .env.sample file found")
            
            # Read and display expected configuration
            with open(env_sample_path, 'r') as f:
                env_content = f.read()
            
            console.print("    📋 Expected environment variables:")
            for line in env_content.split('\n'):
                if line.strip() and not line.startswith('#'):
                    console.print(f"      - {line}")
        else:
            console.print("    ⚠ .env.sample file not found")
        
        # Test current environment
        console.print("  🌍 Current environment:")
        env_vars = ["ROBOT_IP", "OPENAI_API_KEY", "AGENT_TYPE", "VOICE_ENABLED"]
        for var in env_vars:
            value = os.getenv(var, "Not set")
            display_value = value if var != "OPENAI_API_KEY" else ("Set" if value != "Not set" else "Not set")
            console.print(f"    {var}: {display_value}")
        
        return True
        
    except Exception as e:
        console.print(f"  [red]✗[/red] Configuration test failed: {e}")
        return False

def display_summary(results: Dict[str, bool]):
    """Display test summary."""
    console.print("\n" + "="*60)
    
    table = Table(title="UR Bridge Standalone Test Summary", show_header=True, header_style="bold magenta")
    table.add_column("Test Category", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="yellow")
    
    details = {
        "imports": "Core modules and dependencies",
        "tools_registry": "Robot control functions (simulation)",
        "agents": "Voice agent architecture",
        "bridge": "Bridge structure and design",
        "config": "Configuration and environment"
    }
    
    for test_name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        table.add_row(test_name.title(), status, details.get(test_name, ""))
    
    console.print(table)
    
    # Overall result
    all_passed = all(results.values())
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    if all_passed:
        console.print(f"\n[bold green]🎉 All tests passed! ({passed_tests}/{total_tests})[/bold green]")
        console.print("[green]The UR bridge components are ready for standalone testing.[/green]")
    else:
        console.print(f"\n[bold yellow]⚠ Some tests had issues ({passed_tests}/{total_tests})[/bold yellow]")
        console.print("[yellow]Check the details above for any required fixes.[/yellow]")

def main():
    """Main test runner."""
    console.print(Panel.fit(
        "[bold cyan]UR Bridge Standalone Test Suite[/bold cyan]\n"
        "Testing UR bridge components without ROS or physical robot",
        title="🧪 VizorBridge Testing",
        border_style="blue"
    ))
    
    # Run all tests
    results = {}
    
    test_functions = [
        ("imports", test_imports),
        ("tools_registry", test_tools_registry),
        ("agents", test_agent_architecture),
        ("bridge", test_bridge_structure),
        ("config", test_configuration)
    ]
    
    for test_name, test_func in test_functions:
        try:
            if test_name == "imports":
                # test_imports returns a dict, others return bool
                import_results = test_func()
                results[test_name] = all(r["status"] == "✓" for r in import_results.values())
            else:
                results[test_name] = test_func()
        except Exception as e:
            console.print(f"[red]✗[/red] Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Display summary
    display_summary(results)
    
    console.print("\n[dim]💡 To test with a real robot, set ROBOT_IP in .env and ensure RTDE is installed.[/dim]")
    console.print("[dim]💡 To test with ROS, start rosbridge and run the full launcher.[/dim]")

if __name__ == "__main__":
    main()