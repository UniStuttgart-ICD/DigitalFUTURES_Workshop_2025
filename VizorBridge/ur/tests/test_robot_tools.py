#!/usr/bin/env python3
"""
Interactive Robot Tools Tester
=============================

Numbered menu system for testing robot tools with dummy inputs.

Usage:
    python test_robot_tools.py           # Interactive mode
    python test_robot_tools.py --run-all # Run all tools automatically

Features:
- Numbered menu selection (just type the number)
- Pre-filled dummy inputs (no need to type parameters)
- Proper tool descriptions
- Rich console output
- Run-all mode for batch testing

Author: Generated for VizorBridge testing
"""

import sys
import time
import argparse
import subprocess
import platform
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


# Rich imports for better console output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

# Import tools directly from their modules
from ur.tools.movement_tools import move_relative_xyz, move_to_absolute_position, move_home, get_robot_state, stop_robot
from ur.tools.gripper_tools import control_gripper, get_supply_element, place_element_at_position, release_element
# Import robot config constants
from ur.config.robot_config import WAKEWORD, ROBOT_IP
from ur.core.connection import cleanup_global_robot

console = Console()

class RobotToolsTester:
    """Interactive tester for robot tools with numbered selection."""
    
    def __init__(self):
        self.console = console
        self.wake_phrase = f"test with {WAKEWORD}"
          # Get robot IP from environment or config
        self.robot_ip = os.getenv('UR_ROBOT_IP', ROBOT_IP)
        self.ping_timeout = int(os.getenv('PING_TIMEOUT', '3'))
        
        # Validate that we have a valid IP
        if not self.robot_ip:
            self.robot_ip = ROBOT_IP
            console.print(f"[yellow]⚠[/yellow] No robot IP configured, using default: {ROBOT_IP}")
        
        # Define tools with proper descriptions and dummy parameters
        self.tools = {
            1: {
                'name': 'get_robot_state',
                'function': get_robot_state,
                'description': 'Get current robot joint positions and TCP pose',
                'dummy_params': {},
                'category': 'Status'
            },
            2: {
                'name': 'move_relative_xyz',
                'function': move_relative_xyz,
                'description': 'Move robot relative in X, Y, Z directions',
                'dummy_params': {'dx_m': 0.1, 'dy_m': 0.1, 'dz_m': 0, 'wake_phrase': self.wake_phrase},
                'category': 'Movement'
            },
            3: {
                'name': 'move_to_absolute_position',
                'function': move_to_absolute_position,
                'description': 'Move robot to absolute position in Cartesian space',
                'dummy_params': {'x_m': 0.3, 'y_m': 0.0, 'z_m': 0.15, 'wake_phrase': self.wake_phrase},
                'category': 'Movement'
            },
            4: {
                'name': 'move_home',
                'function': move_home,
                'description': 'Move robot to predefined home position',
                'dummy_params': {'wake_phrase': self.wake_phrase},
                'category': 'Movement'
            },            5: {
                'name': 'control_gripper',
                'function': control_gripper,
                'description': 'Open or close the robot gripper',
                'dummy_params': {'action': 'open', 'wake_phrase': self.wake_phrase},
                'category': 'Gripper'
            },
            6: {
                'name': 'get_supply_element',
                'function': get_supply_element,
                'description': 'Get element from supply station',
                'dummy_params': {'element_length': '40cm', 'wake_phrase': self.wake_phrase},
                'category': 'Supply'
            },
            7: {
                'name': 'place_element_at_position',
                'function': place_element_at_position,
                'description': 'Place held element at specified position',
                'dummy_params': {'x': 0.3, 'y': 0.0, 'z': 0.15, 'wake_phrase': self.wake_phrase},
                'category': 'Supply'
            },
            8: {
                'name': 'release_element',
                'function': release_element,
                'description': 'Release held element and return to safe position',
                'dummy_params': {'wake_phrase': self.wake_phrase},
                'category': 'Supply'
            },
            9: {
                'name': 'stop_robot',
                'function': stop_robot,
                'description': 'Emergency stop - halt all robot movements immediately',
                'dummy_params': {'wake_phrase': self.wake_phrase},
                'category': 'Safety'
            }
        }
    
    def show_tools_menu(self):
        """Display a numbered menu of all available tools."""
        table = Table(title="🤖 Robot Tools Menu")
        table.add_column("#", style="bold cyan", width=3)
        table.add_column("Tool Name", style="yellow", no_wrap=True)
        table.add_column("Category", style="magenta", width=10)
        table.add_column("Description", style="green")
        
        for num, tool_info in self.tools.items():
            # Check if tool is available (for tools that might not exist in new structure)
            if not tool_info.get('available', True):
                description = f"[dim]{tool_info['description']} (UNAVAILABLE)[/dim]"
                name_style = "[dim]" + tool_info['name'] + "[/dim]"
                category_style = "[dim]" + tool_info['category'] + "[/dim]"
            else:
                description = tool_info['description']
                name_style = tool_info['name']
                category_style = tool_info['category']
                
            table.add_row(
                str(num),
                name_style,
                category_style,
                description
            )
        
        self.console.print(table)
    
    def show_tool_details(self, tool_num: int):
        """Show detailed information about a specific tool."""
        if tool_num not in self.tools:
            self.console.print(f"[red]✗[/red] Tool number {tool_num} not found")
            return
        
        tool_info = self.tools[tool_num]
        
        content = f"""[bold cyan]Tool:[/bold cyan] {tool_info['name']}
[bold cyan]Category:[/bold cyan] {tool_info['category']}
[bold cyan]Description:[/bold cyan] {tool_info['description']}

[bold cyan]Dummy Parameters:[/bold cyan]"""
        
        for param, value in tool_info['dummy_params'].items():
            content += f"\n  • {param}: [yellow]{value}[/yellow]"
        
        panel = Panel(content, title=f"🔧 Tool #{tool_num} Details", expand=False)
        self.console.print(panel)
    
    def execute_tool_with_dummy_params(self, tool_num: int, auto_confirm: bool = False):
        """Execute a tool using dummy parameters."""
        if tool_num not in self.tools:
            self.console.print(f"[red]✗[/red] Tool number {tool_num} not found")
            return False
        
        tool_info = self.tools[tool_num]
        
        # Check if tool is available
        if not tool_info.get('available', True):
            self.console.print(f"[red]✗[/red] Tool '{tool_info['name']}' is not available in the current configuration")
            return False
        
        # For robot movement tools, require ping test first (except get_robot_state which is read-only)
        if tool_info['name'] != 'get_robot_state' and not auto_confirm:
            self.console.print(f"\n[cyan]🏓 Verifying network connectivity...[/cyan]")
            if not self.ping_robot_ip():
                self.console.print("[red]❌ Cannot execute robot tool without network connectivity![/red]")
                return False
        
        self.console.print(f"\n[bold cyan]🔧 Executing tool #{tool_num}: {tool_info['name']}[/bold cyan]")
        
        # Show parameters that will be used
        if not auto_confirm:
            self.console.print("[yellow]Using dummy parameters:[/yellow]")
            for param, value in tool_info['dummy_params'].items():
                self.console.print(f"  {param}: [green]{value}[/green]")
        
        # Confirm execution (skip if auto_confirm is True)
        if not auto_confirm and not Confirm.ask(f"\nExecute {tool_info['name']} with these parameters?", default=True):
            self.console.print("[yellow]Execution cancelled[/yellow]")
            return False
        
        # Execute tool
        try:
            self.console.print("\n[cyan]🚀 Executing...[/cyan]" if not auto_confirm else f"[cyan]🚀[/cyan] Executing {tool_info['name']}...")
            start_time = time.time()
            
            result = tool_info['function'](**tool_info['dummy_params'])
            
            execution_time = time.time() - start_time
            
            # Display result and determine success
            # Handle both 'status': 'success' and 'success': True formats
            success = False
            if isinstance(result, dict):
                success = (result.get('status') == 'success' or 
                          result.get('success') == True)
            status_icon = "[green]✓[/green]" if success else "[yellow]⚠[/yellow]"
            
            if auto_confirm:
                # Compact output for run-all mode
                self.console.print(f"{status_icon} {tool_info['name']} ({execution_time:.2f}s)")
                if not success and isinstance(result, dict):
                    message = result.get('message', 'Unknown error')
                    self.console.print(f"  [red]Error:[/red] {message}")
                elif success and isinstance(result, dict):
                    # Show success message for successful operations
                    message = result.get('message', 'Success')
                    if len(message) > 60:  # Truncate very long messages
                        message = message[:57] + "..."
                    self.console.print(f"  [green]✓[/green] {message}")
            else:
                # Detailed output for individual execution
                self.console.print(f"\n[green]✓[/green] Execution completed in {execution_time:.2f}s")
                self.console.print("[bold cyan]Result:[/bold cyan]")
                
                if isinstance(result, dict):
                    # Pretty print dictionary results
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 3:
                            # Format long lists (like joint positions) nicely
                            formatted_value = f"[{', '.join([f'{x:.3f}' if isinstance(x, float) else str(x) for x in value[:3]])}...]"
                            self.console.print(f"  {key}: [yellow]{formatted_value}[/yellow]")
                        else:
                            self.console.print(f"  {key}: [yellow]{value}[/yellow]")
                else:
                    self.console.print(f"  [yellow]{result}[/yellow]")
            
            # Return success status for counting
            return success
                
        except Exception as e:
            error_msg = f"Execution failed: {e}"
            if auto_confirm:
                self.console.print(f"[red]✗[/red] {tool_info['name']} - {error_msg}")
            else:
                self.console.print(f"\n[red]✗[/red] {error_msg}")
                self.console.print(f"[red]Error type:[/red] {type(e).__name__}")
            return False
    
    def run_all_tools(self):
        """Execute all tools sequentially with dummy parameters."""
        self.console.print("\n[bold magenta]🚀 Running All Tools Mode[/bold magenta]")
        self.console.print("[yellow]This will execute all robot tools sequentially with dummy parameters[/yellow]")
        
        if not Confirm.ask("Are you sure you want to run ALL tools?", default=False):
            self.console.print("[yellow]Run-all cancelled[/yellow]")
            return
        
        # Test network and connection first
        self.console.print("\n[cyan]🔗 Testing network and connection before starting...[/cyan]")
        ping_success = self.ping_robot_ip()
        if not ping_success:
            self.console.print("[red]❌ Cannot continue without successful ping test![/red]")
            self.console.print("[red]Please check robot network connection and try again.[/red]")
            return
        
        self.test_robot_connection()
        
        self.console.print(f"\n[bold cyan]📋 Executing {len(self.tools)} tools...[/bold cyan]")
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        # Execute each tool
        for tool_num in sorted(self.tools.keys()):
            try:
                tool_info = self.tools[tool_num]  # Get tool info for current tool
                
                if self.execute_tool_with_dummy_params(tool_num, auto_confirm=True):
                    successful += 1
                else:
                    failed += 1
                    
                # Delay between tools for safety and proper completion
                # Longer delay for movement/gripper operations, shorter for status checks
                if tool_info['category'] in ['Movement', 'Gripper', 'Supply']:
                    time.sleep(1.0)  # 1 second for physical operations (reduced since tools now wait internally)
                else:
                    time.sleep(0.3)  # 0.3 seconds for status/info operations
                
            except KeyboardInterrupt:
                self.console.print(f"\n[yellow]⚠[/yellow] Run-all interrupted by user at tool #{tool_num}")
                break
        
        total_time = time.time() - start_time
        
        # Summary
        self.console.print(f"\n[bold cyan]📊 Run-All Summary[/bold cyan]")
        self.console.print(f"  [green]✓[/green] Successful: {successful}")
        self.console.print(f"  [red]✗[/red] Failed: {failed}")
        self.console.print(f"  [cyan]⏱[/cyan] Total time: {total_time:.2f}s")
        
        if successful > 0:
            self.console.print(f"[green]🎉 Run-all completed! {successful}/{successful + failed} tools executed successfully[/green]")
        else:
            self.console.print(f"[red]💥 All tools failed. Check robot connection.[/red]")
    
    def ping_robot_ip(self, ip_address=None, timeout=None):
        """Ping the robot IP address to test network connectivity."""
        # Use provided values or fall back to environment/default values
        ip_address = ip_address or self.robot_ip
        timeout = timeout or self.ping_timeout
        
        self.console.print(f"\n[cyan]🏓 Pinging robot IP: {ip_address}...[/cyan]")
        self.console.print(f"[dim]Using IP from: {'environment (.env)' if os.getenv('ROBOT_IP') else 'default value'}[/dim]")
        
        try:
            # Determine ping command based on OS
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", "1", "-w", str(timeout * 1000), ip_address]
            else:
                cmd = ["ping", "-c", "1", "-W", str(timeout), ip_address]
            
            # Execute ping
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 2)
            
            if result.returncode == 0:
                self.console.print(f"[green]✓[/green] Ping successful! Robot IP {ip_address} is reachable")
                # Extract ping time if available
                output = result.stdout.lower()
                if "time=" in output:
                    # Try to extract ping time
                    try:
                        import re
                        if platform.system().lower() == "windows":
                            match = re.search(r'time[<=]\s*(\d+)ms', output)
                        else:
                            match = re.search(r'time=(\d+\.?\d*)\s*ms', output)
                        
                        if match:
                            ping_time = match.group(1)
                            self.console.print(f"  Response time: [yellow]{ping_time}ms[/yellow]")
                    except:
                        pass  # If we can't parse ping time, that's OK
                return True
            else:
                self.console.print(f"[red]✗[/red] Ping failed! Robot IP {ip_address} is not reachable")
                self.console.print(f"  [red]Error:[/red] {result.stderr.strip() if result.stderr else 'Network unreachable'}")
                return False
                
        except subprocess.TimeoutExpired:
            self.console.print(f"[red]✗[/red] Ping timeout! No response from {ip_address} within {timeout}s")
            return False
        except Exception as e:
            self.console.print(f"[red]✗[/red] Ping test failed: {e}")
            return False

    def test_robot_connection(self):
        """Test basic robot connection using get_robot_state."""
        self.console.print("\n[cyan]🔗 Testing robot connection...[/cyan]")
        
        try:
            result = get_robot_state()
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                self.console.print(f"[yellow]⚠[/yellow] Unexpected result type: {type(result)}")
                self.console.print(f"[yellow]Result:[/yellow] {result}")
                return
            
            if result.get('status') == 'success':
                self.console.print("[green]✓[/green] Robot connection successful!")
                self.console.print(f"  Connected: [yellow]{result.get('connected')}[/yellow]")
                if result.get('tcp_pose'):
                    pose = result['tcp_pose']
                    if isinstance(pose, list) and len(pose) >= 3:
                        self.console.print(f"  TCP Position: [yellow]({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})[/yellow]")
                    else:
                        self.console.print(f"  TCP Pose: [yellow]{pose}[/yellow]")
            else:
                self.console.print("[yellow]⚠[/yellow] Robot connection failed or not available")
                self.console.print("[yellow]Running in simulation mode[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]✗[/red] Connection test failed: {e}")
    
    def run_interactive_session(self):
        """Run the interactive testing session."""
        # Display header
        panel = Panel(
            "[bold cyan]🤖 Robot Tools Interactive Tester[/bold cyan]\n\n"
            f"Robot IP: [yellow]{self.robot_ip}[/yellow] [dim]({'from .env' if os.getenv('ROBOT_IP') else 'default'})[/dim]\n"
            f"Wake phrase: [green]{self.wake_phrase}[/green]\n"
            "[red]⚠ Network ping test REQUIRED before any robot operations![/red]\n"
            "Just type a number (1-9) to run a tool!",
            title="VizorBridge Robot Tools Tester",
            expand=False
        )
        self.console.print(panel)
        
        # Test network and connection - REQUIRED
        ping_success = self.ping_robot_ip()
        if not ping_success:
            self.console.print("[red]❌ Cannot start interactive session without successful ping test![/red]")
            self.console.print("[red]Please check robot network connection and try again.[/red]")
            return
        
        self.test_robot_connection()
          # Show available tools
        self.show_tools_menu()
        
        # Interactive loop
        while True:
            try:
                self.console.print(f"\n[bold blue]Quick Commands:[/bold blue]")
                self.console.print("  [yellow]1-9[/yellow] : Execute tool by number")
                self.console.print("  [yellow]all[/yellow]  : Run ALL tools sequentially")
                self.console.print("  [yellow]i3[/yellow]   : Info about tool #3")
                self.console.print("  [yellow]h[/yellow]    : Show menu again")
                self.console.print("  [yellow]p[/yellow]    : Ping robot IP")
                self.console.print("  [yellow]c[/yellow]    : Test connection")
                self.console.print("  [yellow]q[/yellow]    : Quit")
                
                command = Prompt.ask("\n[bold blue]Enter number or command[/bold blue]", default="h").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                    
                elif command in ['help', 'h']:
                    self.show_tools_menu()
                    
                elif command == 'c':
                    self.test_robot_connection()
                    
                elif command == 'p':
                    self.ping_robot_ip()
                    
                elif command in ['all', 'run-all', 'runall']:
                    self.run_all_tools()
                    
                elif command.startswith('i') and len(command) > 1:
                    # Info command (e.g., 'i3')
                    try:
                        tool_num = int(command[1:])
                        self.show_tool_details(tool_num)
                    except ValueError:
                        self.console.print(f"[red]✗[/red] Invalid tool number: {command[1:]}")
                        
                elif command.isdigit():
                    # Direct tool number
                    tool_num = int(command)
                    if tool_num in self.tools:
                        self.execute_tool_with_dummy_params(tool_num)
                    else:
                        self.console.print(f"[red]✗[/red] Tool number {tool_num} not found. Valid range: 1-{len(self.tools)}")
                        
                else:
                    self.console.print(f"[red]✗[/red] Unknown command: {command}")
                    self.console.print("Type a number (1-9) to run a tool, or 'h' for help")
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted by user[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]✗[/red] Unexpected error: {e}")


def main():
    """Main function to run the robot tools tester."""
    try:
        parser = argparse.ArgumentParser(description="Interactive Robot Tools Tester")
        parser.add_argument("--run-all", action="store_true", help="Run all tools automatically")
        args = parser.parse_args()

        tester = RobotToolsTester()
        if args.run_all:
            tester.run_all_tools()
        else:
            tester.run_interactive_session()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Exiting...[/yellow]")
    except Exception as e:
        console.print(f"[red]✗[/red] Fatal error: {e}")
        raise
    finally:
        # Clean up robot connections
        try:
            cleanup_global_robot()
            console.print("[green]✓[/green] Robot connections cleaned up")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Warning during cleanup: {e}")


if __name__ == "__main__":
    main()