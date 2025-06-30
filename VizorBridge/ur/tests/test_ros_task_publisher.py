#!/usr/bin/env python3
"""
Interactive script to load and publish YAML task files and string commands to ROS topics using roslibpy.

Features:
- Rich CLI interface for file and command selection
- Automatic discovery of YAML files in data_messages folder
- Publishing to multiple ROS topics (tasks and commands) via rosbridge
- Support for predefined commands from PlantUML workflow
- Works without full ROS installation

Usage:
    python test_ros_task_publisher.py [--host HOST] [--port PORT]
"""
import os
import sys
import yaml
import glob
import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
import time

# ROS imports via roslibpy
try:
    import roslibpy
except ImportError:
    print("roslibpy not available - install with: uv add roslibpy")
    sys.exit(1)

# Ensure project root is on PYTHONPATH before importing internal packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ur.config.topics import (
    TASK_EXECUTE_TOPIC, TASK_EXECUTE_MSG_TYPE,
    COMMAND_TOPIC, STATUS_TOPIC, STD_STRING_MSG_TYPE
)
from ur.config import TEST_DATA_MESSAGES_DIR

console = Console()

# Predefined commands from the PlantUML workflow
PREDEFINED_COMMANDS = {
    "commands": [
        "start_fabrication",
        "end_fabrication",
        "stop",
        "gripper_open",
        "gripper_close",
        "home",
    ],
    "status": [
        "success",
        "failed",
        "stop",
        "fabrication_started",
        "fabrication_complete",
        "moving_to_supply",
        "gripper_open",
        "gripper_close",
        "at_assembly",
        "complete_adjust_position"
    ]
}

def load_yaml(file_path: str) -> dict:
    """Load and parse a YAML file into a Python dict."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def discover_yaml_files(data_dir: str) -> list[tuple[str, str]]:
    """Discover all YAML files in the data directory.
    
    Returns:
        List of tuples (filename, full_path)
    """
    if not os.path.exists(data_dir):
        console.print(f"[red]Error: Directory '{data_dir}' does not exist![/red]")
        return []
    
    yaml_pattern = os.path.join(data_dir, "*.yaml")
    yaml_files = glob.glob(yaml_pattern)
    
    if not yaml_files:
        console.print(f"[yellow]No YAML files found in '{data_dir}'[/yellow]")
        return []
    
    return [(os.path.basename(f), f) for f in sorted(yaml_files)]


def display_main_menu() -> str:
    """Display the main menu and get user choice."""
    console.print("\n")
    console.print(Panel.fit("📡 [bold blue]ROS Multi-Topic Publisher (roslibpy)[/bold blue] 📡"))
    
    # Create menu table
    table = Table(title="📋 Publication Options", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="green")
    
    table.add_row("1", "Publish Task Message", f"Send YAML task files to {TASK_EXECUTE_TOPIC}")
    table.add_row("2", "Publish Command", f"Send command strings to {COMMAND_TOPIC}")
    table.add_row("3", "Publish Status", f"Send status strings to {STATUS_TOPIC}")
    table.add_row("q", "Quit", "Exit the publisher")
    
    console.print(table)
    console.print()
    
    choice = Prompt.ask(
        "Select publication type",
        choices=["1", "2", "3", "q"],
        default="1"
    )
    
    return choice


def display_file_selection(yaml_files: list[tuple[str, str]]) -> str:
    """Display available YAML files and let user select one.
    
    Returns:
        Selected file path or None if cancelled
    """
    console.print(Panel.fit("🎯 [bold green]Task File Selection[/bold green] 🎯"))
    
    # Create table of available files
    table = Table(title="📁 Available Task Files", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Task Type", style="yellow")
    
    for i, (filename, filepath) in enumerate(yaml_files, 1):
        # Get file size
        size = os.path.getsize(filepath)
        size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
        
        # Try to get task type from filename
        task_type = "Unknown"
        if "pick" in filename.lower():
            task_type = "Pick"
        elif "place" in filename.lower():
            task_type = "Place"
        elif "home" in filename.lower():
            task_type = "Home"
        
        table.add_row(str(i), filename, size_str, task_type)
    
    console.print(table)
    console.print()
    
    while True:
        try:
            choice = Prompt.ask(
                "Select task file to publish",
                choices=[str(i) for i in range(1, len(yaml_files) + 1)] + ["b", "q"],
                default="1"
            )
            
            if choice.lower() == 'q':
                return "quit"
            elif choice.lower() == 'b':
                return "back"
            
            selected_idx = int(choice) - 1
            selected_file = yaml_files[selected_idx][1]
            filename = yaml_files[selected_idx][0]
            
            console.print(f"\n[green]✓ Selected:[/green] {filename}")
            return selected_file
            
        except (ValueError, IndexError):
            console.print("[red]Invalid selection. Please try again.[/red]")


def display_command_selection(command_type: str) -> str:
    """Display available commands and let user select one."""
    commands = PREDEFINED_COMMANDS.get(command_type, [])
    
    console.print(Panel.fit(f"⚡ [bold yellow]{command_type.title()} Selection[/bold yellow] ⚡"))
    
    # Create table of available commands
    table = Table(title=f"📋 Available {command_type.title()} Messages", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="green")
    
    # Add predefined commands
    for i, cmd in enumerate(commands, 1):
        description = ""
        if "START" in cmd:
            description = "Start fabrication process"
        elif "END" in cmd:
            description = "End fabrication process"
        elif "SUCCESS" in cmd:
            description = "Task completion status"
        elif "GRIPPER" in cmd.upper():
            description = "Gripper state change"
        elif "MOVING" in cmd:
            description = "Robot movement status"
        else:
            description = "System status message"
            
        table.add_row(str(i), cmd, description)
    
    # Add custom option
    table.add_row(str(len(commands) + 1), "[Custom]", "Enter your own command string")
    
    console.print(table)
    console.print()
    
    while True:
        try:
            choice = Prompt.ask(
                f"Select {command_type} to publish",
                choices=[str(i) for i in range(1, len(commands) + 2)] + ["b", "q"],
                default="1"
            )
            
            if choice.lower() == 'q':
                return "quit"
            elif choice.lower() == 'b':
                return "back"
            
            choice_idx = int(choice) - 1
            
            if choice_idx < len(commands):
                selected_command = commands[choice_idx]
                console.print(f"\n[green]✓ Selected:[/green] {selected_command}")
                return selected_command
            else:
                # Custom command
                custom_command = Prompt.ask("Enter custom command string")
                console.print(f"\n[green]✓ Custom command:[/green] {custom_command}")
                return custom_command
            
        except (ValueError, IndexError):
            console.print("[red]Invalid selection. Please try again.[/red]")


def publish_task_message(filepath: str, topic) -> bool:
    """Publish the task message to ROS topic."""
    try:
        # Load task data
        data = load_yaml(filepath)
        console.print(f"\n[blue]📋 Task Info:[/blue]")
        
        # Display basic task information
        if 'trajectory' in data:
            points = data.get('trajectory', {}).get('joint_trajectory', {}).get('points', [])
            console.print(f"  • Trajectory points: {len(points)}")
        
        if 'task_name' in data:
            console.print(f"  • Task name: {data['task_name']}")
        
        console.print(f"  • Publishing to topic: {TASK_EXECUTE_TOPIC}")
        console.print(f"  • Message type: {TASK_EXECUTE_MSG_TYPE}")
        
        # Create message for roslibpy - pass the YAML data directly as message fields
        message = roslibpy.Message(data)
        
        console.print("\n[bold green]📡 Publishing task message...[/bold green]")
        topic.publish(message)
        
        # Give some time for the message to be published
        time.sleep(0.5)
        
        console.print("[bold green]✅ Task message published successfully![/bold green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error publishing task message: {e}[/red]")
        return False


def publish_string_message(command: str, topic, topic_name: str) -> bool:
    """Publish a string message to ROS topic."""
    try:
        console.print(f"\n[blue]📋 {topic_name} Info:[/blue]")
        console.print(f"  • Command: {command}")
        console.print(f"  • Publishing to topic: {topic_name}")
        console.print(f"  • Message type: {STD_STRING_MSG_TYPE}")
        
        # Create string message for roslibpy
        message = roslibpy.Message({"data": command})
        
        console.print(f"\n[bold yellow]📡 Publishing {topic_name.split('/')[-1]} message...[/bold yellow]")
        topic.publish(message)
        
        # Give some time for the message to be published
        time.sleep(0.5)
        
        console.print(f"[bold green]✅ {topic_name.split('/')[-1].title()} message published successfully![/bold green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error publishing {topic_name.split('/')[-1]} message: {e}[/red]")
        return False


def main():
    parser = argparse.ArgumentParser(description="ROS Multi-Topic Publisher using roslibpy")
    parser.add_argument('--host', default='localhost', help='ROS bridge host (default: localhost)')
    parser.add_argument('--port', type=int, default=9090, help='ROS bridge port (default: 9090)')
    args = parser.parse_args()

    console.print(Panel.fit("🚀 [bold blue]Initializing ROS Multi-Topic Publisher (roslibpy)[/bold blue] 🚀"))
    
    # Connect to ROS via rosbridge
    try:
        console.print(f"[blue]🔗 Connecting to ROS bridge at {args.host}:{args.port}...[/blue]")
        client = roslibpy.Ros(host=args.host, port=args.port)
        client.run()
        console.print("[green]✅ Connected to ROS bridge successfully![/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to connect to ROS bridge: {e}[/red]")
        console.print(f"[yellow]💡 Make sure rosbridge is running: roslaunch rosbridge_server rosbridge_websocket.launch[/yellow]")
        return 1
    
    # Create topics for publishing
    topics = {}
    try:
        # Task topic
        topics['task'] = roslibpy.Topic(client, TASK_EXECUTE_TOPIC, TASK_EXECUTE_MSG_TYPE)
        console.print(f"[green]✅ Task topic created: {TASK_EXECUTE_TOPIC}[/green]")
        
        # Command topic
        topics['command'] = roslibpy.Topic(client, COMMAND_TOPIC, STD_STRING_MSG_TYPE)
        console.print(f"[green]✅ Command topic created: {COMMAND_TOPIC}[/green]")
        
        # Status topic
        topics['status'] = roslibpy.Topic(client, STATUS_TOPIC, STD_STRING_MSG_TYPE)
        console.print(f"[green]✅ Status topic created: {STATUS_TOPIC}[/green]")
        
        # Wait a moment for topics to be ready
        time.sleep(1)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create topics: {e}[/red]")
        client.terminate()
        return 1

    try:
        while True:
            try:
                # Main menu
                choice = display_main_menu()
                
                if choice == 'q':
                    console.print("[yellow]👋 Exiting...[/yellow]")
                    break
                
                success = False
                
                if choice == '1':
                    # Task publishing
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    data_dir = os.path.join(script_dir, TEST_DATA_MESSAGES_DIR)
                    
                    yaml_files = discover_yaml_files(data_dir)
                    if not yaml_files:
                        continue
                    
                    selected_file = display_file_selection(yaml_files)
                    if selected_file == "quit":
                        break
                    elif selected_file == "back":
                        continue
                    elif selected_file:
                        if Confirm.ask("\n📡 Publish this task message to ROS?"):
                            success = publish_task_message(selected_file, topics['task'])
                        else:
                            console.print("[yellow]Publication cancelled[/yellow]")
                            continue
                
                elif choice == '2':
                    # Command publishing
                    selected_command = display_command_selection("commands")
                    if selected_command == "quit":
                        break
                    elif selected_command == "back":
                        continue
                    elif selected_command:
                        if Confirm.ask(f"\n⚡ Publish command '{selected_command}' to ROS?"):
                            success = publish_string_message(selected_command, topics['command'], COMMAND_TOPIC)
                        else:
                            console.print("[yellow]Publication cancelled[/yellow]")
                            continue
                
                elif choice == '3':
                    # Status publishing
                    selected_status = display_command_selection("status")
                    if selected_status == "quit":
                        break
                    elif selected_status == "back":
                        continue
                    elif selected_status:
                        if Confirm.ask(f"\n📊 Publish status '{selected_status}' to ROS?"):
                            success = publish_string_message(selected_status, topics['status'], STATUS_TOPIC)
                        else:
                            console.print("[yellow]Publication cancelled[/yellow]")
                            continue
                
                if success:
                    console.print("\n[blue]💡 Tip: Use the subscriber script to verify the message was published[/blue]")
                
                # Ask user if they want to publish another message
                console.print()
                if not Confirm.ask("🔄 Publish another message?", default=True):
                    console.print("[yellow]👋 Exiting...[/yellow]")
                    break
                
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Exiting...[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
                break
    
    finally:
        # Clean up connection
        console.print("[blue]🔌 Disconnecting from ROS bridge...[/blue]")
        client.terminate()
        console.print("[green]✅ Disconnected successfully![/green]")

    return 0


if __name__ == '__main__':
    main() 