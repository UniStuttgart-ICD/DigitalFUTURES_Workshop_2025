#!/usr/bin/env python3
"""
Interactive script to subscribe and listen to ROS task and command messages using roslibpy.

Features:
- Rich CLI interface for displaying messages
- Real-time message reception from multiple ROS topics via rosbridge
- Handles both task messages (/UR10/task/execute) and command messages (/UR10/command, /Robot/status/physical)
- JSON formatting and syntax highlighting
- Works without full ROS installation

Usage:
    python test_ros_task_subscriber.py [--host HOST] [--port PORT]
"""
import os
import sys
import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.panel import Panel
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint
import time
from datetime import datetime

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

console = Console()

class MessageLogger:
    def __init__(self):
        self.messages = []
        self.message_count = 0
        self.task_count = 0
        self.command_count = 0
        self.status_count = 0
        
    def add_message(self, message_data, topic_name, message_type="task"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.message_count += 1
        
        if message_type == "task":
            self.task_count += 1
        elif message_type == "command":
            self.command_count += 1
        elif message_type == "status":
            self.status_count += 1
        
        # Handle the message data based on type
        try:
            if message_type in ["command", "status"]:
                # String messages - extract the 'data' field if it exists
                if isinstance(message_data, dict) and 'data' in message_data:
                    parsed_data = message_data['data']
                    info = {"type": f"{message_type.title()}: {parsed_data}", "points": "N/A", "topic": topic_name}
                else:
                    parsed_data = str(message_data)
                    info = {"type": f"{message_type.title()}: {parsed_data}", "points": "N/A", "topic": topic_name}
            else:
                # Task messages - handle as before
                if isinstance(message_data, dict):
                    parsed_data = message_data
                else:
                    parsed_data = json.loads(message_data) if isinstance(message_data, str) else message_data
                info = self.extract_task_info(parsed_data)
                info["topic"] = topic_name
        except:
            parsed_data = message_data
            info = {"type": f"{message_type.title()}: Unknown", "points": "N/A", "topic": topic_name}
        
        message_entry = {
            "id": self.message_count,
            "timestamp": timestamp,
            "data": parsed_data,
            "raw": json.dumps(parsed_data, indent=2) if isinstance(parsed_data, dict) else str(parsed_data),
            "message_info": info,
            "message_type": message_type
        }
        
        self.messages.append(message_entry)
        
        # Keep only last 50 messages to prevent memory issues
        if len(self.messages) > 50:
            self.messages.pop(0)
    
    def extract_task_info(self, data):
        """Extract basic task information from message data."""
        info = {"type": "Unknown", "points": "N/A"}
        
        if isinstance(data, dict):
            # Try to identify task type from various fields
            if 'name' in data:
                info["type"] = data['name']
            elif 'task_name' in data:
                info["type"] = data['task_name']
            elif 'trajectory' in data:
                points = data.get('trajectory', {}).get('joint_trajectory', {}).get('points', [])
                info["points"] = len(points)
                # Guess task type from name or content
                if 'name' in data:
                    if 'pick' in data['name'].lower():
                        info["type"] = "Pick Task"
                    elif 'place' in data['name'].lower():
                        info["type"] = "Place Task"
                    elif 'home' in data['name'].lower():
                        info["type"] = "Home Task"
                    else:
                        info["type"] = data['name']
                else:
                    info["type"] = "Trajectory"
        
        return info
    
    def get_latest_messages(self, count=10):
        return self.messages[-count:] if self.messages else []


def create_message_display(logger: MessageLogger):
    """Create a rich display for incoming messages."""
    layout = Layout()
    
    # Header with counts for each message type
    header_table = Table.grid()
    header_table.add_column(style="bold blue")
    header_table.add_column(style="bold green")
    header_table.add_column(style="bold yellow")
    header_table.add_column(style="bold magenta")
    header_table.add_column(style="bold cyan")
    header_table.add_row(
        f"📡 ROS Multi-Topic Subscriber",
        f"Tasks: {logger.task_count}",
        f"Commands: {logger.command_count}",
        f"Status: {logger.status_count}",
        f"Total: {logger.message_count}"
    )
    
    # Recent messages table
    messages_table = Table(title="🔄 Recent Messages", show_header=True, header_style="bold magenta")
    messages_table.add_column("#", style="dim", width=4)
    messages_table.add_column("Time", style="cyan", width=12)
    messages_table.add_column("Type", style="green", width=8)
    messages_table.add_column("Content", style="yellow", width=20)
    messages_table.add_column("Topic", style="blue", width=20)
    messages_table.add_column("Size", style="magenta", width=8)
    
    recent_messages = logger.get_latest_messages(10)
    for msg in recent_messages:
        size = len(msg["raw"])
        size_str = f"{size / 1024:.1f}KB" if size > 1024 else f"{size}B"
        
        # Color code by message type
        type_style = {"task": "bold green", "command": "bold yellow", "status": "bold blue"}.get(msg["message_type"], "white")
        
        messages_table.add_row(
            str(msg["id"]),
            msg["timestamp"],
            f"[{type_style}]{msg['message_type'].upper()}[/{type_style}]",
            str(msg["message_info"]["type"])[:20],
            str(msg["message_info"]["topic"])[:20],
            size_str
        )
    
    if not recent_messages:
        messages_table.add_row("", "", "", "[dim]Waiting for messages...[/dim]", "", "")
    
    layout.split_column(
        Layout(Panel(header_table, title="Status", border_style="blue"), size=3),
        Layout(Panel(messages_table, title="Message Log", border_style="green"))
    )
    
    return layout


def create_message_callback(logger: MessageLogger, topic_name: str, message_type: str):
    """Create a callback function for a specific topic and message type."""
    def message_callback(message):
        """Callback function for received messages."""
        if message_type == "task":
            console.print(f"[bold green]🎯 TASK received on {topic_name}[/bold green]")
        elif message_type == "command":
            console.print(f"[bold yellow]⚡ COMMAND received on {topic_name}[/bold yellow]")
        elif message_type == "status":
            console.print(f"[bold blue]📊 STATUS received on {topic_name}[/bold blue]")
        
        # Handle the message data
        if isinstance(message, dict):
            message_data = message
            console.print(f"[dim]Message fields: {list(message.keys())}[/dim]")
        else:
            message_data = str(message)
            console.print(f"[dim]Message as string: {str(message)[:100]}...[/dim]")
        
        logger.add_message(message_data, topic_name, message_type)
        
        # Print message details to console
        console.print(f"[bold cyan]📨 Message #{logger.message_count} at {logger.messages[-1]['timestamp']}[/bold cyan]")
        
        # Show message info
        message_info = logger.messages[-1]['message_info']
        console.print(f"[blue]Content:[/blue] {message_info['type']}")
        if message_info['points'] != "N/A":
            console.print(f"[blue]Trajectory Points:[/blue] {message_info['points']}")
        
        # Show formatted content
        try:
            if isinstance(message_data, dict):
                formatted_json = json.dumps(message_data, indent=2)
                syntax = Syntax(formatted_json, "json", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"📄 {message_type.title()} Content", border_style="cyan"))
            else:
                console.print(Panel(str(message_data), title=f"📄 {message_type.title()} Content", border_style="cyan"))
        except:
            console.print(f"[yellow]Raw message:[/yellow] {str(message_data)[:200]}...")
        
        console.print("[dim]" + "─" * 80 + "[/dim]")
    
    return message_callback


def main():
    parser = argparse.ArgumentParser(description="ROS Multi-Topic Subscriber using roslibpy")
    parser.add_argument('--host', default='localhost', help='ROS bridge host (default: localhost)')
    parser.add_argument('--port', type=int, default=9090, help='ROS bridge port (default: 9090)')
    parser.add_argument('--live', action='store_true', help='Use live updating display')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--tasks-only', action='store_true', help='Subscribe only to task messages')
    parser.add_argument('--commands-only', action='store_true', help='Subscribe only to command messages')
    args = parser.parse_args()

    console.print(Panel.fit("🎧 [bold blue]Initializing ROS Multi-Topic Subscriber (roslibpy)[/bold blue] 🎧"))
    
    # Connect to ROS via rosbridge
    try:
        console.print(f"[blue]🔗 Connecting to ROS bridge at {args.host}:{args.port}...[/blue]")
        client = roslibpy.Ros(host=args.host, port=args.port)
        client.run()
        console.print("[green]✅ Connected to ROS bridge successfully![/green]")
        
        if args.debug:
            console.print(f"[yellow]🔧 DEBUG: Client connected: {client.is_connected}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to connect to ROS bridge: {e}[/red]")
        console.print(f"[yellow]💡 Make sure rosbridge is running: roslaunch rosbridge_server rosbridge_websocket.launch[/yellow]")
        return 1
    
    # Create message logger
    logger = MessageLogger()
    
    # List to keep track of active listeners
    listeners = []
    
    # Subscribe to topics based on arguments
    topics_to_subscribe = []
    
    if not args.commands_only:
        topics_to_subscribe.append((TASK_EXECUTE_TOPIC, TASK_EXECUTE_MSG_TYPE, "task"))
    
    if not args.tasks_only:
        topics_to_subscribe.append((COMMAND_TOPIC, STD_STRING_MSG_TYPE, "command"))
        topics_to_subscribe.append((STATUS_TOPIC, STD_STRING_MSG_TYPE, "status"))
    
    console.print(f"[blue]📋 Subscribing to {len(topics_to_subscribe)} topics...[/blue]")
    
    # Create subscribers for each topic
    for topic_name, msg_type, message_type in topics_to_subscribe:
        try:
            listener = roslibpy.Topic(client, topic_name, msg_type)
            callback = create_message_callback(logger, topic_name, message_type)
            
            if args.debug:
                def debug_wrapper(message, tn=topic_name, mt=message_type, cb=callback):
                    console.print(f"[yellow]🔧 DEBUG: Raw callback on {tn} ({mt}): {message}[/yellow]")
                    cb(message)
                listener.subscribe(debug_wrapper)
            else:
                listener.subscribe(callback)
            
            listeners.append(listener)
            console.print(f"[green]✅ Subscribed to {topic_name} ({msg_type})[/green]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Failed to subscribe to {topic_name}: {e}[/yellow]")
            # Try without message type validation
            try:
                listener = roslibpy.Topic(client, topic_name)
                callback = create_message_callback(logger, topic_name, message_type)
                
                if args.debug:
                    def debug_wrapper(message, tn=topic_name, mt=message_type, cb=callback):
                        console.print(f"[yellow]🔧 DEBUG: Raw callback on {tn} ({mt}): {message}[/yellow]")
                        cb(message)
                    listener.subscribe(debug_wrapper)
                else:
                    listener.subscribe(callback)
                
                listeners.append(listener)
                console.print(f"[green]✅ Subscribed to {topic_name} (flexible type)[/green]")
                
            except Exception as e2:
                console.print(f"[red]❌ Failed to subscribe to {topic_name}: {e2}[/red]")

    if not listeners:
        console.print("[red]❌ No topics subscribed successfully![/red]")
        client.terminate()
        return 1

    console.print("\n[bold green]🎧 Listening for messages... Press Ctrl+C to stop[/bold green]")
    console.print("[blue]💡 Tip: Use the publisher script to send test messages[/blue]")
    
    if args.debug:
        console.print("[yellow]🔧 DEBUG MODE: Additional debugging output enabled[/yellow]")
    
    console.print("[dim]" + "─" * 80 + "[/dim]")

    try:
        if args.live:
            # Live updating display mode
            with Live(create_message_display(logger), refresh_per_second=2) as live:
                while True:
                    time.sleep(0.5)
                    live.update(create_message_display(logger))
                    if args.debug and logger.message_count == 0:
                        if not client.is_connected:
                            console.print("[red]🔧 DEBUG: Lost connection to ROS bridge![/red]")
        else:
            # Simple console output mode
            iteration = 0
            while True:
                time.sleep(1)
                iteration += 1
                if args.debug and iteration % 10 == 0:  # Every 10 seconds
                    console.print(f"[dim]🔧 DEBUG: Still listening... (iteration {iteration}, messages: {logger.message_count})[/dim]")
                    if not client.is_connected:
                        console.print("[red]🔧 DEBUG: Lost connection to ROS bridge![/red]")
                        break
                
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Stopping subscriber...[/yellow]")
    
    finally:
        # Clean up
        console.print("[blue]🔌 Disconnecting from ROS bridge...[/blue]")
        try:
            for listener in listeners:
                listener.unsubscribe()
        except:
            pass
        client.terminate()
        console.print("[green]✅ Disconnected successfully![/green]")
        
        # Show summary
        console.print(f"\n[blue]📊 Session Summary:[/blue]")
        console.print(f"  • Total messages received: {logger.message_count}")
        console.print(f"  • Task messages: {logger.task_count}")
        console.print(f"  • Command messages: {logger.command_count}")
        console.print(f"  • Status messages: {logger.status_count}")
        if logger.messages:
            console.print(f"  • First message: {logger.messages[0]['timestamp']}")
            console.print(f"  • Last message: {logger.messages[-1]['timestamp']}")

    return 0


if __name__ == '__main__':
    main() 