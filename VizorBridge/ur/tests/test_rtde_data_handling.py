#!/usr/bin/env python3
"""
Interactive script to load and execute YAML trajectory files using the UR RTDE Python library.

Features:
- Rich CLI interface for file selection
- Automatic discovery of YAML files in data_messages folder
- Real-time trajectory execution with progress display

Usage:
    python test_ros_data_rtde.py [--ip IP] [--port PORT] [--speed SPEED] [--acceleration ACC]
"""
import os
import argparse
import sys
import yaml
import math
import glob
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
import time
import threading

# Ensure project root is on PYTHONPATH before importing internal packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ur.config import (
    ROBOT_IP,
    ROBOT_ACCELERATION,
    ROBOT_MOVE_SPEED,
    RTDE_CONTROL_PORT,
    TEST_DATA_MESSAGES_DIR
)

console = Console()

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


def display_file_selection(yaml_files: list[tuple[str, str]]) -> str:
    """Display available YAML files and let user select one.
    
    Returns:
        Selected file path or None if cancelled
    """
    console.print("\n")
    console.print(Panel.fit("🤖 [bold blue]UR Robot Trajectory Executor[/bold blue] 🤖"))
    
    # Create table of available files
    table = Table(title="📁 Available Trajectory Files", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Points", style="yellow")
    
    file_info = []
    for i, (filename, filepath) in enumerate(yaml_files, 1):
        # Get file size
        size = os.path.getsize(filepath)
        size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
        
        # Try to get number of trajectory points
        try:
            data = load_yaml(filepath)
            points = data.get('trajectory', {}).get('joint_trajectory', {}).get('points', [])
            points_count = len(points)
        except:
            points_count = "?"
        
        table.add_row(str(i), filename, size_str, str(points_count))
        file_info.append((filename, filepath, points_count))
    
    console.print(table)
    console.print()
    
    while True:
        try:
            choice = Prompt.ask(
                "Select trajectory file",
                choices=[str(i) for i in range(1, len(yaml_files) + 1)] + ["q"],
                default="1"
            )
            
            if choice.lower() == 'q':
                console.print("[yellow]Operation cancelled[/yellow]")
                return None
            
            selected_idx = int(choice) - 1
            selected_file = yaml_files[selected_idx][1]
            filename = yaml_files[selected_idx][0]
            
            console.print(f"\n[green]✓ Selected:[/green] {filename}")
            return selected_file
            
        except (ValueError, IndexError):
            console.print("[red]Invalid selection. Please try again.[/red]")


def stop_trajectory(rtde_c):
    """Emergency stop: halt all robot movements immediately."""
    try:
        rtde_c.stopL()
    except Exception:
        pass
    try:
        rtde_c.stopJ()
    except Exception:
        pass


def execute_trajectory(filepath: str, rtde_c: RTDEControlInterface, speed: float, acceleration: float, stop_event=None):
    """Execute trajectory with progress display. Can be stopped via stop_event."""
    # Load trajectory data
    try:
        data = load_yaml(filepath)
    except Exception as e:
        console.print(f"[red]Error loading file: {e}[/red]")
        return False
    
    points = data.get('trajectory', {}).get('joint_trajectory', {}).get('points', [])
    if not points:
        console.print("[red]No joint points found in the YAML file.[/red]")
        return False
    
    console.print(f"\n[blue]📋 Trajectory Info:[/blue]")
    console.print(f"  • Points: {len(points)}")
    console.print(f"  • Speed: {speed} m/s")
    console.print(f"  • Acceleration: {acceleration} m/s²")
    
    console.print("\n[bold green]🎯 Executing trajectory...[/bold green]")
    
    # Execute with progress bar
    with Progress() as progress:
        task = progress.add_task("Executing trajectory", total=len(points))
        
        for idx, point in enumerate(points):
            if stop_event and stop_event.is_set():
                console.print("\n[yellow]🛑 Trajectory execution stopped by user.[/yellow]")
                stop_trajectory(rtde_c)
                return False
            joints = point.get('positions', [])
            if not joints:
                console.print(f"[yellow]⚠️  Skipping point {idx}: no positions defined[/yellow]")
                continue
            
            # Joint data from YAML is always in degrees; convert to radians for RTDE
            joints_rad = [math.radians(j) for j in joints]
            console.print(f"[cyan]Target joints (degrees):[/cyan] {joints}")
            console.print(f"[magenta]Target joints (radians):[/magenta] {joints_rad}")
            
            joint_names = ["Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
            table = Table(title=f"Target Joint Values (Point {idx+1})", show_header=True, header_style="bold blue")
            table.add_column("Joint", style="cyan")
            table.add_column("Degrees", style="magenta")
            table.add_column("Radians", style="green")
            for name, deg, rad in zip(joint_names, joints, joints_rad):
                table.add_row(name, f"{deg:.2f}", f"{rad:.4f}")
            console.print(table)
            
            progress.update(task, description=f"Moving to point {idx + 1}/{len(points)}")
            
            try:
                # Use synchronous movement - simpler and more reliable
                # Check for stop before starting each movement
                if stop_event and stop_event.is_set():
                    console.print("\n[yellow]🛑 Trajectory execution stopped by user.[/yellow]")
                    return False
                
                rtde_c.moveJ(joints_rad, speed, acceleration, asynchronous=False)
                progress.advance(task)
            except Exception as e:
                console.print(f"\n[red]❌ Error at point {idx}: {e}[/red]")
                return False
    
    console.print("\n[bold green]✅ Trajectory execution completed successfully![bold green]")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Interactive UR robot trajectory executor"
    )
    parser.add_argument(
        '--ip', default=ROBOT_IP, help='Robot IP address'
    )
    parser.add_argument(
        '--port', type=int, default=RTDE_CONTROL_PORT, help='RTDE control port'
    )
    parser.add_argument(
        '--speed', type=float, default=ROBOT_MOVE_SPEED, help='Joint speed (m/s)'
    )
    parser.add_argument(
        '--acceleration', type=float, default=ROBOT_ACCELERATION, help='Joint acceleration (m/s^2)'
    )
    parser.add_argument(
        '--data-dir', default=None, help='Directory containing YAML trajectory files'
    )
    args = parser.parse_args()

    while True:
        try:
            # Set data directory relative to script location if not specified
            if args.data_dir is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                data_dir = os.path.join(script_dir, TEST_DATA_MESSAGES_DIR)
            else:
                data_dir = args.data_dir
            
            # Discover available YAML files
            yaml_files = discover_yaml_files(data_dir)
            if not yaml_files:
                return 1

            # Let user select a file
            selected_file = display_file_selection(yaml_files)
            if not selected_file:
                return 0  # User cancelled

            # Display connection info
            console.print(f"\n[blue]🔗 Connecting to robot...[blue]")
            console.print(f"  • IP: {args.ip}")
            console.print(f"  • Port: {args.port}")

            # Connect to the robot via RTDE
            try:
                rtde_c = RTDEControlInterface(args.ip, args.port)
                console.print("[green]✅ Connected to robot successfully![green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to connect to robot: {e}[red]")
                return 1

            # Load trajectory and show info in main thread
            try:
                data = load_yaml(selected_file)
            except Exception as e:
                console.print(f"[red]Error loading file: {e}[/red]")
                continue
            points = data.get('trajectory', {}).get('joint_trajectory', {}).get('points', [])
            if not points:
                console.print("[red]No joint points found in the YAML file.[/red]")
                continue
            console.print(f"\n[blue]📋 Trajectory Info:[/blue]")
            console.print(f"  • Points: {len(points)}")
            console.print(f"  • Speed: {args.speed} m/s")
            console.print(f"  • Acceleration: {args.acceleration} m/s²")
            if not Confirm.ask("\n🚀 Start trajectory execution?"):
                console.print("[yellow]Trajectory execution cancelled[/yellow]")
                continue

            stop_event = threading.Event()
            traj_thread = threading.Thread(target=execute_trajectory, args=(selected_file, rtde_c, args.speed, args.acceleration, stop_event))
            traj_thread.start()
            
            console.print("[yellow]Press [bold]Enter[/bold] or [bold]Ctrl+C[/bold] to stop execution and return to file selection.[/yellow]")
            
            try:
                # Wait for user to press Enter
                input()
            except KeyboardInterrupt:
                # Ctrl+C was pressed
                pass
            
            # --- Stop Logic (runs for both Enter and Ctrl+C) ---
            if not stop_event.is_set():
                console.print("\n[yellow]Stopping trajectory execution...[/yellow]")
                stop_event.set()

            traj_thread.join()
            console.print("[blue]🔌 Disconnecting from robot...[/blue]")
            rtde_c.disconnect()
            console.print("[green]✅ Disconnected successfully![green]")
            
            # Ask user if they want to run another trajectory
            console.print()
            if not Confirm.ask("🔄 Run another trajectory?", default=True):
                console.print("[yellow]👋 Exiting...[/yellow]")
                break
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]💥 Unexpected error: {e}[red]")
            break


if __name__ == '__main__':
    main()
