#!/usr/bin/env python3
"""
RTDE Connection Diagnostic & Cleanup Utility
============================================

Diagnoses and fixes "RTDE input registers are already in use" errors.
Automatically detects zombie connections and provides cleanup options.

Usage:
    python test_rtde_cleanup.py
    # or
    uv run ur/tests/test_rtde_cleanup.py
"""

import subprocess
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm

console = Console()

def run_cmd(command: str, capture_output: bool = True) -> tuple[int, str, str]:
    """Run a system command and return (exit_code, stdout, stderr)."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=capture_output, 
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if capture_output else 0
            )
        else:
            result = subprocess.run(
                command.split(), 
                capture_output=capture_output, 
                text=True
            )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def check_zombie_connections(robot_ip: str = "192.168.56.101") -> list[dict]:
    """Check for zombie RTDE connections to the robot."""
    console.print(f"🔍 Checking for zombie RTDE connections to {robot_ip}...")
    
    if sys.platform == "win32":
        cmd = f"netstat -ano | findstr {robot_ip} | findstr ESTABLISHED"
    else:
        cmd = f"netstat -an | grep {robot_ip} | grep ESTABLISHED"
    
    exit_code, stdout, stderr = run_cmd(cmd)
    
    connections = []
    if exit_code == 0 and stdout.strip():
        lines = stdout.strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                local_addr = parts[1]
                remote_addr = parts[2]
                if sys.platform == "win32" and len(parts) >= 5:
                    pid = parts[4]
                else:
                    pid = "N/A"
                
                # Parse remote port to identify RTDE service
                remote_port = remote_addr.split(':')[-1]
                service = "Unknown"
                if remote_port == "29999":
                    service = "RTDE Control"
                elif remote_port == "30004":
                    service = "RTDE Receive"
                elif remote_port == "30003":
                    service = "RTDE I/O"
                
                connections.append({
                    'local': local_addr,
                    'remote': remote_addr,
                    'pid': pid,
                    'service': service,
                    'port': remote_port
                })
    
    return connections

def get_process_info(pid: str) -> str:
    """Get process name for a given PID."""
    if pid == "N/A":
        return "N/A"
    
    if sys.platform == "win32":
        cmd = f"tasklist /FI \"PID eq {pid}\""
        exit_code, stdout, stderr = run_cmd(cmd)
        if exit_code == 0:
            lines = stdout.strip().split('\n')
            for line in lines:
                if pid in line:
                    parts = line.split()
                    if len(parts) >= 1:
                        return parts[0]
    else:
        cmd = f"ps -p {pid} -o comm="
        exit_code, stdout, stderr = run_cmd(cmd)
        if exit_code == 0:
            return stdout.strip()
    
    return "Unknown"

def kill_process(pid: str) -> bool:
    """Kill a process by PID."""
    if pid == "N/A":
        return False
    
    console.print(f"🔥 Killing process {pid}...")
    
    if sys.platform == "win32":
        cmd = f"taskkill /PID {pid} /F"
    else:
        cmd = f"kill -9 {pid}"
    
    exit_code, stdout, stderr = run_cmd(cmd)
    
    if exit_code == 0:
        console.print(f"[green]✅ Process {pid} terminated successfully[/green]")
        return True
    else:
        console.print(f"[red]❌ Failed to kill process {pid}: {stderr}[/red]")
        return False

def test_robot_connection() -> bool:
    """Test if robot connection works."""
    console.print("🤖 Testing robot connection...")
    
    try:
        import sys
        import os
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from ur.core.connection import get_robot
        robot = get_robot()
        if robot.is_connected:
            console.print("[green]✅ Robot connected successfully![/green]")
            
            # Get some basic data to verify connection
            joints = robot.get_joints()
            tcp = robot.get_tcp_pose()
            
            if joints and tcp:
                console.print(f"[dim]   Joints: {[round(j, 3) for j in joints]}[/dim]")
                console.print(f"[dim]   TCP: {[round(t, 3) for t in tcp[:3]]}[/dim]")
            return True
        else:
            console.print("[red]❌ Robot connection failed[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Connection test failed: {e}[/red]")
        return False

def cleanup_connections() -> bool:
    """Run the built-in connection cleanup."""
    console.print("🧹 Running built-in connection cleanup...")
    
    try:
        import sys
        import os
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from ur.core.connection import cleanup_all
        cleanup_all()
        console.print("[green]✅ Built-in cleanup completed[/green]")
        return True
    except Exception as e:
        console.print(f"[red]❌ Built-in cleanup failed: {e}[/red]")
        return False

def main():
    """Main diagnostic and cleanup routine."""
    console.print(Panel(
        "[bold cyan]RTDE Connection Diagnostic & Cleanup Utility[/bold cyan]\n"
        "Automatically detects and fixes zombie RTDE connections",
        expand=False
    ))
    
    robot_ip = "192.168.56.101"  # Default robot IP
    
    # Step 1: Check for zombie connections
    connections = check_zombie_connections(robot_ip)
    
    if not connections:
        console.print("[green]✅ No zombie RTDE connections found[/green]")
        
        # Test connection anyway
        if test_robot_connection():
            console.print("\n[bold green]🎉 Everything looks good![/bold green]")
            return
        else:
            console.print("\n[yellow]⚠️ No zombie connections but robot still won't connect[/yellow]")
            console.print("This might be a robot-side issue or network problem.")
            return
    
    # Step 2: Display found connections
    console.print(f"\n[red]⚠️ Found {len(connections)} zombie RTDE connection(s):[/red]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("PID", style="yellow")
    table.add_column("Process", style="cyan")
    table.add_column("Service", style="green")
    table.add_column("Remote Port", style="blue")
    table.add_column("Connection", style="dim")
    
    unique_pids = set()
    for conn in connections:
        process_name = get_process_info(conn['pid'])
        table.add_row(
            conn['pid'],
            process_name,
            conn['service'],
            conn['port'],
            f"{conn['local']} → {conn['remote']}"
        )
        if conn['pid'] != "N/A":
            unique_pids.add(conn['pid'])
    
    console.print(table)
    
    # Step 3: Offer cleanup options
    if unique_pids:
        console.print(f"\nFound processes holding RTDE connections: {', '.join(unique_pids)}")
        
        if Confirm.ask("\n🔥 Kill zombie processes to free RTDE connections?", default=True):
            killed_any = False
            for pid in unique_pids:
                if kill_process(pid):
                    killed_any = True
                    time.sleep(0.5)  # Brief pause between kills
            
            if killed_any:
                console.print("\n⏳ Waiting for connections to close...")
                time.sleep(2)
                
                # Check if connections are gone
                remaining = check_zombie_connections(robot_ip)
                if not remaining:
                    console.print("[green]✅ All zombie connections cleared![/green]")
                else:
                    console.print(f"[yellow]⚠️ {len(remaining)} connection(s) still remain[/yellow]")
    
    # Step 4: Run built-in cleanup
    console.print()
    if Confirm.ask("🧹 Run built-in connection cleanup?", default=True):
        cleanup_connections()
        time.sleep(1)
    
    # Step 5: Test final connection
    console.print()
    success = test_robot_connection()
    
    # Final summary
    console.print("\n" + "="*50)
    if success:
        console.print("[bold green]🎉 RTDE Connection Issue RESOLVED![/bold green]")
        console.print("Your robot tools should now work properly.")
    else:
        console.print("[bold red]❌ Issue NOT resolved[/bold red]")
        console.print("\nPossible causes:")
        console.print("• Robot is not powered on or reachable")
        console.print("• Another application is using RTDE")
        console.print("• Robot needs to be restarted")
        console.print("• Network connectivity issues")
    console.print("="*50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {e}[/red]") 