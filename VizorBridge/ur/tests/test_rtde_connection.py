import time
import rtde_control
import rtde_receive
import rtde_io

# Rich imports for better console output
from rich.console import Console
from rich.panel import Panel

# Import robot configuration directly to avoid circular imports
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ur.config.robot_config import ROBOT_IP

from ur.config.robot_config import (
    GRIPPER_DIGITAL_OUT, 
    GRIPPER_OPEN_STATE, 
    GRIPPER_CLOSE_STATE,
    GRIPPER_OPEN_TIME,
    GRIPPER_CLOSE_TIME,
    GRIPPER_IO_TYPE
)

console = Console()

def control_gripper(rtde_io_interface, open_gripper: bool = True) -> None:
    """Control gripper using digital output."""
    state = GRIPPER_OPEN_STATE if open_gripper else GRIPPER_CLOSE_STATE
    action = "open" if open_gripper else "close"
    wait_time = GRIPPER_OPEN_TIME if open_gripper else GRIPPER_CLOSE_TIME
    
    console.print(f"[yellow]Setting gripper to {action} (Digital Output {GRIPPER_DIGITAL_OUT} = {state})...[/yellow]")
    
    # Set digital output based on configured IO type
    if GRIPPER_IO_TYPE == "standard":
        rtde_io_interface.setStandardDigitalOut(GRIPPER_DIGITAL_OUT, state)
    elif GRIPPER_IO_TYPE == "configurable":
        rtde_io_interface.setConfigurableDigitalOut(GRIPPER_DIGITAL_OUT, state)
    elif GRIPPER_IO_TYPE == "tool":
        rtde_io_interface.setToolDigitalOut(GRIPPER_DIGITAL_OUT, state)
    else:
        console.print(f"[red]✗[/red] Unknown gripper IO type: {GRIPPER_IO_TYPE}")
        return
    
    console.print(f"[green]✓[/green] Gripper {action} command sent, waiting {wait_time}s...")
    time.sleep(wait_time)

def main(robot_ip: str) -> None:
    """Simple connectivity check with a UR robot via RTDE including gripper control."""
    console.print(Panel(f"[bold cyan]RTDE Connection & Gripper Test[/bold cyan]\nRobot IP: {robot_ip}", expand=False))
    
    # Initialize control, receive, and IO interfaces
    console.print("[cyan]Initializing RTDE interfaces...[/cyan]")
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    rtde_io_interface = rtde_io.RTDEIOInterface(robot_ip)

    try:        # Get current joint positions
        actual_q = rtde_r.getActualQ()
        console.print(f"[green]✓[/green] Current joint positions: {actual_q}")
        
        # Get current TCP pose
        actual_tcp_pose = rtde_r.getActualTCPPose()
        console.print(f"[green]✓[/green] Current TCP pose: {actual_tcp_pose}")
        
        # Test movement - move to current position (safe ping)
        console.print("[yellow]Sending moveJ to current position as connectivity test...[/yellow]")
        rtde_c.moveJ(actual_q)
        time.sleep(0.5)
        
        # Move TCP up by 150mm (0.15m) in Z-axis
        console.print("[yellow]Moving TCP up by 150mm...[/yellow]")
        target_pose = actual_tcp_pose.copy()
        target_pose[2] += 0.15  # Move up 150mm in Z-axis
        rtde_c.moveL(target_pose, 0.25, 0.3)  # Speed: 0.25 m/s, Acceleration: 0.3 m/s²
        time.sleep(1.0)
          # Move back to original position
        console.print("[yellow]Moving back to original position...[/yellow]")
        rtde_c.moveL(actual_tcp_pose, 0.25, 0.3)
        time.sleep(1.0)
        
        console.print("[green]✓[/green] RTDE communication and TCP movement successful!")
        
        # Test gripper control
        console.print("\n[cyan]Testing gripper control...[/cyan]")
        
        # Close gripper
        control_gripper(rtde_io_interface, open_gripper=False)
        
        # Open gripper  
        control_gripper(rtde_io_interface, open_gripper=True)
        
        # Close gripper again
        control_gripper(rtde_io_interface, open_gripper=False)
        
        console.print("[green]✓[/green] Gripper control test completed!")
        console.print("\n[bold green]🎉 All RTDE and gripper tests completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error during RTDE communication or gripper control: {e}")
        raise
    finally:
        # Always stop the script to clean up
        rtde_c.stopScript()


if __name__ == "__main__":
    import sys

    ip = sys.argv[1] if len(sys.argv) > 1 else ROBOT_IP
    main(ip)
