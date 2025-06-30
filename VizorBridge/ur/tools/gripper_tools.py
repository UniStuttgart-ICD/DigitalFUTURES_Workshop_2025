"""
Gripper and Assembly Tools for LLM Agents
=========================================

Gripper control and assembly-related robot tools for LLM integration.
"""

from typing import Dict, Any
from smolagents import tool
from rich.console import Console
from ur.core.connection import get_robot, get_bridge
from ur.config.robot_config import (
    GRIPPER_DIGITAL_OUT, 
    GRIPPER_OPEN_TIME, 
    GRIPPER_CLOSE_TIME, 
    GRIPPER_OPEN_STATE,
    GRIPPER_CLOSE_STATE,
    GRIPPER_IO_TYPE,
    WAKEWORD
)
from ur.config.system_config import generate_gripper_status, generate_supply_status, generate_movement_status, STATUS_SUCCESS
import time
from ur.utils import validate_wakeword, send_immediate_response

console = Console()

def _publish_status_if_bridge_available(status_message: str) -> None:
    """Publish status message via bridge if available, otherwise skip gracefully."""
    bridge = get_bridge()
    if bridge and hasattr(bridge, 'publish_status'):
        bridge.publish_status(status_message)
        print(f"📡 [TOOL STATUS] {status_message}")

def _set_gripper_output(rtde_io_interface, state: bool) -> None:
    """Set gripper digital output based on configured IO type."""
    if GRIPPER_IO_TYPE == "standard":
        rtde_io_interface.setStandardDigitalOut(GRIPPER_DIGITAL_OUT, state)
    elif GRIPPER_IO_TYPE == "configurable":
        rtde_io_interface.setConfigurableDigitalOut(GRIPPER_DIGITAL_OUT, state)
    elif GRIPPER_IO_TYPE == "tool":
        rtde_io_interface.setToolDigitalOut(GRIPPER_DIGITAL_OUT, state)
    else:
        raise ValueError(f"Unknown gripper IO type: {GRIPPER_IO_TYPE}")

@tool
def control_gripper(action: str, wake_phrase: str = "") -> Dict[str, Any]:
    """Control robot gripper - opens or closes the gripper
    
    Args:
        action: Action to perform ('open' or 'close')
        wake_phrase: Safety wake phrase (must contain 'mave')
    
    Returns:
        Dict with operation result
    """
    print(f"[ALERT] Running control_gripper tool")
    if not validate_wakeword(wake_phrase):
        return {
            "success": False,
            "message": f"I'd love to {action} the gripper, but you forgot to say 'mave'. Safety protocols!",
            "action": "wakeword_required"
        }
    
    try:
        send_immediate_response(f"{action} gripper")
        robot = get_robot()
        if not robot.ensure_connected() or not robot.rtde_io:
            return {
                "success": False,
                "message": "Gripper control unavailable - RTDE IO interface not connected",
                "action": "gripper_unavailable"
            }
        
        # Determine open or close
        op = action.lower()
        if op == 'open':
            state = GRIPPER_OPEN_STATE
            wait_time = GRIPPER_OPEN_TIME
            message = "Gripper opened successfully."
            # Publish gripper progress status
            progress_status = generate_gripper_status("OPEN", "PROGRESS")
            _publish_status_if_bridge_available(progress_status)
        elif op == 'close':
            state = GRIPPER_CLOSE_STATE
            wait_time = GRIPPER_CLOSE_TIME
            message = "Gripper closed successfully."
            # Publish gripper progress status
            progress_status = generate_gripper_status("CLOSE", "PROGRESS")
            _publish_status_if_bridge_available(progress_status)
        else:
            return {
                "success": False,
                "message": f"Unknown gripper action: {action}",
                "action": "gripper_unknown"
            }
        
        # Set gripper state using configured IO type
        console.print(f"[yellow]Setting gripper to {op} (Digital Output {GRIPPER_DIGITAL_OUT} = {state})...[/yellow]")
        _set_gripper_output(robot.rtde_io, state)
        console.print(f"[green]✓[/green] Gripper {op} command sent, waiting {wait_time}s...")
        time.sleep(wait_time)
        console.print(f"[green]✓[/green] {message}")
        
        # Publish gripper completion status
        completion_status = generate_gripper_status(op.upper(), "COMPLETE")
        _publish_status_if_bridge_available(completion_status)
        
        return {
            "success": True,
            "message": message,
            "action": f"gripper_{op}",
            "wake_phrase_validated": True
        }
        
    except Exception as e:
        error_msg = f"Error controlling gripper: {str(e)}"
        console.print(f"[red]✗[/red] {error_msg}")
        return {
            "success": False,
            "message": f"Gripper malfunction detected! {error_msg}",
            "error": str(e)
        }

#@tool
def get_supply_element(element_length: str, wake_phrase: str = "") -> Dict[str, Any]:
    """Get element from supply station based on length
    
    Args:
        element_length: Length of element to retrieve (e.g., '40cm', '50cm')
        wake_phrase: Safety wake phrase (must contain 'mave')
    
    Returns:
        Dict with operation result
    """
    print(f"[ALERT] Running get_supply_element tool")
    if not validate_wakeword(wake_phrase):
        return {
            "success": False,
            "message": f"I'd love to get the {element_length} element, but you forgot to say 'mave'. Safety first!",
            "action": "wakeword_required"
        }
    
    try:
        send_immediate_response(f"retrieving {element_length} element from supply station")
        
        # Publish supply operation progress status
        progress_status = generate_supply_status(element_length, "RETRIEVE", "PROGRESS")
        _publish_status_if_bridge_available(progress_status)
        
        # Publish movement to supply station
        supply_movement_status = generate_movement_status("LINEAR", "SUPPLY", "PROGRESS")
        _publish_status_if_bridge_available(supply_movement_status)
        
        # Placeholder implementation with status updates
        console.print(f"[cyan]🤖[/cyan] Opening gripper for element pickup")
        console.print(f"[cyan]🤖[/cyan] Moving to {element_length} supply station")
        console.print(f"[cyan]🤖[/cyan] Closing gripper to grab element")
        
        # Publish supply operation completion status
        completion_status = generate_supply_status(element_length, "RETRIEVE", "COMPLETE")
        _publish_status_if_bridge_available(completion_status)
        
        message = f"Got the {element_length} element! *satisfied robot noises* Ready for placement."
        console.print(f"[green]✓[/green] {message}")
        
        return {
            "success": True,
            "message": message,
            "element_length": element_length,
            "action": "element_retrieved",
            "wake_phrase_validated": True
        }
        
    except Exception as e:
        error_msg = f"Error retrieving {element_length} element: {str(e)}"
        console.print(f"[red]✗[/red] {error_msg}")
        return {
            "success": False,
            "message": f"Supply station malfunction! {error_msg}",
            "error": str(e)
        }

#@tool
def place_element_at_position(x: float, y: float, z: float, wake_phrase: str = "") -> Dict[str, Any]:
    """Place currently held element at specified position
    
    Args:
        x: X coordinate in meters for element placement
        y: Y coordinate in meters for element placement  
        z: Z coordinate in meters for element placement
        wake_phrase: Safety wake phrase (must contain 'mave')
    
    Returns:
        Dict with operation result
    """
    print(f"[ALERT] Running place_element_at_position tool")
    if not validate_wakeword(wake_phrase):
        return {
            "success": False,
            "message": f"I'd love to place the element, but you forgot to say 'mave'. Safety protocols!",
            "action": "wakeword_required"
        }
    
    try:
        send_immediate_response(f"placing element at position ({x}, {y}, {z})")
        
        # Publish movement to assembly area
        assembly_movement_status = generate_movement_status("LINEAR", "ASSEMBLY", "PROGRESS")
        _publish_status_if_bridge_available(assembly_movement_status)
        
        # Placeholder implementation with status updates
        console.print(f"[cyan]🤖[/cyan] Moving to approach position")
        console.print(f"[cyan]🤖[/cyan] Moving to placement position")
        
        # Publish assembly area arrival
        at_assembly_status = generate_movement_status("LINEAR", "ASSEMBLY", "COMPLETE")
        _publish_status_if_bridge_available(at_assembly_status)
        
        message = f"Element placed at ({x:.2f}, {y:.2f}, {z:.2f}). I'll hold it steady until you secure it!"
        console.print(f"[green]✓[/green] {message}")
        
        return {
            "success": True,
            "message": message,
            "position": [x, y, z],
            "action": "element_placed",
            "holding": True,
            "wake_phrase_validated": True
        }
        
    except Exception as e:
        error_msg = f"Error placing element: {str(e)}"
        console.print(f"[red]✗[/red] {error_msg}")
        return {
            "success": False,
            "message": f"Placement failed! {error_msg}",
            "error": str(e)
        }

@tool
def confirm_task(wake_phrase: str = "") -> Dict[str, Any]:
    """
    Confirm the current task. Requires the wake-word 'mave' to publish success status.

    Args:
        wake_phrase: Safety wake phrase (must contain 'mave')

    Returns:
        Dict with operation result
    """
    print(f"[ALERT] Running confirm_task tool")
    # Immediate acknowledgment
    send_immediate_response(f"confirm task {wake_phrase}")

    # Wake word validation
    if not validate_wakeword(wake_phrase):
        return {
            "success": False,
            "message": f"Wake word '{WAKEWORD}' required for task confirmation",
            "action": "wakeword_required"
        }

    # Publish success status via bridge
    _publish_status_if_bridge_available(STATUS_SUCCESS)

    return {
        "success": True,
        "message": "Task confirmed. Success status published.",
        "action": "task_confirmed"
    }
