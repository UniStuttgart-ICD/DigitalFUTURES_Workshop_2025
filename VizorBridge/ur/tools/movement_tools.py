"""
Movement Tools for LLM Agents
=============================

Movement-related robot control tools for LLM integration.
"""

from typing import Dict, Any
from smolagents import tool

from ur.core.connection import get_robot, get_bridge
from ur.config.robot_config import WAKEWORD, HOME_POSITION, ROBOT_MOVE_SPEED, ROBOT_ACCELERATION, ROBOT_HOME_SPEED, ROBOT_HOME_ACCELERATION, MAX_RELATIVE_MOVE, MAX_ROBOT_REACH
from ur.config.system_config import generate_movement_status, generate_status_message
from ur.utils import validate_wakeword, send_immediate_response
import math

def _publish_status_if_bridge_available(status_message: str) -> None:
    """Publish status message via bridge if available, otherwise skip gracefully."""
    bridge = get_bridge()
    if bridge and hasattr(bridge, 'publish_status'):
        bridge.publish_status(status_message)
        print(f"📡 [TOOL STATUS] {status_message}")

def is_robot_at_home_position(joints: list | None) -> bool:
    """Check if robot is currently at home position."""
    if joints is None:
        return False
    
    # Convert HOME_POSITION from degrees to radians for comparison
    home_joints_rad = [math.radians(angle) for angle in HOME_POSITION]
    
    # Check if current joints are close to home position (tolerance: 0.1 radians ≈ 5.7 degrees)
    tolerance = 0.1
    
    if len(joints) != len(home_joints_rad):
        return False
    
    for current, home in zip(joints, home_joints_rad):
        if abs(current - home) > tolerance:
            return False
    
    return True

@tool
def move_relative_xyz(dx_m: float, dy_m: float, dz_m: float, wake_phrase: str = "") -> Dict[str, Any]:
    """Move the robot TCP relative to its current pose in X, Y, Z directions.
    
    Args:
        dx_m: Relative distance in X direction (meters)
        dy_m: Relative distance in Y direction (meters)
        dz_m: Relative distance in Z direction (meters)
        wake_phrase: Must contain the wake word 'mave' for safety
    
    Returns:
        Dictionary with status and TCP pose information [x, y, z, rx, ry, rz]
    """
    print(f"[ALERT] Running move_relative_xyz tool")
    # Send immediate acknowledgment
    send_immediate_response(f"move ({dx_m}, {dy_m}, {dz_m})m relative {wake_phrase}")
    
    # Wake word validation
    if not validate_wakeword(wake_phrase):
        return {
            "status": "error",
            "message": f"Wake word '{WAKEWORD}' required for robot movement",
            "pose": None
        }
    
    robot = get_robot()
    # Ensure robot connection
    if not robot.ensure_connected() or not getattr(robot, 'rtde_c', None):
        return {"status": "error", "message": "Robot not connected", "pose": None}
    
    # Validate relative move limits
    for val, name in ((dx_m, 'X'), (dy_m, 'Y'), (dz_m, 'Z')):
        if abs(val) > MAX_RELATIVE_MOVE:
            return {"status": "error", "message": f"Invalid relative {name} movement: {val}m exceeds maximum {MAX_RELATIVE_MOVE}m", "pose": None}
    
    # Get current TCP pose
    current_pose = robot.get_tcp_pose()
    if current_pose is None:
        return {"status": "error", "message": "Failed to read current robot pose", "pose": None}
    
    # Compute target pose
    target_pose = [
        current_pose[0] + dx_m,
        current_pose[1] + dy_m,
        current_pose[2] + dz_m,
        current_pose[3],
        current_pose[4],
        current_pose[5]
    ]
    
    try:
        # Publish movement start status
        movement_status = generate_movement_status("RELATIVE", status_type="PROGRESS")
        _publish_status_if_bridge_available(movement_status)
        
        robot.rtde_c.moveL(target_pose, ROBOT_MOVE_SPEED, ROBOT_ACCELERATION, asynchronous=True)
        
        # Wait for movement to complete
        if robot.wait_for_movement_completion():
            # Publish movement completion status
            completion_status = generate_movement_status("RELATIVE", status_type="COMPLETE")
            _publish_status_if_bridge_available(completion_status)
            
            return {"status": "success", "message": f"Moved ({dx_m}, {dy_m}, {dz_m})m in (X,Y,Z) directions", "pose": target_pose}
        else:
            return {"status": "error", "message": "Movement timed out or failed to complete", "pose": None}
    except Exception as e:
        return {"status": "error", "message": f"Movement failed: {e}", "pose": None}

#@tool
def move_to_absolute_position(x_m: float, y_m: float, z_m: float, wake_phrase: str = "") -> Dict[str, Any]:
    """Move the robot TCP to an absolute pose in Cartesian space.
    
    Args:
        x_m: Target TCP X position in meters
        y_m: Target TCP Y position in meters
        z_m: Target TCP Z position in meters
        wake_phrase: Must contain the wake word 'mave' for safety
    
    Returns:
        Dictionary with status and TCP pose information [x, y, z, rx, ry, rz]
    """
    print(f"[ALERT] Running move_to_absolute_position tool")
    
    # Send immediate acknowledgment
    send_immediate_response(f"move to absolute position ({x_m}, {y_m}, {z_m})m {wake_phrase}")
    
    # Wake word validation
    if not validate_wakeword(wake_phrase):
        return {
            "status": "error",
            "message": f"Wake word '{WAKEWORD}' required for robot movement",
            "pose": None
        }
    
    robot = get_robot()
    # Ensure robot connection
    if not robot.ensure_connected() or not getattr(robot, 'rtde_c', None):
        return {"status": "error", "message": "Robot not connected", "pose": None}
    
    # Validate absolute reach limits
    for val, name in ((x_m, 'X'), (y_m, 'Y'), (z_m, 'Z')):
        if abs(val) > MAX_ROBOT_REACH:
            return {"status": "error", "message": f"Invalid absolute {name} position: {val}m exceeds maximum reach {MAX_ROBOT_REACH}m", "pose": None}
    
    # Form target pose [x, y, z, rx, ry, rz]
    target_pose = [x_m, y_m, z_m, 0, 3.14, 0]
    
    try:
        # Publish movement start status
        movement_status = generate_movement_status("ABSOLUTE", status_type="PROGRESS")
        _publish_status_if_bridge_available(movement_status)
        
        robot.rtde_c.moveL(target_pose, ROBOT_MOVE_SPEED, ROBOT_ACCELERATION, asynchronous=True)
        
        # Wait for movement to complete
        if robot.wait_for_movement_completion():
            # Publish movement completion status
            completion_status = generate_movement_status("ABSOLUTE", status_type="COMPLETE")
            _publish_status_if_bridge_available(completion_status)
            
            return {"status": "success", "message": f"Moved to absolute position ({x_m}, {y_m}, {z_m})m", "pose": target_pose}
        else:
            return {"status": "error", "message": "Movement timed out or failed to complete", "pose": None}
    except Exception as e:
        return {"status": "error", "message": f"Movement failed: {e}", "pose": None}

@tool
def move_home(wake_phrase: str = "") -> Dict[str, Any]:
    """Send the robot to its predefined home position using joint movement.
    
    Args:
        wake_phrase: Must contain the wake word 'mave' for safety
    
    Returns:
        Dictionary with status information
    """
    print(f"[ALERT] Running move_home tool")
    # Send immediate acknowledgment
    send_immediate_response(f"checking position and moving home {wake_phrase}")
    
    # Wake word validation
    if not validate_wakeword(wake_phrase):
        return {
            "status": "error",
            "message": f"Wake word '{WAKEWORD}' required for robot movement"
        }
    
    robot = get_robot()
    # Check current joints
    current_joints = robot.get_joints()
    if is_robot_at_home_position(current_joints):
        # Publish status for already at home
        home_status = generate_movement_status("HOME", status_type="COMPLETE")
        _publish_status_if_bridge_available(home_status)
        
        return {"status": "success", "message": "Robot is already at home position! No movement needed.", "already_at_home": True}
    
    # Prepare home joint positions in radians
    home_joints_rad = [math.radians(angle) for angle in HOME_POSITION]
    
    try:
        # Publish movement start status
        movement_status = generate_movement_status("HOME", status_type="PROGRESS")
        _publish_status_if_bridge_available(movement_status)
        
        robot.rtde_c.moveJ(home_joints_rad, ROBOT_HOME_SPEED, ROBOT_HOME_ACCELERATION, asynchronous=True)
        
        # Wait for movement to complete
        if robot.wait_for_movement_completion():
            # Publish movement completion status
            completion_status = generate_movement_status("HOME", status_type="COMPLETE")
            _publish_status_if_bridge_available(completion_status)
            
            return {"status": "success", "message": "Robot moved to home position", "already_at_home": False}
        else:
            return {"status": "error", "message": "Home movement timed out or failed to complete", "already_at_home": False}
    except Exception as e:
        return {"status": "error", "message": f"Failed to move robot home: {e}", "already_at_home": False}

@tool
def get_robot_state() -> Dict[str, Any]:
    """Get current robot joint positions and TCP pose.
    
    Returns:
        Dictionary with robot state information including home position status
    """
    print(f"[ALERT] Running get_robot_state tool")
    # Send immediate acknowledgment
    send_immediate_response("robot status check")
    
    robot = get_robot()
    joints = robot.get_joints()
    tcp_pose = robot.get_tcp_pose()
    
    # Check if robot is at home position
    at_home = is_robot_at_home_position(joints)
    
    return {
        "status": "success" if (joints is not None and tcp_pose is not None) else "error",
        "joints": joints,
        "tcp_pose": tcp_pose,
        "connected": robot.is_connected,
        "at_home_position": at_home,
        "home_position_joints": HOME_POSITION
    }

@tool
def stop_robot(wake_phrase: str = "") -> Dict[str, Any]:
    """Emergency stop all robot movements and halt execution.
    
    Args:
        wake_phrase: Must contain the wake word 'mave' for safety
    
    Returns:
        Dictionary with stop operation status
    """
    print(f"[ALERT] Running stop_robot tool")
    # Emergency stop doesn't need wake word validation - safety first!
    send_immediate_response(f"EMERGENCY STOP {wake_phrase}")
    
    robot = get_robot()
    # Ensure robot connection
    if not robot.ensure_connected() or not getattr(robot, 'rtde_c', None):
        return {"status": "error", "message": "Robot not connected - cannot send stop command", "stopped": False}
    
    try:
        # Stop all movements immediately
        robot.rtde_c.stopL()  # Stop linear movements
        robot.rtde_c.stopJ()  # Stop joint movements
        robot.rtde_c.stopScript()  # Stop any running scripts
        
        # Publish emergency stop status
        stop_status = generate_status_message("EMERGENCY_STOP", status_type="COMPLETE")
        _publish_status_if_bridge_available(stop_status)
        
        return {"status": "success", "message": "Robot stopped successfully - all movements halted", "stopped": True}
    except Exception as e:
        return {"status": "error", "message": f"Failed to stop robot: {e}", "stopped": False}