"""
UR Robot RTDE Connection Management
==================================

Handles RTDE connections, basic robot communication, and bridge management for tool integration.
"""

import os
import sys
import threading
from typing import Optional, List, Any
from rich.console import Console

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ur.config.robot_config import ROBOT_IP

# Try to import RTDE libraries
try:
    import rtde_control
    import rtde_receive
    import rtde_io
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False

console = Console()

# Global RTDE connection lock to prevent concurrent access
_rtde_lock = threading.Lock()
_rtde_instances = {
    'control': None,
    'receive': None,
    'io': None
}

def _get_or_create_rtde_interface(interface_type: str, robot_ip: str):
    """Get or create a singleton RTDE interface instance.
    
    This ensures only one instance of each RTDE interface type exists globally,
    preventing the "RTDE input registers are already in use" error.
    """
    global _rtde_instances
    
    with _rtde_lock:
        # Return existing instance if available and healthy
        existing = _rtde_instances.get(interface_type)
        if existing is not None:
            try:
                # Test if the connection is still alive
                if interface_type == 'receive':
                    # Quick health check
                    joints = existing.getActualQ()
                    if joints is not None and len(joints) == 6:
                        return existing
                elif interface_type == 'control':
                    # For control interface, check if it's responsive
                    if existing.isSteady() is not None:
                        return existing
                elif interface_type == 'io':
                    # For IO interface, just return it (harder to test)
                    return existing
            except:
                # Connection is dead, will create new one
                pass
        
        # Create new instance
        try:
            if interface_type == 'control':
                instance = rtde_control.RTDEControlInterface(robot_ip)
            elif interface_type == 'receive':
                instance = rtde_receive.RTDEReceiveInterface(robot_ip)
            elif interface_type == 'io':
                instance = rtde_io.RTDEIOInterface(robot_ip)
            else:
                raise ValueError(f"Unknown interface type: {interface_type}")
            
            _rtde_instances[interface_type] = instance
            return instance
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to create {interface_type} interface: {e}")
            _rtde_instances[interface_type] = None
            raise

def _cleanup_rtde_interfaces():
    """Clean up all RTDE interface instances."""
    global _rtde_instances
    
    with _rtde_lock:
        for interface_type, instance in _rtde_instances.items():
            if instance is not None:
                try:
                    instance.disconnect()
                except:
                    pass
        
        _rtde_instances = {
            'control': None,
            'receive': None,
            'io': None
        }

class URConnection:
    """RTDE connection manager for UR robot."""
    
    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self.rtde_c: Optional[Any] = None
        self.rtde_r: Optional[Any] = None
        self.rtde_io: Optional[Any] = None
        self.is_connected = False
        self._is_shutting_down = False
        self._connect()
    
    def _connect(self):
        """Establish RTDE connections using singleton interfaces."""
        if not RTDE_AVAILABLE:
            console.print("[yellow]⚠[/yellow] RTDE libraries not available - simulation mode")
            return
        
        # Check if connections are already active and working
        if self._test_existing_connections():
            console.print("[green]✓[/green] Existing robot connections are healthy")
            self.is_connected = True
            return
        
        try:
            # Use singleton RTDE interfaces to prevent conflicts
            self.rtde_c = _get_or_create_rtde_interface('control', self.robot_ip)
            self.rtde_r = _get_or_create_rtde_interface('receive', self.robot_ip)
            self.rtde_io = _get_or_create_rtde_interface('io', self.robot_ip)
            self.is_connected = True
            console.print(f"[green]✓[/green] Robot connected to {self.robot_ip}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to connect to robot: {e}")
            self.is_connected = False
            # Don't cleanup global interfaces on individual connection failure
            self.rtde_c = None
            self.rtde_r = None
            self.rtde_io = None
    
    def _test_existing_connections(self) -> bool:
        """Test if existing RTDE connections are still valid and responsive."""
        try:
            # Test if all required interfaces exist
            if not (hasattr(self, 'rtde_c') and hasattr(self, 'rtde_r') and hasattr(self, 'rtde_io')):
                return False
            
            if not (self.rtde_c and self.rtde_r and self.rtde_io):
                return False
            
            # Test connection by trying to get basic data
            # This is a lightweight check that verifies RTDE is responding
            joints = self.rtde_r.getActualQ()
            if joints is None or len(joints) != 6:
                return False
                
            # Connection appears healthy
            return True
        except Exception:
            # Any exception means connection is not healthy
            return False
    
    def _cleanup_connections(self):
        """Clean up local RTDE connection references."""
        # Only clean up local references, not the global singletons
        # The global singletons are managed by _cleanup_rtde_interfaces()
        self.rtde_c = None
        self.rtde_r = None
        self.rtde_io = None
    
    def ensure_connected(self) -> bool:
        """Ensure robot connection is active."""
        # First, test if we're actually connected even if flag says we're not
        if not self.is_connected and self._test_existing_connections():
            console.print("[green]✓[/green] Connection recovered - was actually healthy")
            self.is_connected = True
            return True
            
        # Only reconnect if we're truly disconnected
        if not self.is_connected:
            console.print("[yellow]🔄[/yellow] Attempting to reconnect...")
            self._connect()
        return self.is_connected
    
    def get_joints(self) -> Optional[List[float]]:
        """Get current joint positions."""
        if not self.ensure_connected() or not self.rtde_r:
            return None
        try:
            joints = self.rtde_r.getActualQ()
            # Validate the data before returning
            if joints is None or len(joints) != 6:
                console.print(f"[yellow]⚠[/yellow] Invalid joint data received")
                return None
            return joints
        except Exception as e:
            console.print(f"[red]✗[/red] Error getting joints: {e}")
            # Don't immediately mark as disconnected - might be temporary
            # self.is_connected = False
            return None
    
    def get_tcp_pose(self) -> Optional[List[float]]:
        """Get current TCP pose."""
        if not self.ensure_connected() or not self.rtde_r:
            return None
        try:
            pose = self.rtde_r.getActualTCPPose()
            # Validate the data before returning
            if pose is None or len(pose) != 6:
                console.print(f"[yellow]⚠[/yellow] Invalid TCP pose data received")
                return None
            return pose
        except Exception as e:
            console.print(f"[red]✗[/red] Error getting TCP pose: {e}")
            # Don't immediately mark as disconnected - might be temporary
            # self.is_connected = False
            return None
    
    def wait_for_movement_completion(self, timeout: float = 60.0, show_progress: bool = True) -> bool:
        import time
        start = time.time()
        last_pct = -1.0

        # Short initial sleep to let the move start
        time.sleep(0.1)

        while time.time() - start < timeout:
            # Primary: Ex version
            try:
                status = self.rtde_c.getAsyncOperationProgressEx()
                if show_progress:
                    pct = int(status.progress * 100)
                    if pct != last_pct and pct % 10 == 0:
                        console.print(f"[blue]📈[/blue] Move progress: {pct}%")
                        last_pct = pct
                if status.completed:
                    console.print("[green]✓[/green] Movement completed")
                    return True
                time.sleep(0.1)
                continue
            except Exception:
                # If not supported, fall back
                pass

            # Fallback: basic progress
            try:
                prog = self.rtde_c.getAsyncOperationProgress()
                # Show raw percentage if available
                if show_progress and isinstance(prog, (int, float)):
                    pct = int(prog)
                    if pct != last_pct and pct % 10 == 0:
                        console.print(f"[blue]📈[/blue] Move progress: {pct}%")
                        last_pct = pct
                # Completion signals: negative or full 100%
                if prog is not None and (prog < 0 or prog >= 100):
                    console.print("[green]✓[/green] Movement completed (fallback)")
                    return True
                time.sleep(0.1)
                continue
            except Exception:
                pass

            # Velocity check
            speed = self.rtde_r.getActualTCPSpeed() if self.rtde_r else None
            frac  = self.rtde_r.getTargetSpeedFraction() if self.rtde_r else None
            if speed is not None and speed < 1e-3 and frac is not None and frac == 0.0:
                console.print("[green]✓[/green] Movement completed (velocity)")
                return True

            # Steady & program check
            if self.rtde_c.isSteady() and not self.rtde_c.isProgramRunning():
                console.print("[green]✓[/green] Movement completed (steady & idle)")
                return True

            time.sleep(0.1)

        console.print(f"[yellow]⚠[/yellow] Timeout after {timeout}s")
        return False

    
    def move_l(self, pose: List[float], speed: float = 0.25, acceleration: float = 1.2) -> bool:
        """Move robot linearly to target pose."""
        if not self.ensure_connected() or not self.rtde_c:
            return False
        try:
            self.rtde_c.moveL(pose, speed, acceleration, asynchronous=True)
            return self.wait_for_movement_completion()
        except Exception as e:
            console.print(f"[red]✗[/red] Linear movement failed: {e}")
            self.is_connected = False
            return False
    
    def move_j(self, joint_positions: List[float], speed: float = 0.25, acceleration: float = 1.2) -> bool:
        """Move robot in joint space to target positions."""
        if not self.ensure_connected() or not self.rtde_c:
            return False
        try:
            self.rtde_c.moveJ(joint_positions, speed, acceleration, asynchronous=True)
            return self.wait_for_movement_completion()
        except Exception as e:
            console.print(f"[red]✗[/red] Joint movement failed: {e}")
            self.is_connected = False
            return False
    
    def cleanup(self):
        """Gracefully stop robot and clean up connections."""
        console.print("[yellow]🧹[/yellow] Cleaning up robot connection...")
        self._is_shutting_down = True
        
        # Stop any running scripts or movements safely
        if self.rtde_c and self.is_connected:
            try:
                if self.rtde_c.isProgramRunning():
                    self.rtde_c.stopScript()
                    console.print("[blue]🛑[/blue] Robot scripts stopped")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to stop robot script during cleanup: {e}")
        
        # Clean up local RTDE interface references
        self._cleanup_connections()
        self.is_connected = False
        console.print("[green]✓[/green] Robot connection cleaned up")


# Global robot instance management
_robot: Optional[URConnection] = None

def get_robot() -> URConnection:
    """Get or create global robot instance."""
    global _robot
    if _robot is None:
        _robot = URConnection(ROBOT_IP)
    elif not _robot.is_connected:
        # Try to reconnect if not connected
        _robot.ensure_connected()
    return _robot

def cleanup_global_robot():
    """Clean up global robot connection and RTDE interfaces."""
    global _robot
    if _robot is not None:
        _robot.cleanup()
        _robot = None
    # Also cleanup the global RTDE interface singletons
    _cleanup_rtde_interfaces()
    console.print("[green]✅[/green] Global robot and RTDE interfaces cleaned up")


# Bridge management for tool integration
# =====================================

_bridge_instance: Optional[Any] = None
_bridge_lock = threading.Lock()

def set_bridge(bridge) -> None:
    """Register the bridge instance for tool access.
    
    Args:
        bridge: URBridge instance that tools can use for ROS communication
        
    Thread-safe: Yes - uses threading.Lock for atomic operations
    """
    global _bridge_instance
    with _bridge_lock:
        _bridge_instance = bridge
        if bridge:
            console.print(f"[green]✓[/green] Bridge registered for tool access: {bridge.name}")
        else:
            console.print("[yellow]⚠[/yellow] Bridge unregistered")

def get_bridge() -> Optional[Any]:
    """Get the current bridge instance for tool access.
    
    Returns:
        URBridge instance if registered, None otherwise (graceful fallback)
        
    Thread-safe: Yes - uses threading.Lock for consistent reads
    
    Note: 
        Tools should check if bridge is None and handle gracefully.
        This allows tools to work in testing scenarios without ROS.
    """
    with _bridge_lock:
        return _bridge_instance

def cleanup_bridge():
    """Clean up bridge reference (called during shutdown)."""
    global _bridge_instance
    with _bridge_lock:
        if _bridge_instance:
            console.print("[yellow]🧹[/yellow] Bridge reference cleaned up")
        _bridge_instance = None

def is_bridge_available() -> bool:
    """Check if bridge is available for tool use.
    
    Returns:
        True if bridge is registered and available, False otherwise
    """
    with _bridge_lock:
        return _bridge_instance is not None

def cleanup_all():
    """Clean up all global connections and references."""
    cleanup_global_robot()
    cleanup_bridge()
    console.print("[green]🧹[/green] All connections cleaned up")
