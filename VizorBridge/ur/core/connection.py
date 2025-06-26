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
    
    def __init__(self, robot_ip: str = "192.168.56.101"):
        self.robot_ip = robot_ip
        self.rtde_c: Optional[Any] = None
        self.rtde_r: Optional[Any] = None
        self.rtde_io: Optional[Any] = None
        self.is_connected = False
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
        """Wait for robot movement to complete using smart detection methods.
        
        Uses multiple detection methods for robust movement completion detection:
        1. Primary: getAsyncOperationProgress() for real-time 0-100% completion
        2. Secondary: isSteady() with velocity verification 
        3. Tertiary: isProgramRunning() for program state awareness
        4. Quaternary: Velocity-based detection using getActualQd()
        
        Args:
            timeout: Maximum time to wait in seconds (increased to 60s for slow movements)
            show_progress: Whether to display progress updates
            
        Returns:
            True if movement completed, False if timeout or error
        """
        if not self.ensure_connected() or not self.rtde_c:
            return False
            
        import time
        start_time = time.time()
        last_progress = -1
        consecutive_steady_checks = 0
        steady_required = 3  # Require 3 consecutive steady checks for reliability
        
        try:
            # Wait a short time to ensure movement has started
            time.sleep(0.1)
            
            while time.time() - start_time < timeout:
                elapsed = time.time() - start_time
                
                # Method 1: Primary - getAsyncOperationProgress() 
                try:
                    progress = self.rtde_c.getAsyncOperationProgress()
                    if progress is not None:
                        # Progress is 0-100, where 100 means completed
                        if show_progress and int(progress) != last_progress and int(progress) % 10 == 0:
                            console.print(f"[blue]📈[/blue] Movement progress: {int(progress)}%")
                            last_progress = int(progress)
                        
                        if progress >= 100.0:
                            console.print(f"[green]✓[/green] Movement completed (progress: 100%)")
                            return True
                        
                        # If we have good progress data, continue monitoring
                        time.sleep(0.1)
                        continue
                except Exception as e:
                    # getAsyncOperationProgress might not be available, fall back to other methods
                    if elapsed < 1.0:  # Only log this once early on
                        console.print(f"[yellow]⚠[/yellow] Progress monitoring unavailable, using fallback methods")
                
                # Method 2: Secondary - Enhanced isSteady() with velocity check
                try:
                    is_steady = self.rtde_c.isSteady()
                    velocities = self.rtde_r.getActualQd() if self.rtde_r else None
                    
                    # Check if robot is steady AND velocities are near zero
                    velocity_near_zero = True
                    if velocities and len(velocities) == 6:
                        # Check if all joint velocities are below threshold (rad/s)
                        velocity_threshold = 0.01  # Very small velocity threshold
                        velocity_near_zero = all(abs(v) < velocity_threshold for v in velocities)
                    
                    if is_steady and velocity_near_zero:
                        consecutive_steady_checks += 1
                        if consecutive_steady_checks >= steady_required:
                            console.print(f"[green]✓[/green] Movement completed (steady + zero velocity)")
                            return True
                    else:
                        consecutive_steady_checks = 0
                        
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Steady check failed: {e}")
                
                # Method 3: Tertiary - Program running state check
                try:
                    # If no program is running, movement should be done
                    program_running = self.rtde_c.isProgramRunning()
                    if program_running is False:
                        # Double-check with steady state
                        if self.rtde_c.isSteady():
                            console.print(f"[green]✓[/green] Movement completed (no program running + steady)")
                            return True
                except Exception:
                    pass  # This method is optional
                
                # Show periodic updates for long movements
                if show_progress and elapsed > 10 and int(elapsed) % 10 == 0:
                    console.print(f"[blue]⏱[/blue] Still moving... {int(elapsed)}s elapsed")
                
                time.sleep(0.1)  # Check every 100ms for responsiveness
                
            # Timeout reached - try final verification and diagnostics
            console.print(f"[yellow]⚠[/yellow] Movement timeout after {timeout}s, doing final checks...")
            
            # Final verification: Check if robot is actually steady despite timeout
            try:
                is_steady = self.rtde_c.isSteady()
                velocities = self.rtde_r.getActualQd() if self.rtde_r else None
                program_running = self.rtde_c.isProgramRunning()
                
                # Check robot status for safety issues
                robot_status = None
                safety_status = None
                try:
                    robot_status = self.rtde_c.getRobotStatus() if hasattr(self.rtde_c, 'getRobotStatus') else None
                    safety_status = self.rtde_r.getSafetyStatusBits() if self.rtde_r else None
                except:
                    pass
                
                if is_steady and velocities and all(abs(v) < 0.01 for v in velocities):
                    console.print(f"[green]✓[/green] Movement actually completed (post-timeout verification)")
                    return True
                
                # Diagnose why movement failed
                console.print(f"[blue]🔍[/blue] Movement failure diagnostics:")
                console.print(f"  - Robot steady: {is_steady}")
                console.print(f"  - Program running: {program_running}")
                
                if velocities:
                    max_vel = max(abs(v) for v in velocities)
                    console.print(f"  - Max joint velocity: {max_vel:.4f} rad/s")
                else:
                    console.print(f"  - Joint velocities: unavailable")
                
                # Check for safety issues
                if safety_status is not None:
                    if safety_status & 0x4:  # Bit 2: Protective stopped
                        console.print(f"[red]🛡[/red] Robot is in PROTECTIVE STOP - likely collision detected!")
                    elif safety_status & 0x10:  # Bit 4: Safeguard stopped
                        console.print(f"[yellow]🛡[/yellow] Robot is in SAFEGUARD STOP")
                    elif safety_status & 0x100:  # Bit 8: Emergency stopped
                        console.print(f"[red]🚨[/red] Robot is in EMERGENCY STOP")
                    elif safety_status & 0x200:  # Bit 9: Violation
                        console.print(f"[red]⚠[/red] Safety violation detected")
                    elif safety_status & 0x400:  # Bit 10: Fault
                        console.print(f"[red]⚠[/red] Safety fault detected")
                    else:
                        console.print(f"  - Safety status: 0x{safety_status:04x} (normal)")
                
                if robot_status is not None:
                    if not (robot_status & 0x1):  # Bit 0: Power on
                        console.print(f"[red]⚡[/red] Robot power is OFF")
                    if robot_status & 0x2:  # Bit 1: Program running
                        console.print(f"[blue]📋[/blue] Program still running")
                
                # Provide helpful suggestions
                if program_running:
                    console.print(f"[red]⚠[/red] Robot still executing movement after timeout!")
                    console.print(f"[yellow]💡[/yellow] Suggestion: Increase timeout or check for obstacles")
                else:
                    console.print(f"[yellow]⚠[/yellow] Robot stopped but may not have reached target")
                    if safety_status and (safety_status & 0x4):
                        console.print(f"[yellow]💡[/yellow] Suggestion: Check for self-collision or workspace limits")
                    else:
                        console.print(f"[yellow]💡[/yellow] Suggestion: Check target position validity or increase speeds")
                    
            except Exception as e:
                console.print(f"[red]✗[/red] Final verification failed: {e}")
            
            return False
            
        except Exception as e:
            console.print(f"[red]✗[/red] Error waiting for movement: {e}")
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
        """Clean up connections."""
        try:
            if self.rtde_c:
                # Stop any running scripts and movements
                try:
                    self.rtde_c.stopL()  # Stop linear movements
                    self.rtde_c.stopJ()  # Stop joint movements
                    console.print("[yellow]🛑[/yellow] Robot movements stopped")
                except:
                    pass
                
                try:
                    self.rtde_c.stopScript()  # Stop any running scripts
                    console.print("[yellow]📄[/yellow] Robot scripts stopped")
                except:
                    pass
        except:
            pass
        finally:
            self.is_connected = False
            self._cleanup_connections()
            console.print("[green]✅[/green] Robot connection cleaned up")


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
