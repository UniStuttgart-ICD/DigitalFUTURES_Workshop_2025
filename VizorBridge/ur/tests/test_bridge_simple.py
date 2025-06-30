#!/usr/bin/env python3
"""
Simple Bridge Test - No LLM Integration

A minimal robot bridge that:
- Receives ROS messages (tasks and commands)
- Executes robot tasks with gripper control
- Publishes status and position updates
- Handles START/END fabrication commands
- No LLM agent integration

Usage:
    python test_bridge_simple.py
"""

import time
import roslibpy
import sys
import os
import math
import threading
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.live import Live
from rich.errors import LiveError
from rich.panel import Panel
from rich.text import Text

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from RobotInterface import RobotInterface
from ur.core.connection import get_robot, set_bridge
from ur.config.robot_config import (
    ROBOT_NAME,
    ROBOT_IP,
    ROBOT_MOVE_SPEED,
    ROBOT_ACCELERATION,
    ROBOT_HOME_SPEED,
    ROBOT_HOME_ACCELERATION,
    HOME_POSITION,
    GRIPPER_DIGITAL_OUT,
    GRIPPER_OPEN_TIME,
    GRIPPER_CLOSE_TIME,
    GRIPPER_OPEN_STATE,
    GRIPPER_CLOSE_STATE,
)
from ur.config.topics import (
    TASK_EXECUTE_TOPIC,
    COMMAND_TOPIC,
    STATUS_TOPIC,
    POSITION_TOPIC,
    TASK_EXECUTE_MSG_TYPE,
    STD_STRING_MSG_TYPE,
    POSITION_MSG_TYPE,
    ROS_HOST
)
from ur.config.system_config import (
    COMMAND_START,
    COMMAND_END,
    STATUS_FABRICATION_STARTED,
    STATUS_FABRICATION_COMPLETE,
    STATUS_GRIPPER_OPEN,
    STATUS_GRIPPER_CLOSE,
    generate_task_status,
    STATUS_SUCCESS,
    STATUS_FAILED,
    generate_movement_status,
)
from ur.tools.gripper_tools import _set_gripper_output


class SimpleBridge(RobotInterface):
    """
    Simple robot bridge without LLM integration.
    
    Handles:
    - ROS task execution 
    - Command processing (START/END fabrication)
    - Status and position publishing
    - Robot movement and gripper control
    """

    def __init__(
        self,
        name: str = ROBOT_NAME,
        client: roslibpy.Ros = None,
        robot_ip: str | None = None,
        task_topic: str = TASK_EXECUTE_TOPIC,
    ):
        super().__init__(name, client)
        self.robot_ip = robot_ip or ROBOT_IP
        
        # Robot connection
        self.robot_connection = get_robot()
        self.is_connected = self.robot_connection.is_connected

        # Simple fabrication state
        self.fabrication_active = False
        
        # Rich console for better output
        self.console = Console()
        self.current_progress = None
        self.current_task_id = None
        self._should_stop_movement = False  # Flag to break progress bar loop
        self._in_task_progress = False  # Flag to prevent nested progress bars

        # Subscribe to task topic
        self.task_listener = roslibpy.Topic(
            self.client,
            task_topic,
            TASK_EXECUTE_MSG_TYPE,
        )
        self.task_listener.subscribe(self.handle_task)

        # Publishers for status and position
        self.status_publisher = roslibpy.Topic(
            self.client,
            STATUS_TOPIC,
            STD_STRING_MSG_TYPE,
        )
        
        self.position_publisher = roslibpy.Topic(
            self.client,
            POSITION_TOPIC,
            POSITION_MSG_TYPE,
        )

        # Register this bridge for tool access
        set_bridge(self)
        self.console.print(f"✅ [bold green][SIMPLE BRIDGE][/bold green] Initialized: {self.name}")

    def connect(self) -> bool:
        """Establish RTDE connection to robot."""
        if not self.robot_connection.is_connected:
            self.robot_connection._connect()
        self.is_connected = self.robot_connection.is_connected
        
        if self.is_connected:
            self.publish_status(STATUS_FABRICATION_STARTED)
            print(f"🤖 [SIMPLE BRIDGE] Connected to robot at {self.robot_ip}")
        
        return self.is_connected

    def publish_status(self, status_message: str):
        """Publish robot status updates."""
        try:
            self.status_publisher.publish({'data': status_message})
            print(f"📡 [STATUS] {status_message}")
        except Exception as e:
            print(f"❌ [STATUS ERROR] {e}")

    def publish_joint_state(self):
        """Publish current joint state."""
        if not self.is_connected or not self.robot_connection:
            return
        
        try:
            joint_positions = self.robot_connection.get_joints()
            if joint_positions:
                joint_state_msg = {
                    'header': {
                        'stamp': {'secs': int(time.time()), 'nsecs': 0},
                        'frame_id': 'base_link'
                    },
                    'name': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
                    'position': joint_positions,
                    'velocity': [],
                    'effort': []
                }
                self.position_publisher.publish(joint_state_msg)
        except Exception as e:
            print(f"❌ [POSITION ERROR] {e}")

    def handle_task(self, msg: dict):
        """Handle incoming ROS tasks."""
        threading.Thread(target=lambda: self._handle_task_thread(msg), daemon=True).start()

    def _handle_task_thread(self, msg: dict):
        """Background thread for task processing."""
        data = msg.get("msg", msg)
        task_name = data.get('name', 'Unknown')

        print(f"\n🎯 [TASK] {task_name}")
        
        # Execute the task
        success = self.execute_task(
            task_name,
            data.get("tcp", []),
            data.get("trajectory")
        )
        
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {status}\n")
        # Notify user that we're ready for the next command
        self.console.print("⏳ [dim]Waiting for next command...[/dim]")

    def stop_robot(self):
        """Immediately stop all robot movements and break progress bar."""
        stopped = False
        self._should_stop_movement = True  # Signal progress bar to exit
        try:
            if hasattr(self.robot_connection, 'rtde_c') and self.robot_connection.rtde_c:
                try:
                    self.robot_connection.rtde_c.stopL()
                    stopped = True
                except Exception:
                    pass
                try:
                    self.robot_connection.rtde_c.stopJ()
                    stopped = True
                except Exception:
                    pass
            if stopped:
                self.console.print("🛑 [bold red]Robot movements stopped![/bold red]")
            else:
                self.console.print("⚠️ [yellow]No active movement to stop or stop not supported.[/yellow]")
        except Exception as e:
            self.console.print(f"❌ [red]Error stopping robot:[/red] {e}")

    def process_command(self, msg: dict):
        """Process ROS commands."""
        command = msg.get('data')
        self.console.print(f"\n🎛️  [COMMAND] {command}")
        
        if command == COMMAND_START:
            self._start_fabrication()
        elif command == COMMAND_END:
            self._end_fabrication()
        elif command == "pause":
            self.console.print("   ⏸️  Executing: Pause robot operations")
            self.pause = True
        elif command == "resume":
            self.console.print("   ▶️  Executing: Resume robot operations")
            self.pause = False
        elif command == "stop":
            self.console.print("   🛑 Executing: Stop robot operations")
            self.stop_robot()
        elif command == "gripper_open":
            self.console.print("   🔓 Executing: Open gripper")
            self.control_gripper("open")
        elif command == "gripper_close":
            self.console.print("   🔒 Executing: Close gripper")
            self.control_gripper("close")
        elif command == "home":
            self.console.print("   🏠 Executing: Move to home position")
            self.go_to_home()
        else:
            self.console.print(f"   ❓ Unknown: {command}")
        self.console.print()

    def _start_fabrication(self):
        """Start fabrication mode."""
        print("🚀 [FABRICATION] Starting...")
        self.connect()
        self.fabrication_active = True
        self.publish_status("FABRICATION_STARTED")

    def _end_fabrication(self):
        """End fabrication mode."""
        print("🏁 [FABRICATION] Ending...")
        self.fabrication_active = False
        self.publish_status(STATUS_FABRICATION_COMPLETE)

    def cleanup(self):
        """Clean up resources."""
        print("🧹 [SIMPLE BRIDGE] Cleaning up...")
        
        # Publish final status
        try:
            self.publish_status(STATUS_FABRICATION_COMPLETE)
        except Exception as e:
            print(f"⚠️ [CLEANUP] Error publishing final status: {e}")
        
        # Unsubscribe from topics
        try:
            if hasattr(self, 'task_listener'):
                self.task_listener.unsubscribe()
        except Exception as e:
            print(f"⚠️ [CLEANUP] Error unsubscribing: {e}")
        
        self.is_connected = False
        print("✅ [SIMPLE BRIDGE] Cleanup complete")

    # === Robot Control Methods ===

    def get_current_state(self) -> dict | None:
        """Get current TCP pose and joint positions."""
        if not self.is_connected or not self.robot_connection:
            return None
        try:
            tcp_pose = self.robot_connection.get_tcp_pose()
            joint_positions = self.robot_connection.get_joints()
            return {
                "tcp_pose": tcp_pose,
                "joint_positions": joint_positions
            }
        except Exception:
            return None

    def go_to_home(self) -> bool:
        """Move robot to home position."""
        if not self.is_connected or not self.robot_connection:
            return False
        try:
            home_joints_rad = [math.radians(angle) for angle in HOME_POSITION]

            # Run in background so UI stays responsive
            def _home_move():
                try:
                    if hasattr(self.robot_connection, "rtde_c") and self.robot_connection.rtde_c:
                        # Start asynchronous move
                        self.robot_connection.rtde_c.moveJ(
                            home_joints_rad,
                            ROBOT_HOME_SPEED,
                            ROBOT_HOME_ACCELERATION,
                            asynchronous=True,
                        )
                        # Wait until it actually finishes (no progress bar)
                        self.robot_connection.wait_for_movement_completion(
                            timeout=90, show_progress=False
                        )
                    else:
                        # Fallback helper – already blocks internally
                        self.robot_connection.move_j(
                            home_joints_rad,
                            ROBOT_HOME_SPEED,
                            ROBOT_HOME_ACCELERATION,
                        )

                    # On completion publish state / status
                    self.publish_joint_state()
                    try:
                        self.publish_status(
                            generate_movement_status("HOME", status_type="complete")
                        )
                    except Exception:
                        self.publish_status("at_home_position")

                    print("✅ Home position reached")
                except Exception as e:
                    print(f"❌ Home move failed: {e}")
                    self.publish_status(STATUS_FAILED)

            threading.Thread(target=_home_move, daemon=True).start()
            # Immediately acknowledge command success (movement in progress)
            return True
        except Exception:
            return False

    def _wait_for_movement_no_progress(self, timeout: float = 60.0) -> bool:
        """Wait for robot movement completion without rich progress bar."""
        if not hasattr(self.robot_connection, 'rtde_c') or not self.robot_connection.rtde_c:
            return False
        start = time.time()
        while time.time() - start < timeout:
            try:
                prog = self.robot_connection.rtde_c.getAsyncOperationProgress()
                if prog is not None and prog >= 100:
                    return True
            except Exception:
                pass
            try:
                if self.robot_connection.rtde_c.isSteady():
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        # Final check
        try:
            return self.robot_connection.rtde_c.isSteady()
        except Exception:
            return False

    def moveL(self, tcp_pose: list, speed: float = ROBOT_MOVE_SPEED, acceleration: float = ROBOT_ACCELERATION) -> bool:
        """Linear movement to TCP pose with optional task-level progress integration."""
        if not self.is_connected or not self.robot_connection:
            self.console.print("[red]moveL failed: not connected[/red]")
            return False
        try:
            self.console.print(f"[yellow]moveL: Sending TCP pose: {tcp_pose}")
            if hasattr(self.robot_connection, 'rtde_c') and self.robot_connection.rtde_c:
                self.robot_connection.rtde_c.moveL(tcp_pose, speed, acceleration, asynchronous=True)
                if self._in_task_progress:
                    success = self._wait_for_movement_no_progress()
                else:
                    success = self._wait_for_movement_with_progress("Linear Movement")
            else:
                success = self.robot_connection.move_l(tcp_pose, speed, acceleration)
            self.console.print(f"[cyan]moveL result: {success}")
            if success:
                self.publish_joint_state()
            return success
        except Exception as e:
            self.console.print(f"❌ [red]Linear movement failed:[/red] {e}")
            return False

    def moveJ(self, joint_positions: list, speed: float = ROBOT_MOVE_SPEED, acceleration: float = ROBOT_ACCELERATION) -> bool:
        """Joint space movement with optional task-level progress integration."""
        if not self.is_connected or not self.robot_connection:
            self.console.print("[red]moveJ failed: not connected[/red]")
            return False
        try:
            self.console.print(f"[yellow]moveJ: Sending joint positions: {joint_positions}")
            if hasattr(self.robot_connection, 'rtde_c') and self.robot_connection.rtde_c:
                self.robot_connection.rtde_c.moveJ(joint_positions, speed, acceleration, asynchronous=True)
                if self._in_task_progress:
                    success = self._wait_for_movement_no_progress()
                else:
                    success = self._wait_for_movement_with_progress("Joint Movement")
            else:
                success = self.robot_connection.move_j(joint_positions, speed, acceleration)
            self.console.print(f"[cyan]moveJ result: {success}")
            if success:
                self.publish_joint_state()
            return success
        except Exception as e:
            self.console.print(f"❌ [red]Joint movement failed:[/red] {e}")
            return False

    def execute_trajectory_msg(self, trajectory: dict) -> bool:
        """Execute ROS joint trajectory."""
        if not self.is_connected or not self.robot_connection:
            return False

        joint_trajectory = trajectory.get("joint_trajectory", {})
        points = joint_trajectory.get("points", [])
        
        if not points:
            return True

        try:
            for point in points:
                positions = point.get("positions", [])
                if len(positions) == 6:
                    positions_rad = [math.radians(p) for p in positions]
                    if not self.moveJ(positions_rad):
                        return False
                else:
                    return False
            return True
        except Exception:
            return False

    def control_gripper(self, action: str, timeout: float = 5.0) -> bool:
        """Control gripper - open or close, with timeout and debug prints."""
        if not self.is_connected or not self.robot_connection:
            self.console.print("[red]Gripper control failed: not connected[/red]")
            return False
        self.console.print(f"[yellow]Attempting to {action} gripper...[/yellow]")
        try:
            if action.lower() == "open":
                state, wait_time, status_msg = GRIPPER_OPEN_STATE, GRIPPER_OPEN_TIME, STATUS_GRIPPER_OPEN
            elif action.lower() == "close":
                state, wait_time, status_msg = GRIPPER_CLOSE_STATE, GRIPPER_CLOSE_TIME, STATUS_GRIPPER_CLOSE
            else:
                self.console.print(f"[red]Unknown gripper action: {action}[/red]")
                return False
            if not hasattr(self.robot_connection, 'rtde_io') or not self.robot_connection.rtde_io:
                self.publish_status(status_msg)
                self.console.print(f"[yellow]No rtde_io, published status: {status_msg}[/yellow]")
                return True
            # Set gripper output and wait, but with a timeout
            _set_gripper_output(self.robot_connection.rtde_io, state)
            start = time.time()
            while time.time() - start < timeout:
                time.sleep(0.1)
            self.publish_status(status_msg)
            self.console.print(f"[green]Gripper {action} complete, published status: {status_msg}[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]Gripper control error: {e}[/red]")
            return False

    def execute_task(self, task_name: str, tcps: list = None, trajectory: dict = None) -> bool:
        """Execute a task with a Rich progress bar for all steps (gripper + moves)."""
        task_start_time = time.time()

        task_name_lower = task_name.lower()
        steps = []
        # Determine steps for progress bar
        if "pickup" in task_name_lower or "pick" in task_name_lower:
            steps.append(("Open gripper", lambda: self.control_gripper("open")))
            # Add move steps
            move_points = []
            if tcps:
                move_points = tcps if isinstance(tcps, list) else [tcps]
            elif trajectory:
                joint_trajectory = trajectory.get("joint_trajectory", {})
                points = joint_trajectory.get("points", [])
                move_points = points
            for i, pt in enumerate(move_points):
                steps.append((f"Move to point {i+1}", lambda pt=pt: self._move_point(pt, tcps, trajectory)))
            steps.append(("Close gripper", lambda: self.control_gripper("close")))
        elif "place" in task_name_lower:
            move_points = []
            if tcps:
                move_points = tcps if isinstance(tcps, list) else [tcps]
            elif trajectory:
                joint_trajectory = trajectory.get("joint_trajectory", {})
                points = joint_trajectory.get("points", [])
                move_points = points
            for i, pt in enumerate(move_points):
                steps.append((f"Move to point {i+1}", lambda pt=pt: self._move_point(pt, tcps, trajectory)))
            steps.append(("Open gripper", lambda: self.control_gripper("open")))
        elif "gripper" in task_name_lower and "open" in task_name_lower:
            steps.append(("Open gripper", lambda: self.control_gripper("open")))
        elif "gripper" in task_name_lower and "close" in task_name_lower:
            steps.append(("Close gripper", lambda: self.control_gripper("close")))
        
        else:
            # Fallback: only handle joint trajectory
            if trajectory:
                steps.append(("Move trajectory", lambda: self.execute_trajectory_msg(trajectory)))
        # Run steps with progress bar
        success = True
        self._in_task_progress = True
        try:
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=self.console,
                    refresh_per_second=10
                ) as progress:
                    task = progress.add_task(f"🤖 {task_name}", total=len(steps))
                    for i, (desc, fn) in enumerate(steps):
                        progress.update(task, description=f"{desc}")
                        step_result = fn()
                        if not step_result:
                            self.console.print(f"[red]Step failed: {desc}[/red]")
                            success = False
                            break
                        progress.advance(task)
                    progress.update(task, completed=len(steps))
            except LiveError:
                # Fallback: no live display available, run steps sequentially
                for i, (desc, fn) in enumerate(steps):
                    self.console.print(f"🤖 {desc}")
                    step_result = fn()
                    if not step_result:
                        self.console.print(f"[red]Step failed: {desc}[/red]")
                        success = False
                        break
        finally:
            self._in_task_progress = False
        # Publish status if successful
        if success and task_name:
            self.publish_status(STATUS_SUCCESS)
            self.console.print(f"✅ [TASK SUCCESS] {STATUS_SUCCESS}")
        if not success:
            self.publish_status(STATUS_FAILED)
            self.console.print(f"❌ [TASK FAILED] {STATUS_FAILED}")

        # --- Timing end & report ---
        elapsed = time.time() - task_start_time
        self.console.print(f"⏱️  Task '{task_name}' finished in {elapsed:.2f} seconds")
        return success

    def _move_point(self, pt, tcps, trajectory):
        """Move to a single point, handling both TCP and joint points."""
        # If pt is a dict with 'positions', treat as joint
        if isinstance(pt, dict) and 'positions' in pt:
            positions = pt['positions']
            if len(positions) == 6:
                positions_rad = [math.radians(p) for p in positions]
                return self.moveJ(positions_rad)
            else:
                return False
        # Otherwise, treat as TCP
        elif isinstance(pt, (list, tuple)) and len(pt) == 6:
            return self.moveL(pt)
        else:
            return False

    def _wait_for_movement_with_progress(self, movement_type: str, timeout: float = 30.0) -> bool:
        """Wait for robot movement completion with Rich progress bar and fallback estimation."""
        if not self.robot_connection or not hasattr(self.robot_connection, 'rtde_c'):
            return False
        self._should_stop_movement = False
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=10
        ) as progress:
            task = progress.add_task(f"🤖 {movement_type}", total=100)
            start_time = time.time()
            last_progress = -1
            consecutive_steady_checks = 0
            steady_required = 3
            real_progress_seen = False
            try:
                # Wait a short time to ensure movement has started
                time.sleep(0.1)
                while time.time() - start_time < timeout:
                    if self._should_stop_movement:
                        progress.update(task, completed=0)
                        self.console.print(f"🛑 [red]{movement_type} stopped by user![/red]")
                        return False
                    elapsed = time.time() - start_time
                    # Method 1: Primary - getAsyncOperationProgress()
                    try:
                        robot_progress = self.robot_connection.rtde_c.getAsyncOperationProgress()
                        if robot_progress is not None:
                            real_progress_seen = True
                            progress.update(task, completed=robot_progress)
                            if robot_progress >= 100.0:
                                progress.update(task, completed=100)
                                self.console.print(f"✅ [green]{movement_type} completed successfully![/green]")
                                return True
                            time.sleep(0.1)
                            continue
                    except Exception:
                        if elapsed < 1.0:
                            progress.update(task, description=f"🤖 {movement_type} (fallback mode)")
                    # Method 2: Secondary - Enhanced isSteady() with velocity check
                    try:
                        is_steady = self.robot_connection.rtde_c.isSteady()
                        velocities = self.robot_connection.rtde_r.getActualQd() if hasattr(self.robot_connection, 'rtde_r') and self.robot_connection.rtde_r else None
                        # Estimate progress based on time (fallback)
                        estimated_progress = min(99, (elapsed / (timeout * 0.95)) * 99)
                        progress.update(task, completed=estimated_progress)
                        # Check if robot is steady AND velocities are near zero
                        velocity_near_zero = True
                        if velocities and len(velocities) == 6:
                            velocity_threshold = 0.01
                            velocity_near_zero = all(abs(v) < velocity_threshold for v in velocities)
                        if is_steady and velocity_near_zero:
                            consecutive_steady_checks += 1
                            if consecutive_steady_checks >= steady_required:
                                progress.update(task, completed=100)
                                self.console.print(f"✅ [green]{movement_type} completed (steady + zero velocity)![/green]")
                                return True
                        else:
                            consecutive_steady_checks = 0
                    except Exception as e:
                        progress.update(task, description=f"🤖 {movement_type} (monitoring issues)")
                    # Method 3: Program running state check
                    try:
                        program_running = self.robot_connection.rtde_c.isProgramRunning()
                        if program_running is False and self.robot_connection.rtde_c.isSteady():
                            progress.update(task, completed=100)
                            self.console.print(f"✅ [green]{movement_type} completed (no program running)![/green]")
                            return True
                    except Exception:
                        pass
                    time.sleep(0.1)
                # Timeout reached
                progress.update(task, description=f"⚠️ {movement_type} (timeout)")
                self.console.print(f"⚠️ [yellow]{movement_type} timeout after {timeout}s[/yellow]")
                if not real_progress_seen:
                    self.console.print(f"⚠️ [yellow]No progress updates received from robot. Check connection or robot state.[/yellow]")
                # Final verification
                try:
                    is_steady = self.robot_connection.rtde_c.isSteady()
                    velocities = self.robot_connection.rtde_r.getActualQd() if hasattr(self.robot_connection, 'rtde_r') and self.robot_connection.rtde_r else None
                    if is_steady and velocities and all(abs(v) < 0.01 for v in velocities):
                        progress.update(task, completed=100)
                        self.console.print(f"✅ [green]{movement_type} actually completed (post-timeout verification)![/green]")
                        return True
                except Exception:
                    pass
                return False
            except Exception as e:
                progress.update(task, description=f"❌ {movement_type} (error)")
                self.console.print(f"❌ [red]Error during {movement_type}: {e}[/red]")
                return False


def main():
    """Main test function."""
    console = Console()
    
    # Create a nice startup panel
    startup_text = Text()
    startup_text.append("🚀 Simple Bridge Test ", style="bold magenta")
    startup_text.append("(No LLM Integration)", style="dim")
    
    console.print(Panel(startup_text, title="[bold blue]Starting...[/bold blue]", border_style="blue"))
    
    # Connect to ROS
    try:
        client = roslibpy.Ros(host=ROS_HOST, port=9090)
        client.run()
        console.print("✅ [green]Connected to ROS bridge[/green]")
    except Exception as e:
        console.print(f"❌ [red]Failed to connect to ROS:[/red] {e}")
        return
    
    # Create simple bridge
    try:
        bridge = SimpleBridge("UR10", client)
        console.print("✅ [green]Simple bridge created[/green]")
        
        # Create info panel
        info_panel = Panel.fit(
            "[bold cyan]🎯 Simple Bridge Active[/bold cyan] - Listening for ROS messages...\n\n"
            "[bold]Topics:[/bold]\n"
            f"   • [blue]Task Topic:[/blue] {TASK_EXECUTE_TOPIC}\n"
            f"   • [blue]Command Topic:[/blue] {COMMAND_TOPIC}\n"
            f"   • [blue]Status Topic:[/blue] {STATUS_TOPIC}\n"
            f"   • [blue]Position Topic:[/blue] {POSITION_TOPIC}\n\n"
            "[bold]📋 Available Commands:[/bold]\n"
            "   • [green]START_FABRICATION[/green] - Start fabrication mode\n"
            "   • [red]END_FABRICATION[/red] - End fabrication mode\n\n"
            "[dim]Press Ctrl+C to exit[/dim]",
            title="[bold green]Bridge Status[/bold green]",
            border_style="green"
        )
        console.print(info_panel)
        
        # Keep alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n🛑 [yellow]Shutting down...[/yellow]")
        
    except Exception as e:
        console.print(f"❌ [red]Error creating bridge:[/red] {e}")
    
    finally:
        # Cleanup
        try:
            if 'bridge' in locals():
                bridge.cleanup()
            client.terminate()
            console.print("✅ [green]Cleanup complete[/green]")
        except Exception as e:
            console.print(f"⚠️ [yellow]Cleanup error:[/yellow] {e}")


if __name__ == "__main__":
    main() 