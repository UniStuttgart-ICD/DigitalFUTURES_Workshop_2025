import time
import roslibpy
import sys
import os
import math
import asyncio
import threading
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.progress import TimeElapsedColumn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RobotInterface import RobotInterface
from ur.core.connection import get_robot, set_bridge
from ur.config.robot_config import (
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
    GRIPPER_IO_TYPE,
)
from ur.config.topics import (
    TASK_EXECUTE_TOPIC,
    COMMAND_TOPIC,
    STATUS_TOPIC,
    POSITION_TOPIC,
    TASK_EXECUTE_MSG_TYPE,
    STD_STRING_MSG_TYPE,
    POSITION_MSG_TYPE,
    ROS_HOST,
    ROS_PORT
)
from ur.config.system_config import (
    COMMAND_START,
    COMMAND_END,
    STATUS_FABRICATION_STARTED,
    STATUS_FABRICATION_COMPLETE,
    STATUS_GRIPPER_OPEN,
    STATUS_GRIPPER_CLOSE,
    STATUS_SUCCESS,
    STATUS_FAILED,
    STATUS_STOP,
)
from ur.tools.gripper_tools import _set_gripper_output

class URBridge(RobotInterface):
    """
    A pure robot controller for Universal Robots.

    This bridge connects to a UR robot via RTDE for direct control and
    listens to a ROS topic for tasks. It provides a clean Python API for
    robot operations that can be called by higher-level agents (like a
    voice or planning agent). It does not contain any agent-specific logic
    or UI code.
    """

    def __init__(
        self,
        name: str,
        client: roslibpy.Ros,
        robot_ip: str | None = None,
        task_topic: str = TASK_EXECUTE_TOPIC,
        command_topic: str = COMMAND_TOPIC,
    ):
        # Initialize base RobotInterface with name and ROS client
        super().__init__(name, client)
        self.pause = False
        self.stop = False
        self.robot_ip = robot_ip or ROBOT_IP
        
        # Use shared global robot connection to avoid RTDE conflicts
        self.robot_connection = get_robot()
        self.is_connected = self.robot_connection.is_connected

        # Agent notification system
        self.agent_ref = None  # Will be set by the agent

        # Fabrication state management - NEW
        self.fabrication_active = False
        self.voice_agent = None
        self.voice_agent_thread = None
        self.agent_event_loop = None
        self.launcher_ref = None  # Reference back to launcher for agent creation

        # Subscribe to the specified task topic
        self.task_listener = roslibpy.Topic(
            self.client,
            task_topic,
            TASK_EXECUTE_MSG_TYPE,
        )
        self.task_listener.subscribe(self.handle_task)

        # Subscribe to the command topic (moved from RobotInterface)
        self.command_listener = roslibpy.Topic(
            self.client,
            command_topic,
            STD_STRING_MSG_TYPE,
        )
        self.command_listener.subscribe(self.process_command)

        # Add publishers for status and position as defined in PlantUML
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

        # Register this bridge instance for tool access (Phase 1 integration)
        set_bridge(self)
        print(f"✅ [BRIDGE] Registered for tool access: {self.name}")

        # Rich console for improved movement and progress output
        self.console = Console()
        self._should_stop_movement = False
        # Disable rich.Progress for movement tasks to avoid UI conflicts
        self._in_task_progress = True

    def set_agent_reference(self, agent):
        """Allow agent to register for task notifications."""
        self.agent_ref = agent
        print(f"🤖 [BRIDGE] Agent registered for task notifications: {type(agent).__name__}")

    def set_launcher_reference(self, launcher):
        """Allow launcher to register for voice agent creation."""
        self.launcher_ref = launcher
        print(f"🔗 [BRIDGE] Launcher registered for agent creation: {type(launcher).__name__}")

    def _notify_agent_sync(self, event_type: str, event_data: dict):
        """Synchronous agent notification wrapper."""
        if self.agent_ref and hasattr(self.agent_ref, 'handle_task_event'):
            try:
                import asyncio
                # Try to get existing event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, schedule the coroutine
                    asyncio.create_task(self.agent_ref.handle_task_event(event_type, event_data))
                except RuntimeError:
                    # No running loop, create one for this call
                    asyncio.run(self.agent_ref.handle_task_event(event_type, event_data))
            except Exception as e:
                print(f"⚠️ [BRIDGE] Agent notification failed: {e}")

    def publish_status(self, status_message: str):
        """Publish robot status updates to /Robot/status/physical"""
        try:
            self.status_publisher.publish({'data': status_message})
            print(f"📡 [ROS PUBLISH] Status: {status_message}")
        except Exception as e:
            print(f"❌ [ROS ERROR] Failed to publish status: {e}")

    def publish_joint_state(self):
        """Publish current joint state to position topic"""
        if not self.is_connected or not self.robot_connection:
            return
        
        try:
            joint_positions = self.robot_connection.get_joints()
            if joint_positions:
                # Create JointState message format
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
            print(f"Failed to publish joint state: {e}")

    def connect(self) -> bool:
        """
        Establishes the RTDE connection to the robot.
        Returns True on success, False on failure.
        """
        if not self.robot_connection.is_connected:
            self.robot_connection._connect()
        self.is_connected = self.robot_connection.is_connected
        
        if self.is_connected:
            self.publish_status(STATUS_FABRICATION_STARTED)
        
        return self.is_connected

    def stop(self):
        """Safely stops the robot and disconnects."""
        self.cleanup()
    
    def shutdown(self):
        """Public method for external shutdown calls (avoids parent class conflicts)."""
        self.cleanup()
    
    def cleanup(self, include_fabrication=True, include_connections=True):
        """Unified cleanup method for all bridge resources."""
        print("🧹 [BRIDGE] Starting cleanup...")
        
        if include_fabrication:
            self._cleanup_fabrication()
        
        if include_connections:
            self._cleanup_connections()
        
        print("🧹 [BRIDGE] Cleanup complete")
    
    def _cleanup_fabrication(self):
        """Clean up fabrication-specific resources."""
        if self.fabrication_active:
            print("🏁 [BRIDGE] Cleaning up fabrication mode...")
            
            # Generate goodbye message if voice agent is running
            if self.voice_agent:
                try:
                    self._generate_and_display_fabrication_message("end")
                except Exception as e:
                    print(f"⚠️ [BRIDGE] Error generating goodbye message: {e}")
            
            # Note: Voice agent cleanup is handled by the launcher/cleanup manager
            # We just mark fabrication as inactive
            self.fabrication_active = False
            self.voice_agent = None
            self.voice_agent_thread = None
            self.agent_event_loop = None
    
    def _cleanup_connections(self):
        """Clean up ROS connections and robot state."""
        print("📡 [BRIDGE] Cleaning up connections...")
        
        # Publish final status
        try:
            self.publish_status(STATUS_FABRICATION_COMPLETE)
        except Exception as e:
            print(f"⚠️ [BRIDGE] Error publishing final status: {e}")
        
        # Unsubscribe from ROS topics
        try:
            if hasattr(self, 'task_listener'):
                self.task_listener.unsubscribe()
        except Exception as e:
            print(f"⚠️ [BRIDGE] Error unsubscribing from tasks: {e}")
        
        # Mark as disconnected
        self.is_connected = False

        # Unregister bridge from global registry
        try:
            from ur.core.connection import cleanup_bridge
            cleanup_bridge()
            print("✅ [BRIDGE] Unregistered from tool access")
        except Exception as e:
            print(f"⚠️ [BRIDGE] Error during global cleanup: {e}")

    def handle_task(self, msg: dict):
        """Callback for incoming ROS tasks with agent notification."""
        # Process the task in a separate thread to avoid blocking the ROS client
        thread = threading.Thread(target=self._handle_task_thread, args=(msg,))
        thread.start()
        print("🔄 [BRIDGE] Task received, processing in background thread")

    def _handle_task_thread(self, msg: dict):
        """Task processing logic that runs in a background thread."""
        print(f"🔄 [BRIDGE] Task received, processing in background thread")
        # ROSlibpy wraps message payload under 'msg', mimic test behavior
        data = msg.get('msg', msg)
        task_name = data.get('name', '').lower()
        tcps = data.get('tcps')
        trajectory = data.get('trajectory')  # Extract full trajectory object
        
        # Notify the agent about the task if present
        if self.voice_agent and hasattr(self.voice_agent, 'handle_task_event'):
            print(f"🗣️  [BRIDGE] Notifying voice agent of task: {task_name}")
            event_data = {"task_name": task_name, "tcps": tcps, "trajectory": trajectory}
            coro = self.voice_agent.handle_task_event("task_received", event_data)
            if self.agent_event_loop and self.agent_event_loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, self.agent_event_loop)
                print(f"✅ [BRIDGE] Agent notification scheduled for task: {task_name}")
            else:
                try:
                    asyncio.run(coro)
                except Exception as e:
                    print(f"⚠️ [BRIDGE] Agent notification error: {e}")
        # Always execute the task to ensure robot movement
        success = self.execute_task(task_name=task_name, tcps=tcps, trajectory=trajectory)

        if success:
            #NOTE: Real success message will be now handled by the agent
            status_message = f"{STATUS_SUCCESS}_{task_name}"
            self._notify_agent_sync('task_completed', {'task_name': task_name})
        else:
            status_message = STATUS_FAILED
            self._notify_agent_sync('task_failed', {'task_name': task_name, 'reason': 'Execution failed'})
         
        self.publish_status(status_message)
        print(f"✅ [TASK_EXECUTION] Finished processing task: {task_name}, Success: {success}")

    def process_command(self, msg: dict):
        """Override parent's command handling with fabrication lifecycle management."""
        command = msg.get('data', '').lower()
        print(f"\n🎛️  [ROS COMMAND] {command}")
        
        # Notify agent about command with system prompt context (if agent exists)
        if self.agent_ref:
            self._notify_agent_sync('command_received', {
                'command': command,
                'context_type': 'command',  # ROS topic context
                'source_topic': 'COMMAND_TOPIC'
            })
        
        if command == COMMAND_START:
            # Always start (or restart) fabrication mode
            self.connect()
            self._start_fabrication()
        elif command == COMMAND_END:
            # Signal the launcher to initiate a full system shutdown.
            # Publish STOP status
            print("🔄 [BRIDGE] END_FABRICATION received, ending fabrication")
            self.publish_status(STATUS_STOP)
            if self.launcher_ref:
                self.launcher_ref.shutdown_requested = True
            else:
                # Fallback if no launcher is present (e.g., standalone bridge test)
                print("⚠️ [BRIDGE] No launcher reference, performing self-cleanup.")
                self.cleanup()
        elif command == "pause":
            print("   ⏸️  Executing: Pause robot operations")
            self.pause = True
        elif command == "resume":
            self.pause = False
            self.publish_status("Robot resumed")
            print("▶️ [BRIDGE] Robot resumed")
        elif command == "stop":
            self.stop = True
            self._should_stop_movement = True
            # Publish STOP status
            self.publish_status(STATUS_STOP)
            print("⏹️ [BRIDGE] Stop command received, initiating graceful stop...")
            # Immediately stop robot movement if active
            self.stop_robot()
            # Direct cleanup if no task is running
            if not self.fabrication_active:
                self.cleanup()
        elif command == "home":
            print("   🏠 Executing: Move to home position")
            self.go_to_home()
        else:
            print(f"   ❓ Unknown command: {command}")

    def _start_fabrication(self):
        """Start fabrication mode - create and start voice agent."""
        print("🚀 [FABRICATION] Starting fabrication mode...")
        
        # Recreate and re-subscribe to task topic (unsubscribe previous listener first)
        try:
            if hasattr(self, 'task_listener'):
                try:
                    self.task_listener.unsubscribe()
                except Exception:
                    pass
            self.task_listener = roslibpy.Topic(
                self.client,
                TASK_EXECUTE_TOPIC,
                TASK_EXECUTE_MSG_TYPE,
            )
            self.task_listener.subscribe(self.handle_task)
            print("🔄 [BRIDGE] Task listener unsubscribed and re-subscribed")
        except Exception as e:
            print(f"⚠️ [BRIDGE] Failed to re-subscribe to task listener: {e}")
        
        # Ensure any previous session is completely terminated
        if self.voice_agent:
            print("⚠️ [FABRICATION] Previous voice agent still exists - cleaning up first...")
            self._stop_voice_agent_completely()
            # Wait for complete cleanup
            time.sleep(1)
            
        self.fabrication_active = True
        
        # Set UI to active state (white) - update both launcher UI and voice agent UI
        if hasattr(self, 'launcher_ref') and self.launcher_ref:
            if hasattr(self.launcher_ref, 'ui') and self.launcher_ref.ui:
                self.launcher_ref.ui.set_fabrication_active(True)
        
        # Reset shutdown flag for restart capability
        if hasattr(self, 'launcher_ref') and self.launcher_ref:
            self.launcher_ref.shutdown_requested = False
        
        if self.launcher_ref and not self.voice_agent:
            try:
                # Create voice agent via launcher
                self.voice_agent = self.launcher_ref.create_voice_agent(self)
                if self.voice_agent:
                    # Start agent in background thread
                    self._start_voice_agent_thread()
                    
                    # Update voice agent UI to active state
                    if hasattr(self.voice_agent, 'ui'):
                        self.voice_agent.ui.set_fabrication_active(True)
                else:
                    print("⚠️ [FABRICATION] Failed to create voice agent")
            except Exception as e:
                print(f"❌ [FABRICATION] Error starting voice agent: {e}")
        else:
            if not self.launcher_ref:
                print("❌ [BRIDGE] No launcher reference, cannot create voice agent.")
            if self.voice_agent:
                print("⚠️ [FABRICATION] Voice agent already running")

    def _end_fabrication(self):
        """End fabrication mode - stop voice agent."""
        print("🏁 [FABRICATION] Ending fabrication mode...")
        
        if self.voice_agent:
            try:
                # Stop the voice agent properly
                self._stop_voice_agent_completely()
            except Exception as e:
                print(f"❌ [FABRICATION] Error stopping voice agent: {e}")
        else:
            print("⚠️ [FABRICATION] Voice agent not running")
            
        self.fabrication_active = False
        
        # Set UI to inactive state (black) - update both launcher UI and voice agent UI
        if hasattr(self, 'launcher_ref') and self.launcher_ref:
            if hasattr(self.launcher_ref, 'ui') and self.launcher_ref.ui:
                self.launcher_ref.ui.set_fabrication_active(False)
        
        # Also update the voice agent's UI if it exists
        if self.voice_agent and hasattr(self.voice_agent, 'ui'):
            self.voice_agent.ui.set_fabrication_active(False)
        
        # Signal the launcher to shut down the entire system
        if hasattr(self, 'launcher_ref') and self.launcher_ref:
            print("🔄 [FABRICATION] Requesting system shutdown...")
            self.launcher_ref.shutdown_requested = True

    def _start_voice_agent_thread(self):
        """Start the voice agent in a background thread with its own asyncio loop."""
        if not self.voice_agent:
            return

        def _run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.agent_event_loop = loop
            try:
                loop.run_until_complete(self.voice_agent.start())
            except Exception as e:
                print(f"❌ [AGENT] Voice agent error: {e}")
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self.voice_agent_thread = threading.Thread(
            target=_run_loop,
            daemon=True,
            name="VoiceAgentThread"
        )
        self.voice_agent_thread.start()
        print("✅ [FABRICATION] Voice agent thread started")

    def _stop_voice_agent_completely(self):
        """Stop the voice agent session and clean up thread."""
        if self.voice_agent:
            print("🔇 [FABRICATION] Stopping voice agent...")
            
            # Use synchronous stop method to avoid event loop conflicts
            if hasattr(self.voice_agent, 'stop_sync'):
                try:
                    self.voice_agent.stop_sync()
                    print("✅ [FABRICATION] Voice agent stopped synchronously")
                except Exception as e:
                    print(f"⚠️ [FABRICATION] Error in synchronous stop: {e}")
            else:
                # Fallback: Set stop flag manually
                if hasattr(self.voice_agent, 'should_stop'):
                    self.voice_agent.should_stop = True
                if hasattr(self.voice_agent, 'session') and self.voice_agent.session:
                    self.voice_agent.session.should_stop = True
                    self.voice_agent.session.is_connected = False
                print("✅ [FABRICATION] Voice agent stop flags set")
            
            # Wait for the agent thread to finish gracefully
            if self.voice_agent_thread and self.voice_agent_thread.is_alive():
                print("⏳ [FABRICATION] Waiting for voice agent thread to finish...")
                try:
                    self.voice_agent_thread.join(timeout=3.0)
                    if self.voice_agent_thread.is_alive():
                        print("⚠️ [FABRICATION] Voice agent thread didn't stop within timeout")
                    else:
                        print("✅ [FABRICATION] Voice agent thread stopped gracefully")
                except Exception as e:
                    print(f"⚠️ [FABRICATION] Error joining voice agent thread: {e}")
            
            # Clean up references
            self.voice_agent = None
            self.voice_agent_thread = None
            print("✅ [FABRICATION] Voice agent stopped successfully")
    
    def _stop_voice_agent(self):
        """Stop the voice agent and clean up thread."""
        # Note: This method is deprecated - voice agent cleanup is now handled 
        # by the centralized cleanup manager for better coordination
        print("⚠️ [BRIDGE] _stop_voice_agent is deprecated, use cleanup manager instead")
        
        if self.voice_agent:
            # Set stop flag for graceful shutdown
            if hasattr(self.voice_agent, 'should_stop'):
                self.voice_agent.should_stop = True
            
            # Clean up references (actual stopping handled by cleanup manager)
            self.voice_agent = None
            self.voice_agent_thread = None
            print("✅ [FABRICATION] Voice agent references cleared")

    def _generate_and_display_fabrication_message(self, message_type: str):
        """Generate and speak fabrication start/end message via LLM."""
        if not self.voice_agent:
            return
            
        try:
            if hasattr(self.voice_agent, 'generate_fabrication_message'):
                # Run the async message generation
                message = asyncio.run(self.voice_agent.generate_fabrication_message(message_type))
                if message_type == "start":
                    print(f"🤖 [WELCOME] {message}")
                else:
                    print(f"🤖 [GOODBYE] {message}")
                
                # Wait a moment for the session to be fully connected before speaking
                import time
                if message_type == "start":
                    # For start messages, wait a bit longer for connection to stabilize
                    time.sleep(2)
                else:
                    # For end messages, connection should already be stable
                    time.sleep(0.5)
                
                # Make the voice agent actually speak the message
                self._speak_fabrication_message(message)
                
            else:
                # Fallback messages
                fallback_message = ""
                if message_type == "start":
                    fallback_message = "Voice agent online! Ready for fabrication."
                    print("🤖 [WELCOME] Voice agent online! Ready for fabrication.")
                else:
                    fallback_message = "Fabrication complete! Voice agent going offline."
                    print("🤖 [GOODBYE] Fabrication complete! Voice agent going offline.")
                
                # Wait and speak the fallback message
                import time
                time.sleep(1)
                self._speak_fabrication_message(fallback_message)
                
        except Exception as e:
            print(f"⚠️ [MESSAGE] Error generating fabrication message: {e}")
            # Fallback messages
            fallback_message = ""
            if message_type == "start":
                fallback_message = "Voice agent online! Ready for fabrication."
                print("🤖 [WELCOME] Voice agent online! Ready for fabrication.")
            else:
                fallback_message = "Fabrication complete! Voice agent going offline."
                print("🤖 [GOODBYE] Fabrication complete! Voice agent going offline.")
            
            # Wait and speak the fallback message
            import time
            time.sleep(1)
            self._speak_fabrication_message(fallback_message)

    def _speak_fabrication_message(self, message: str):
        """Make the voice agent speak the fabrication message, waiting for connection if necessary."""
        if not self.voice_agent or not message:
            return

        async def _wait_and_speak():
            """Wait for the agent session to be ready, then send the message."""
            # Wait up to 5 seconds for the session to become connected
            for _ in range(10): # 10 * 0.5s = 5s timeout
                if (hasattr(self.voice_agent, 'session') and self.voice_agent.session and
                    hasattr(self.voice_agent.session, 'is_connected') and self.voice_agent.session.is_connected):
                    break
                await asyncio.sleep(0.5)
            else:
                print(f"⚠️ [SPEAK] Agent session not ready in time. Cannot speak message: {message[:50]}...")
                return

            try:
                # Always use agent send_message to speak via OpenAI voice
                await self.voice_agent.session.send_message(message)
                print("🔊 Assistant speaking via agent")
            except Exception as e:
                print(f"⚠️ [SPEAK] Error sending message to agent: {e}")

        try:
            # Run the async wrapper function
            asyncio.run(_wait_and_speak())
        except Exception as e:
            print(f"⚠️ [SPEAK] Error making agent speak: {e}")

    def get_current_state(self) -> dict | None:
        """Returns the current TCP pose and joint positions, or None on failure."""
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
        """Moves the robot to a predefined home position in joint space."""
        if not self.is_connected or not self.robot_connection:
            return False
        try:
            home_joints_rad = [math.radians(angle) for angle in HOME_POSITION]

            # Spawn a background thread to avoid blocking command handler
            def _home_move():
                try:
                    if hasattr(self.robot_connection, "rtde_c") and self.robot_connection.rtde_c:
                        self.robot_connection.rtde_c.moveJ(
                            home_joints_rad,
                            ROBOT_HOME_SPEED,
                            ROBOT_HOME_ACCELERATION,
                            asynchronous=True,
                        )
                        self.robot_connection.wait_for_movement_completion(
                            timeout=90, show_progress=False
                        )
                    else:
                        self.robot_connection.move_j(
                            home_joints_rad,
                            ROBOT_HOME_SPEED,
                            ROBOT_HOME_ACCELERATION,
                        )

                    self.publish_joint_state()
                    try:
                        from ur.config.system_config import generate_movement_status

                        self.publish_status(
                            generate_movement_status("HOME", status_type="complete")
                        )
                    except Exception:
                        self.publish_status("at_home_position")

                    self.console.print("[green]✅ Robot at home position[/green]")
                except Exception as e:
                    self.console.print(f"[red]❌ Home move failed: {e}[/red]")
                    self.publish_status(STATUS_FAILED)

            threading.Thread(target=_home_move, daemon=True).start()
            # Immediately acknowledge – movement in background
            return True
        except Exception:
            return False

    def moveL(self, tcp_pose: list, speed: float = ROBOT_MOVE_SPEED, acceleration: float = ROBOT_ACCELERATION) -> bool:
        """Moves the robot linearly to a given TCP pose with Rich progress support."""
        if not self.is_connected or not self.robot_connection:
            return False
        try:
            # Send movement command
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
        """Moves the robot in joint space to the given positions with Rich progress support."""
        if not self.is_connected or not self.robot_connection:
            return False
        try:
            # Send joint movement command
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
        """Executes a ROS-style JointTrajectory from a dictionary."""
        if not self.is_connected or not self.robot_connection:
            return False

        # Handle the actual YAML structure: trajectory.joint_trajectory.points
        joint_trajectory = trajectory.get("joint_trajectory", {})
        points = joint_trajectory.get("points", [])
        
        if not points:
            return True  # Empty trajectory is a success

        try:
            for point in points:
                # Check for stop command before each trajectory point (but allow task to start fresh)
                if getattr(self, '_should_stop_movement', False):
                    print("🛑 [TRAJECTORY] Stop command received, aborting trajectory execution")
                    return False
                    
                positions = point.get("positions", [])
                if len(positions) == 6:
                    positions_rad = [math.radians(p) for p in positions]
                    if not self.moveJ(positions_rad):
                        # Stop if any single move fails
                        return False
                else:
                    # Invalid point, stop execution
                    print(f"❌ [TRAJECTORY] Invalid point: {point}")
                    return False
            return True
        except Exception:
            return False

    def control_gripper(self, action: str) -> bool:
        """Control the robot gripper - open or close."""
        if not self.is_connected or not self.robot_connection:
            return False
        
        try:
            if action.lower() == "open":
                state, wait_time, status_msg = GRIPPER_OPEN_STATE, GRIPPER_OPEN_TIME, STATUS_GRIPPER_OPEN
            elif action.lower() == "close":
                state, wait_time, status_msg = GRIPPER_CLOSE_STATE, GRIPPER_CLOSE_TIME, STATUS_GRIPPER_CLOSE
            else:
                return False
            
            if not hasattr(self.robot_connection, 'rtde_io') or not self.robot_connection.rtde_io:
                self.publish_status(status_msg)
                return True
            
            _set_gripper_output(self.robot_connection.rtde_io, state)
            time.sleep(wait_time)
            self.publish_status(status_msg)
            return True
        except Exception:
            return False

    def execute_task(self, task_name: str, tcps: list = None, trajectory: dict = None) -> bool:
        """Execute a task based on its name pattern with appropriate gripper control."""
        # Reset movement stop flag for new task execution (keep self.stop for command processing)
        self._should_stop_movement = False
        
        # Import status generation function here to avoid circular imports
        from ur.config.system_config import generate_task_status
        
        # --- Timing start ---
        task_start_time = time.time()
        
        task_name_lower = task_name.lower()
        success = False
        
        # Handle pickup/place/home as before
        if "pickup" in task_name_lower or "pick" in task_name_lower:
            success = self._handle_pickup_task(tcps, trajectory)
        elif "place" in task_name_lower:
            success = self._handle_place_task(tcps, trajectory)
        elif "home" in task_name_lower:
            success = self._handle_home_task(tcps, trajectory)
        # Handle pure gripper commands (no trajectory)
        elif "gripper" in task_name_lower:
            if "open" in task_name_lower:
                success = self.control_gripper("open")
            elif "close" in task_name_lower:
                success = self.control_gripper("close")
            else:
                success = False
        # Fallback: execute trajectory if provided
        else:
            success = self._execute_trajectory_only(tcps, trajectory)
        
        # CRITICAL: Only send SUCCESS status when entire task completes successfully
        if success and task_name:
            #TODO: Uncommented to use voice to send 
            # self.publish_status(STATUS_SUCCESS)
            print(f"✅ [TASK SUCCESS] {STATUS_SUCCESS}")
        
        # --- Timing end & report ---
        elapsed = time.time() - task_start_time
        try:
            self.console.print(f"⏱️  Task '{task_name}' finished in {elapsed:.2f} seconds")
        except Exception:
            print(f"⏱️  Task '{task_name}' finished in {elapsed:.2f} seconds")
        
        return success

    def execute_task_with_commentary(self, task_name: str, tcps: list = None, trajectory: dict = None) -> bool:
        """Execute task with LLM-generated commentary."""
        
        # Reset movement stop flag for new task execution (keep self.stop for command processing)
        self._should_stop_movement = False
        
        # --- Timing start ---
        task_start_time = time.time()
        
        # Notify agent task is starting
        if self.agent_ref:
            self._notify_agent_sync('task_starting', {
                'task_name': task_name,
                'tcps': tcps,
                'trajectory': trajectory
            })
        
        # Execute the task with commentary
        task_name_lower = task_name.lower()
        success = False
        
        if "pickup" in task_name_lower or "pick" in task_name_lower:
            success = self._handle_pickup_task_with_commentary(tcps, trajectory, task_name)
        elif "place" in task_name_lower:
            success = self._handle_place_task_with_commentary(tcps, trajectory, task_name)
        elif "home" in task_name_lower:
            success = self._handle_home_task_with_commentary(tcps, trajectory, task_name)
        else:
            success = self.execute_task(task_name, tcps, trajectory)
        
        # Notify agent task completed
        if self.agent_ref:
            self._notify_agent_sync('task_completed', {
                'task_name': task_name,
                'success': success
            })
        
        # CRITICAL: Only send SUCCESS status when entire task completes successfully
        if success and task_name:
            from ur.config.system_config import generate_task_status
            success_status = generate_task_status(task_name, status_type="SUCCESS")
            self.publish_status(success_status)
            print(f"✅ [TASK SUCCESS] {success_status}")
        
        if not success:
            self.publish_status(STATUS_FAILED)
            print(f"❌ [TASK FAILED] {STATUS_FAILED}")
        
        # --- Timing end & report ---
        elapsed = time.time() - task_start_time
        try:
            self.console.print(f"⏱️  Task '{task_name}' finished in {elapsed:.2f} seconds")
        except Exception:
            print(f"⏱️  Task '{task_name}' finished in {elapsed:.2f} seconds")
        
        return success

    def _handle_pickup_task(self, tcps: list = None, trajectory: dict = None) -> bool:
        """Handle pickup tasks: Open gripper → Execute trajectory → Close gripper."""
        return (self.control_gripper("open") and 
                self._execute_trajectory_only(tcps, trajectory) 
                #and self.control_gripper("close")
                )

    def _handle_place_task(self, tcps: list = None, trajectory: dict = None) -> bool:
        """Handle place tasks: Execute trajectory → Open gripper."""
        return (self._execute_trajectory_only(tcps, trajectory)
                #and self.control_gripper("open")
                )

    def _handle_home_task(self, tcps: list = None, trajectory: dict = None) -> bool:
        """Handle home tasks: Execute trajectory only (no gripper control)."""
        return self._execute_trajectory_only(tcps, trajectory)

    def _execute_trajectory_only(self, tcps: list = None, trajectory: dict = None) -> bool:
        """Execute joint trajectory without gripper control. TCP frames not supported."""
        if trajectory:
            return self.execute_trajectory_msg(trajectory)
        else:
            return False

    def _handle_pickup_task_with_commentary(self, tcps: list = None, trajectory: dict = None, task_name: str = "") -> bool:
        """Handle pickup with dynamic LLM commentary."""
        # (Automated task) No human permission request for ROS tasks
        # Open gripper with announcement
        gripper_open = self.control_gripper("open")
        if gripper_open and self.agent_ref:
            self._notify_agent_sync('robot_action', {
                'action_type': 'gripper_opened',
                'task_name': task_name,
                'context_type': 'task_execute'
            })
        
        # Announce movement
        if self.agent_ref:
            self._notify_agent_sync('robot_action', {
                'action_type': 'moving_to_pickup',
                'task_name': task_name,
                'context_type': 'task_execute'
            })
        
        # Execute trajectory
        trajectory_success = self._execute_trajectory_only(tcps, trajectory)
        
        # Close gripper with announcement
        #gripper_close = self.control_gripper("close")
        if self.agent_ref:
            self._notify_agent_sync('robot_action', {
                'action_type': 'element_secured',
                'task_name': task_name,
                'context_type': 'task_execute'
            })
        
        return gripper_open and trajectory_success #and gripper_close

    def _handle_place_task_with_commentary(self, tcps: list = None, trajectory: dict = None, task_name: str = "") -> bool:
        """Handle place with dynamic LLM commentary."""
        # Announce moving to assembly
        if self.agent_ref:
            self._notify_agent_sync('robot_action', {
                'action_type': 'moving_to_assembly',
                'task_name': task_name,
                'context_type': 'task_execute'
            })
        
        trajectory_success = self._execute_trajectory_only(tcps, trajectory)
        
        # (Automated task) Skipping human permission request
        
        # gripper_open = self.control_gripper("open")
        
        return trajectory_success # and gripper_open

    def _handle_home_task_with_commentary(self, tcps: list = None, trajectory: dict = None, task_name: str = "") -> bool:
        """Handle home with commentary."""
        
        if self.agent_ref:
            self._notify_agent_sync('robot_action', {
                'action_type': 'returning_home',
                'task_name': task_name,
                'context_type': 'task_execute'
            })
        
        return self._execute_trajectory_only(tcps, trajectory)

    def stop_robot(self):
        """Stop robot movement and break any ongoing progress."""
        self._should_stop_movement = True
        try:
            if hasattr(self.robot_connection, 'rtde_c') and self.robot_connection.rtde_c:
                try:
                    self.robot_connection.rtde_c.stopL()
                except Exception:
                    pass
                try:
                    self.robot_connection.rtde_c.stopJ()
                except Exception:
                    pass
                self.console.print("🛑 [bold red]Robot movements stopped![/bold red]")
            else:
                self.console.print("⚠️ [yellow]No active movement to stop or stop not supported.[/yellow]")
        except Exception as e:
            self.console.print(f"❌ [red]Error stopping robot:[/red] {e}")

    def _wait_for_movement_no_progress(self, timeout: float = 60.0) -> bool:
        """Wait for robot movement completion without progress bar."""
        if not hasattr(self.robot_connection, 'rtde_c') or not self.robot_connection.rtde_c:
            return False
        start = time.time()
        while time.time() - start < timeout:
            if self._should_stop_movement:
                return False
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
        try:
            return self.robot_connection.rtde_c.isSteady()
        except Exception:
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
                # Small delay to ensure the movement command has been accepted
                time.sleep(0.1)
                while time.time() - start_time < timeout:
                    if self._should_stop_movement:
                        progress.update(task, completed=0)
                        self.console.print(f"🛑 [red]{movement_type} stopped by user![/red]")
                        return False
                    elapsed = time.time() - start_time
                    # --- Method 1: Primary progress from getAsyncOperationProgress() ---
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
                        # If we're in the very first second, we can silently ignore errors
                        if elapsed < 1.0:
                            progress.update(task, description=f"🤖 {movement_type} (fallback mode)")
                    # --- Method 2: Fallback estimation combined with isSteady & velocity ---
                    try:
                        is_steady = self.robot_connection.rtde_c.isSteady()
                        velocities = None
                        if hasattr(self.robot_connection, 'rtde_r') and self.robot_connection.rtde_r:
                            velocities = self.robot_connection.rtde_r.getActualQd()
                        # Estimate progress linearly as a last resort (capped below 100)
                        estimated_progress = min(99, (elapsed / (timeout * 0.95)) * 99)
                        progress.update(task, completed=estimated_progress)
                        # Consider the robot done if it's steady for several consecutive checks AND near-zero velocity
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
                    except Exception:
                        progress.update(task, description=f"🤖 {movement_type} (monitoring issues)")
                    # --- Method 3: Program running state ---
                    try:
                        program_running = self.robot_connection.rtde_c.isProgramRunning()
                        if program_running is False and self.robot_connection.rtde_c.isSteady():
                            progress.update(task, completed=100)
                            self.console.print(f"✅ [green]{movement_type} completed (no program running)![/green]")
                            return True
                    except Exception:
                        pass
                    time.sleep(0.1)
                # --- Timeout reached ---
                progress.update(task, description=f"⚠️ {movement_type} (timeout)")
                self.console.print(f"⚠️ [yellow]{movement_type} timeout after {timeout}s[/yellow]")
                if not real_progress_seen:
                    self.console.print(f"⚠️ [yellow]No progress updates received from robot. Check connection or robot state.[/yellow]")
                # Final verification: steady & velocities after timeout
                try:
                    is_steady = self.robot_connection.rtde_c.isSteady()
                    velocities = None
                    if hasattr(self.robot_connection, 'rtde_r') and self.robot_connection.rtde_r:
                        velocities = self.robot_connection.rtde_r.getActualQd()
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