import time
import roslibpy
import sys
import os
import math
import asyncio
import threading

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
)
from ur.config.system_config import (
    COMMAND_START,
    COMMAND_END,
    STATUS_FABRICATION_STARTED,
    STATUS_FABRICATION_COMPLETE,
    STATUS_GRIPPER_OPEN,
    STATUS_GRIPPER_CLOSE,
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
    ):
        super().__init__(name, client)
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
        self.launcher_ref = None  # Reference back to launcher for agent creation

        # Subscribe to the specified task topic
        self.task_listener = roslibpy.Topic(
            self.client,
            task_topic,
            TASK_EXECUTE_MSG_TYPE,
        )
        self.task_listener.subscribe(self.handle_task)

        # NOTE: Command topic subscription is handled by parent RobotInterface class
        # We override process_command method to handle fabrication lifecycle

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
            self.status_publisher.publish(roslibpy.Message({'data': status_message}))
            print(f"📡 [ROS PUBLISH] Status: {status_message}")
        except Exception as e:
            print(f"❌ [ROS ERROR] Failed to publish status: {e}")

    def publish_joint_state(self):
        """Publish current joint state to /UR10/set_position"""
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
                self.position_publisher.publish(roslibpy.Message(joint_state_msg))
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
        # Offload task processing to background thread to avoid blocking ROS subscriber
        threading.Thread(target=lambda: self._handle_task_thread(msg), daemon=True).start()
        print("🔄 [BRIDGE] Task received, processing in background thread")

    def _handle_task_thread(self, msg: dict):
        """Background thread handler for ROS tasks."""
        data = msg.get("msg", msg)
        task_name = data.get('name', 'Unknown')

        print(f"\n🎯 [ROS TASK] {task_name}")
        # Notify agent about incoming task with system prompt context
        if self.agent_ref:
            self._notify_agent_sync('task_received', {
                'task_name': task_name,
                'task_data': data,
                'context_type': 'task_execute',
                'source_topic': 'TASK_EXECUTE_TOPIC'
            })
        # Execute the task with commentary
        success = self.execute_task_with_commentary(
            task_name,
            data.get("tcp", []),
            data.get("trajectory")
        )
        print(f"   {'✅' if success else '❌'} {'Success' if success else 'Failed'}\n")

    def process_command(self, msg: dict):
        """Override parent's command handling with fabrication lifecycle management."""
        command = msg.get('data')
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
            if self.fabrication_active:
                self._end_fabrication()
            # End fabrication stops the bridge gracefully
            self.cleanup(include_fabrication=True, include_connections=True)
        else:
            # Handle other commands using parent's logic for backwards compatibility
            if command == "pause":
                print("   ⏸️  Executing: Pause robot operations")
                self.pause = True
            elif command == "resume":
                print("   ▶️  Executing: Resume robot operations")
                self.pause = False
            elif command == "stop":
                print("   🛑 Executing: Stop robot operations")
                self.stop = True  # This is the parent's flag, not our method
            elif command == "home":
                print("   🏠 Executing: Move to home position")
                self.go_to_home()
            else:
                print(f"   ❓ Unknown: {command}")
        print()

    def _start_fabrication(self):
        """Start fabrication mode - create and start voice agent."""
        print("🚀 [FABRICATION] Starting fabrication mode...")
        
        # Recreate and re-subscribe to task topic in case it was unsubscribed
        try:
            self.task_listener = roslibpy.Topic(
                self.client,
                TASK_EXECUTE_TOPIC,
                TASK_EXECUTE_MSG_TYPE,
            )
            self.task_listener.subscribe(self.handle_task)
            print("🔄 [BRIDGE] Re-subscribed to task listener")
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
                self.voice_agent = self.launcher_ref.create_agent()
                if self.voice_agent:
                    # Start agent in background thread
                    self._start_voice_agent_thread()
                    
                    # Update voice agent UI to active state
                    if hasattr(self.voice_agent, 'ui'):
                        self.voice_agent.ui.set_fabrication_active(True)
                    
                    # Generate and display welcome message
                    self._generate_and_display_fabrication_message("start")
                else:
                    print("⚠️ [FABRICATION] Failed to create voice agent")
            except Exception as e:
                print(f"❌ [FABRICATION] Error starting voice agent: {e}")
        else:
            if not self.launcher_ref:
                print("⚠️ [FABRICATION] No launcher reference available")
            if self.voice_agent:
                print("⚠️ [FABRICATION] Voice agent already running")

    def _end_fabrication(self):
        """End fabrication mode - stop voice agent."""
        print("🏁 [FABRICATION] Ending fabrication mode...")
        
        if self.voice_agent:
            try:
                # Generate and display goodbye message first
                self._generate_and_display_fabrication_message("end")
                
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
        """Start the voice agent in a background thread."""
        if not self.voice_agent:
            return
            
        async def run_agent_async():
            try:
                await self.voice_agent.start()
            except Exception as e:
                print(f"❌ [AGENT] Voice agent error: {e}")

        self.voice_agent_thread = threading.Thread(
            target=lambda: asyncio.run(run_agent_async()),
            daemon=True,
            name="VoiceAgentThread"
        )
        self.voice_agent_thread.start()
        print("✅ [FABRICATION] Voice agent started successfully")

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
        """Make the voice agent speak the fabrication message."""
        if not self.voice_agent or not message:
            return
            
        try:
            # For OpenAI agents that have a session with direct connection
            if hasattr(self.voice_agent, 'session') and self.voice_agent.session:
                # Check if session is connected and ready
                if (hasattr(self.voice_agent.session, 'connection') and 
                    self.voice_agent.session.connection and
                    hasattr(self.voice_agent.session, 'is_connected') and
                    self.voice_agent.session.is_connected):
                    
                    # Create a conversation item as if the assistant is saying this message
                    async def speak_message():
                        await self.voice_agent.session.connection.send({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "text", "text": message}]
                            }
                        })
                        # Trigger a response to make the assistant speak
                        await self.voice_agent.session.connection.send({"type": "response.create"})
                    
                    # Run the async function
                    asyncio.run(speak_message())
                    print("🔊 Assistant speaking...")
                    return
                    
                elif hasattr(self.voice_agent.session, 'send_message'):
                    # Fallback to send_message approach if connection not ready
                    try:
                        asyncio.run(self.voice_agent.session.send_message(message))
                        print("🔊 Assistant speaking...")
                        return
                    except Exception as e:
                        print(f"⚠️ [SPEAK] Send message failed: {e}")
                
                else:
                    print(f"⚠️ [SPEAK] Session not connected yet - message: {message[:50]}...")
            
            # For SmolAgent with TTS capability  
            if hasattr(self.voice_agent, 'session') and hasattr(self.voice_agent.session, 'config'):
                if getattr(self.voice_agent.session.config, 'enable_tts', False):
                    # Use SmolAgent TTS
                    if hasattr(self.voice_agent.session, 'voice_processor'):
                        try:
                            audio_data = self.voice_agent.session.voice_processor.text_to_speech(message)
                            if audio_data:
                                self.voice_agent.session.voice_processor.play_audio(audio_data)
                                print("🔊 Assistant speaking...")
                        except Exception as e:
                            print(f"⚠️ [TTS] Error with SmolAgent TTS: {e}")
            
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
            # Convert configured home position (in degrees) to radians
            home_joints_rad = [math.radians(angle) for angle in HOME_POSITION]
            success = self.robot_connection.move_j(home_joints_rad, ROBOT_HOME_SPEED, ROBOT_HOME_ACCELERATION)
            if success:
                self.publish_joint_state()
            return success
        except Exception:
            return False

    def moveL(self, tcp_pose: list, speed: float = ROBOT_MOVE_SPEED, acceleration: float = ROBOT_ACCELERATION) -> bool:
        """Moves the robot linearly to a given TCP pose."""
        if not self.is_connected or not self.robot_connection:
            return False
        try:
            success = self.robot_connection.move_l(tcp_pose, speed, acceleration)
            if success:
                self.publish_joint_state()
            return success
        except Exception:
            return False

    def moveJ(self, joint_positions: list, speed: float = ROBOT_MOVE_SPEED, acceleration: float = ROBOT_ACCELERATION) -> bool:
        """Moves the robot in joint space to the given positions."""
        if not self.is_connected or not self.robot_connection:
            return False
        try:
            success = self.robot_connection.move_j(joint_positions, speed, acceleration)
            if success:
                self.publish_joint_state()
            return success
        except Exception:
            return False

    def execute_tcp_frames(self, frames: list) -> bool:
        """Executes a sequence of linear movements to given TCP frames."""
        if not self.is_connected or not self.robot_connection:
            return False
        try:
            for frame in frames:
                if not self.moveL(frame):
                    # Stop if any single move fails
                    return False
            return True
        except Exception:
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
                positions = point.get("positions", [])
                if len(positions) == 6:
                    positions_rad = [math.radians(p) for p in positions]
                    if not self.moveJ(positions_rad):
                        # Stop if any single move fails
                        return False
                else:
                    # Invalid point, stop execution
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
        # Import status generation function here to avoid circular imports
        from ur.config.system_config import generate_task_status
        
        task_name_lower = task_name.lower()
        success = False
        
        if "pickup" in task_name_lower or "pick" in task_name_lower:
            success = self._handle_pickup_task(tcps, trajectory)
        elif "place" in task_name_lower:
            success = self._handle_place_task(tcps, trajectory)
        elif "home" in task_name_lower:
            success = self._handle_home_task(tcps, trajectory)
        else:
            success = self._execute_trajectory_only(tcps, trajectory)
        
        # CRITICAL: Only send SUCCESS status when entire task completes successfully
        if success and task_name:
            success_status = generate_task_status(task_name, status_type="SUCCESS")
            self.publish_status(success_status)
            print(f"✅ [TASK SUCCESS] {success_status}")
        
        return success

    def execute_task_with_commentary(self, task_name: str, tcps: list = None, trajectory: dict = None) -> bool:
        """Execute task with LLM-generated commentary."""
        
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
            success = self._execute_trajectory_only(tcps, trajectory)
        
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
        
        return success

    def _handle_pickup_task(self, tcps: list = None, trajectory: dict = None) -> bool:
        """Handle pickup tasks: Open gripper → Execute trajectory → Close gripper."""
        return (self.control_gripper("open") and 
                self._execute_trajectory_only(tcps, trajectory) and 
                self.control_gripper("close"))

    def _handle_place_task(self, tcps: list = None, trajectory: dict = None) -> bool:
        """Handle place tasks: Execute trajectory → Open gripper."""
        return (self._execute_trajectory_only(tcps, trajectory) and 
                self.control_gripper("open"))

    def _handle_home_task(self, tcps: list = None, trajectory: dict = None) -> bool:
        """Handle home tasks: Execute trajectory only (no gripper control)."""
        return self._execute_trajectory_only(tcps, trajectory)

    def _execute_trajectory_only(self, tcps: list = None, trajectory: dict = None) -> bool:
        """Execute either TCP frames or joint trajectory without gripper control."""
        if tcps:
            return self.execute_tcp_frames(tcps if isinstance(tcps, list) else [tcps])
        elif trajectory:
            return self.execute_trajectory_msg(trajectory)
        else:
            return False

    def _handle_pickup_task_with_commentary(self, tcps: list = None, trajectory: dict = None, task_name: str = "") -> bool:
        """Handle pickup with dynamic LLM commentary."""
        
        # Request human assistance
        if self.agent_ref:
            self._notify_agent_sync('request_human_action', {
                'action': 'place_element_on_supply_station',
                'task_name': task_name
            })
        
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
        gripper_close = self.control_gripper("close")
        if gripper_close and self.agent_ref:
            self._notify_agent_sync('robot_action', {
                'action_type': 'element_secured',
                'task_name': task_name,
                'context_type': 'task_execute'
            })
        
        return gripper_open and trajectory_success and gripper_close

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
        
        # Request human to secure screws
        if trajectory_success and self.agent_ref:
            self._notify_agent_sync('request_human_action', {
                'action': 'secure_screws',
                'task_name': task_name
            })
        
        gripper_open = self.control_gripper("open")
        
        return trajectory_success and gripper_open

    def _handle_home_task_with_commentary(self, tcps: list = None, trajectory: dict = None, task_name: str = "") -> bool:
        """Handle home with commentary."""
        
        if self.agent_ref:
            self._notify_agent_sync('robot_action', {
                'action_type': 'returning_home',
                'task_name': task_name,
                'context_type': 'task_execute'
            })
        
        return self._execute_trajectory_only(tcps, trajectory)