from abc import abstractmethod
import time
import threading
import roslibpy


class RobotInterface:
    """Base interface for robot bridges."""

    def __init__(self, name, client):
        self.name = name
        self.pause = False
        self.stop = False
        self.client = client

        exec_listener = roslibpy.Topic(self.client, "/UR10/task/execute", "vizor_package/GeneralTask")
        exec_listener.subscribe(self.execute_trajectory)

        command_listener = roslibpy.Topic(self.client, "/UR10/command", "std_msgs/String")
        command_listener.subscribe(self.process_command)

        print(f"{self.name} Connected: {self.client.is_connected}")

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def go_to_home(self):
        pass

    @abstractmethod
    def execute(self, frames):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def execute_trajectory(self, msg):
        # Handle the actual YAML structure: msg.trajectory.joint_trajectory.points
        trajectory = msg.get("trajectory", {})
        joint_trajectory = trajectory.get("joint_trajectory", {})
        frames = joint_trajectory.get("points", [])
        
        # Extract task information for better visibility
        task_name = msg.get("name", "Unknown Task")
        task_id = msg.get("id", "N/A")
        
        print(f"\n🚀 [ROS ACTION] Trajectory execution received!")
        print(f"   📋 Task: {task_name} (ID: {task_id})")
        print(f"   📊 Points: {len(frames)} trajectory points")
        print(f"   🎯 Target: {msg.get('target', 'Unknown')}")
        print("   ⚡ Starting execution thread...")
        
        thread = threading.Thread(target=self.execute, args=(frames,))
        thread.start()
        print("   ✅ Execution thread started successfully!\n")

    def process_command(self, msg):
        command = msg['data']
        print(f"\n🎛️  [ROS COMMAND] Received: {command}")

        if command == "START_FABRICATION":
            print("   🏠 Executing: Move to home position")
            self.go_to_home()
        elif command == "pause":
            print("   ⏸️  Executing: Pause robot operations")
            self.pause = True
        elif command == "resume":
            print("   ▶️  Executing: Resume robot operations")
            self.pause = False
        elif command == "stop":
            print("   🛑 Executing: Stop robot operations")
            self.stop = True
        elif command == "END_FABRICATION":
            print("   🏁 Executing: End fabrication sequence")
            self.stop = True
        elif command == "home":
            print("   🏠 Executing: Move to home position")
            self.go_to_home()
        else:
            print(f"   ❓ Unknown command: {command}")
        print()

    def init_home(self, auto_home):
        self.get_current_state()
        if auto_home:
            time.sleep(2)
            self.go_to_home()

    def cleanup_connections(self):
        self.client.terminate()
        self.client.close()

    def moveJ(self, joints):
        """Move robot in joint space."""
        print(f"{self.name} moveJ {joints}")

    def moveL(self, tcp):
        """Move robot linearly to a TCP."""
        print(f"{self.name} moveL {tcp}")
