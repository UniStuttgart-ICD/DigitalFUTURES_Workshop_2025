"""
Robot Configuration Constants
============================

Configuration settings for UR robot control.
"""

# Environment switch
# Set this to True to use the real robot settings, False for simulation
USE_REAL_ROBOT = True

# Robot connection
if USE_REAL_ROBOT:
    ROBOT_IP = "192.168.2.3"
else:
    ROBOT_IP = "192.168.56.101"
    
ROBOT_NAME = "UR10"
RTDE_CONTROL_PORT = 30004  # Default RTDE control port

# Movement parameters
if USE_REAL_ROBOT:
    # Real robot speeds
    ROBOT_MOVE_SPEED = 0.2 # m/s
    ROBOT_ACCELERATION = 1.0   # m/s²
    ROBOT_HOME_SPEED = 0.1  # m/s
    ROBOT_HOME_ACCELERATION = 1.0 # m/s²
else:
    # Simulation speeds
    ROBOT_MOVE_SPEED = 2.5 # m/s
    ROBOT_ACCELERATION = 1.2   # m/s² - ur_rtde default acceleration
    ROBOT_HOME_SPEED = 1.0 # m/s - Slower speed for home movements
    ROBOT_HOME_ACCELERATION = 1.5 # m/s² - Slower acceleration for home movements

# Safety settings
WAKEWORD = "mave"
HOME_POSITION = [-167.412124, -99.038841, -78.558301, 87.597142, -90, 94.612093]

# Safety margins
SAFETY_MARGIN_JOINTS = 0.1  # radians - margin for joint limits
SAFETY_MARGIN_POSITION = 0.05  # meters - margin for position limits

# Robot reach validation (typical UR robot reach is ~1.85m)
MAX_ROBOT_REACH = 2.0  # meters - maximum reasonable robot reach
MAX_RELATIVE_MOVE = 1.0  # meters - maximum reasonable relative movement

# Gripper configuration
GRIPPER_DIGITAL_OUT = 4  # Digital output pin for gripper stepper motor
GRIPPER_OPEN_STATE = False   # Digital signal state for open gripper
GRIPPER_CLOSE_STATE = True # Digital signal state for close gripper
GRIPPER_OPEN_TIME = 0.2   # Time to fully open gripper (seconds)
GRIPPER_CLOSE_TIME = 0.2  # Time to fully close gripper (seconds)
GRIPPER_IO_TYPE = "standard"  # Type of digital output: "standard", "configurable", or "tool"

# Test data paths
TEST_DATA_MESSAGES_DIR = "data_messages"  # Directory for test trajectory files

# Robot Agent Identity
ROBOT_AGENT_NAME = ROBOT_NAME  # use the actual robot name constant
ROBOT_AGENT_UI_NAME = "OpenAI-Agent"

# Robot tool threading
ROBOT_TOOL_THREAD_PREFIX = "robot_tool" 