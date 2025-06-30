"""
ROS Topic configuration for VizorBridge.
Defines topic names and their associated message types.
"""

from ur.config.robot_config import ROBOT_NAME

#ROS_HOST = "192.168.137.1"
ROS_HOST = "localhost"
ROS_PORT = 9090

# Topic names
TASK_EXECUTE_TOPIC = f"/{ROBOT_NAME}/task/execute"
TASK_EXECUTE_MSG_TYPE = "vizor_package/GeneralTask"

COMMAND_TOPIC = f"/{ROBOT_NAME}/command"
STATUS_TOPIC = "/Robot/status/physical"
STD_STRING_MSG_TYPE = "std_msgs/String"


POSITION_TOPIC = f"/{ROBOT_NAME}/set_position"
POSITION_MSG_TYPE = "sensor_msgs/JointState" 
