"""
ROS Topic configuration for VizorBridge.
Defines topic names and their associated message types.
"""

# Topic names
TASK_EXECUTE_TOPIC = "/UR10/task/execute"
TASK_EXECUTE_MSG_TYPE = "vizor_package/GeneralTask"

COMMAND_TOPIC = "/UR10/command"
STATUS_TOPIC = "/Robot/status/physical"
STD_STRING_MSG_TYPE = "std_msgs/String"


POSITION_TOPIC = "/UR10/set_position"
POSITION_MSG_TYPE = "sensor_msgs/JointState" 
