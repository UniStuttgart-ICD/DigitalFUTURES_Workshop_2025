import os
from typing import Optional

# Launcher settings (environment-driven)

AGENT_TYPE = "openai"
VOICE_ENABLED = True

# SmolAgent settings
SMOL_MODEL_ID = "gpt-4o-mini"
SMOL_PROVIDER = "openai"

# Default OpenAI realtime model ID (overridable via environment)
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini-realtime-preview-2024-12-17")

# OpenAI Realtime API constants
OPENAI_TRANSCRIPTION_MODEL = "whisper-1"

# Threading system settings  
TOOL_EXECUTOR_MAX_WORKERS = 2

# Session management
SESSION_ID_PREFIX = "session"

# ROS bridge settings
ROS_HOST = "localhost"
ROS_PORT = 9090

# Bridge command constants
COMMAND_START = "START_FABRICATION"
COMMAND_END = "END_FABRICATION"

# Essential base status constants (cannot be generated dynamically)
STATUS_FABRICATION_STARTED = "FABRICATION_STARTED"
STATUS_FABRICATION_COMPLETE = "FABRICATION_COMPLETE"
STATUS_MOVING_TO_SUPPLY = "MOVING_TO_SUPPLY"
STATUS_AT_ASSEMBLY = "AT_ASSEMBLY"
STATUS_GRIPPER_OPEN = "GRIPPER_OPEN"
STATUS_GRIPPER_CLOSE = "GRIPPER_CLOSE"

# Additional status constants from PlantUML requirements
STATUS_MOVING_TO_POSITION = "MOVING_TO_POSITION"
STATUS_AT_TARGET_POSITION = "AT_TARGET_POSITION"
STATUS_MOVING_TO_HOME = "MOVING_TO_HOME"
STATUS_COMPLETE_ADJUST_POSITION = "COMPLETE_ADJUST_POSITION"
STATUS_ELEMENT_RETRIEVED = "ELEMENT_RETRIEVED"
STATUS_ELEMENT_PLACED = "ELEMENT_PLACED"

# Dynamic Status Generation System
# ================================
# CRITICAL RULE: SUCCESS status messages must ONLY be sent at task completion!
# Do not use SUCCESS for intermediate operations within a task.

def generate_status_message(
    action: str, 
    task_name: Optional[str] = None, 
    element_info: Optional[str] = None,
    status_type: str = "INFO"
) -> str:
    """Generate dynamic status messages based on action and context.
    
    Args:
        action: Base action being performed (e.g., "PICKUP", "PLACE", "MOVE")
        task_name: Optional task name for context (e.g., "PickUp_40", "ReturnHome")
        element_info: Optional element information (e.g., "40cm", "50cm")
        status_type: Type of status message ("SUCCESS", "PROGRESS", "COMPLETE", "INFO", "ERROR")
    
    Returns:
        Formatted status message string
        
    Examples:
        generate_status_message("PICKUP", "PickUp_40", status_type="SUCCESS") 
        -> "SUCCESS_PickUp_40"  # ONLY at complete task success!
        
        generate_status_message("MOVING", element_info="40cm", status_type="PROGRESS")
        -> "MOVING_40CM_ELEMENT"
        
        generate_status_message("ADJUST_POSITION", status_type="COMPLETE")
        -> "COMPLETE_ADJUST_POSITION"
        
    WARNING: Only use status_type="SUCCESS" when the ENTIRE task is completed successfully!
    """
    # Build base message
    base_msg = action.upper()
    
    # Add element information if provided
    if element_info:
        # Clean element info (remove spaces, convert to uppercase)
        clean_element = element_info.replace(" ", "").replace("cm", "CM").upper()
        base_msg = f"{base_msg}_{clean_element}_ELEMENT"
    
    # Add task name if provided (for specific task tracking)
    if task_name:
        base_msg = f"{base_msg}_{task_name}" if not element_info else base_msg
    
    # Apply status type prefix/suffix
    if status_type == "SUCCESS":
        return f"SUCCESS_{task_name}" if task_name else f"SUCCESS_{base_msg}"
    elif status_type == "PROGRESS":
        return f"{base_msg}_IN_PROGRESS"
    elif status_type == "COMPLETE":
        return f"COMPLETE_{base_msg}" if not base_msg.startswith("COMPLETE") else base_msg
    elif status_type == "ERROR":
        return f"ERROR_{base_msg}"
    else:  # INFO or default
        return base_msg

def generate_movement_status(
    movement_type: str,
    target: Optional[str] = None,
    status_type: str = "PROGRESS"
) -> str:
    """Generate movement-specific status messages.
    
    Args:
        movement_type: Type of movement ("LINEAR", "JOINT", "HOME", "RELATIVE", "ABSOLUTE")
        target: Optional target description ("HOME", "SUPPLY", "ASSEMBLY", "40cm")
        status_type: Status type ("PROGRESS", "COMPLETE", "SUCCESS")
    
    Returns:
        Movement-specific status message
        
    Examples:
        generate_movement_status("LINEAR", "SUPPLY") -> "MOVING_TO_SUPPLY"
        generate_movement_status("HOME", status_type="COMPLETE") -> "AT_HOME_POSITION"
    """
    if movement_type.upper() == "HOME":
        if status_type == "PROGRESS":
            return "MOVING_TO_HOME"
        elif status_type in ["COMPLETE", "SUCCESS"]:
            return "AT_HOME_POSITION"
    
    if target:
        target_upper = target.upper()
        if status_type == "PROGRESS":
            return f"MOVING_TO_{target_upper}"
        elif status_type in ["COMPLETE", "SUCCESS"]:
            return f"AT_{target_upper}"
    
    # Fallback for generic movement
    if status_type == "PROGRESS":
        return f"MOVING_{movement_type.upper()}"
    elif status_type in ["COMPLETE", "SUCCESS"]:
        return f"{movement_type.upper()}_COMPLETE"
    
    return f"{movement_type.upper()}_MOVEMENT"

def generate_gripper_status(action: str, status_type: str = "COMPLETE") -> str:
    """Generate gripper-specific status messages.
    
    Args:
        action: Gripper action ("OPEN", "CLOSE", "GRAB", "RELEASE")
        status_type: Status type ("PROGRESS", "COMPLETE", "SUCCESS")
    
    Returns:
        Gripper-specific status message
    """
    action_upper = action.upper()
    
    if status_type == "PROGRESS":
        return f"GRIPPER_{action_upper}_IN_PROGRESS"
    elif status_type in ["COMPLETE", "SUCCESS"]:
        return f"GRIPPER_{action_upper}"
    else:
        return f"GRIPPER_{action_upper}_{status_type.upper()}"

def generate_task_status(task_name: str, status_type: str = "PROGRESS") -> str:
    """Generate task-specific status messages.
    
    Args:
        task_name: Name of the task (e.g., "CheckElements", "PickUp_40")
        status_type: Status type ("SUCCESS", "PROGRESS", "COMPLETE", "ERROR")
    
    Returns:
        Task-specific status message
        
    Examples:
        generate_task_status("PickUp_40", "SUCCESS") -> "SUCCESS_PickUp_40"  # ONLY at task completion
        generate_task_status("CheckElements", "PROGRESS") -> "CheckElements_IN_PROGRESS"
    
    Note:
        SUCCESS status should ONLY be used when the entire task is completed successfully.
        Use PROGRESS for intermediate operations within a task.
    """
    if status_type == "SUCCESS":
        return f"SUCCESS_{task_name}"
    elif status_type == "PROGRESS":
        return f"{task_name}_IN_PROGRESS"
    elif status_type == "COMPLETE":
        return f"{task_name}_COMPLETE"
    elif status_type == "ERROR":
        return f"ERROR_{task_name}"
    else:
        return f"{task_name}_{status_type.upper()}"

def generate_supply_status(
    element_length: str,
    action: str = "RETRIEVE",
    status_type: str = "SUCCESS"
) -> str:
    """Generate supply station status messages.
    
    Args:
        element_length: Length of element (e.g., "40cm", "50cm")
        action: Supply action ("RETRIEVE", "PLACE", "GET")
        status_type: Status type ("SUCCESS", "PROGRESS", "COMPLETE")
    
    Returns:
        Supply-specific status message
        
    Examples:
        generate_supply_status("40cm", "RETRIEVE", "SUCCESS") -> "SUCCESS_RETRIEVE_40CM_ELEMENT"
        generate_supply_status("50cm", "PLACE", "PROGRESS") -> "PLACING_50CM_ELEMENT"
    """
    # Clean element length
    clean_length = element_length.replace(" ", "").replace("cm", "CM").upper()
    action_upper = action.upper()
    
    if status_type == "SUCCESS":
        return f"SUCCESS_{action_upper}_{clean_length}_ELEMENT"
    elif status_type == "PROGRESS":
        if action_upper == "RETRIEVE":
            return f"RETRIEVING_{clean_length}_ELEMENT"
        elif action_upper == "PLACE":
            return f"PLACING_{clean_length}_ELEMENT"
        else:
            return f"{action_upper}ING_{clean_length}_ELEMENT"
    elif status_type == "COMPLETE":
        return f"{action_upper}_{clean_length}_ELEMENT_COMPLETE"
    else:
        return f"{action_upper}_{clean_length}_ELEMENT_{status_type.upper()}" 