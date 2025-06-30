import os
from typing import Optional

# Launcher settings (environment-driven)

AGENT_TYPE = "openai"
VOICE_ENABLED = True

# SmolAgent settings
SMOL_MODEL_ID = "gpt-4o-mini"
SMOL_PROVIDER = "openai"

# Default OpenAI realtime model ID (overridable via environment)
OPENAI_MODEL_ID = "gpt-4o-realtime-preview"

# OpenAI Realtime API constants
OPENAI_TRANSCRIPTION_MODEL = "whisper-1"

# Threading system settings  
TOOL_EXECUTOR_MAX_WORKERS = 2

# Session management
SESSION_ID_PREFIX = "session"


# Bridge command constants
COMMAND_START = "start_fabrication"
COMMAND_END = "end_fabrication"

# Essential base status constants (cannot be generated dynamically)
STATUS_FABRICATION_STARTED = "fabrication_started"
STATUS_FABRICATION_COMPLETE = "fabrication_complete"
STATUS_MOVING_TO_SUPPLY = "moving_to_supply"
STATUS_AT_ASSEMBLY = "at_assembly"
STATUS_GRIPPER_OPEN = "gripper_open"
STATUS_GRIPPER_CLOSE = "gripper_close"

# Additional status constants from PlantUML requirements
STATUS_MOVING_TO_POSITION = "moving_to_position"
STATUS_AT_TARGET_POSITION = "at_target_position"
STATUS_MOVING_TO_HOME = "moving_to_home"
STATUS_COMPLETE_ADJUST_POSITION = "complete_adjust_position"
STATUS_ELEMENT_RETRIEVED = "element_retrieved"
STATUS_ELEMENT_PLACED = "element_placed"

# Dynamic Status Generation System
# ================================
# CRITICAL RULE: success status messages must ONLY be sent at task completion!
# Do not use success for intermediate operations within a task.

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_STOP = "stop"

def generate_status_message(
    action: str, 
    task_name: Optional[str] = None, 
    element_info: Optional[str] = None,
    status_type: str = "info"
) -> str:
    """Generate dynamic status messages based on action and context.
    
    Args:
        action: Base action being performed (e.g., "pickup", "place", "move")
        task_name: Optional task name for context (e.g., "pickup_40", "returnhome")
        element_info: Optional element information (e.g., "40cm", "50cm")
        status_type: Type of status message ("success", "progress", "complete", "info", "error")
    
    Returns:
        Formatted status message string
        
    Examples:
        generate_status_message("pickup", "pickup_40", status_type="success") 
        -> "success_pickup_40"  # ONLY at complete task success!
        
        generate_status_message("moving", element_info="40cm", status_type="progress")
        -> "moving_40cm_element"
        
        generate_status_message("adjust_position", status_type="complete")
        -> "complete_adjust_position"
        
    WARNING: Only use status_type="success" when the ENTIRE task is completed successfully!
    """
    # Build base message
    base_msg = action.lower()
    
    # Add element information if provided
    if element_info:
        # Clean element info (remove spaces, convert to lowercase)
        clean_element = element_info.replace(" ", "").replace("cm", "cm").lower()
        base_msg = f"{base_msg}_{clean_element}_element"
    
    # Add task name if provided (for specific task tracking)
    if task_name:
        base_msg = f"{base_msg}_{task_name.lower()}" if not element_info else base_msg
    
    # Apply status type prefix/suffix
    status_type = status_type.lower()
    if status_type == "success":
        return f"success_{task_name.lower()}" if task_name else f"success_{base_msg}"
    elif status_type == "progress":
        return f"{base_msg}_in_progress"
    elif status_type == "complete":
        return f"complete_{base_msg}" if not base_msg.startswith("complete") else base_msg
    elif status_type == "error":
        return f"error_{base_msg}"
    else:  # info or default
        return base_msg

def generate_movement_status(
    movement_type: str,
    target: Optional[str] = None,
    status_type: str = "progress"
) -> str:
    """Generate movement-specific status messages.
    
    Args:
        movement_type: Type of movement ("linear", "joint", "home", "relative", "absolute")
        target: Optional target description ("home", "supply", "assembly", "40cm")
        status_type: Status type ("progress", "complete", "success")
    
    Returns:
        Movement-specific status message
        
    Examples:
        generate_movement_status("linear", "supply") -> "moving_to_supply"
        generate_movement_status("home", status_type="complete") -> "at_home_position"
    """
    movement_type = movement_type.lower()
    status_type = status_type.lower()

    if movement_type == "home":
        if status_type == "progress":
            return "moving_to_home"
        elif status_type in ["complete", "success"]:
            return "at_home_position"
    
    if target:
        target_lower = target.lower()
        if status_type == "progress":
            return f"moving_to_{target_lower}"
        elif status_type in ["complete", "success"]:
            return f"at_{target_lower}"
    
    # Fallback for generic movement
    if status_type == "progress":
        return f"moving_{movement_type}"
    elif status_type in ["complete", "success"]:
        return f"{movement_type}_complete"
    
    return f"{movement_type}_movement"

def generate_gripper_status(action: str, status_type: str = "complete") -> str:
    """Generate gripper-specific status messages.
    
    Args:
        action: Gripper action ("open", "close", "grab", "release")
        status_type: Status type ("progress", "complete", "success")
    
    Returns:
        Gripper-specific status message
    """
    action_lower = action.lower()
    status_type = status_type.lower()
    
    if status_type == "progress":
        return f"gripper_{action_lower}_in_progress"
    elif status_type in ["complete", "success"]:
        return f"gripper_{action_lower}"
    else:
        return f"gripper_{action_lower}_{status_type}"

def generate_task_status(task_name: str, status_type: str = "progress") -> str:
    """Generate task-specific status messages.
    
    Args:
        task_name: Name of the task (e.g., "checkelements", "pickup_40")
        status_type: Status type ("success", "progress", "complete", "error")
    
    Returns:
        Task-specific status message
        
    Examples:
        generate_task_status("pickup_40", "success") -> "success_pickup_40"  # ONLY at task completion
        generate_task_status("checkelements", "progress") -> "checkelements_in_progress"
    
    Note:
        success status should ONLY be used when the entire task is completed successfully.
        Use progress for intermediate operations within a task.
    """
    task_name = task_name.lower()
    status_type = status_type.lower()

    if status_type == "success":
        return f"success_{task_name}"
    elif status_type == "progress":
        return f"{task_name}_in_progress"
    elif status_type == "complete":
        return f"{task_name}_complete"
    elif status_type == "error":
        return f"error_{task_name}"
    else:
        return f"{task_name}_{status_type}"

def generate_supply_status(
    element_length: str,
    action: str = "retrieve",
    status_type: str = "success"
) -> str:
    """Generate supply station status messages.
    
    Args:
        element_length: Length of element (e.g., "40cm", "50cm")
        action: Supply action ("retrieve", "place", "get")
        status_type: Status type ("success", "progress", "complete")
    
    Returns:
        Supply-specific status message
        
    Examples:
        generate_supply_status("40cm", "retrieve", "success") -> "success_retrieve_40cm_element"
        generate_supply_status("50cm", "place", "progress") -> "placing_50cm_element"
    """
    # Clean element length
    clean_length = element_length.replace(" ", "").replace("cm", "cm").lower()
    action_lower = action.lower()
    status_type = status_type.lower()
    
    if status_type == "success":
        return f"success_{action_lower}_{clean_length}_element"
    elif status_type == "progress":
        if action_lower == "retrieve":
            return f"retrieving_{clean_length}_element"
        elif action_lower == "place":
            return f"placing_{clean_length}_element"
        else:
            return f"{action_lower}ing_{clean_length}_element"
    elif status_type == "complete":
        return f"{action_lower}_{clean_length}_element_complete"
    else:
        return f"{action_lower}_{clean_length}_element_{status_type}" 