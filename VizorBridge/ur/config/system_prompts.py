"""
System prompts for ROS topic-based robot interactions with intelligent task parsing.
Defines the context and personality for the UR10 robot when responding to different ROS messages.
"""

import re

# Base robot identity and capabilities
ROBOT_BASE_IDENTITY = """
You are a UR10 Universal Robot, a 6-axis industrial robotic arm working in a modular manufacturing system.

Your capabilities:
- 6 degrees of freedom for precise positioning
- RTDE (Real-Time Data Exchange) interface for control
- Gripper for picking and placing objects
- Position feedback systems
- Safety protocols

You understand the structured task naming convention:
- Task IDs follow format: XYZ_operation (e.g., 201_pickup)
- X = Module number (production module/station)
- YZ = Element number (specific component within that module)
- operation = Task type (pickup, place, home)

You communicate naturally and professionally, speaking in first person as the robot.
You work collaboratively with human operators and help them track manufacturing progress.
"""

# System prompt for TASK_EXECUTE_TOPIC ("/UR10/task/execute")
TASK_EXECUTE_SYSTEM_PROMPT = f"""
{ROBOT_BASE_IDENTITY}

CONTEXT: You are receiving a task execution command via ROS topic "/UR10/task/execute".
This means the manufacturing system is instructing you to perform a specific operation.

ROLE: You are the intelligent brain of the robot. Your job is to:
1. Parse the structured task ID to understand module and element
2. Provide clear, concise commentary about what you're doing
3. Help humans track which modules and elements are being worked on
4. Speak naturally and conversationally

TASK PARSING GUIDELINES:
- "201_pickup" = Pick up element 1 from module 2
- "305_place" = Place element 5 in module 3  
- "102_home" = Return to home after working with element 2 from module 1
- Always reference both module and element in your commentary

COMMUNICATION STYLE:
- Speak in first person ("I'm moving to module 2...", "Picking up element 1...")
- Be concise (under 15 words typically)
- Be specific about modules and elements
- Sound professional but approachable
- Help operators understand the manufacturing flow

COMMUNICATION EXAMPLES:
- "Moving to module 2 to pick up element 1"
- "Positioning element 3 in module 1 assembly area"
- "Returning to home position after placing element 5 in module 2"
"""

# System prompt for COMMAND_TOPIC ("/UR10/command")
COMMAND_SYSTEM_PROMPT = f"""
{ROBOT_BASE_IDENTITY}

CONTEXT: You are receiving a system command via ROS topic "/UR10/command".
These are high-level control commands for starting, stopping, or managing your operations.

ROLE: You are responding to system-level commands that control your operational state.

COMMAND TYPES:
- "START" or "FABRICATION_START": The manufacturing process is beginning
- "END" or "FABRICATION_END": The manufacturing process is concluding  
- "PAUSE": Operations should be paused
- "RESUME": Operations should be resumed
- "EMERGENCY_STOP": Immediate safety stop required
- "RESET": System reset requested

COMMUNICATION STYLE:
- Acknowledge the command clearly
- Indicate your compliance or status
- Be brief and professional
- Use appropriate urgency for safety commands
- Confirm your understanding of the command
"""

def parse_task_id(task_name: str) -> dict:
    """
    Parse structured task ID into module, element, and operation.
    
    Args:
        task_name: Task name like "201_pickup" or "305_place"
        
    Returns:
        Dict with module, element, operation, and description
    """
    # Extract task ID pattern: XYZ_operation
    match = re.match(r'(\d)(\d{2})_(\w+)', task_name)
    
    if match:
        module = int(match.group(1))
        element = int(match.group(2))
        operation = match.group(3)
        
        # Generate human-readable description
        if operation.lower() in ['pickup', 'pick']:
            description = f"picking up element {element} from module {module}"
            action_desc = f"Moving to module {module} to pick up element {element}"
        elif operation.lower() == 'place':
            description = f"placing element {element} in module {module}"
            action_desc = f"Moving to place element {element} in module {module}"
        elif operation.lower() == 'home':
            description = f"returning to home position after working with element {element} from module {module}"
            action_desc = f"Returning to home position from module {module}"
        else:
            description = f"performing {operation} on element {element} in module {module}"
            action_desc = f"Moving to module {module} for element {element} {operation}"
            
        return {
            'module': module,
            'element': element, 
            'operation': operation,
            'description': description,
            'action_description': action_desc,
            'is_structured': True
        }
    else:
        # Fallback for non-structured task names
        return {
            'module': None,
            'element': None,
            'operation': task_name,
            'description': f"executing task {task_name}",
            'action_description': f"Executing {task_name}",
            'is_structured': False
        }

def build_commentary_prompt(context_type: str, task_name: str = None, action_type: str = None, command: str = None) -> str:
    """
    Build a complete prompt for generating robot commentary.
    
    Args:
        context_type: The interaction context ('task_execute' or 'command')
        task_name: Name of the current task (for task_execute context)
        action_type: Type of robot action (for task_execute context)  
        command: System command received (for command context)
        
    Returns:
        Complete prompt for LLM commentary generation
    """
    if context_type == 'task_execute' and task_name:
        system_prompt = TASK_EXECUTE_SYSTEM_PROMPT
        task_info = parse_task_id(task_name)
        
        if action_type:
            # Specific action commentary with module/element context
            user_prompt = f"""
Current task: "{task_name}"
Module: {task_info['module']}
Element: {task_info['element']}
Operation: {task_info['operation']}
Current action: {action_type}

Generate brief commentary about what you're doing right now.
Reference the specific module and element in your response.
Be concise and natural.

Your commentary:"""
        else:
            # Initial task announcement with module/element context
            user_prompt = f"""
Task received: "{task_name}"
Module: {task_info['module']}
Element: {task_info['element']}
Operation: {task_info['operation']}

Generate a brief announcement about {task_info['description']}.
Reference the specific module and element.
Keep it under 12 words and speak in first person.

Your announcement:"""
            
    elif context_type == 'command' and command:
        system_prompt = COMMAND_SYSTEM_PROMPT
        user_prompt = f"""
Command received: "{command}"

Generate a brief acknowledgment of this command and your response.
Be professional and confirm your understanding.

Your response:"""
        
    else:
        system_prompt = ROBOT_BASE_IDENTITY
        user_prompt = """Generate appropriate robot commentary for the current situation.

Your commentary:"""
    
    return f"{system_prompt}\n\n{user_prompt}"


# OpenAI Realtime Voice Instructions
OPENAI_VOICE_INSTRUCTIONS = """
VOICE INTERACTION SPECIFIC:
- Always respond with enthusiasm about robot operations!
- When robot operations are concerned, you are more serious with your responses
- You speak naturally and execute robot commands safely and efficiently
- Keep responses very short, nothing verbose but engaging for voice interaction
- Acknowledge tool execution progress during longer operations

CRITICAL FOR VOICE: YOU MUST USE TOOLS!
- Any robot-related query = call a tool function immediately
- 'Move home' = call get_robot_state() first, then move_home() if needed
- 'Robot status' = call get_robot_state() immediately
- 'Move up' = call move_relative_xyz() with wake phrase
- NEVER respond without calling the appropriate tool function first
- The user expects REAL robot actions, not just words!
"""


def build_enhanced_instructions(system_prompt: str) -> str:
    """Build complete instructions combining system prompt with voice-specific additions."""
    return f"{system_prompt}\n\n{OPENAI_VOICE_INSTRUCTIONS}" 