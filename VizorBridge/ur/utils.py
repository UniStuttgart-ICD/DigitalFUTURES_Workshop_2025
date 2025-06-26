from ur.config.robot_config import WAKEWORD

def validate_wakeword(text: str) -> bool:
    """Check if the wake word is present in the text."""
    return WAKEWORD in text.lower()


def send_immediate_response(command_text: str):
    """Immediate acknowledgment for robot commands."""
    print(f"Processing: {command_text}") 