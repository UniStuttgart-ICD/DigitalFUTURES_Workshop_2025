from abc import ABC, abstractmethod

class BaseVoiceAgent(ABC):
    """
    Abstract base class for a voice agent.

    This class defines the common interface that all voice agent
    implementations must adhere to, allowing them to be used interchangeably
    by the main application.
    """

    def __init__(self, bridge_ref, ui_ref):
        """
        Initializes the agent.

        Args:
            bridge_ref: A reference to the URBridge instance for robot control.
            ui_ref: A reference to the VoiceAgentStatus UI instance for status updates.
        """
        self.bridge = bridge_ref
        self.ui = ui_ref
        self.should_stop = False  # Flag for graceful shutdown coordination
        
        # Register with bridge for task notifications
        if bridge_ref and hasattr(bridge_ref, 'set_agent_reference'):
            bridge_ref.set_agent_reference(self)

    @abstractmethod
    async def start(self):
        """
        Starts the voice agent's main loop and begins the session.
        This method should handle all setup, connection, and event processing.
        """
        pass

    @abstractmethod
    async def stop(self):
        """
        Gracefully stops the voice agent's session and cleans up resources.
        This should set should_stop=True and clean up any active sessions.
        """
        pass

    @abstractmethod
    async def handle_task_event(self, event_type: str, event_data: dict):
        """
        Handle task events from the bridge and generate appropriate responses.
        
        Args:
            event_type: Type of event ('task_received', 'task_starting', 'robot_action', etc.)
            event_data: Dictionary containing event-specific data
        """
        pass 