"""
UR Core Module
=============

Core functionality for UR robot bridge and control.
"""

from .connection import URConnection
# from .safety import SafetyValidator  # TODO: Create safety module
# from .bridge import URBridge  # TODO: Move bridge from parent directory

__all__ = ['URConnection'] 