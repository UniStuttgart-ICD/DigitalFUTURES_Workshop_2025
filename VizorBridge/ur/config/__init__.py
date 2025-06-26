"""
UR Configuration Module
======================

Configuration constants and settings for UR robot control.
"""

from .robot_config import *
from .supply_stations import SUPPLY_STATIONS
from .topics import *
from .voice_config import *

__all__ = [
    'ROBOT_IP',
    'ROBOT_MOVE_SPEED', 
    'ROBOT_ACCELERATION',
    'RTDE_CONTROL_PORT',
    'TEST_DATA_MESSAGES_DIR',
    'WAKEWORD',
    'HOME_POSITION',
    'SUPPLY_STATIONS',
] 