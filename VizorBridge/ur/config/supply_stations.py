"""
Supply Station Configuration
===========================

Supply station TCP poses and configurations.
"""

# Supply station TCP poses in ur_rtde format [x, y, z, rx, ry, rz]
# Positions in meters, rotations in radians (pointing down: ry=3.14)
SUPPLY_STATIONS = {
    "10cm": {"position": [0.3, -0.2, 0.15, 0, 3.14, 0], "approach_height": 0.05},
    "20cm": {"position": [0.3, -0.1, 0.15, 0, 3.14, 0], "approach_height": 0.05},
    "30cm": {"position": [0.3, 0.0, 0.15, 0, 3.14, 0], "approach_height": 0.05},
    "40cm": {"position": [0.3, 0.1, 0.15, 0, 3.14, 0], "approach_height": 0.05},
    "50cm": {"position": [0.3, 0.2, 0.15, 0, 3.14, 0], "approach_height": 0.05},
    "60cm": {"position": [0.3, 0.3, 0.15, 0, 3.14, 0], "approach_height": 0.05},
} 