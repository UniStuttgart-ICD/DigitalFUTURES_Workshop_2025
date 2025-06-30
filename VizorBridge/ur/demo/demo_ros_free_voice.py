#!/usr/bin/env python3
"""
ROS-Free Robot Voice Agent Demo
==============================

A minimal demo using the core UR library with OpenAI Realtime API voice interaction.
No ROS dependencies - just direct RTDE robot connection and voice control.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ur.config.voice_config import load_config_from_env
from ur.agents.openai_agent import create_robot_agent, RealtimeSession
from ur.core.connection import get_robot, cleanup_global_robot
from ur.config.robot_config import ROBOT_IP


async def main():
    # Load config from environment (.env if present)
    config = load_config_from_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is required")
        return

    # Connect to robot
    robot = get_robot()
    if robot.is_connected:
        print(f"✅ Connected to robot at {ROBOT_IP}")
    else:
        print("⚠️ Robot not connected - running in simulation mode")

    # Initialize agent and realtime session
    agent = create_robot_agent(config)
    session = RealtimeSession(agent, api_key, bridge_ref=robot, config=config)
    try:
        # Connect and start the realtime session (handles audio and text I/O)
        await session.connect()
        await session.start_session_with_connection()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        raise
    finally:
        # Ensure robot connection is cleaned up
        cleanup_global_robot()


if __name__ == "__main__":
    print("🤖 ROS-Free Robot Voice Agent Demo")
    print("==================================")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    finally:
        cleanup_global_robot() 