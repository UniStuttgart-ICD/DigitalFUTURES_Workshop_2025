import time
import roslibpy
import os
import sys
#from abb.bridge import ABBGoFa
#from sensor.ForceSensor import LoacellSensor
from ur.launcher import URVoiceSystem
# from kuka.bridge import KUKAVarProxy

# KUKA_REMOTE_HOST = '192.168.2.3' 

def run_fabrication_system():
    """Run the fabrication system with clean restart capability."""
    # ros client
    client = roslibpy.Ros(host= "127.0.0.1", port=9090)
    
    try:
        client.run()
        print("✅ ROS Bridge connected successfully")
    except Exception as e:
        print(f"⚠️  ROS Bridge connection failed: {e}")
        print("ℹ️  Make sure rosbridge_server is running: roslaunch rosbridge_server rosbridge_websocket.launch")
        print("🔄 Continuing without ROS Bridge...")

    # sensors
    # sensor1 = LoacellSensor("ForceSensor1", client, serial_port='COM5', baudrate=57600)
    # sensor2 = LoacellSensor("ForceSensor2", client, serial_port='COM4', baudrate=57600)

    # robots
    # abb = ABBGoFa("GoFa1", client, auto_home = True)
    ur_system = URVoiceSystem(client)
    # kr210 = KUKAVarProxy("KR210", KUKA_REMOTE_HOST, auto_home = False)
    
    try:
        while True:
            time.sleep(1)
            # Check if system requested shutdown (e.g., after END_FABRICATION)
            if ur_system.shutdown_requested:
                print("\n🔄 Fabrication session ended - resetting to waiting state...")
                print("🧹 Performing session cleanup...")
                
                # Stop voice agent properly if it exists
                try:
                    if hasattr(ur_system, 'voice_agent') and ur_system.voice_agent:
                        ur_system.voice_agent.stop()
                        print("✅ Voice agent stopped")
                except Exception as e:
                    print(f"⚠️ Voice agent stop warning: {e}")
                
                # Reset the system state without disconnecting robot or ROS
                try:
                    ur_system.reset_for_restart()
                    print("✅ System state reset")
                except Exception as e:
                    print(f"⚠️ Reset warning: {e}")
                
                print("🛌 System ready for next START_FABRICATION command...")
                print("📡 Robot and ROS connections maintained")
                
                # Continue in the main loop - don't restart process
            # print ("listening for trajectories ...")
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        # Only cleanup objects that are actually defined
        # sensor1.cleanup()
        # sensor2.cleanup()
        # abb.cleanup()
        ur_system.cleanup()
        # kr210.cleanup()
        
        # Clean up global robot connection
        from ur.core.connection import cleanup_global_robot
        cleanup_global_robot()
        
        # Disconnect ROS client if connected
        if client.is_connected:
            client.terminate()
        print("✅ Cleanup complete")

if __name__ == '__main__':
    # Main loop with restart capability
    while True:
        try:
            run_fabrication_system()
            # If we reach here, it means normal exit (not restart)
            break
        except (SystemExit, KeyboardInterrupt):
            # Handle Ctrl+C or system exit
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("🔄 Restarting system...")
            continue