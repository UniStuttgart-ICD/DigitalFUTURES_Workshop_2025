# VizorBridge

This project provides simple bridges for connecting robots and sensors to ROS via `roslibpy`. The new **LLM agent** mode allows tasks to be sent over WebSockets and executed on a UR robot.

## Usage

Create a virtual environment and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install websockets asyncio ur-rtde roslibpy
```

To run the bridge without ROS force-torque publishing:

```bash
python main.py --robot-ip 192.168.1.50 --ws-host 0.0.0.0 --ws-port 8765
```

Enable force-torque forwarding into ROS with `--use-ros`. If you have a local
Ollama installation you can point the bridge to a model with `--ollama-model`:

```bash
source /opt/ros/noetic/setup.bash
python main.py --robot-ip 192.168.1.50 --ws-host 0.0.0.0 --ws-port 8765 --use-ros --ollama-model llama2
```

A JSON task sent over the WebSocket should look like:

```json
{
  "description": "Move to pick pose",
  "targetTcp": {"x": 0.1, "y": 0.2, "z": 0.3, "rx": 0, "ry": 0, "rz": 0},
  "trajectory": [[0,0,0,0,0,0]]
}
```

The agent validates the task and dispatches either `moveL` or `moveJ` commands to the robot interface. If the `openai` package is installed or a local Ollama model is specified, a short description of the task will be generated.
