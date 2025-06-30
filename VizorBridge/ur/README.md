# UR Robot Bridge System

A comprehensive Python framework for controlling Universal Robots through voice commands and intelligent agents. This system provides multiple AI agent implementations for natural language robot control with safety validation and real-time interaction.

## 🚀 Features

- **Multiple Agent Types**: Choose from OpenAI Realtime API or HuggingFace SmolAgents
- **Voice Control**: Real-time voice commands using FastRTC
- **Safety First**: Wake-word validation and comprehensive safety checks
- **ROS Integration**: Seamless integration with ROS for task coordination
- **Code-First Agents**: SmolAgents use intelligent code generation for robot control
- **Flexible Models**: Support for OpenAI, HuggingFace, Anthropic, and local models

## 📋 Requirements

### System Requirements
- Python 3.8+
- ROS (Robot Operating System) with rosbridge
- Universal Robot (UR3, UR5, UR10) with RTDE interface
- Microphone and speakers for voice interaction

### Python Dependencies
```bash
# Core dependencies
uv add roslibpy ur-rtde rich dotenv

# For SmolAgents
uv add smolagents

# For voice interaction
uv add fastrtc

# For OpenAI agent
uv add openai websockets sounddevice numpy
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/VizorBridge.git
cd VizorBridge
```

2. **Install dependencies with uv**:
```bash
uv sync
```

3. **Set up environment variables**:
```bash
cp .env.sample .env
# Edit .env with your configuration
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Robot Configuration
ROBOT_IP=192.168.56.101
ROBOT_NAME=UR10

# Agent Configuration
AGENT_TYPE=smol                    # Options: openai, smol, smolagents
VOICE_ENABLED=true

# OpenAI Agent (if using AGENT_TYPE=openai)
OPENAI_API_KEY=your_openai_key_here

# SmolAgent Configuration (if using AGENT_TYPE=smol)
SMOL_MODEL_ID=gpt-4o-mini         # Model to use
SMOL_PROVIDER=openai              # Options: openai, huggingface, litellm
HF_TOKEN=your_hf_token_here       # For HuggingFace models

# Safety
WAKEWORD=mave                   # Required wake word for robot movements
DEBUG=false
```

## 🎯 Quick Start

### 1. Start ROS Bridge
```bash
# Terminal 1: Start roscore
roscore

# Terminal 2: Start rosbridge
roslaunch rosbridge_server rosbridge_websocket.launch
```

### 2. Launch Voice Agent

#### Using the Launcher (Recommended)
```bash
# SmolAgent with OpenAI
AGENT_TYPE=smol SMOL_PROVIDER=openai python ur/launcher.py

# SmolAgent with HuggingFace
AGENT_TYPE=smol SMOL_PROVIDER=huggingface python ur/launcher.py

# OpenAI Realtime Agent
AGENT_TYPE=openai python ur/launcher.py
```

#### Using Demo Scripts
```bash
# CLI mode (text-based)
python ur/demo_smol_agent.py --mode cli --model gpt-4o-mini --provider openai

# Voice mode
python ur/demo_smol_agent.py --mode voice --model gpt-4o-mini --provider openai

# With HuggingFace models
python ur/demo_smol_agent.py --mode cli --model Qwen/Qwen2.5-Coder-32B-Instruct --provider huggingface
```

## 🤖 Agent Types

### SmolAgent (Recommended)
Uses HuggingFace's SmolAgents framework with code-first approach.

**Advantages:**
- Intelligent code generation for robot control
- Support for multiple model providers
- Better reasoning for complex tasks
- Open-source and customizable

**Configuration:**
```bash
AGENT_TYPE=smol
SMOL_MODEL_ID=gpt-4o-mini
SMOL_PROVIDER=openai
```

**Supported Models:**
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **HuggingFace**: `Qwen/Qwen2.5-Coder-32B-Instruct`, `microsoft/DialoGPT-medium`
- **Anthropic** (via LiteLLM): `claude-3-sonnet`, `claude-3-haiku`

### OpenAI Agent
Uses OpenAI's Realtime API for direct voice interaction.

**Advantages:**
- Ultra-low latency voice interaction
- Native voice processing
- Optimized for conversational AI

**Configuration:**
```bash
AGENT_TYPE=openai
OPENAI_API_KEY=your_key_here
```

## 🎙️ Voice Commands

### Basic Commands
```
"mave move home"                    # Move robot to home position
"mave get robot state"              # Get current robot pose
"mave move up 5 centimeters"        # Relative movement
"mave open gripper"                 # Control gripper
"mave stop robot"                   # Emergency stop
```

### Advanced Commands
```
"mave move to position x 0.3 y 0.2 z 0.15"  # Absolute positioning
"mave get element from 20cm supply station"   # Pick up elements
"mave place element at x 0.4 y 0.1 z 0.2"    # Place elements
```

### Safety Notes
- **ALL robot movements require the wake word "mave"**
- Commands without the wake word will be rejected
- Emergency stop is available with "mave stop robot"

## 🛡️ Safety Features

### Wake Word Validation
All movement commands must include the wake word "mave" for safety.

### Safety Checks
- Joint limit validation using UR RTDE safety functions
- Pose safety validation before movements
- TCP orientation and robot range checking
- Emergency stop capability

### Error Handling
- Graceful failure handling for all robot operations
- Automatic connection recovery
- Comprehensive logging and status reporting

## 🔧 Robot Tools API

The system provides a comprehensive set of robot control functions:

### Movement Functions
- `move_home()` - Move to predefined home position
- `move_relative_xyz(dx, dy, dz)` - Relative Cartesian movement
- `move_to_absolute_position(x, y, z)` - Absolute positioning
- `move_to_supply_station(distance)` - Move to supply stations

### Gripper Control
- `control_gripper(action)` - Open/close gripper
- `get_supply_element(length)` - Pick up elements
- `place_element_at_position(x, y, z)` - Place elements
- `release_element()` - Release held elements

### State & Safety
- `get_robot_state()` - Current pose and joint positions
- `stop_robot()` - Emergency stop

## 📁 Project Structure

```
ur/
├── bridge.py              # Core UR robot bridge
├── launcher.py            # Agent launcher system
├── demo_smol_agent.py     # SmolAgent demo script
├── robot_tools.py         # Robot control functions
├── agents/
│   ├── base_agent.py      # Abstract base agent
│   ├── openai_agent.py    # OpenAI Realtime agent
│   └── smol_agent.py      # SmolAgents implementation
└── ui/
    └── console.py         # Console UI for status
```

## 🧪 Testing

### Test Robot Connection
```bash
python ur/test_bridge_standalone.py
```

### Test Individual Components
```bash
# Test robot tools
python -c "from ur.robot_tools import get_robot_state; print(get_robot_state())"

# Test SmolAgent setup
python ur/demo_smol_agent.py --mode cli --model gpt-4o-mini
```

## 🔧 Troubleshooting

### Common Issues

**Robot Connection Failed**
- Verify `ROBOT_IP` in `.env`
- Check network connectivity to robot
- Ensure RTDE interface is enabled on robot

**ROS Connection Failed**
- Start `roscore` and `rosbridge_server`
- Check ROS bridge is running on port 9090
- Verify `roslibpy` connection

**Voice Interface Not Working**
- Check microphone permissions
- Verify FastRTC dependencies are installed
- Ensure proper audio device configuration

**SmolAgent Model Issues**
- Verify API keys are set correctly
- Check model availability for selected provider
- Ensure sufficient API quota/credits

### Debug Mode
Enable detailed logging:
```bash
DEBUG=true python ur/demo_smol_agent.py --mode cli
```

## 🚀 Advanced Usage

### Custom Robot Tools
Add custom robot functions by decorating with `@tool`:

```python
from smolagents import tool

@tool
def custom_robot_action(param: str) -> Dict[str, Any]:
    """Custom robot action description."""
    # Your implementation
    return {"status": "success", "result": "action completed"}
```

### Multi-Agent Systems
Use the unified robot agent for complex workflows:

```python
from samples.voice_agents.demo_unified_robot_agent import UnifiedRobotAgent

agent = UnifiedRobotAgent()
await agent.start()
```

### Custom Model Providers
Add support for new model providers by extending the agent initialization:

```python
# In smol_agent.py
elif self.provider == "custom":
    model = CustomModel(model_id=self.model_id)
```

## 📖 Examples

### Example 1: Basic Robot Control
```python
import asyncio
from ur.demo_smol_agent import CLISmolAgentDemo

async def robot_demo():
    demo = CLISmolAgentDemo(model_id="gpt-4o-mini", provider="openai")
    await demo.setup()
    await demo.run_cli_demo()

asyncio.run(robot_demo())
```

### Example 2: Voice-Controlled Workflow
```bash
# Start voice demo
python ur/demo_smol_agent.py --mode voice

# Say: "mave move home"
# Say: "mave get element from 30cm supply station"
# Say: "mave place element at x 0.4 y 0.2 z 0.1"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check the wiki for detailed guides
- **Community**: Join our Discord for discussions

---

**Happy Robot Controlling! 🤖**
