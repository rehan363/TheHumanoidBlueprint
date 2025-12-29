---
sidebar_position: 11
title: "Week 11 - Voice-to-Action"
description: "Build natural language interfaces for robots using speech recognition, LLMs, and action execution"
keywords: [voice control, speech recognition, LLM, natural language, robotics, Whisper, GPT, Claude, ROS 2]
last_updated: "2025-12-29"
estimated_reading_time: 28
---

# Week 11: Voice-to-Action

Welcome to Module 4! We now explore how to give robots the ability to understand and respond to natural language commands. By combining speech recognition, Large Language Models (LLMs), and robot action systems, we create intuitive human-robot interaction that goes far beyond simple keyword matching.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the Voice-to-Action pipeline architecture
- Implement speech recognition using Whisper
- Use LLMs to interpret natural language commands
- Design action grammars for robot capabilities
- Build a ROS 2 voice control system
- Handle ambiguity and clarification in commands
- Implement safety constraints for voice-controlled robots

---

## The Voice-to-Action Pipeline

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Voice-to-Action Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                           │
│  │    User     │                                           │
│  │   Speech    │                                           │
│  └──────┬──────┘                                           │
│         │ Audio                                             │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Speech Recognition (ASR)                │   │
│  │              (Whisper, Google, Azure)                │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │ Text                          │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Natural Language Understanding             │   │
│  │              (LLM: GPT-4, Claude, Llama)            │   │
│  │                                                       │   │
│  │  "Go to the kitchen and pick up the red cup"        │   │
│  │                     ↓                                 │   │
│  │  Intent: navigate + manipulate                       │   │
│  │  Entities: location=kitchen, object=red_cup          │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │ Structured Command            │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Action Planning & Execution             │   │
│  │                                                       │   │
│  │  1. Navigate to kitchen                              │   │
│  │  2. Detect red cup                                   │   │
│  │  3. Plan grasp                                       │   │
│  │  4. Execute pick                                     │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│                    ┌─────────────┐                         │
│                    │    Robot    │                         │
│                    │   Actions   │                         │
│                    └─────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Voice Control for Robots?

| Traditional Interface | Voice Interface |
|----------------------|-----------------|
| Joystick/keyboard required | Hands-free operation |
| Technical knowledge needed | Natural conversation |
| Limited expressiveness | Rich, contextual commands |
| Pre-defined commands only | Flexible interpretation |
| Sequential inputs | Complex multi-step commands |

### Challenges in Voice-to-Action

```
┌─────────────────────────────────────────────────────────────┐
│              Voice Control Challenges                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. SPEECH RECOGNITION                                      │
│     • Background noise in real environments                 │
│     • Accents and speech variations                         │
│     • Domain-specific vocabulary                            │
│                                                             │
│  2. LANGUAGE UNDERSTANDING                                  │
│     • Ambiguous commands ("put it there")                   │
│     • Implicit context ("do it again")                      │
│     • Multi-step instructions                               │
│                                                             │
│  3. GROUNDING                                               │
│     • Mapping words to physical objects                     │
│     • Spatial references ("next to", "behind")             │
│     • Dynamic environment changes                           │
│                                                             │
│  4. SAFETY                                                  │
│     • Misheard commands with dangerous outcomes             │
│     • Unauthorized voice commands                           │
│     • Confirmation for critical actions                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Speech Recognition with Whisper

### What is Whisper?

**OpenAI Whisper** is a state-of-the-art automatic speech recognition (ASR) model:

- Trained on 680,000 hours of multilingual audio
- Robust to noise, accents, and technical language
- Multiple model sizes (tiny to large)
- Runs locally or via API

### Model Sizes

| Model | Parameters | VRAM | Relative Speed | Use Case |
|-------|------------|------|----------------|----------|
| tiny | 39M | ~1 GB | 32x | Edge devices |
| base | 74M | ~1 GB | 16x | Quick prototyping |
| small | 244M | ~2 GB | 6x | Good balance |
| medium | 769M | ~5 GB | 2x | High accuracy |
| large | 1550M | ~10 GB | 1x | Best quality |

### Installing Whisper

```bash
# Install OpenAI Whisper
pip install openai-whisper

# Or faster-whisper (optimized implementation)
pip install faster-whisper

# For GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Whisper Usage

```python
import whisper

# Load model
model = whisper.load_model("base")

# Transcribe audio file
result = model.transcribe("audio.wav")
print(result["text"])

# With language specification
result = model.transcribe("audio.wav", language="en")

# Get word-level timestamps
result = model.transcribe("audio.wav", word_timestamps=True)
for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

### Real-Time Speech Recognition

```python
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
import threading

class RealtimeSpeechRecognizer:
    def __init__(self, model_size="base", device="cuda"):
        self.model = WhisperModel(model_size, device=device)
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.chunk_duration = 2.0  # seconds
        self.running = False

    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """Process audio chunks from queue"""
        buffer = np.array([], dtype=np.float32)

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                buffer = np.concatenate([buffer, chunk.flatten()])

                # Process when buffer is long enough
                if len(buffer) >= self.sample_rate * self.chunk_duration:
                    # Transcribe
                    segments, info = self.model.transcribe(
                        buffer,
                        beam_size=5,
                        language="en",
                        vad_filter=True  # Voice activity detection
                    )

                    for segment in segments:
                        text = segment.text.strip()
                        if text:
                            self.on_transcription(text)

                    # Keep last 0.5s for context
                    buffer = buffer[-int(self.sample_rate * 0.5):]

            except queue.Empty:
                continue

    def on_transcription(self, text: str):
        """Override this method to handle transcriptions"""
        print(f"Transcribed: {text}")

    def start(self):
        """Start real-time recognition"""
        self.running = True

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
        )
        self.stream.start()

    def stop(self):
        """Stop recognition"""
        self.running = False
        self.stream.stop()
        self.stream.close()
        self.process_thread.join()


# Usage
recognizer = RealtimeSpeechRecognizer(model_size="small")
recognizer.start()

# Run for 30 seconds
import time
time.sleep(30)

recognizer.stop()
```

---

## Natural Language Understanding with LLMs

### The Role of LLMs

LLMs transform free-form natural language into structured robot commands:

```
┌─────────────────────────────────────────────────────────────┐
│                 LLM Command Interpretation                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "Could you please go to the living room and        │
│          fetch me the blue book from the coffee table?"    │
│                                                             │
│                         ↓ LLM                               │
│                                                             │
│  Output:                                                    │
│  {                                                          │
│    "intent": "fetch_object",                               │
│    "steps": [                                               │
│      {                                                      │
│        "action": "navigate",                               │
│        "destination": "living_room"                        │
│      },                                                     │
│      {                                                      │
│        "action": "locate",                                 │
│        "object": "book",                                   │
│        "attributes": {"color": "blue"},                    │
│        "location": "coffee_table"                          │
│      },                                                     │
│      {                                                      │
│        "action": "pick",                                   │
│        "object": "book"                                    │
│      },                                                     │
│      {                                                      │
│        "action": "navigate",                               │
│        "destination": "user_location"                      │
│      },                                                     │
│      {                                                      │
│        "action": "handover",                               │
│        "object": "book"                                    │
│      }                                                      │
│    ],                                                       │
│    "confirmation_required": false                          │
│  }                                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Choosing an LLM

| Model | Latency | Cost | Quality | Local Option |
|-------|---------|------|---------|--------------|
| GPT-4o | ~500ms | $$$ | Excellent | No |
| Claude 3.5 Sonnet | ~400ms | $$ | Excellent | No |
| GPT-4o-mini | ~200ms | $ | Very Good | No |
| Llama 3.1 70B | ~300ms | Free | Very Good | Yes |
| Llama 3.1 8B | ~100ms | Free | Good | Yes |
| Phi-3 | ~50ms | Free | Decent | Yes (edge) |

### LLM Integration

```python
from anthropic import Anthropic
import json

class RobotCommandInterpreter:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.robot_capabilities = self._load_capabilities()

    def _load_capabilities(self) -> dict:
        """Define robot's action vocabulary"""
        return {
            "actions": [
                {
                    "name": "navigate",
                    "description": "Move to a location",
                    "parameters": {
                        "destination": "string (room name or coordinates)"
                    }
                },
                {
                    "name": "pick",
                    "description": "Pick up an object",
                    "parameters": {
                        "object": "string (object name)",
                        "attributes": "dict (color, size, etc.)"
                    }
                },
                {
                    "name": "place",
                    "description": "Place held object",
                    "parameters": {
                        "location": "string (surface or container)"
                    }
                },
                {
                    "name": "speak",
                    "description": "Say something to user",
                    "parameters": {
                        "message": "string"
                    }
                },
                {
                    "name": "wait",
                    "description": "Wait for duration or event",
                    "parameters": {
                        "duration_seconds": "float",
                        "until_event": "string (optional)"
                    }
                }
            ],
            "known_locations": [
                "kitchen", "living_room", "bedroom", "bathroom",
                "garage", "entrance", "charging_station"
            ],
            "known_objects": [
                "cup", "bottle", "book", "remote", "phone",
                "keys", "bag", "box", "tool"
            ]
        }

    def interpret_command(self, user_input: str, context: dict = None) -> dict:
        """Convert natural language to robot commands"""

        system_prompt = f"""You are a robot command interpreter. Convert natural language commands into structured JSON actions.

Robot Capabilities:
{json.dumps(self.robot_capabilities, indent=2)}

Current Context:
- Robot location: {context.get('robot_location', 'unknown') if context else 'unknown'}
- Held object: {context.get('held_object', 'none') if context else 'none'}
- Known objects in view: {context.get('visible_objects', []) if context else []}

Rules:
1. Only use actions from the capabilities list
2. Break complex commands into sequential steps
3. If a command is ambiguous, set "needs_clarification": true and include "clarification_question"
4. If a command seems dangerous or impossible, set "safety_concern": true
5. Always respond with valid JSON

Output format:
{{
  "understood": true/false,
  "intent": "brief description",
  "steps": [
    {{"action": "action_name", "parameters": {{...}}}}
  ],
  "needs_clarification": true/false,
  "clarification_question": "question if needed",
  "safety_concern": true/false,
  "safety_reason": "reason if concerned"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_input}
            ]
        )

        # Parse JSON response
        try:
            result = json.loads(response.content[0].text)
            return result
        except json.JSONDecodeError:
            return {
                "understood": False,
                "error": "Failed to parse LLM response"
            }


# Usage example
interpreter = RobotCommandInterpreter(api_key="your-api-key")

context = {
    "robot_location": "kitchen",
    "held_object": None,
    "visible_objects": ["cup", "plate", "bottle"]
}

result = interpreter.interpret_command(
    "Bring me the cup from the counter",
    context=context
)
print(json.dumps(result, indent=2))
```

### Local LLM with Ollama

```python
import requests
import json

class LocalLLMInterpreter:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def interpret_command(self, user_input: str, system_prompt: str) -> dict:
        """Use local Ollama model for interpretation"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{system_prompt}\n\nUser command: {user_input}\n\nJSON response:",
                "stream": False,
                "format": "json"
            }
        )

        if response.status_code == 200:
            result = response.json()
            try:
                return json.loads(result["response"])
            except json.JSONDecodeError:
                return {"error": "Invalid JSON from model"}
        else:
            return {"error": f"Ollama error: {response.status_code}"}


# Install and run Ollama first:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull llama3.1:8b
# ollama serve
```

---

## Action Grammar Design

### Defining Robot Actions

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

class ActionType(Enum):
    NAVIGATE = "navigate"
    PICK = "pick"
    PLACE = "place"
    SPEAK = "speak"
    WAIT = "wait"
    LOOK = "look"
    OPEN = "open"
    CLOSE = "close"

@dataclass
class RobotAction:
    action_type: ActionType
    parameters: Dict[str, Any]
    preconditions: List[str]
    effects: List[str]
    estimated_duration: float  # seconds

class ActionGrammar:
    """Defines valid robot actions and their constraints"""

    def __init__(self):
        self.actions = self._define_actions()

    def _define_actions(self) -> Dict[ActionType, dict]:
        return {
            ActionType.NAVIGATE: {
                "parameters": {
                    "destination": {"type": "location", "required": True},
                    "speed": {"type": "float", "default": 0.5, "range": [0.1, 1.0]}
                },
                "preconditions": ["robot_not_holding_fragile"],
                "effects": ["robot_at(destination)"],
                "duration_estimate": lambda params: 10.0  # Simplified
            },
            ActionType.PICK: {
                "parameters": {
                    "object": {"type": "object", "required": True},
                    "grasp_type": {"type": "enum", "values": ["power", "precision"], "default": "power"}
                },
                "preconditions": ["gripper_empty", "object_visible", "object_reachable"],
                "effects": ["holding(object)", "not gripper_empty"],
                "duration_estimate": lambda params: 5.0
            },
            ActionType.PLACE: {
                "parameters": {
                    "location": {"type": "location", "required": True},
                    "careful": {"type": "bool", "default": False}
                },
                "preconditions": ["holding_object"],
                "effects": ["gripper_empty", "object_at(location)"],
                "duration_estimate": lambda params: 4.0
            },
            ActionType.SPEAK: {
                "parameters": {
                    "message": {"type": "string", "required": True},
                    "volume": {"type": "float", "default": 0.7, "range": [0.0, 1.0]}
                },
                "preconditions": [],
                "effects": [],
                "duration_estimate": lambda params: len(params.get("message", "")) * 0.1
            },
            ActionType.WAIT: {
                "parameters": {
                    "duration": {"type": "float", "required": False},
                    "until_event": {"type": "string", "required": False}
                },
                "preconditions": [],
                "effects": [],
                "duration_estimate": lambda params: params.get("duration", 5.0)
            }
        }

    def validate_action(self, action: RobotAction, robot_state: dict) -> tuple[bool, str]:
        """Check if action is valid given current robot state"""

        if action.action_type not in self.actions:
            return False, f"Unknown action type: {action.action_type}"

        action_def = self.actions[action.action_type]

        # Check required parameters
        for param_name, param_def in action_def["parameters"].items():
            if param_def.get("required", False) and param_name not in action.parameters:
                return False, f"Missing required parameter: {param_name}"

        # Check preconditions
        for precondition in action_def["preconditions"]:
            if not self._check_precondition(precondition, robot_state):
                return False, f"Precondition not met: {precondition}"

        return True, "Action valid"

    def _check_precondition(self, precondition: str, state: dict) -> bool:
        """Evaluate precondition against robot state"""
        precondition_checks = {
            "gripper_empty": lambda s: s.get("held_object") is None,
            "holding_object": lambda s: s.get("held_object") is not None,
            "object_visible": lambda s: True,  # Would check perception
            "object_reachable": lambda s: True,  # Would check kinematics
            "robot_not_holding_fragile": lambda s: not s.get("holding_fragile", False)
        }

        if precondition in precondition_checks:
            return precondition_checks[precondition](state)
        return True  # Unknown preconditions pass by default
```

---

## ROS 2 Voice Control Node

### Complete Voice Control Package

```python
#!/usr/bin/env python3
"""
ROS 2 Voice Control Node
Integrates speech recognition, LLM interpretation, and action execution
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import json
import threading
import queue
from anthropic import Anthropic


class VoiceControlNode(Node):
    def __init__(self):
        super().__init__('voice_control_node')

        # Parameters
        self.declare_parameter('whisper_model', 'small')
        self.declare_parameter('language', 'en')
        self.declare_parameter('llm_api_key', '')
        self.declare_parameter('confirmation_required', True)

        # Load parameters
        whisper_model = self.get_parameter('whisper_model').value
        self.language = self.get_parameter('language').value
        api_key = self.get_parameter('llm_api_key').value
        self.confirmation_required = self.get_parameter('confirmation_required').value

        # Initialize speech recognition
        self.get_logger().info(f'Loading Whisper model: {whisper_model}')
        self.whisper = WhisperModel(whisper_model, device="cuda")

        # Initialize LLM
        self.llm_client = Anthropic(api_key=api_key)

        # Audio settings
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.listening = False

        # Robot state
        self.robot_state = {
            "location": "unknown",
            "held_object": None,
            "battery_level": 100,
            "visible_objects": []
        }

        # Publishers
        self.speech_pub = self.create_publisher(String, '/voice_control/speech_text', 10)
        self.command_pub = self.create_publisher(String, '/voice_control/command', 10)
        self.status_pub = self.create_publisher(String, '/voice_control/status', 10)

        # Subscribers
        self.create_subscription(String, '/voice_control/robot_state', self.state_callback, 10)

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # TTS publisher (for robot speech)
        self.tts_pub = self.create_publisher(String, '/tts/say', 10)

        # Pending command for confirmation
        self.pending_command = None

        self.get_logger().info('Voice Control Node initialized')

    def state_callback(self, msg: String):
        """Update robot state from external source"""
        try:
            self.robot_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid robot state JSON')

    def audio_callback(self, indata, frames, time, status):
        """Handle incoming audio"""
        if status:
            self.get_logger().warn(f'Audio status: {status}')
        if self.listening:
            self.audio_queue.put(indata.copy())

    def start_listening(self):
        """Start audio capture"""
        self.listening = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.1)
        )
        self.stream.start()
        self.get_logger().info('Started listening for voice commands')

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio_loop)
        self.process_thread.start()

    def stop_listening(self):
        """Stop audio capture"""
        self.listening = False
        self.stream.stop()
        self.stream.close()

    def process_audio_loop(self):
        """Continuously process audio"""
        buffer = np.array([], dtype=np.float32)

        while self.listening:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                buffer = np.concatenate([buffer, chunk.flatten()])

                # Process every 2 seconds of audio
                if len(buffer) >= self.sample_rate * 2:
                    self.process_speech(buffer)
                    buffer = buffer[-int(self.sample_rate * 0.3):]  # Keep overlap

            except queue.Empty:
                continue

    def process_speech(self, audio: np.ndarray):
        """Transcribe and interpret speech"""
        # Transcribe with Whisper
        segments, info = self.whisper.transcribe(
            audio,
            beam_size=5,
            language=self.language,
            vad_filter=True
        )

        text = " ".join([seg.text for seg in segments]).strip()

        if not text:
            return

        # Publish transcription
        self.speech_pub.publish(String(data=text))
        self.get_logger().info(f'Heard: "{text}"')

        # Check for confirmation responses
        if self.pending_command:
            self.handle_confirmation(text)
            return

        # Interpret command
        self.interpret_and_execute(text)

    def interpret_and_execute(self, text: str):
        """Use LLM to interpret command and execute"""

        system_prompt = self._create_system_prompt()

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": text}]
        )

        try:
            command = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            self.speak("Sorry, I didn't understand that command.")
            return

        # Publish interpreted command
        self.command_pub.publish(String(data=json.dumps(command)))

        # Handle different outcomes
        if not command.get("understood", False):
            self.speak("I'm not sure what you want me to do. Could you rephrase?")
            return

        if command.get("needs_clarification", False):
            self.speak(command.get("clarification_question", "Could you clarify?"))
            return

        if command.get("safety_concern", False):
            self.speak(f"I can't do that safely. {command.get('safety_reason', '')}")
            return

        # Execute or request confirmation
        if self.confirmation_required and self._is_significant_action(command):
            self.pending_command = command
            self.speak(f"You want me to {command.get('intent', 'perform this action')}. Should I proceed?")
        else:
            self.execute_command(command)

    def handle_confirmation(self, text: str):
        """Handle yes/no confirmation"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["yes", "yeah", "correct", "proceed", "do it", "go ahead"]):
            self.execute_command(self.pending_command)
            self.pending_command = None
        elif any(word in text_lower for word in ["no", "cancel", "stop", "don't", "never mind"]):
            self.speak("Okay, I won't do that.")
            self.pending_command = None
        else:
            self.speak("Please say yes to confirm or no to cancel.")

    def execute_command(self, command: dict):
        """Execute the interpreted command"""
        steps = command.get("steps", [])

        for step in steps:
            action = step.get("action")
            params = step.get("parameters", {})

            self.get_logger().info(f'Executing: {action} with {params}')

            if action == "navigate":
                self.execute_navigate(params)
            elif action == "pick":
                self.execute_pick(params)
            elif action == "place":
                self.execute_place(params)
            elif action == "speak":
                self.speak(params.get("message", ""))
            elif action == "wait":
                self.execute_wait(params)
            else:
                self.get_logger().warn(f'Unknown action: {action}')

    def execute_navigate(self, params: dict):
        """Send navigation goal"""
        destination = params.get("destination", "")

        # Map location names to coordinates (simplified)
        locations = {
            "kitchen": (2.0, 3.0, 0.0),
            "living_room": (5.0, 2.0, 1.57),
            "bedroom": (1.0, 6.0, 3.14),
            "entrance": (0.0, 0.0, 0.0),
            "charging_station": (-1.0, 0.0, 0.0)
        }

        if destination not in locations:
            self.speak(f"I don't know where {destination} is.")
            return

        x, y, yaw = locations[destination]

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = np.cos(yaw / 2)
        goal.pose.pose.orientation.z = np.sin(yaw / 2)

        self.speak(f"Navigating to {destination}")
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal)

    def execute_pick(self, params: dict):
        """Execute pick action (simplified)"""
        obj = params.get("object", "object")
        self.speak(f"Picking up the {obj}")
        # Would integrate with manipulation stack
        self.robot_state["held_object"] = obj

    def execute_place(self, params: dict):
        """Execute place action (simplified)"""
        location = params.get("location", "here")
        self.speak(f"Placing object at {location}")
        self.robot_state["held_object"] = None

    def execute_wait(self, params: dict):
        """Execute wait action"""
        duration = params.get("duration", 5.0)
        self.speak(f"Waiting for {duration} seconds")
        # In real implementation, would use ROS timer

    def speak(self, message: str):
        """Make robot speak"""
        self.get_logger().info(f'Speaking: "{message}"')
        self.tts_pub.publish(String(data=message))
        self.status_pub.publish(String(data=f"SPEAKING: {message}"))

    def _create_system_prompt(self) -> str:
        """Create LLM system prompt with current context"""
        return f"""You are a robot command interpreter. Convert voice commands to JSON actions.

Available actions:
- navigate: Move to location (kitchen, living_room, bedroom, entrance, charging_station)
- pick: Pick up an object
- place: Put down held object
- speak: Say something
- wait: Wait for duration

Current state:
- Location: {self.robot_state.get('location', 'unknown')}
- Holding: {self.robot_state.get('held_object', 'nothing')}
- Battery: {self.robot_state.get('battery_level', 100)}%

Respond with JSON:
{{
  "understood": true/false,
  "intent": "brief description",
  "steps": [{{"action": "name", "parameters": {{}}}}],
  "needs_clarification": false,
  "safety_concern": false
}}"""

    def _is_significant_action(self, command: dict) -> bool:
        """Check if action requires confirmation"""
        significant_actions = ["navigate", "pick", "place"]
        steps = command.get("steps", [])
        return any(step.get("action") in significant_actions for step in steps)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceControlNode()

    try:
        node.start_listening()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_listening()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Launch File

```python
# File: voice_control_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('whisper_model', default_value='small'),
        DeclareLaunchArgument('language', default_value='en'),
        DeclareLaunchArgument('llm_api_key', default_value=''),
        DeclareLaunchArgument('confirmation_required', default_value='true'),

        Node(
            package='voice_control',
            executable='voice_control_node',
            name='voice_control',
            parameters=[{
                'whisper_model': LaunchConfiguration('whisper_model'),
                'language': LaunchConfiguration('language'),
                'llm_api_key': LaunchConfiguration('llm_api_key'),
                'confirmation_required': LaunchConfiguration('confirmation_required'),
            }],
            output='screen'
        ),
    ])
```

---

## Handling Ambiguity and Context

### Contextual Understanding

```python
class ConversationContext:
    """Maintains conversation context for multi-turn interactions"""

    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
        self.entities = {}  # Track mentioned entities
        self.last_action = None
        self.last_object = None
        self.last_location = None

    def add_turn(self, user_input: str, robot_response: dict):
        """Add a conversation turn"""
        self.history.append({
            "user": user_input,
            "robot": robot_response,
            "timestamp": time.time()
        })

        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Extract entities
        if robot_response.get("understood"):
            for step in robot_response.get("steps", []):
                action = step.get("action")
                params = step.get("parameters", {})

                self.last_action = action

                if "object" in params:
                    self.last_object = params["object"]
                    self.entities[params["object"]] = time.time()

                if "destination" in params or "location" in params:
                    loc = params.get("destination") or params.get("location")
                    self.last_location = loc
                    self.entities[loc] = time.time()

    def resolve_reference(self, reference: str) -> str:
        """Resolve pronouns and references"""
        reference_lower = reference.lower()

        # "it", "that", "the object"
        if reference_lower in ["it", "that", "this", "the object", "that thing"]:
            return self.last_object or reference

        # "there", "that place"
        if reference_lower in ["there", "that place", "the same place"]:
            return self.last_location or reference

        # "again", "the same"
        if reference_lower in ["again", "the same", "repeat"]:
            return self.last_action or reference

        return reference

    def get_context_prompt(self) -> str:
        """Generate context for LLM"""
        recent = self.history[-3:] if self.history else []

        context_lines = ["Recent conversation:"]
        for turn in recent:
            context_lines.append(f"User: {turn['user']}")
            if turn['robot'].get('understood'):
                context_lines.append(f"Robot action: {turn['robot'].get('intent', 'unknown')}")

        if self.last_object:
            context_lines.append(f"Last mentioned object: {self.last_object}")
        if self.last_location:
            context_lines.append(f"Last mentioned location: {self.last_location}")

        return "\n".join(context_lines)
```

### Clarification Dialogues

```python
class ClarificationManager:
    """Handles ambiguous commands through dialogue"""

    def __init__(self, speak_callback):
        self.speak = speak_callback
        self.pending_clarification = None

    def request_clarification(self, ambiguity_type: str, options: list = None) -> dict:
        """Generate clarification request"""

        clarifications = {
            "object_ambiguous": {
                "question": f"Which one do you mean? I see: {', '.join(options or [])}",
                "type": "select",
                "options": options
            },
            "location_ambiguous": {
                "question": "Where exactly should I go?",
                "type": "location",
                "options": options
            },
            "action_ambiguous": {
                "question": "What would you like me to do with it?",
                "type": "action",
                "options": ["pick it up", "move it", "describe it"]
            },
            "missing_object": {
                "question": "What object are you referring to?",
                "type": "object",
                "options": None
            },
            "missing_location": {
                "question": "Where should I put it?",
                "type": "location",
                "options": None
            }
        }

        clarification = clarifications.get(ambiguity_type, {
            "question": "Could you please clarify?",
            "type": "open",
            "options": None
        })

        self.pending_clarification = {
            "type": ambiguity_type,
            "original_type": clarification["type"],
            "options": clarification["options"]
        }

        self.speak(clarification["question"])
        return clarification

    def handle_clarification_response(self, response: str) -> dict:
        """Process clarification response"""
        if not self.pending_clarification:
            return {"resolved": False}

        clarification_type = self.pending_clarification["type"]
        options = self.pending_clarification["options"]

        # Try to match response to options
        if options:
            response_lower = response.lower()
            for option in options:
                if option.lower() in response_lower:
                    self.pending_clarification = None
                    return {
                        "resolved": True,
                        "value": option,
                        "type": clarification_type
                    }

        # Accept free-form response
        self.pending_clarification = None
        return {
            "resolved": True,
            "value": response,
            "type": clarification_type
        }
```

---

## Safety Considerations

### Safety Filter

```python
class SafetyFilter:
    """Filter dangerous or unauthorized commands"""

    def __init__(self):
        self.dangerous_patterns = [
            "throw", "drop", "destroy", "break",
            "hit", "attack", "hurt", "harm",
            "maximum speed", "ignore obstacle",
            "override safety", "disable"
        ]

        self.restricted_areas = [
            "stairs", "balcony", "road", "outside boundary"
        ]

        self.authorization_required = [
            "unlock", "open door", "access", "admin"
        ]

    def check_command(self, command: dict) -> tuple[bool, str]:
        """Check if command is safe to execute"""

        intent = command.get("intent", "").lower()
        steps = command.get("steps", [])

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern in intent:
                return False, f"Command contains dangerous action: {pattern}"

        # Check each step
        for step in steps:
            action = step.get("action", "")
            params = step.get("parameters", {})

            # Check navigation to restricted areas
            if action == "navigate":
                dest = params.get("destination", "").lower()
                for restricted in self.restricted_areas:
                    if restricted in dest:
                        return False, f"Cannot navigate to restricted area: {restricted}"

            # Check for high-speed commands
            if params.get("speed", 0) > 0.8:
                return False, "Speed too high for safety"

            # Check for authorization-required actions
            for auth_pattern in self.authorization_required:
                if auth_pattern in action.lower():
                    return False, f"Action requires authorization: {action}"

        return True, "Command passed safety check"

    def sanitize_command(self, command: dict) -> dict:
        """Modify command to be safer"""

        for step in command.get("steps", []):
            params = step.get("parameters", {})

            # Limit speed
            if "speed" in params:
                params["speed"] = min(params["speed"], 0.5)

            # Add careful flag for manipulation
            if step.get("action") in ["pick", "place"]:
                params["careful"] = True

        return command
```

### Voice Authentication

```python
import numpy as np
from scipy.spatial.distance import cosine

class VoiceAuthenticator:
    """Simple voice authentication using speaker embeddings"""

    def __init__(self):
        self.authorized_embeddings = {}
        self.threshold = 0.3  # Cosine distance threshold

    def enroll_user(self, user_id: str, audio_samples: list):
        """Enroll a new authorized user"""
        # In production, use a proper speaker embedding model
        embeddings = [self._extract_embedding(audio) for audio in audio_samples]
        self.authorized_embeddings[user_id] = np.mean(embeddings, axis=0)

    def verify_speaker(self, audio: np.ndarray) -> tuple[bool, str]:
        """Verify if speaker is authorized"""
        if not self.authorized_embeddings:
            return True, "no_auth_configured"

        embedding = self._extract_embedding(audio)

        best_match = None
        best_distance = float('inf')

        for user_id, enrolled_embedding in self.authorized_embeddings.items():
            distance = cosine(embedding, enrolled_embedding)
            if distance < best_distance:
                best_distance = distance
                best_match = user_id

        if best_distance < self.threshold:
            return True, best_match
        else:
            return False, "unauthorized"

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding (simplified)"""
        # In production, use models like:
        # - SpeechBrain's ECAPA-TDNN
        # - Resemblyzer
        # - pyannote.audio

        # Simplified: use MFCC-based features
        from scipy.fftpack import dct

        # Frame the audio
        frame_length = 400
        hop_length = 160
        n_mfcc = 13

        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frames.append(frame)

        # Very simplified embedding
        if frames:
            features = np.array([np.mean(f) for f in frames[:100]])
            return np.pad(features, (0, max(0, 100 - len(features))))
        return np.zeros(100)
```

---

## Text-to-Speech Integration

### TTS Options

```python
import pyttsx3
from gtts import gTTS
import os
import tempfile
import pygame

class TextToSpeech:
    """Multi-backend TTS system"""

    def __init__(self, backend: str = "pyttsx3"):
        self.backend = backend

        if backend == "pyttsx3":
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)

        elif backend == "gtts":
            pygame.mixer.init()

    def speak(self, text: str, blocking: bool = True):
        """Speak the given text"""

        if self.backend == "pyttsx3":
            if blocking:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                self.engine.say(text)
                self.engine.startLoop(False)

        elif self.backend == "gtts":
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                tts.save(f.name)
                pygame.mixer.music.load(f.name)
                pygame.mixer.music.play()
                if blocking:
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                os.unlink(f.name)

    def set_voice(self, voice_id: str = None, gender: str = None):
        """Set voice properties"""
        if self.backend == "pyttsx3":
            voices = self.engine.getProperty('voices')
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            elif gender:
                for voice in voices:
                    if gender.lower() in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
```

---

## Practical Exercise: Complete Voice-Controlled Robot

### Goal

Build a voice-controlled robot that can:
1. Understand natural language navigation commands
2. Describe what it sees
3. Pick up and deliver objects
4. Maintain conversation context

### Project Structure

```
voice_robot_ws/
├── src/
│   └── voice_control/
│       ├── voice_control/
│       │   ├── __init__.py
│       │   ├── voice_control_node.py
│       │   ├── speech_recognition.py
│       │   ├── llm_interpreter.py
│       │   ├── action_executor.py
│       │   ├── safety_filter.py
│       │   └── tts.py
│       ├── config/
│       │   └── voice_control_params.yaml
│       ├── launch/
│       │   └── voice_control.launch.py
│       ├── package.xml
│       └── setup.py
```

### Testing Commands

```bash
# Terminal 1: Launch robot simulation
ros2 launch my_robot_sim simulation.launch.py

# Terminal 2: Launch Nav2
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

# Terminal 3: Launch voice control
ros2 launch voice_control voice_control.launch.py \
  llm_api_key:=$ANTHROPIC_API_KEY

# Test commands:
# "Go to the kitchen"
# "What do you see?"
# "Pick up the red cup"
# "Bring it to me"
# "Stop"
```

---

## Summary

In this chapter, you learned:

- **Voice-to-Action Pipeline**: Speech → Text → Understanding → Action
- **Speech Recognition**: Whisper for robust, real-time transcription
- **LLM Interpretation**: Using Claude/GPT to parse natural language into structured commands
- **Action Grammar**: Defining robot capabilities and constraints
- **ROS 2 Integration**: Building a complete voice control node
- **Context Handling**: Multi-turn conversations and reference resolution
- **Safety**: Filtering dangerous commands and voice authentication
- **TTS**: Making robots speak back to users

Voice control transforms how humans interact with robots, making them accessible to everyone regardless of technical expertise.

---

## Further Reading

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper
- [Anthropic Claude API](https://docs.anthropic.com/) - LLM API documentation
- [SpeechBrain](https://speechbrain.github.io/) - Speech processing toolkit
- [ROS 2 Audio Common](https://github.com/ros-drivers/audio_common) - ROS audio packages

---

## Next Week Preview

In **Week 12**, we explore **Cognitive Planning with VLMs**:
- Vision-Language Models for scene understanding
- Grounding language in visual perception
- Task planning from natural language goals
- Multi-modal reasoning for robotics
