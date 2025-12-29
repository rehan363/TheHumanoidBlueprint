---
sidebar_position: 12
title: "Week 12 - Cognitive Planning with VLMs"
description: "Build intelligent robot planners using Vision-Language Models for scene understanding, task decomposition, and multi-modal reasoning"
keywords: [VLM, cognitive planning, GPT-4V, Claude Vision, task planning, scene understanding, robotics, ROS 2, multi-modal AI]
last_updated: "2025-12-30"
estimated_reading_time: 32
---

# Week 12: Cognitive Planning with VLMs

Welcome to the culmination of Vision-Language-Action integration! This chapter explores how Vision-Language Models (VLMs) enable robots to understand scenes, reason about tasks, and generate executable plans from natural language goals. We move beyond simple voice commands to sophisticated cognitive architectures that see, think, and act.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain how VLMs combine vision and language understanding
- Use GPT-4V/Claude Vision for robotic scene analysis
- Implement grounded task planning from natural language
- Design hierarchical task decomposition systems
- Build a cognitive planning pipeline in ROS 2
- Handle planning failures and replanning strategies
- Integrate perception feedback into plan execution

---

## Vision-Language Models for Robotics

### What are VLMs?

Vision-Language Models process both images and text, enabling rich multi-modal understanding:

```
┌─────────────────────────────────────────────────────────────────┐
│                  Vision-Language Model Architecture              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐        ┌──────────────┐                     │
│   │    Image     │        │    Text      │                     │
│   │   Encoder    │        │   Encoder    │                     │
│   │  (ViT/CLIP)  │        │ (Transformer)│                     │
│   └──────┬───────┘        └──────┬───────┘                     │
│          │                       │                              │
│          │   Visual Tokens       │   Text Tokens               │
│          │                       │                              │
│          └───────────┬───────────┘                              │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────┐                              │
│          │   Cross-Modal Fusion  │                              │
│          │     (Attention)       │                              │
│          └───────────┬───────────┘                              │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────┐                              │
│          │   Language Model      │                              │
│          │   (Decoder/LLM)       │                              │
│          └───────────┬───────────┘                              │
│                      │                                          │
│                      ▼                                          │
│              Generated Response                                  │
│     (Scene descriptions, plans, answers)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### VLM Capabilities for Robotics

| Capability | Description | Robot Application |
|------------|-------------|-------------------|
| Scene Description | Describe what's in an image | Environment awareness |
| Object Detection | Identify and locate objects | Manipulation targets |
| Spatial Reasoning | Understand object relationships | Navigation planning |
| Action Recognition | Identify ongoing activities | Human-robot collaboration |
| Anomaly Detection | Spot unusual situations | Safety monitoring |
| Task Grounding | Map language to visual elements | Command understanding |

### Available VLMs

| Model | Provider | Strengths | Latency | Local Option |
|-------|----------|-----------|---------|--------------|
| GPT-4V | OpenAI | Best overall understanding | ~2s | No |
| Claude 3.5 Sonnet | Anthropic | Strong reasoning, safety | ~1.5s | No |
| Gemini Pro Vision | Google | Fast, good spatial | ~1s | No |
| LLaVA 1.6 | Open Source | Good quality, customizable | ~500ms | Yes |
| CogVLM | Open Source | Strong visual grounding | ~600ms | Yes |
| Qwen-VL | Alibaba | Multilingual support | ~400ms | Yes |

---

## Scene Understanding with VLMs

### Basic Scene Analysis

```python
import anthropic
import base64
from pathlib import Path

class RobotSceneAnalyzer:
    """Analyze robot camera images using VLMs"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def analyze_scene(self, image_path: str, query: str = None) -> dict:
        """Analyze a scene from robot's camera"""

        image_data = self.encode_image(image_path)

        system_prompt = """You are a robot vision system analyzing scenes for task planning.

When analyzing images, provide:
1. Objects detected with approximate locations (left/center/right, near/far)
2. Spatial relationships between objects
3. Potential manipulation targets
4. Navigation obstacles or paths
5. Any safety concerns

Be precise and structured in your analysis."""

        user_prompt = query or "Analyze this scene for robot task planning. What objects do you see and how are they arranged?"

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )

        return {
            "analysis": response.content[0].text,
            "model": "claude-sonnet-4-20250514",
            "query": user_prompt
        }

    def identify_objects(self, image_path: str) -> list:
        """Extract structured object list from scene"""

        image_data = self.encode_image(image_path)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": """List all objects visible in this image as JSON array.
For each object include:
- name: object type
- color: primary color
- position: {horizontal: left/center/right, depth: near/mid/far}
- graspable: true/false
- size: small/medium/large

Respond ONLY with valid JSON array."""
                        }
                    ]
                }
            ]
        )

        import json
        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return []

    def check_task_feasibility(self, image_path: str, task: str) -> dict:
        """Check if a task is feasible given current scene"""

        image_data = self.encode_image(image_path)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Task: {task}

Analyze if this task is feasible given the current scene. Return JSON:
{{
  "feasible": true/false,
  "confidence": 0.0-1.0,
  "reasons": ["reason1", "reason2"],
  "prerequisites": ["what needs to happen first"],
  "obstacles": ["potential problems"],
  "alternative_approaches": ["if not directly feasible"]
}}"""
                        }
                    ]
                }
            ]
        )

        import json
        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {"feasible": False, "error": "Failed to parse response"}


# Usage
analyzer = RobotSceneAnalyzer(api_key="your-api-key")

# Basic scene analysis
result = analyzer.analyze_scene("camera_frame.jpg")
print(result["analysis"])

# Object detection
objects = analyzer.identify_objects("camera_frame.jpg")
for obj in objects:
    print(f"Found {obj['color']} {obj['name']} at {obj['position']}")

# Task feasibility
feasibility = analyzer.check_task_feasibility(
    "camera_frame.jpg",
    "Pick up the red cup and place it on the shelf"
)
print(f"Feasible: {feasibility['feasible']}")
```

### Spatial Reasoning

```python
class SpatialReasoner:
    """Reason about spatial relationships for robot planning"""

    def __init__(self, vlm_client):
        self.client = vlm_client

    def get_spatial_relations(self, image_path: str, objects: list) -> dict:
        """Extract spatial relationships between objects"""

        image_data = self._encode_image(image_path)

        object_list = ", ".join(objects)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Analyze spatial relationships between these objects: {object_list}

Return JSON with:
{{
  "relations": [
    {{"subject": "obj1", "relation": "on/in/next_to/behind/in_front_of", "object": "obj2"}},
    ...
  ],
  "distances": [
    {{"from": "obj1", "to": "obj2", "distance": "touching/close/medium/far"}}
  ],
  "accessibility": [
    {{"object": "name", "accessible": true/false, "blocked_by": ["obj"] or null}}
  ]
}}"""
                        }
                    ]
                }
            ]
        )

        import json
        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse spatial relations"}

    def find_path_to_object(self, image_path: str, target_object: str) -> dict:
        """Determine navigation approach to reach an object"""

        image_data = self._encode_image(image_path)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""I need to reach the {target_object} in this scene.

Analyze and return JSON:
{{
  "target_visible": true/false,
  "target_location": {{"horizontal": "left/center/right", "depth": "near/mid/far"}},
  "obstacles": ["list of obstacles between robot and target"],
  "approach_direction": "front/left/right/back",
  "navigation_steps": [
    {{"action": "move/turn/avoid", "description": "specific instruction"}}
  ],
  "gripper_approach": "top/side/front"
}}"""
                        }
                    ]
                }
            ]
        )

        import json
        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {"error": "Failed to determine path"}

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
```

---

## Task Planning with VLMs

### The Planning Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                 VLM-Based Task Planning Pipeline                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│  │   Natural   │      │   Camera    │      │   Robot     │    │
│  │  Language   │      │   Image     │      │   State     │    │
│  │    Goal     │      │             │      │             │    │
│  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    │
│         │                    │                    │            │
│         └────────────────────┼────────────────────┘            │
│                              │                                  │
│                              ▼                                  │
│                   ┌─────────────────────┐                      │
│                   │   Vision-Language   │                      │
│                   │       Model         │                      │
│                   └──────────┬──────────┘                      │
│                              │                                  │
│                              ▼                                  │
│                   ┌─────────────────────┐                      │
│                   │  Scene Understanding│                      │
│                   │  + Goal Grounding   │                      │
│                   └──────────┬──────────┘                      │
│                              │                                  │
│                              ▼                                  │
│                   ┌─────────────────────┐                      │
│                   │  Task Decomposition │                      │
│                   │  (High-level Plan)  │                      │
│                   └──────────┬──────────┘                      │
│                              │                                  │
│                              ▼                                  │
│                   ┌─────────────────────┐                      │
│                   │   Action Grounding  │                      │
│                   │ (Executable Steps)  │                      │
│                   └──────────┬──────────┘                      │
│                              │                                  │
│                              ▼                                  │
│                   ┌─────────────────────┐                      │
│                   │  Feasibility Check  │                      │
│                   │   + Safety Filter   │                      │
│                   └──────────┬──────────┘                      │
│                              │                                  │
│                              ▼                                  │
│                      Executable Plan                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hierarchical Task Decomposition

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import json

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class PrimitiveAction:
    """Lowest-level executable action"""
    action_type: str  # navigate, pick, place, look, speak
    parameters: Dict[str, Any]
    preconditions: List[str]
    effects: List[str]
    estimated_duration: float

@dataclass
class SubTask:
    """Mid-level task composed of primitive actions"""
    name: str
    description: str
    actions: List[PrimitiveAction]
    status: TaskStatus = TaskStatus.PENDING

@dataclass
class TaskPlan:
    """High-level plan with subtasks"""
    goal: str
    subtasks: List[SubTask]
    current_subtask_index: int = 0
    status: TaskStatus = TaskStatus.PENDING


class HierarchicalTaskPlanner:
    """Generate hierarchical task plans using VLMs"""

    def __init__(self, vlm_client):
        self.client = vlm_client
        self.action_primitives = self._define_primitives()

    def _define_primitives(self) -> dict:
        """Define available primitive actions"""
        return {
            "navigate": {
                "parameters": ["destination", "speed"],
                "preconditions": ["robot_mobile"],
                "effects": ["robot_at(destination)"]
            },
            "pick": {
                "parameters": ["object", "grasp_type"],
                "preconditions": ["gripper_empty", "object_reachable", "object_graspable"],
                "effects": ["holding(object)", "not(gripper_empty)"]
            },
            "place": {
                "parameters": ["location", "careful"],
                "preconditions": ["holding_object"],
                "effects": ["gripper_empty", "object_at(location)"]
            },
            "look_at": {
                "parameters": ["target"],
                "preconditions": [],
                "effects": ["observing(target)"]
            },
            "open_gripper": {
                "parameters": [],
                "preconditions": [],
                "effects": ["gripper_open"]
            },
            "close_gripper": {
                "parameters": ["force"],
                "preconditions": ["gripper_open"],
                "effects": ["gripper_closed"]
            },
            "move_arm": {
                "parameters": ["pose", "speed"],
                "preconditions": [],
                "effects": ["arm_at(pose)"]
            },
            "wait": {
                "parameters": ["duration"],
                "preconditions": [],
                "effects": []
            },
            "speak": {
                "parameters": ["message"],
                "preconditions": [],
                "effects": []
            }
        }

    def plan_task(self, goal: str, image_path: str, robot_state: dict) -> TaskPlan:
        """Generate a complete hierarchical plan"""

        # Step 1: Understand scene and goal
        scene_analysis = self._analyze_scene_for_goal(image_path, goal)

        # Step 2: High-level decomposition
        subtasks = self._decompose_goal(goal, scene_analysis, robot_state)

        # Step 3: Ground each subtask to primitives
        grounded_subtasks = []
        for subtask in subtasks:
            actions = self._ground_subtask(subtask, scene_analysis)
            grounded_subtasks.append(SubTask(
                name=subtask["name"],
                description=subtask["description"],
                actions=actions
            ))

        return TaskPlan(goal=goal, subtasks=grounded_subtasks)

    def _analyze_scene_for_goal(self, image_path: str, goal: str) -> dict:
        """Analyze scene with goal context"""

        image_data = self._encode_image(image_path)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Goal: {goal}

Analyze this scene to support planning for the given goal. Return JSON:
{{
  "relevant_objects": [
    {{"name": "obj", "location": "description", "state": "open/closed/etc", "relevant_to_goal": true/false}}
  ],
  "goal_object": {{"name": "target object", "visible": true/false, "location": "where"}},
  "obstacles": ["list of obstacles"],
  "workspace_clear": true/false,
  "suggested_approach": "description of how to achieve goal",
  "potential_issues": ["things that might go wrong"]
}}"""
                        }
                    ]
                }
            ]
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {"error": "Failed to analyze scene"}

    def _decompose_goal(self, goal: str, scene: dict, robot_state: dict) -> list:
        """Decompose high-level goal into subtasks"""

        primitives_desc = json.dumps(list(self.action_primitives.keys()))

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"""You are a robot task planner. Decompose goals into subtasks.

Available primitive actions: {primitives_desc}

Current robot state:
- Location: {robot_state.get('location', 'unknown')}
- Gripper: {robot_state.get('gripper_state', 'unknown')}
- Holding: {robot_state.get('held_object', 'nothing')}

Scene analysis: {json.dumps(scene)}""",
            messages=[
                {
                    "role": "user",
                    "content": f"""Decompose this goal into subtasks: "{goal}"

Return JSON array of subtasks:
[
  {{
    "name": "subtask_name",
    "description": "what this achieves",
    "required_primitives": ["action1", "action2"],
    "dependencies": ["names of subtasks that must complete first"],
    "success_criteria": "how to verify completion"
  }}
]

Order subtasks by execution sequence. Be thorough but avoid unnecessary steps."""
                }
            ]
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return []

    def _ground_subtask(self, subtask: dict, scene: dict) -> List[PrimitiveAction]:
        """Convert subtask to executable primitive actions"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"""Convert subtasks to primitive robot actions.

Available primitives and parameters:
{json.dumps(self.action_primitives, indent=2)}

Scene context: {json.dumps(scene)}""",
            messages=[
                {
                    "role": "user",
                    "content": f"""Subtask: {subtask['name']}
Description: {subtask['description']}

Convert to primitive actions. Return JSON array:
[
  {{
    "action_type": "primitive_name",
    "parameters": {{"param": "value"}},
    "estimated_duration": seconds
  }}
]"""
                }
            ]
        )

        try:
            action_dicts = json.loads(response.content[0].text)
            actions = []
            for ad in action_dicts:
                action_type = ad["action_type"]
                primitive_def = self.action_primitives.get(action_type, {})
                actions.append(PrimitiveAction(
                    action_type=action_type,
                    parameters=ad.get("parameters", {}),
                    preconditions=primitive_def.get("preconditions", []),
                    effects=primitive_def.get("effects", []),
                    estimated_duration=ad.get("estimated_duration", 5.0)
                ))
            return actions
        except (json.JSONDecodeError, KeyError):
            return []

    def _encode_image(self, image_path: str) -> str:
        import base64
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")


# Usage Example
planner = HierarchicalTaskPlanner(vlm_client)

robot_state = {
    "location": "living_room",
    "gripper_state": "open",
    "held_object": None
}

plan = planner.plan_task(
    goal="Get the red cup from the kitchen table and bring it to me",
    image_path="current_view.jpg",
    robot_state=robot_state
)

print(f"Goal: {plan.goal}")
for i, subtask in enumerate(plan.subtasks):
    print(f"\nSubtask {i+1}: {subtask.name}")
    print(f"  Description: {subtask.description}")
    for action in subtask.actions:
        print(f"    - {action.action_type}: {action.parameters}")
```

---

## ROS 2 Cognitive Planning Node

### Complete Planning System

```python
#!/usr/bin/env python3
"""
ROS 2 Cognitive Planning Node
Integrates VLM-based scene understanding with task planning and execution
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import base64
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from enum import Enum
import anthropic


class PlanStatus(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutableAction:
    action_type: str
    parameters: dict
    status: str = "pending"
    result: Optional[str] = None


class CognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('cognitive_planning_node')

        # Parameters
        self.declare_parameter('vlm_api_key', '')
        self.declare_parameter('vlm_model', 'claude-sonnet-4-20250514')
        self.declare_parameter('planning_rate', 1.0)
        self.declare_parameter('replan_on_failure', True)

        api_key = self.get_parameter('vlm_api_key').value
        self.vlm_model = self.get_parameter('vlm_model').value
        self.replan_on_failure = self.get_parameter('replan_on_failure').value

        # Initialize VLM client
        self.vlm_client = anthropic.Anthropic(api_key=api_key)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # State
        self.current_image = None
        self.current_plan = None
        self.plan_status = PlanStatus.IDLE
        self.current_action_index = 0
        self.robot_state = {
            "location": "unknown",
            "gripper_state": "open",
            "held_object": None,
            "battery": 100
        }

        # Callback group for concurrent callbacks
        self.cb_group = ReentrantCallbackGroup()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw',
            self.image_callback, 10,
            callback_group=self.cb_group
        )

        self.goal_sub = self.create_subscription(
            String, '/cognitive_planner/goal',
            self.goal_callback, 10,
            callback_group=self.cb_group
        )

        self.state_sub = self.create_subscription(
            String, '/robot/state',
            self.state_callback, 10
        )

        # Publishers
        self.plan_pub = self.create_publisher(String, '/cognitive_planner/plan', 10)
        self.status_pub = self.create_publisher(String, '/cognitive_planner/status', 10)
        self.action_pub = self.create_publisher(String, '/cognitive_planner/current_action', 10)
        self.tts_pub = self.create_publisher(String, '/tts/say', 10)

        # Action clients
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self.cb_group
        )

        # Execution timer
        self.execution_timer = self.create_timer(
            0.5, self.execution_loop,
            callback_group=self.cb_group
        )

        # Location mapping
        self.known_locations = {
            "kitchen": {"x": 5.0, "y": 2.0, "yaw": 0.0},
            "living_room": {"x": 0.0, "y": 0.0, "yaw": 1.57},
            "bedroom": {"x": -3.0, "y": 4.0, "yaw": 3.14},
            "entrance": {"x": 0.0, "y": -2.0, "yaw": -1.57},
            "charging_station": {"x": -1.0, "y": -1.0, "yaw": 0.0}
        }

        self.get_logger().info('Cognitive Planning Node initialized')

    def image_callback(self, msg: Image):
        """Store latest camera image"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')

    def state_callback(self, msg: String):
        """Update robot state"""
        try:
            self.robot_state.update(json.loads(msg.data))
        except json.JSONDecodeError:
            pass

    def goal_callback(self, msg: String):
        """Handle new goal request"""
        goal = msg.data
        self.get_logger().info(f'Received goal: {goal}')

        if self.plan_status == PlanStatus.EXECUTING:
            self.speak("I'm currently executing a task. Please wait or say cancel.")
            return

        # Start planning in background thread
        threading.Thread(
            target=self.plan_and_execute,
            args=(goal,),
            daemon=True
        ).start()

    def plan_and_execute(self, goal: str):
        """Main planning pipeline"""
        self.plan_status = PlanStatus.PLANNING
        self.publish_status()

        self.speak(f"Planning how to {goal}")

        # Get current scene image
        if self.current_image is None:
            self.speak("I can't see anything. Please check my camera.")
            self.plan_status = PlanStatus.FAILED
            return

        # Encode image
        _, buffer = cv2.imencode('.jpg', self.current_image)
        image_b64 = base64.standard_b64encode(buffer).decode('utf-8')

        # Generate plan using VLM
        try:
            plan = self.generate_plan(goal, image_b64)

            if not plan or not plan.get("actions"):
                self.speak("I couldn't figure out how to do that. Could you rephrase?")
                self.plan_status = PlanStatus.FAILED
                return

            # Store and publish plan
            self.current_plan = plan
            self.current_action_index = 0
            self.plan_pub.publish(String(data=json.dumps(plan)))

            # Check for safety concerns
            if plan.get("safety_concerns"):
                concerns = ", ".join(plan["safety_concerns"])
                self.speak(f"Warning: {concerns}. Should I proceed?")
                # In production, wait for confirmation

            # Start execution
            self.speak(f"I'll {plan.get('summary', goal)}. Starting now.")
            self.plan_status = PlanStatus.EXECUTING
            self.publish_status()

        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')
            self.speak("Sorry, I had trouble planning that task.")
            self.plan_status = PlanStatus.FAILED

    def generate_plan(self, goal: str, image_b64: str) -> dict:
        """Use VLM to generate executable plan"""

        system_prompt = f"""You are a cognitive robot planner. Generate executable action plans.

Robot capabilities:
- navigate: Move to location (kitchen, living_room, bedroom, entrance, charging_station)
- pick: Pick up an object (requires: object visible, reachable, gripper empty)
- place: Place held object at location
- look_at: Turn to look at target
- speak: Say something to user
- wait: Wait for specified seconds

Current robot state:
{json.dumps(self.robot_state, indent=2)}

Rules:
1. Only use available actions and known locations
2. Verify preconditions before actions
3. Include look_at before pick to ensure visibility
4. Add speak actions for user feedback
5. Be safe and conservative"""

        user_message = f"""Goal: {goal}

Based on the current camera view and goal, generate an action plan.

Return JSON:
{{
  "goal": "original goal",
  "summary": "brief description of plan",
  "feasible": true/false,
  "reasoning": "why this plan will work",
  "actions": [
    {{"action": "action_name", "params": {{}}, "description": "what this does"}}
  ],
  "expected_duration": seconds,
  "safety_concerns": ["any concerns"],
  "success_criteria": "how to verify completion"
}}"""

        response = self.vlm_client.messages.create(
            model=self.vlm_model,
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64
                            }
                        },
                        {"type": "text", "text": user_message}
                    ]
                }
            ]
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse VLM response')
            return None

    def execution_loop(self):
        """Execute current plan step by step"""
        if self.plan_status != PlanStatus.EXECUTING:
            return

        if not self.current_plan or not self.current_plan.get("actions"):
            return

        actions = self.current_plan["actions"]

        if self.current_action_index >= len(actions):
            # Plan completed
            self.speak("Task completed successfully!")
            self.plan_status = PlanStatus.COMPLETED
            self.publish_status()
            return

        current_action = actions[self.current_action_index]

        # Publish current action
        self.action_pub.publish(String(data=json.dumps(current_action)))

        # Execute action
        success = self.execute_action(current_action)

        if success:
            self.current_action_index += 1
        else:
            if self.replan_on_failure:
                self.get_logger().warn(f'Action failed, attempting replan')
                self.handle_execution_failure(current_action)
            else:
                self.speak("I couldn't complete that action. Task failed.")
                self.plan_status = PlanStatus.FAILED
                self.publish_status()

    def execute_action(self, action: dict) -> bool:
        """Execute a single action"""
        action_type = action.get("action")
        params = action.get("params", {})

        self.get_logger().info(f'Executing: {action_type} with {params}')

        if action_type == "navigate":
            return self.execute_navigate(params)
        elif action_type == "pick":
            return self.execute_pick(params)
        elif action_type == "place":
            return self.execute_place(params)
        elif action_type == "look_at":
            return self.execute_look_at(params)
        elif action_type == "speak":
            return self.execute_speak(params)
        elif action_type == "wait":
            return self.execute_wait(params)
        else:
            self.get_logger().warn(f'Unknown action: {action_type}')
            return False

    def execute_navigate(self, params: dict) -> bool:
        """Execute navigation action"""
        destination = params.get("destination", "").lower()

        if destination not in self.known_locations:
            self.speak(f"I don't know where {destination} is.")
            return False

        loc = self.known_locations[destination]

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = loc["x"]
        goal_msg.pose.pose.position.y = loc["y"]
        goal_msg.pose.pose.orientation.w = np.cos(loc["yaw"] / 2)
        goal_msg.pose.pose.orientation.z = np.sin(loc["yaw"] / 2)

        self.speak(f"Moving to {destination}")

        # Send goal and wait
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)

        # Update state
        self.robot_state["location"] = destination
        return True

    def execute_pick(self, params: dict) -> bool:
        """Execute pick action (simplified)"""
        obj = params.get("object", "object")

        if self.robot_state.get("held_object"):
            self.speak("My gripper is not empty.")
            return False

        self.speak(f"Picking up the {obj}")
        # In production: call manipulation action server

        # Simulate success
        self.robot_state["held_object"] = obj
        self.robot_state["gripper_state"] = "closed"
        return True

    def execute_place(self, params: dict) -> bool:
        """Execute place action (simplified)"""
        location = params.get("location", "here")

        if not self.robot_state.get("held_object"):
            self.speak("I'm not holding anything.")
            return False

        obj = self.robot_state["held_object"]
        self.speak(f"Placing {obj} at {location}")
        # In production: call manipulation action server

        # Simulate success
        self.robot_state["held_object"] = None
        self.robot_state["gripper_state"] = "open"
        return True

    def execute_look_at(self, params: dict) -> bool:
        """Execute look_at action"""
        target = params.get("target", "forward")
        self.speak(f"Looking at {target}")
        # In production: control head/camera pan-tilt
        return True

    def execute_speak(self, params: dict) -> bool:
        """Execute speak action"""
        message = params.get("message", "")
        if message:
            self.speak(message)
        return True

    def execute_wait(self, params: dict) -> bool:
        """Execute wait action"""
        duration = params.get("duration", 1.0)
        import time
        time.sleep(duration)
        return True

    def handle_execution_failure(self, failed_action: dict):
        """Handle action failure with replanning"""
        self.get_logger().info('Attempting to replan after failure')

        # Get fresh image
        if self.current_image is not None:
            _, buffer = cv2.imencode('.jpg', self.current_image)
            image_b64 = base64.standard_b64encode(buffer).decode('utf-8')

            # Ask VLM for recovery plan
            recovery = self.get_recovery_plan(failed_action, image_b64)

            if recovery and recovery.get("can_recover"):
                self.speak(f"Adjusting plan: {recovery.get('recovery_description', '')}")

                # Insert recovery actions
                remaining_actions = self.current_plan["actions"][self.current_action_index:]
                new_actions = recovery.get("recovery_actions", []) + remaining_actions
                self.current_plan["actions"] = (
                    self.current_plan["actions"][:self.current_action_index] +
                    new_actions
                )
            else:
                self.speak("I couldn't recover from the failure.")
                self.plan_status = PlanStatus.FAILED

    def get_recovery_plan(self, failed_action: dict, image_b64: str) -> dict:
        """Ask VLM for recovery strategy"""

        response = self.vlm_client.messages.create(
            model=self.vlm_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Action failed: {json.dumps(failed_action)}
Current robot state: {json.dumps(self.robot_state)}

Can you suggest a recovery? Return JSON:
{{
  "can_recover": true/false,
  "failure_reason": "why it probably failed",
  "recovery_description": "what to do",
  "recovery_actions": [
    {{"action": "name", "params": {{}}, "description": ""}}
  ]
}}"""
                        }
                    ]
                }
            ]
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {"can_recover": False}

    def speak(self, message: str):
        """Publish TTS message"""
        self.tts_pub.publish(String(data=message))
        self.get_logger().info(f'Speaking: {message}')

    def publish_status(self):
        """Publish current status"""
        status = {
            "status": self.plan_status.value,
            "current_action_index": self.current_action_index,
            "total_actions": len(self.current_plan["actions"]) if self.current_plan else 0,
            "robot_state": self.robot_state
        }
        self.status_pub.publish(String(data=json.dumps(status)))


def main(args=None):
    rclpy.init(args=args)
    node = CognitivePlanningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Prompt Engineering for Robots

### Effective Prompting Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│              Prompt Engineering for Robot VLMs                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CONTEXT SETTING                                             │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ "You are a mobile manipulator robot with:           │    │
│     │  - 7-DOF arm with parallel gripper                  │    │
│     │  - RGB-D camera at head                             │    │
│     │  - Differential drive base                          │    │
│     │  - Max payload: 2kg"                                │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  2. CAPABILITY CONSTRAINTS                                      │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ "Available actions: navigate, pick, place, look     │    │
│     │  Known locations: kitchen, living_room, bedroom     │    │
│     │  Graspable objects: cups, books, bottles (<500g)"   │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  3. STATE INJECTION                                             │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ "Current state:                                      │    │
│     │  - Location: kitchen                                 │    │
│     │  - Gripper: holding 'red_cup'                       │    │
│     │  - Battery: 65%                                      │    │
│     │  - Arm: retracted position"                         │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  4. OUTPUT FORMAT SPECIFICATION                                 │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ "Return ONLY valid JSON:                            │    │
│     │  {                                                   │    │
│     │    'actions': [...],                                │    │
│     │    'reasoning': '...'                               │    │
│     │  }"                                                  │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  5. SAFETY RULES                                                │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ "Never:                                              │    │
│     │  - Pick up hot objects                               │    │
│     │  - Navigate near stairs                              │    │
│     │  - Exceed speed 0.5 m/s indoors                     │    │
│     │  Always confirm before actions near humans"          │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Prompt Templates

```python
class RobotPromptTemplates:
    """Prompt templates for different robot tasks"""

    @staticmethod
    def scene_analysis(robot_capabilities: dict) -> str:
        return f"""You are analyzing a scene for a robot with these capabilities:
{json.dumps(robot_capabilities, indent=2)}

When analyzing scenes:
1. Identify all objects relevant to robot manipulation
2. Note spatial relationships (on, in, next to, behind)
3. Assess reachability from robot's perspective
4. Identify potential obstacles or hazards
5. Suggest optimal approach directions for manipulation

Be precise with positions using: left/center/right and near/mid/far."""

    @staticmethod
    def task_planning(robot_state: dict, available_actions: list) -> str:
        return f"""You are a robot task planner.

Current robot state:
{json.dumps(robot_state, indent=2)}

Available actions:
{json.dumps(available_actions, indent=2)}

Planning rules:
1. Only use available actions
2. Check preconditions before each action
3. Order actions to satisfy dependencies
4. Include verification steps
5. Plan for common failure modes
6. Minimize unnecessary movements

Output structured JSON with clear action sequence."""

    @staticmethod
    def failure_recovery(failed_action: dict, error_type: str) -> str:
        return f"""A robot action failed and needs recovery.

Failed action: {json.dumps(failed_action)}
Error type: {error_type}

Analyze the situation and suggest:
1. Most likely cause of failure
2. Whether recovery is possible
3. Specific recovery actions if possible
4. Alternative approaches to achieve the goal

Be conservative - suggest aborting if safety is a concern."""

    @staticmethod
    def human_interaction(context: str) -> str:
        return f"""You help a robot interact naturally with humans.

Context: {context}

Guidelines:
1. Be concise - robot speech should be brief
2. Confirm understanding of commands
3. Ask clarifying questions when ambiguous
4. Provide status updates during long tasks
5. Apologize appropriately for failures
6. Never promise what the robot cannot do

Generate appropriate robot dialogue."""
```

---

## Multi-Modal Reasoning

### Combining Vision, Language, and Memory

```python
class MultiModalReasoner:
    """Combine multiple information sources for reasoning"""

    def __init__(self, vlm_client):
        self.client = vlm_client
        self.memory = SemanticMemory()
        self.conversation_history = []

    def reason_about_task(
        self,
        goal: str,
        current_image: str,  # base64
        robot_state: dict,
        environment_map: dict = None
    ) -> dict:
        """Comprehensive multi-modal reasoning"""

        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(goal, k=3)
        memory_context = self._format_memories(relevant_memories)

        # Build comprehensive context
        context = {
            "goal": goal,
            "robot_state": robot_state,
            "environment_knowledge": environment_map,
            "relevant_experiences": memory_context,
            "conversation_history": self.conversation_history[-5:]
        }

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": current_image
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Comprehensive task reasoning request.

Context:
{json.dumps(context, indent=2)}

Based on:
1. The current visual scene
2. Robot's current state
3. Known environment information
4. Past relevant experiences
5. Conversation history

Provide thorough reasoning:
{{
  "scene_understanding": "what you see relevant to the goal",
  "goal_interpretation": "specific interpretation of the goal",
  "feasibility_assessment": {{
    "feasible": true/false,
    "confidence": 0.0-1.0,
    "blockers": ["issues preventing completion"]
  }},
  "plan": {{
    "approach": "high-level strategy",
    "steps": ["step1", "step2"],
    "alternatives": ["backup approaches"]
  }},
  "risks": ["potential problems"],
  "clarifications_needed": ["questions if goal is ambiguous"],
  "relevant_past_experience": "how past experience informs this"
}}"""
                        }
                    ]
                }
            ]
        )

        try:
            result = json.loads(response.content[0].text)

            # Store this interaction in memory
            self.memory.store({
                "goal": goal,
                "reasoning": result,
                "outcome": "pending"  # Updated later
            })

            return result
        except json.JSONDecodeError:
            return {"error": "Failed to parse reasoning"}

    def _format_memories(self, memories: list) -> str:
        if not memories:
            return "No relevant past experiences."

        formatted = []
        for mem in memories:
            formatted.append(f"- Goal: {mem['goal']}, Outcome: {mem.get('outcome', 'unknown')}")
        return "\n".join(formatted)


class SemanticMemory:
    """Simple semantic memory for robot experiences"""

    def __init__(self):
        self.memories = []

    def store(self, experience: dict):
        """Store an experience"""
        experience["timestamp"] = time.time()
        self.memories.append(experience)

    def retrieve(self, query: str, k: int = 5) -> list:
        """Retrieve relevant memories (simplified - use embeddings in production)"""
        # Simple keyword matching - use vector similarity in production
        scored = []
        query_words = set(query.lower().split())

        for mem in self.memories:
            mem_text = json.dumps(mem).lower()
            score = len(query_words.intersection(mem_text.split()))
            scored.append((score, mem))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [mem for score, mem in scored[:k] if score > 0]

    def update_outcome(self, goal: str, outcome: str):
        """Update the outcome of a recent experience"""
        for mem in reversed(self.memories):
            if mem.get("goal") == goal and mem.get("outcome") == "pending":
                mem["outcome"] = outcome
                break
```

---

## Handling Planning Failures

### Robust Replanning System

```python
class RobustPlanner:
    """Planner with failure handling and replanning"""

    def __init__(self, vlm_client):
        self.client = vlm_client
        self.max_replan_attempts = 3
        self.failure_history = []

    def execute_with_monitoring(
        self,
        plan: dict,
        execute_action_fn,
        get_observation_fn
    ) -> dict:
        """Execute plan with monitoring and replanning"""

        actions = plan.get("actions", [])
        results = []
        replan_count = 0

        i = 0
        while i < len(actions):
            action = actions[i]

            # Pre-execution check
            observation = get_observation_fn()
            if not self.verify_preconditions(action, observation):
                # Preconditions not met - need to replan
                self.failure_history.append({
                    "action": action,
                    "reason": "preconditions_not_met",
                    "observation": observation
                })

                if replan_count < self.max_replan_attempts:
                    new_plan = self.replan_from_failure(
                        plan["goal"],
                        actions[i:],
                        observation,
                        "preconditions_not_met"
                    )
                    if new_plan:
                        actions = actions[:i] + new_plan["actions"]
                        replan_count += 1
                        continue

                return {
                    "success": False,
                    "completed_actions": results,
                    "failed_at": i,
                    "reason": "preconditions_not_met"
                }

            # Execute action
            success, result = execute_action_fn(action)
            results.append({"action": action, "success": success, "result": result})

            if not success:
                self.failure_history.append({
                    "action": action,
                    "reason": "execution_failed",
                    "result": result
                })

                if replan_count < self.max_replan_attempts:
                    observation = get_observation_fn()
                    new_plan = self.replan_from_failure(
                        plan["goal"],
                        actions[i:],
                        observation,
                        f"execution_failed: {result}"
                    )
                    if new_plan:
                        actions = actions[:i] + new_plan["actions"]
                        replan_count += 1
                        continue

                return {
                    "success": False,
                    "completed_actions": results,
                    "failed_at": i,
                    "reason": f"execution_failed: {result}"
                }

            # Post-execution verification
            observation = get_observation_fn()
            if not self.verify_effects(action, observation):
                self.failure_history.append({
                    "action": action,
                    "reason": "effects_not_achieved",
                    "observation": observation
                })

                if replan_count < self.max_replan_attempts:
                    new_plan = self.replan_from_failure(
                        plan["goal"],
                        actions[i:],
                        observation,
                        "effects_not_achieved"
                    )
                    if new_plan:
                        actions = actions[:i+1] + new_plan["actions"]
                        replan_count += 1
                        i += 1
                        continue

            i += 1

        return {
            "success": True,
            "completed_actions": results,
            "replan_count": replan_count
        }

    def verify_preconditions(self, action: dict, observation: dict) -> bool:
        """Verify action preconditions using VLM"""

        action_type = action.get("action")

        # Define preconditions
        preconditions = {
            "pick": ["gripper_empty", "object_visible", "object_reachable"],
            "place": ["holding_object", "location_clear"],
            "navigate": ["path_clear"]
        }

        required = preconditions.get(action_type, [])

        for precond in required:
            if precond == "gripper_empty":
                if observation.get("held_object"):
                    return False
            elif precond == "holding_object":
                if not observation.get("held_object"):
                    return False
            # Add more precondition checks

        return True

    def verify_effects(self, action: dict, observation: dict) -> bool:
        """Verify action effects were achieved"""

        action_type = action.get("action")
        params = action.get("params", {})

        if action_type == "pick":
            expected_object = params.get("object")
            return observation.get("held_object") == expected_object

        elif action_type == "place":
            return observation.get("held_object") is None

        elif action_type == "navigate":
            expected_location = params.get("destination")
            return observation.get("location") == expected_location

        return True

    def replan_from_failure(
        self,
        original_goal: str,
        remaining_actions: list,
        observation: dict,
        failure_reason: str
    ) -> dict:
        """Generate recovery plan after failure"""

        # Include failure history for context
        recent_failures = self.failure_history[-3:]

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""Robot task replanning needed.

Original goal: {original_goal}
Remaining planned actions: {json.dumps(remaining_actions)}
Current observation: {json.dumps(observation)}
Failure reason: {failure_reason}
Recent failure history: {json.dumps(recent_failures)}

Generate a recovery plan. Consider:
1. What went wrong and why
2. Whether the goal is still achievable
3. What alternative approaches might work
4. Whether to retry, skip, or abort

Return JSON:
{{
  "can_recover": true/false,
  "recovery_strategy": "retry/skip/alternative/abort",
  "reasoning": "explanation",
  "actions": [
    {{"action": "name", "params": {{}}, "description": ""}}
  ]
}}"""
                }
            ]
        )

        try:
            result = json.loads(response.content[0].text)
            if result.get("can_recover") and result.get("actions"):
                return result
            return None
        except json.JSONDecodeError:
            return None
```

---

## Local VLM Deployment

### Running VLMs Locally with Ollama

```python
import requests
import base64

class LocalVLM:
    """Interface for local VLM using Ollama"""

    def __init__(
        self,
        model: str = "llava:13b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Analyze image with local VLM"""

        # Encode image
        with open(image_path, "rb") as f:
            image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
        )

        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code}"

    def generate_plan(self, image_path: str, goal: str, robot_state: dict) -> dict:
        """Generate task plan using local VLM"""

        prompt = f"""You are a robot task planner. Analyze this image and create a plan.

Goal: {goal}
Robot state: {json.dumps(robot_state)}

Return a JSON plan with actions the robot should take.
Format: {{"actions": [{{"action": "name", "params": {{}}}}], "reasoning": "..."}}"""

        response_text = self.analyze_image(image_path, prompt)

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"error": "No JSON found in response", "raw": response_text}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw": response_text}


# Installation and setup:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull llava:13b
# ollama serve

# Usage
local_vlm = LocalVLM(model="llava:13b")

result = local_vlm.generate_plan(
    image_path="scene.jpg",
    goal="Pick up the cup",
    robot_state={"gripper": "empty", "location": "kitchen"}
)
print(result)
```

### LLaVA with Transformers

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

class LLaVAPlanner:
    """Local VLM planner using LLaVA"""

    def __init__(self, model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    def analyze(self, image_path: str, prompt: str) -> str:
        """Analyze image with prompt"""

        image = Image.open(image_path)

        # Format prompt for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            formatted_prompt,
            image,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )

        response = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract assistant response
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        return response

    def plan_task(self, image_path: str, goal: str) -> dict:
        """Generate task plan"""

        prompt = f"""Analyze this image for robot task planning.

Goal: {goal}

Identify:
1. Relevant objects and their locations
2. Required actions to achieve the goal
3. Any obstacles or concerns

Provide a step-by-step plan in JSON format:
{{"steps": ["step1", "step2"], "objects": ["obj1"], "concerns": []}}"""

        response = self.analyze(image_path, prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {"raw_response": response}
```

---

## Practical Exercise: Cognitive Robot Assistant

### Project Goal

Build a cognitive robot assistant that can:
1. Understand natural language requests with visual context
2. Generate and execute multi-step plans
3. Handle failures gracefully with replanning
4. Maintain conversation context

### Package Structure

```
cognitive_robot_ws/
├── src/
│   └── cognitive_robot/
│       ├── cognitive_robot/
│       │   ├── __init__.py
│       │   ├── planning_node.py
│       │   ├── scene_analyzer.py
│       │   ├── task_planner.py
│       │   ├── executor.py
│       │   ├── memory.py
│       │   └── prompts.py
│       ├── config/
│       │   └── cognitive_robot.yaml
│       ├── launch/
│       │   └── cognitive_robot.launch.py
│       ├── package.xml
│       └── setup.py
```

### Configuration

```yaml
# config/cognitive_robot.yaml
cognitive_robot:
  ros__parameters:
    # VLM settings
    vlm_provider: "anthropic"  # or "openai", "local"
    vlm_model: "claude-sonnet-4-20250514"
    vlm_api_key: ""

    # Local VLM settings (if provider is "local")
    local_vlm_url: "http://localhost:11434"
    local_vlm_model: "llava:13b"

    # Planning settings
    max_replan_attempts: 3
    plan_timeout: 30.0
    action_timeout: 60.0

    # Safety settings
    require_confirmation: true
    max_speed: 0.5
    restricted_areas: ["stairs", "outside"]

    # Memory settings
    enable_memory: true
    max_memory_items: 100
```

### Launch File

```python
# launch/cognitive_robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('cognitive_robot')
    config_file = os.path.join(pkg_dir, 'config', 'cognitive_robot.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('vlm_api_key', default_value=''),

        Node(
            package='cognitive_robot',
            executable='planning_node',
            name='cognitive_planner',
            parameters=[
                config_file,
                {'vlm_api_key': LaunchConfiguration('vlm_api_key')}
            ],
            output='screen'
        ),
    ])
```

### Testing the System

```bash
# Terminal 1: Launch simulation
ros2 launch my_robot simulation.launch.py

# Terminal 2: Launch navigation
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

# Terminal 3: Launch cognitive planner
ros2 launch cognitive_robot cognitive_robot.launch.py \
  vlm_api_key:=$ANTHROPIC_API_KEY

# Terminal 4: Send goals
ros2 topic pub /cognitive_planner/goal std_msgs/String \
  "data: 'Find the red cup on the table and bring it to me'"

# Monitor status
ros2 topic echo /cognitive_planner/status

# Monitor current action
ros2 topic echo /cognitive_planner/current_action
```

---

## Summary

In this chapter, you learned:

- **Vision-Language Models**: How VLMs combine visual and textual understanding
- **Scene Analysis**: Using VLMs to understand robot environments
- **Task Planning**: Generating executable plans from natural language goals
- **Hierarchical Decomposition**: Breaking complex goals into primitive actions
- **ROS 2 Integration**: Building a complete cognitive planning node
- **Prompt Engineering**: Crafting effective prompts for robot VLMs
- **Multi-Modal Reasoning**: Combining vision, language, and memory
- **Failure Handling**: Robust replanning when actions fail
- **Local Deployment**: Running VLMs locally for low-latency robotics

Cognitive planning with VLMs represents the frontier of robot intelligence, enabling natural human-robot collaboration through language and vision.

---

## Further Reading

- [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io/)
- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/)
- [SayCan: Grounding Language in Robotic Affordances](https://say-can.github.io/)
- [Code as Policies](https://code-as-policies.github.io/)
- [LLaVA: Visual Instruction Tuning](https://llava-vl.github.io/)
- [Anthropic Claude Vision Documentation](https://docs.anthropic.com/claude/docs/vision)

---

## Next Week Preview

In **Week 13**, we bring everything together in the **Capstone Project**:
- Integrating all modules: ROS 2, simulation, perception, navigation, and VLA
- Building a complete Physical AI system
- Testing, validation, and documentation
- Presenting your autonomous robot
