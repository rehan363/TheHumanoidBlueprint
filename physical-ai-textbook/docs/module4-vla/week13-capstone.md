---
sidebar_position: 13
title: "Week 13 - Capstone Project"
description: "Build a complete autonomous robot assistant integrating perception, navigation, manipulation, and natural language interaction"
keywords: [capstone, project, Physical AI, autonomous robot, ROS 2, VLA, integration, portfolio]
last_updated: "2025-12-30"
estimated_reading_time: 35
---

# Week 13: Capstone Project

Congratulations on reaching the final chapter! This capstone project brings together everything you've learned across all four modules to build a complete Physical AI system. You'll create an autonomous robot assistant capable of understanding natural language commands, perceiving its environment, planning tasks, and executing complex multi-step actions.

---

## Learning Objectives

By completing this capstone project, you will:

- Integrate ROS 2 nodes across perception, planning, and control
- Build a complete autonomous robot system in simulation
- Implement voice-controlled task execution with VLM reasoning
- Design robust error handling and recovery mechanisms
- Create comprehensive documentation and testing procedures
- Demonstrate your Physical AI system with real-world scenarios

---

## Project Overview

### The Challenge

Build an **Autonomous Home Assistant Robot** that can:

```
┌─────────────────────────────────────────────────────────────────┐
│              Autonomous Home Assistant Robot                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CAPABILITIES:                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. UNDERSTAND    │ Natural language voice commands       │   │
│  ├──────────────────┼───────────────────────────────────────┤   │
│  │ 2. PERCEIVE      │ RGB-D camera scene understanding      │   │
│  ├──────────────────┼───────────────────────────────────────┤   │
│  │ 3. NAVIGATE      │ Autonomous navigation in home         │   │
│  ├──────────────────┼───────────────────────────────────────┤   │
│  │ 4. MANIPULATE    │ Pick and place objects                │   │
│  ├──────────────────┼───────────────────────────────────────┤   │
│  │ 5. COMMUNICATE   │ Verbal feedback and status updates    │   │
│  └──────────────────┴───────────────────────────────────────┘   │
│                                                                 │
│  EXAMPLE TASKS:                                                 │
│  • "Bring me the water bottle from the kitchen"                │
│  • "Find my keys and tell me where they are"                   │
│  • "Clean up the living room by putting toys in the box"       │
│  • "Go to the bedroom and check if the window is open"         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPSTONE SYSTEM ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    USER INTERFACE                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Voice     │  │   Web UI    │  │   RViz      │     │   │
│  │  │   Input     │  │  Dashboard  │  │   Monitor   │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  └─────────┼────────────────┼────────────────┼─────────────┘   │
│            │                │                │                  │
│            └────────────────┴────────────────┘                  │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  COGNITIVE LAYER                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Speech    │  │    VLM      │  │   Task      │     │   │
│  │  │Recognition  │──▶│  Planner   │──▶│  Manager   │     │   │
│  │  │  (Whisper)  │  │  (Claude)   │  │             │     │   │
│  │  └─────────────┘  └─────────────┘  └──────┬──────┘     │   │
│  └───────────────────────────────────────────┼─────────────┘   │
│                                              │                  │
│                                              ▼                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  PERCEPTION LAYER                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Object    │  │   Scene     │  │   Person    │     │   │
│  │  │  Detection  │  │   Graph     │  │  Detection  │     │   │
│  │  │   (YOLO)    │  │             │  │             │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                              │                  │
│                                              ▼                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   ACTION LAYER                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Nav2      │  │  MoveIt2    │  │   Gripper   │     │   │
│  │  │ Navigation  │  │   Motion    │  │  Control    │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                              │                  │
│                                              ▼                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  SIMULATION LAYER                        │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │              Gazebo / Isaac Sim                  │    │   │
│  │  │   ┌───────┐  ┌─────────┐  ┌──────────────┐     │    │   │
│  │  │   │ Robot │  │  Home   │  │   Objects    │     │    │   │
│  │  │   │ Model │  │  World  │  │   & Props    │     │    │   │
│  │  │   └───────┘  └─────────┘  └──────────────┘     │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Requirements

### Minimum Requirements (Pass)

To successfully complete the capstone, your robot must demonstrate:

| Requirement | Description | Verification |
|-------------|-------------|--------------|
| **Voice Command** | Accept at least 5 different voice commands | Demo recording |
| **Navigation** | Navigate to 3+ named locations | Success rate ≥80% |
| **Object Detection** | Detect and locate 5+ object types | mAP ≥0.5 |
| **Pick & Place** | Successfully grasp and place objects | Success rate ≥60% |
| **Task Completion** | Complete 3 multi-step tasks end-to-end | Demo recording |
| **Error Handling** | Recover from at least 2 failure types | Demo recording |
| **Documentation** | README, architecture diagram, demo video | Deliverables |

### Stretch Goals (Excellence)

| Goal | Description | Bonus |
|------|-------------|-------|
| **Multi-Object Tasks** | Handle tasks involving 3+ objects | +10% |
| **Dynamic Replanning** | Replan when environment changes | +10% |
| **Person Following** | Follow and interact with humans | +15% |
| **Web Dashboard** | Real-time monitoring interface | +10% |
| **Real Hardware** | Deploy on physical robot | +25% |

---

## Project Structure

### Recommended Workspace Layout

```
capstone_ws/
├── src/
│   ├── capstone_bringup/           # Launch files and configs
│   │   ├── launch/
│   │   │   ├── simulation.launch.py
│   │   │   ├── robot.launch.py
│   │   │   ├── perception.launch.py
│   │   │   ├── navigation.launch.py
│   │   │   └── full_system.launch.py
│   │   ├── config/
│   │   │   ├── robot_params.yaml
│   │   │   ├── nav2_params.yaml
│   │   │   ├── perception_params.yaml
│   │   │   └── vlm_params.yaml
│   │   ├── worlds/
│   │   │   └── home_environment.world
│   │   ├── rviz/
│   │   │   └── capstone.rviz
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   ├── capstone_description/       # Robot URDF and meshes
│   │   ├── urdf/
│   │   │   ├── robot.urdf.xacro
│   │   │   ├── sensors.urdf.xacro
│   │   │   └── arm.urdf.xacro
│   │   ├── meshes/
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   ├── capstone_perception/        # Perception nodes
│   │   ├── capstone_perception/
│   │   │   ├── __init__.py
│   │   │   ├── object_detector.py
│   │   │   ├── scene_analyzer.py
│   │   │   └── person_detector.py
│   │   ├── package.xml
│   │   └── setup.py
│   │
│   ├── capstone_planning/          # Planning and reasoning
│   │   ├── capstone_planning/
│   │   │   ├── __init__.py
│   │   │   ├── task_planner.py
│   │   │   ├── vlm_interface.py
│   │   │   └── action_executor.py
│   │   ├── package.xml
│   │   └── setup.py
│   │
│   ├── capstone_voice/             # Voice interface
│   │   ├── capstone_voice/
│   │   │   ├── __init__.py
│   │   │   ├── speech_recognition.py
│   │   │   ├── text_to_speech.py
│   │   │   └── voice_controller.py
│   │   ├── package.xml
│   │   └── setup.py
│   │
│   ├── capstone_manipulation/      # Arm control
│   │   ├── capstone_manipulation/
│   │   │   ├── __init__.py
│   │   │   ├── grasp_planner.py
│   │   │   └── arm_controller.py
│   │   ├── package.xml
│   │   └── setup.py
│   │
│   └── capstone_interfaces/        # Custom messages/services
│       ├── msg/
│       │   ├── DetectedObject.msg
│       │   ├── TaskStatus.msg
│       │   └── RobotState.msg
│       ├── srv/
│       │   ├── ExecuteTask.srv
│       │   └── GetSceneInfo.srv
│       ├── action/
│       │   └── PerformTask.action
│       ├── package.xml
│       └── CMakeLists.txt
│
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── SETUP.md
│   └── images/
│
├── tests/
│   ├── test_perception.py
│   ├── test_navigation.py
│   ├── test_manipulation.py
│   └── test_integration.py
│
└── scripts/
    ├── setup_workspace.sh
    ├── run_demo.sh
    └── record_demo.sh
```

---

## Implementation Guide

### Phase 1: Environment Setup (Days 1-2)

#### 1.1 Create the Workspace

```bash
# Create workspace
mkdir -p ~/capstone_ws/src
cd ~/capstone_ws/src

# Create packages
ros2 pkg create --build-type ament_cmake capstone_bringup
ros2 pkg create --build-type ament_cmake capstone_description
ros2 pkg create --build-type ament_cmake capstone_interfaces
ros2 pkg create --build-type ament_python capstone_perception
ros2 pkg create --build-type ament_python capstone_planning
ros2 pkg create --build-type ament_python capstone_voice
ros2 pkg create --build-type ament_python capstone_manipulation

# Build
cd ~/capstone_ws
colcon build --symlink-install
source install/setup.bash
```

#### 1.2 Home Environment World

```xml
<!-- worlds/home_environment.world -->
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="home_environment">

    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- House structure -->
    <model name="house">
      <static>true</static>
      <link name="walls">
        <!-- Living Room walls -->
        <collision name="living_wall_north">
          <pose>0 5 1.25 0 0 0</pose>
          <geometry>
            <box><size>8 0.15 2.5</size></box>
          </geometry>
        </collision>
        <visual name="living_wall_north_visual">
          <pose>0 5 1.25 0 0 0</pose>
          <geometry>
            <box><size>8 0.15 2.5</size></box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
          </material>
        </visual>

        <!-- Add more walls for kitchen, bedroom, etc. -->
        <!-- Kitchen area: x: 3-7, y: 0-5 -->
        <!-- Living room: x: -4-3, y: 0-5 -->
        <!-- Bedroom: x: -4-0, y: -5-0 -->

      </link>
    </model>

    <!-- Furniture -->
    <model name="kitchen_table">
      <static>true</static>
      <pose>5 2.5 0 0 0 0</pose>
      <link name="table">
        <collision name="tabletop">
          <pose>0 0 0.75 0 0 0</pose>
          <geometry>
            <box><size>1.2 0.8 0.05</size></box>
          </geometry>
        </collision>
        <visual name="tabletop_visual">
          <pose>0 0 0.75 0 0 0</pose>
          <geometry>
            <box><size>1.2 0.8 0.05</size></box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
          </material>
        </visual>
        <!-- Table legs -->
        <collision name="leg1">
          <pose>0.5 0.3 0.375 0 0 0</pose>
          <geometry><cylinder><radius>0.03</radius><length>0.75</length></cylinder></geometry>
        </collision>
        <visual name="leg1_visual">
          <pose>0.5 0.3 0.375 0 0 0</pose>
          <geometry><cylinder><radius>0.03</radius><length>0.75</length></cylinder></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient></material>
        </visual>
        <!-- Add 3 more legs -->
      </link>
    </model>

    <!-- Sofa in living room -->
    <model name="sofa">
      <static>true</static>
      <pose>-2 3 0 0 0 0</pose>
      <link name="sofa_link">
        <collision name="seat">
          <pose>0 0 0.25 0 0 0</pose>
          <geometry>
            <box><size>2.0 0.8 0.5</size></box>
          </geometry>
        </collision>
        <visual name="seat_visual">
          <pose>0 0 0.25 0 0 0</pose>
          <geometry>
            <box><size>2.0 0.8 0.5</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.6 1</ambient>
          </material>
        </visual>
        <collision name="back">
          <pose>0 -0.3 0.6 0 0 0</pose>
          <geometry>
            <box><size>2.0 0.2 0.7</size></box>
          </geometry>
        </collision>
        <visual name="back_visual">
          <pose>0 -0.3 0.6 0 0 0</pose>
          <geometry>
            <box><size>2.0 0.2 0.7</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.6 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <!-- Manipulable objects -->
    <model name="red_cup">
      <pose>5.2 2.3 0.8 0 0 0</pose>
      <link name="cup">
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz>
          </inertia>
        </inertial>
        <collision name="cup_collision">
          <geometry>
            <cylinder><radius>0.035</radius><length>0.1</length></cylinder>
          </geometry>
        </collision>
        <visual name="cup_visual">
          <geometry>
            <cylinder><radius>0.035</radius><length>0.1</length></cylinder>
          </geometry>
          <material>
            <ambient>0.9 0.1 0.1 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <model name="water_bottle">
      <pose>5.0 2.6 0.85 0 0 0</pose>
      <link name="bottle">
        <inertial>
          <mass>0.3</mass>
          <inertia>
            <ixx>0.0005</ixx><iyy>0.0005</iyy><izz>0.0002</izz>
          </inertia>
        </inertial>
        <collision name="bottle_collision">
          <geometry>
            <cylinder><radius>0.03</radius><length>0.2</length></cylinder>
          </geometry>
        </collision>
        <visual name="bottle_visual">
          <geometry>
            <cylinder><radius>0.03</radius><length>0.2</length></cylinder>
          </geometry>
          <material>
            <ambient>0.2 0.5 0.9 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <model name="book">
      <pose>-1.5 3.2 0.55 0 0 0.3</pose>
      <link name="book_link">
        <inertial>
          <mass>0.4</mass>
          <inertia>
            <ixx>0.001</ixx><iyy>0.002</iyy><izz>0.002</izz>
          </inertia>
        </inertial>
        <collision name="book_collision">
          <geometry>
            <box><size>0.2 0.15 0.03</size></box>
          </geometry>
        </collision>
        <visual name="book_visual">
          <geometry>
            <box><size>0.2 0.15 0.03</size></box>
          </geometry>
          <material>
            <ambient>0.1 0.5 0.1 1</ambient>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

### Phase 2: Core Perception (Days 3-4)

#### 2.1 Object Detector Node

```python
#!/usr/bin/env python3
"""
Object Detection Node for Capstone Project
Detects and tracks objects in the robot's environment
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Custom message (define in capstone_interfaces)
from capstone_interfaces.msg import DetectedObject, DetectedObjectArray


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_3d: Tuple[float, float, float] = None


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')

        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('target_objects', [
            'cup', 'bottle', 'book', 'remote', 'cell phone',
            'keys', 'bowl', 'apple', 'banana', 'scissors'
        ])

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.target_objects = self.get_parameter('target_objects').value

        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)

        # CV Bridge
        self.bridge = CvBridge()

        # Camera info
        self.camera_info = None
        self.depth_image = None

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw',
            self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            self.depth_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info',
            self.camera_info_callback, 10
        )

        # Publishers
        self.detections_pub = self.create_publisher(
            DetectedObjectArray, '/perception/detected_objects', 10
        )
        self.debug_image_pub = self.create_publisher(
            Image, '/perception/debug_image', 10
        )

        # Detection state
        self.tracked_objects = {}
        self.frame_count = 0

        self.get_logger().info('Object Detector Node initialized')

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def rgb_callback(self, msg: Image):
        self.frame_count += 1

        # Process every 3rd frame for efficiency
        if self.frame_count % 3 != 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        # Run detection
        detections = self.detect_objects(cv_image)

        # Add 3D positions if depth available
        if self.depth_image is not None and self.camera_info is not None:
            detections = self.add_3d_positions(detections)

        # Publish detections
        self.publish_detections(detections, msg.header)

        # Publish debug image
        debug_image = self.draw_detections(cv_image, detections)
        self.debug_image_pub.publish(
            self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
        )

    def detect_objects(self, image: np.ndarray) -> List[Detection]:
        """Run YOLO detection"""
        results = self.model(image, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Filter to target objects
                if class_name.lower() not in [t.lower() for t in self.target_objects]:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2)
                ))

        return detections

    def add_3d_positions(self, detections: List[Detection]) -> List[Detection]:
        """Calculate 3D positions from depth"""
        if self.camera_info is None:
            return detections

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Get depth at center
            if 0 <= center_y < self.depth_image.shape[0] and \
               0 <= center_x < self.depth_image.shape[1]:
                depth = self.depth_image[center_y, center_x]

                if depth > 0 and not np.isnan(depth):
                    # Calculate 3D position in camera frame
                    z = depth
                    x = (center_x - cx) * z / fx
                    y = (center_y - cy) * z / fy
                    det.center_3d = (x, y, z)

        return detections

    def publish_detections(self, detections: List[Detection], header: Header):
        """Publish detection results"""
        msg = DetectedObjectArray()
        msg.header = header

        for det in detections:
            obj_msg = DetectedObject()
            obj_msg.class_name = det.class_name
            obj_msg.confidence = det.confidence
            obj_msg.bbox_x = det.bbox[0]
            obj_msg.bbox_y = det.bbox[1]
            obj_msg.bbox_width = det.bbox[2] - det.bbox[0]
            obj_msg.bbox_height = det.bbox[3] - det.bbox[1]

            if det.center_3d is not None:
                obj_msg.position.x = det.center_3d[0]
                obj_msg.position.y = det.center_3d[1]
                obj_msg.position.z = det.center_3d[2]
                obj_msg.has_3d_position = True
            else:
                obj_msg.has_3d_position = False

            msg.objects.append(obj_msg)

        self.detections_pub.publish(msg)

    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on image"""
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name}: {det.confidence:.2f}"
            if det.center_3d is not None:
                label += f" ({det.center_3d[2]:.2f}m)"

            cv2.putText(
                output, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return output


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()

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

### Phase 3: Task Planning (Days 5-6)

#### 3.1 Task Manager Node

```python
#!/usr/bin/env python3
"""
Task Manager Node
Coordinates task execution using VLM reasoning
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
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
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import anthropic

from capstone_interfaces.msg import DetectedObjectArray, TaskStatus, RobotState
from capstone_interfaces.action import PerformTask
from capstone_interfaces.srv import ExecuteTask, GetSceneInfo


class TaskState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskContext:
    goal: str
    plan: List[Dict] = field(default_factory=list)
    current_step: int = 0
    state: TaskState = TaskState.IDLE
    start_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')

        # Parameters
        self.declare_parameter('vlm_api_key', '')
        self.declare_parameter('vlm_model', 'claude-sonnet-4-20250514')
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('require_confirmation', True)

        api_key = self.get_parameter('vlm_api_key').value
        self.vlm_model = self.get_parameter('vlm_model').value
        self.max_retries = self.get_parameter('max_retries').value
        self.require_confirmation = self.get_parameter('require_confirmation').value

        # VLM client
        self.vlm = anthropic.Anthropic(api_key=api_key)

        # CV Bridge
        self.bridge = CvBridge()

        # Callback group
        self.cb_group = ReentrantCallbackGroup()

        # State
        self.current_task = None
        self.current_image = None
        self.detected_objects = []
        self.robot_state = {
            'location': 'living_room',
            'holding': None,
            'battery': 100,
            'status': 'ready'
        }

        # Known locations
        self.locations = {
            'living_room': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
            'kitchen': {'x': 5.0, 'y': 2.0, 'yaw': 0.0},
            'bedroom': {'x': -2.0, 'y': -3.0, 'yaw': 1.57},
            'entrance': {'x': 0.0, 'y': 4.0, 'yaw': 3.14},
            'charging_station': {'x': -3.0, 'y': 0.0, 'yaw': 0.0}
        }

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw',
            self.image_callback, 10,
            callback_group=self.cb_group
        )
        self.detections_sub = self.create_subscription(
            DetectedObjectArray, '/perception/detected_objects',
            self.detections_callback, 10
        )
        self.voice_sub = self.create_subscription(
            String, '/voice/command',
            self.voice_command_callback, 10,
            callback_group=self.cb_group
        )
        self.confirmation_sub = self.create_subscription(
            String, '/voice/confirmation',
            self.confirmation_callback, 10
        )

        # Publishers
        self.status_pub = self.create_publisher(TaskStatus, '/task/status', 10)
        self.tts_pub = self.create_publisher(String, '/tts/say', 10)
        self.state_pub = self.create_publisher(RobotState, '/robot/state', 10)

        # Action clients
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self.cb_group
        )

        # Action server
        self._action_server = ActionServer(
            self,
            PerformTask,
            'perform_task',
            execute_callback=self.execute_task_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )

        # Task execution timer
        self.task_timer = self.create_timer(
            0.5, self.task_execution_loop,
            callback_group=self.cb_group
        )

        self.get_logger().info('Task Manager Node initialized')

    def image_callback(self, msg: Image):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def detections_callback(self, msg: DetectedObjectArray):
        self.detected_objects = msg.objects

    def voice_command_callback(self, msg: String):
        """Handle voice commands"""
        command = msg.data.strip()
        if not command:
            return

        self.get_logger().info(f'Received voice command: {command}')

        if self.current_task and self.current_task.state == TaskState.EXECUTING:
            self.speak("I'm currently working on a task. Please wait.")
            return

        # Start new task
        self.start_task(command)

    def confirmation_callback(self, msg: String):
        """Handle confirmation responses"""
        if not self.current_task or \
           self.current_task.state != TaskState.WAITING_CONFIRMATION:
            return

        response = msg.data.lower()
        if any(word in response for word in ['yes', 'yeah', 'proceed', 'go ahead', 'do it']):
            self.current_task.state = TaskState.EXECUTING
            self.speak("Okay, starting now.")
        elif any(word in response for word in ['no', 'cancel', 'stop', 'don\'t']):
            self.current_task.state = TaskState.IDLE
            self.speak("Task cancelled.")
            self.current_task = None

    def goal_callback(self, goal_request):
        """Accept task goal"""
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept cancellation"""
        return CancelResponse.ACCEPT

    async def execute_task_callback(self, goal_handle):
        """Execute task action"""
        self.get_logger().info('Executing task via action')

        task_goal = goal_handle.request.task_description
        self.start_task(task_goal)

        # Wait for completion
        while self.current_task and \
              self.current_task.state not in [TaskState.COMPLETED, TaskState.FAILED]:
            await asyncio.sleep(0.5)

            # Publish feedback
            feedback = PerformTask.Feedback()
            feedback.current_step = self.current_task.current_step
            feedback.total_steps = len(self.current_task.plan)
            feedback.status = self.current_task.state.value
            goal_handle.publish_feedback(feedback)

        # Return result
        result = PerformTask.Result()
        if self.current_task:
            result.success = self.current_task.state == TaskState.COMPLETED
            result.message = json.dumps(self.current_task.results)
        else:
            result.success = False
            result.message = "Task was cancelled"

        goal_handle.succeed()
        return result

    def start_task(self, goal: str):
        """Start a new task"""
        self.current_task = TaskContext(goal=goal)
        self.current_task.state = TaskState.PLANNING
        self.current_task.start_time = self.get_clock().now().nanoseconds / 1e9

        self.speak(f"I'll help you with: {goal}")

        # Plan in background
        threading.Thread(
            target=self.plan_task,
            daemon=True
        ).start()

    def plan_task(self):
        """Use VLM to plan the task"""
        if self.current_image is None:
            self.speak("I can't see anything. Please check my camera.")
            self.current_task.state = TaskState.FAILED
            return

        # Encode current image
        _, buffer = cv2.imencode('.jpg', self.current_image)
        image_b64 = base64.standard_b64encode(buffer).decode('utf-8')

        # Format detected objects
        objects_desc = []
        for obj in self.detected_objects:
            obj_str = f"- {obj.class_name} (confidence: {obj.confidence:.2f})"
            if obj.has_3d_position:
                obj_str += f" at ({obj.position.x:.2f}, {obj.position.y:.2f}, {obj.position.z:.2f}m)"
            objects_desc.append(obj_str)

        objects_text = "\n".join(objects_desc) if objects_desc else "No objects currently detected"

        # Build prompt
        system_prompt = f"""You are a home assistant robot task planner.

Robot capabilities:
- navigate: Move to location ({', '.join(self.locations.keys())})
- pick: Pick up a graspable object
- place: Place held object at location
- look_around: Scan the environment
- speak: Say something to the user
- wait: Wait for specified seconds

Current state:
- Location: {self.robot_state['location']}
- Holding: {self.robot_state['holding'] or 'nothing'}
- Battery: {self.robot_state['battery']}%

Detected objects:
{objects_text}

Rules:
1. Only use available actions
2. Verify object is visible before picking
3. Navigate to correct room before manipulating objects
4. Provide verbal feedback during task
5. Handle cases where objects are not found"""

        user_prompt = f"""Task: {self.current_task.goal}

Create a step-by-step plan. Return JSON:
{{
  "understood": true/false,
  "summary": "brief task description",
  "feasible": true/false,
  "reason": "if not feasible, explain why",
  "plan": [
    {{"action": "action_name", "params": {{}}, "description": "what this does"}}
  ],
  "estimated_duration": seconds,
  "potential_issues": ["possible problems"]
}}"""

        try:
            response = self.vlm.messages.create(
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
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )

            plan_data = json.loads(response.content[0].text)

            if not plan_data.get('feasible', False):
                reason = plan_data.get('reason', 'Task not feasible')
                self.speak(f"I can't do that. {reason}")
                self.current_task.state = TaskState.FAILED
                return

            self.current_task.plan = plan_data.get('plan', [])
            self.current_task.results['plan'] = plan_data

            if self.require_confirmation:
                summary = plan_data.get('summary', self.current_task.goal)
                self.speak(f"I'll {summary}. Should I proceed?")
                self.current_task.state = TaskState.WAITING_CONFIRMATION
            else:
                self.speak(f"Starting: {plan_data.get('summary', '')}")
                self.current_task.state = TaskState.EXECUTING

        except Exception as e:
            self.get_logger().error(f'Planning error: {e}')
            self.speak("I had trouble understanding that task.")
            self.current_task.state = TaskState.FAILED

    def task_execution_loop(self):
        """Execute task steps"""
        if not self.current_task:
            return

        if self.current_task.state != TaskState.EXECUTING:
            return

        plan = self.current_task.plan
        step_idx = self.current_task.current_step

        if step_idx >= len(plan):
            self.speak("Task completed successfully!")
            self.current_task.state = TaskState.COMPLETED
            self.publish_status()
            return

        current_step = plan[step_idx]
        action = current_step.get('action')
        params = current_step.get('params', {})

        self.get_logger().info(f'Executing step {step_idx + 1}: {action}')

        success = self.execute_action(action, params)

        if success:
            self.current_task.current_step += 1
        else:
            self.current_task.errors.append(f"Step {step_idx + 1} failed: {action}")
            if len(self.current_task.errors) >= self.max_retries:
                self.speak("I've encountered too many errors. Task failed.")
                self.current_task.state = TaskState.FAILED
            else:
                self.speak("Let me try a different approach.")
                self.current_task.state = TaskState.RECOVERING
                self.attempt_recovery(current_step)

        self.publish_status()

    def execute_action(self, action: str, params: dict) -> bool:
        """Execute a single action"""
        if action == 'navigate':
            return self.execute_navigate(params)
        elif action == 'pick':
            return self.execute_pick(params)
        elif action == 'place':
            return self.execute_place(params)
        elif action == 'look_around':
            return self.execute_look_around(params)
        elif action == 'speak':
            return self.execute_speak(params)
        elif action == 'wait':
            return self.execute_wait(params)
        else:
            self.get_logger().warn(f'Unknown action: {action}')
            return False

    def execute_navigate(self, params: dict) -> bool:
        """Execute navigation"""
        destination = params.get('destination', '').lower()

        if destination not in self.locations:
            self.speak(f"I don't know where {destination} is.")
            return False

        loc = self.locations[destination]

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = loc['x']
        goal.pose.pose.position.y = loc['y']
        goal.pose.pose.orientation.w = np.cos(loc['yaw'] / 2)
        goal.pose.pose.orientation.z = np.sin(loc['yaw'] / 2)

        self.speak(f"Moving to {destination}")

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.result().accepted:
            return False

        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)

        self.robot_state['location'] = destination
        return True

    def execute_pick(self, params: dict) -> bool:
        """Execute pick action"""
        target_object = params.get('object', '')

        if self.robot_state['holding']:
            self.speak("My gripper is already holding something.")
            return False

        # Check if object is detected
        found = False
        for obj in self.detected_objects:
            if target_object.lower() in obj.class_name.lower():
                found = True
                break

        if not found:
            self.speak(f"I can't see the {target_object}.")
            return False

        self.speak(f"Picking up the {target_object}")
        # In production: call manipulation action server

        # Simulate success
        self.robot_state['holding'] = target_object
        return True

    def execute_place(self, params: dict) -> bool:
        """Execute place action"""
        location = params.get('location', 'here')

        if not self.robot_state['holding']:
            self.speak("I'm not holding anything.")
            return False

        self.speak(f"Placing {self.robot_state['holding']} at {location}")
        # In production: call manipulation action server

        self.robot_state['holding'] = None
        return True

    def execute_look_around(self, params: dict) -> bool:
        """Scan environment"""
        self.speak("Looking around")
        # In production: pan camera or rotate base
        import time
        time.sleep(2.0)
        return True

    def execute_speak(self, params: dict) -> bool:
        """Speak to user"""
        message = params.get('message', '')
        if message:
            self.speak(message)
        return True

    def execute_wait(self, params: dict) -> bool:
        """Wait"""
        duration = params.get('duration', 1.0)
        import time
        time.sleep(duration)
        return True

    def attempt_recovery(self, failed_step: dict):
        """Attempt to recover from failure"""
        # Simplified recovery - in production, use VLM for intelligent recovery
        self.speak("Attempting to recover")
        self.current_task.state = TaskState.EXECUTING

    def speak(self, message: str):
        """Publish TTS message"""
        msg = String()
        msg.data = message
        self.tts_pub.publish(msg)
        self.get_logger().info(f'Speaking: {message}')

    def publish_status(self):
        """Publish task status"""
        if not self.current_task:
            return

        msg = TaskStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.task_goal = self.current_task.goal
        msg.state = self.current_task.state.value
        msg.current_step = self.current_task.current_step
        msg.total_steps = len(self.current_task.plan)
        msg.errors = self.current_task.errors

        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()

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

### Phase 4: Voice Interface (Days 7-8)

#### 4.1 Voice Controller Node

```python
#!/usr/bin/env python3
"""
Voice Controller Node
Handles speech recognition and text-to-speech
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import pyttsx3
import queue
import threading


class VoiceControllerNode(Node):
    def __init__(self):
        super().__init__('voice_controller_node')

        # Parameters
        self.declare_parameter('whisper_model', 'base')
        self.declare_parameter('language', 'en')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('wake_word', 'robot')

        model_size = self.get_parameter('whisper_model').value
        self.language = self.get_parameter('language').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.wake_word = self.get_parameter('wake_word').value.lower()

        # Load Whisper
        self.get_logger().info(f'Loading Whisper model: {model_size}')
        self.whisper = WhisperModel(model_size, device="cuda", compute_type="float16")

        # Initialize TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

        # Audio
        self.audio_queue = queue.Queue()
        self.listening = False
        self.wake_word_detected = False

        # Publishers
        self.command_pub = self.create_publisher(String, '/voice/command', 10)
        self.confirmation_pub = self.create_publisher(String, '/voice/confirmation', 10)
        self.transcript_pub = self.create_publisher(String, '/voice/transcript', 10)

        # Subscribers
        self.tts_sub = self.create_subscription(
            String, '/tts/say',
            self.tts_callback, 10
        )

        self.get_logger().info('Voice Controller Node initialized')

    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
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
        self.get_logger().info('Started listening')

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

    def stop_listening(self):
        """Stop audio capture"""
        self.listening = False
        self.stream.stop()
        self.stream.close()

    def process_audio(self):
        """Process audio continuously"""
        buffer = np.array([], dtype=np.float32)

        while self.listening:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                buffer = np.concatenate([buffer, chunk.flatten()])

                # Process every 3 seconds
                if len(buffer) >= self.sample_rate * 3:
                    self.transcribe_and_process(buffer)
                    buffer = buffer[-int(self.sample_rate * 0.5):]

            except queue.Empty:
                continue

    def transcribe_and_process(self, audio: np.ndarray):
        """Transcribe audio and process commands"""
        segments, _ = self.whisper.transcribe(
            audio,
            beam_size=5,
            language=self.language,
            vad_filter=True
        )

        text = " ".join([seg.text for seg in segments]).strip()
        if not text:
            return

        # Publish transcript
        self.transcript_pub.publish(String(data=text))
        self.get_logger().info(f'Heard: "{text}"')

        text_lower = text.lower()

        # Check for wake word
        if self.wake_word in text_lower:
            self.wake_word_detected = True
            # Extract command after wake word
            parts = text_lower.split(self.wake_word, 1)
            if len(parts) > 1 and parts[1].strip():
                command = parts[1].strip()
                self.command_pub.publish(String(data=command))
            else:
                self.speak("Yes? How can I help?")
            return

        # Check for confirmation responses
        if any(word in text_lower for word in ['yes', 'no', 'cancel', 'proceed', 'stop']):
            self.confirmation_pub.publish(String(data=text))
            return

        # If wake word was recently detected, treat as command
        if self.wake_word_detected:
            self.command_pub.publish(String(data=text))
            self.wake_word_detected = False

    def tts_callback(self, msg: String):
        """Handle TTS requests"""
        self.speak(msg.data)

    def speak(self, text: str):
        """Speak text"""
        self.get_logger().info(f'Speaking: "{text}"')

        # Pause listening during speech
        was_listening = self.listening
        if was_listening:
            self.listening = False

        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

        if was_listening:
            self.listening = True


def main(args=None):
    rclpy.init(args=args)
    node = VoiceControllerNode()

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

### Phase 5: Integration & Testing (Days 9-11)

#### 5.1 Main Launch File

```python
# launch/full_system.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    vlm_api_key = LaunchConfiguration('vlm_api_key', default='')

    pkg_bringup = FindPackageShare('capstone_bringup')
    pkg_perception = FindPackageShare('capstone_perception')
    pkg_planning = FindPackageShare('capstone_planning')
    pkg_voice = FindPackageShare('capstone_voice')

    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('vlm_api_key', default_value=''),

        # Simulation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([pkg_bringup, 'launch', 'simulation.launch.py'])
            ]),
            launch_arguments={'use_sim_time': use_sim_time}.items()
        ),

        # Navigation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([pkg_bringup, 'launch', 'navigation.launch.py'])
            ]),
            launch_arguments={'use_sim_time': use_sim_time}.items()
        ),

        # Perception
        Node(
            package='capstone_perception',
            executable='object_detector',
            name='object_detector',
            parameters=[{
                'use_sim_time': use_sim_time,
                'model_path': 'yolov8n.pt',
                'confidence_threshold': 0.5
            }],
            output='screen'
        ),

        # Task Manager
        Node(
            package='capstone_planning',
            executable='task_manager',
            name='task_manager',
            parameters=[{
                'use_sim_time': use_sim_time,
                'vlm_api_key': vlm_api_key,
                'vlm_model': 'claude-sonnet-4-20250514',
                'require_confirmation': True
            }],
            output='screen'
        ),

        # Voice Controller
        Node(
            package='capstone_voice',
            executable='voice_controller',
            name='voice_controller',
            parameters=[{
                'whisper_model': 'base',
                'language': 'en',
                'wake_word': 'robot'
            }],
            output='screen'
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', PathJoinSubstitution([pkg_bringup, 'rviz', 'capstone.rviz'])]
        ),
    ])
```

#### 5.2 Integration Test Script

```python
#!/usr/bin/env python3
"""
Integration Test Suite for Capstone Project
"""

import rclpy
from rclpy.node import Node
import unittest
import time
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from capstone_interfaces.msg import DetectedObjectArray, TaskStatus


class CapstoneIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('integration_test_node')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_01_perception_pipeline(self):
        """Test object detection is publishing"""
        received = []

        def callback(msg):
            received.append(msg)

        sub = self.node.create_subscription(
            DetectedObjectArray,
            '/perception/detected_objects',
            callback,
            10
        )

        # Wait for messages
        timeout = time.time() + 10.0
        while time.time() < timeout and len(received) < 3:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.node.destroy_subscription(sub)
        self.assertGreater(len(received), 0, "No detections received")

    def test_02_navigation_available(self):
        """Test Nav2 is available"""
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient

        client = ActionClient(self.node, NavigateToPose, 'navigate_to_pose')
        available = client.wait_for_server(timeout_sec=10.0)
        self.assertTrue(available, "Nav2 action server not available")

    def test_03_voice_command_processing(self):
        """Test voice commands are processed"""
        pub = self.node.create_publisher(String, '/voice/command', 10)

        received_status = []

        def status_callback(msg):
            received_status.append(msg)

        sub = self.node.create_subscription(
            TaskStatus,
            '/task/status',
            status_callback,
            10
        )

        # Send test command
        time.sleep(1.0)  # Wait for connections
        pub.publish(String(data="look around"))

        # Wait for task status
        timeout = time.time() + 15.0
        while time.time() < timeout and len(received_status) < 1:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.node.destroy_publisher(pub)
        self.node.destroy_subscription(sub)
        self.assertGreater(len(received_status), 0, "No task status received")

    def test_04_navigation_execution(self):
        """Test navigation to known location"""
        pub = self.node.create_publisher(String, '/voice/command', 10)

        final_status = [None]

        def status_callback(msg):
            final_status[0] = msg

        sub = self.node.create_subscription(
            TaskStatus,
            '/task/status',
            status_callback,
            10
        )

        time.sleep(1.0)
        pub.publish(String(data="go to the kitchen"))

        # Wait for completion
        timeout = time.time() + 60.0
        while time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if final_status[0] and final_status[0].state in ['completed', 'failed']:
                break

        self.node.destroy_publisher(pub)
        self.node.destroy_subscription(sub)

        self.assertIsNotNone(final_status[0], "No final status received")
        self.assertEqual(final_status[0].state, 'completed', "Navigation failed")

    def test_05_multi_step_task(self):
        """Test complete multi-step task"""
        pub = self.node.create_publisher(String, '/voice/command', 10)

        task_states = []

        def status_callback(msg):
            task_states.append(msg.state)

        sub = self.node.create_subscription(
            TaskStatus,
            '/task/status',
            status_callback,
            10
        )

        # Send confirmation automatically
        confirm_pub = self.node.create_publisher(String, '/voice/confirmation', 10)

        time.sleep(1.0)
        pub.publish(String(data="get the cup from the kitchen table"))

        # Wait for planning, then confirm
        time.sleep(5.0)
        confirm_pub.publish(String(data="yes"))

        # Wait for completion
        timeout = time.time() + 120.0
        while time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if task_states and task_states[-1] in ['completed', 'failed']:
                break

        self.node.destroy_publisher(pub)
        self.node.destroy_publisher(confirm_pub)
        self.node.destroy_subscription(sub)

        self.assertIn('executing', task_states, "Task never started executing")
        self.assertEqual(task_states[-1], 'completed', "Multi-step task failed")


if __name__ == '__main__':
    unittest.main()
```

### Phase 6: Documentation & Demo (Days 12-13)

#### 6.1 README Template

```markdown
# Capstone Project: Autonomous Home Assistant Robot

## Overview

This project implements a complete autonomous home assistant robot capable of
understanding natural language commands, perceiving its environment, and
executing complex multi-step tasks.

## Features

- **Voice Control**: Wake word activation with natural language commands
- **Object Detection**: Real-time detection of household objects using YOLOv8
- **Autonomous Navigation**: Nav2-based navigation to named locations
- **Task Planning**: VLM-powered cognitive planning with Claude
- **Manipulation**: Pick and place operations for common objects
- **Error Recovery**: Automatic recovery from execution failures

## Demo Video

[Link to demo video]

## System Architecture

![Architecture Diagram](docs/images/architecture.png)

## Requirements

### Hardware (Simulation)
- NVIDIA GPU with 8GB+ VRAM (for YOLO and Whisper)
- 16GB+ RAM
- Microphone (for voice input)

### Software
- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Fortress
- Python 3.10+

### API Keys
- Anthropic API key (for Claude VLM)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/capstone_robot.git
cd capstone_robot

# Install dependencies
./scripts/setup_workspace.sh

# Build
cd ~/capstone_ws
colcon build --symlink-install
source install/setup.bash

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

### Launch Full System

```bash
ros2 launch capstone_bringup full_system.launch.py \
  vlm_api_key:=$ANTHROPIC_API_KEY
```

### Voice Commands

Say "Robot" followed by your command:

- "Robot, go to the kitchen"
- "Robot, find the water bottle"
- "Robot, bring me the cup from the table"
- "Robot, clean up the living room"

### Manual Testing

```bash
# Send command via topic
ros2 topic pub /voice/command std_msgs/String "data: 'go to the kitchen'"

# Monitor status
ros2 topic echo /task/status
```

## Testing

```bash
# Run integration tests
python3 tests/test_integration.py

# Run specific test
python3 -m pytest tests/test_navigation.py -v
```

## Project Structure

```
capstone_ws/
├── src/
│   ├── capstone_bringup/      # Launch files
│   ├── capstone_perception/   # Object detection
│   ├── capstone_planning/     # Task planning
│   ├── capstone_voice/        # Voice interface
│   └── capstone_interfaces/   # Custom messages
├── docs/                       # Documentation
└── tests/                      # Test suite
```

## Demonstration Scenarios

### Scenario 1: Fetch Object
"Robot, bring me the water bottle from the kitchen"
- Robot navigates to kitchen
- Detects water bottle
- Picks up bottle
- Returns to user
- Hands over bottle

### Scenario 2: Find Object
"Robot, find my keys"
- Robot searches rooms systematically
- Reports location when found
- Or reports "not found" after searching

### Scenario 3: Cleanup Task
"Robot, put all the cups in the kitchen sink"
- Robot identifies cups in the room
- Picks each cup
- Navigates to kitchen
- Places in sink area
- Repeats until complete

## Known Limitations

1. Manipulation is simulated (no real MoveIt2 integration)
2. Object recognition limited to COCO dataset classes
3. No multi-floor navigation
4. Voice recognition may struggle in noisy environments

## Future Improvements

- [ ] Real hardware deployment
- [ ] MoveIt2 integration for actual manipulation
- [ ] Multi-floor navigation
- [ ] Improved object tracking across frames
- [ ] Web-based monitoring dashboard

## Acknowledgments

- ROS 2 and Nav2 teams
- Ultralytics (YOLOv8)
- OpenAI (Whisper)
- Anthropic (Claude)

## License

MIT License
```

---

## Evaluation Rubric

### Technical Implementation (60%)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Perception | 15 | Object detection accuracy and reliability |
| Navigation | 15 | Successful autonomous navigation |
| Planning | 15 | Task decomposition and execution |
| Voice Interface | 10 | Speech recognition and TTS |
| Integration | 5 | Smooth system integration |

### Documentation (20%)

| Criterion | Points | Description |
|-----------|--------|-------------|
| README | 5 | Clear setup and usage instructions |
| Architecture | 5 | System design documentation |
| Code Comments | 5 | Well-documented code |
| Demo Video | 5 | Clear demonstration of capabilities |

### Demonstration (20%)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Basic Tasks | 10 | Completion of required scenarios |
| Robustness | 5 | Error handling and recovery |
| Stretch Goals | 5 | Additional features implemented |

---

## Tips for Success

### 1. Start Simple
```
Week 1: Get simulation running with teleop
Week 2: Add perception (static object detection)
Week 3: Integrate navigation (go to fixed points)
Week 4: Add voice commands (simple keywords)
Week 5: Integrate VLM planning
Week 6: Polish and document
```

### 2. Test Incrementally
- Test each component in isolation first
- Add integration tests as you connect components
- Use rosbag to record and replay scenarios

### 3. Common Pitfalls to Avoid
- Don't try to implement everything at once
- Don't skip documentation until the end
- Don't ignore error handling
- Don't forget to test failure cases

### 4. Debugging Tools
```bash
# View all topics
ros2 topic list

# Monitor specific topic
ros2 topic echo /perception/detected_objects

# Check node status
ros2 node list
ros2 node info /task_manager

# Record session
ros2 bag record -a

# View TF tree
ros2 run tf2_tools view_frames
```

---

## Summary

Congratulations on completing the Physical AI & Humanoid Robotics curriculum!

Through this capstone project, you have:

- **Integrated** multiple ROS 2 subsystems into a cohesive robot
- **Implemented** perception, planning, and control pipelines
- **Applied** Vision-Language Models for intelligent task planning
- **Built** a voice-controlled autonomous robot assistant
- **Demonstrated** end-to-end task execution capabilities
- **Documented** a complete robotics project portfolio-ready

### What You've Learned Across All Modules

```
┌─────────────────────────────────────────────────────────────────┐
│              PHYSICAL AI CURRICULUM SUMMARY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MODULE 1: FOUNDATIONS                                          │
│  ├── Physical AI concepts and embodied intelligence            │
│  ├── ROS 2 fundamentals and robot control                      │
│  └── Sensors, actuators, and robot architectures               │
│                                                                 │
│  MODULE 2: SIMULATION                                           │
│  ├── Gazebo environment and physics simulation                 │
│  ├── Unity robotics and visual simulation                      │
│  └── Testing and validation in simulation                      │
│                                                                 │
│  MODULE 3: NVIDIA ISAAC                                         │
│  ├── Isaac Sim for photorealistic simulation                   │
│  ├── Isaac ROS for GPU-accelerated perception                  │
│  └── Nav2 integration for autonomous navigation                │
│                                                                 │
│  MODULE 4: VISION-LANGUAGE-ACTION                               │
│  ├── Voice-to-Action with speech recognition                   │
│  ├── VLM-based cognitive planning                              │
│  └── Complete system integration (CAPSTONE)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Next Steps

1. **Deploy on Real Hardware**: Take your simulation-proven system to a physical robot
2. **Expand Capabilities**: Add more sophisticated manipulation, multi-robot coordination
3. **Explore Research**: Dive deeper into embodied AI, foundation models for robotics
4. **Build Your Portfolio**: Document and share your projects on GitHub
5. **Join the Community**: Contribute to open-source robotics projects

---

## Further Resources

### Competitions & Challenges
- [RoboCup@Home](https://athome.robocup.org/) - Domestic service robot competition
- [Amazon Picking Challenge](https://www.amazonrobotics.com/) - Manipulation challenges
- [DARPA Robotics Challenge](https://www.darpa.mil/) - Advanced robotics

### Research Labs
- [Stanford AI Lab](https://ai.stanford.edu/) - Mobile manipulation research
- [Berkeley AI Research](https://bair.berkeley.edu/) - Robot learning
- [MIT CSAIL](https://www.csail.mit.edu/) - Robotics and AI

### Industry
- [NVIDIA Isaac](https://developer.nvidia.com/isaac-sdk) - Robotics platform
- [Boston Dynamics](https://www.bostondynamics.com/) - Advanced robotics
- [Open Robotics](https://www.openrobotics.org/) - ROS development

---

**You've completed the Physical AI & Humanoid Robotics Textbook!**

Go build amazing robots that help people. The future of Physical AI is in your hands.
