---
sidebar_position: 1
---

# Physical AI & Humanoid Robotics

## Welcome

This course introduces **Physical AI**—AI systems that operate in the physical world and comprehend physical laws. We'll bridge the gap between digital AI models and embodied intelligence, enabling you to design, simulate, and deploy humanoid robots capable of natural human interactions.

**Focus & Theme:** AI Systems in Physical World. Embodied Intelligence.

**Goal:** Bridging the gap between digital brain and physical body. Students apply their AI knowledge to control Humanoid Robots in simulated and real-world environments.

---

## Why Physical AI Matters

Humanoid robots are poised to excel in our human-centered world because they share our physical form and can be trained with abundant data from interacting in human environments. This represents a significant transition from AI models confined to digital environments to **embodied intelligence** that operates in physical space.

Unlike digital AI that processes text, images, or code in isolation, Physical AI systems must:
- Understand physics, gravity, and collision dynamics
- Navigate unstructured human environments
- Manipulate real objects with tactile feedback
- Balance and move through dynamic spaces
- Interact naturally with humans through voice, gesture, and vision

The future of work will be a partnership between people, intelligent agents (AI software), and robots. This course prepares you for that future.

---

## Learning Outcomes

After completing this course, you will be able to:

1. **Understand Physical AI principles** and the concept of embodied intelligence
2. **Master ROS 2** (Robot Operating System) for robotic control and communication
3. **Simulate robots** using Gazebo for physics and Unity for high-fidelity visualization
4. **Develop with NVIDIA Isaac** AI robot platform for perception and navigation
5. **Design humanoid robots** for natural human-robot interactions
6. **Integrate GPT models** for conversational robotics (Voice-to-Action, cognitive planning)

---

## Course Overview: 4 Modules

### Module 1: The Robotic Nervous System (ROS 2)

**Focus:** Middleware for robot control

- ROS 2 Nodes, Topics, and Services
- Bridging Python Agents to ROS controllers using rclpy
- Understanding URDF (Unified Robot Description Format) for humanoids
- Building ROS 2 packages with Python

### Module 2: The Digital Twin (Gazebo & Unity)

**Focus:** Physics simulation and environment building

- Simulating physics, gravity, and collisions in Gazebo
- High-fidelity rendering and human-robot interaction in Unity
- Simulating sensors: LiDAR, Depth Cameras, and IMUs
- URDF and SDF robot description formats

### Module 3: The AI-Robot Brain (NVIDIA Isaac™)

**Focus:** Advanced perception and training

- NVIDIA Isaac Sim: Photorealistic simulation and synthetic data generation
- Isaac ROS: Hardware-accelerated VSLAM (Visual SLAM) and navigation
- Nav2: Path planning for bipedal humanoid movement
- AI-powered perception and manipulation
- Sim-to-real transfer techniques

### Module 4: Vision-Language-Action (VLA)

**Focus:** The convergence of LLMs and Robotics

- Voice-to-Action: Using OpenAI Whisper for voice commands
- Cognitive Planning: Using LLMs to translate natural language into ROS 2 actions
- Multi-modal interaction: Speech, gesture, and vision
- Capstone Project: The Autonomous Humanoid

---

## 13-Week Curriculum Breakdown

| Week | Module | Topic |
|------|---------|-------|
| 1 | ROS 2 | Foundations of Physical AI |
| 2 | ROS 2 | Physical AI Landscape & Sensors |
| 3 | ROS 2 | ROS 2 Architecture & Core Concepts |
| 4 | ROS 2 | Nodes, Topics, and Services |
| 5 | ROS 2 | URDF & Robot Description |
| 6 | Gazebo | Gazebo Simulation Setup |
| 7 | Unity | Unity for Robot Visualization |
| 8 | Isaac | Isaac Sim & SDK |
| 9 | Isaac | Isaac ROS & Perception |
| 10 | Isaac | Navigation 2 & Path Planning |
| 11 | VLA | Voice-to-Action Integration |
| 12 | VLA | Cognitive Planning with LLMs |
| 13 | VLA | Capstone: Autonomous Humanoid |

---

## Hardware Requirements

This course sits at the intersection of three heavy computational loads: **Physics Simulation**, **Visual Perception**, and **Generative AI**. The hardware requirements reflect this complexity.

### Required: Digital Twin Workstation

To run NVIDIA Isaac Sim, Gazebo, and VLA models, you need:

- **GPU:** NVIDIA RTX 4070 Ti (12GB VRAM) or higher
- **CPU:** Intel Core i7 (13th Gen+) or AMD Ryzen 9
- **RAM:** 64 GB DDR5 (32 GB is minimum, may crash during complex scenes)
- **OS:** Ubuntu 22.04 LTS (recommended for ROS 2 compatibility)

### Optional: Physical AI Edge Kit (~$700)

For deploying code to real hardware (Weeks 11-13):

- **The Brain:** NVIDIA Jetson Orin Nano (8GB) - $249
- **The Eyes:** Intel RealSense D435i (RGB + Depth) - $349
- **The Ears:** ReSpeaker USB Mic Array v2.0 - $69
- **Storage:** 128GB microSD card + accessories - $30

### Robot Lab Options

Three tiers depending on budget:

1. **Proxy Approach (Budget):** Unitree Go2 Edu ($1,800-$3,000) - quadruped, excellent ROS 2 support
2. **Miniature Humanoid:** Unitree G1 (~$16k) or Hiwonder TonyPi Pro (~$600, limited to kinematics)
3. **Premium:** Unitree G1 Humanoid (full sim-to-real capability)

*Note: See Week 1-2 content for detailed hardware specifications and cloud alternatives.*

---

## Assessments

Throughout this course, you'll complete:

- **ROS 2 package development project** (Module 1)
- **Gazebo simulation implementation** (Module 2)
- **Isaac-based perception pipeline** (Module 3)
- **Capstone:** Simulated humanoid robot with conversational AI (Module 4)

The capstone project integrates everything you've learned: a simulated humanoid that receives a voice command, plans a path, navigates obstacles using VSLAM, identifies an object using computer vision, and manipulates it with ROS 2.

---

## Getting Started

Ready to begin? Start with **Week 1: Foundations of Physical AI** in the sidebar navigation.

For detailed setup instructions, see the course requirements in Weeks 1-2 for your workstation configuration.

**Estimated Course Duration:** 13 weeks

**Prerequisites:** Python programming, basic Linux command line, familiarity with machine learning concepts
