---
sidebar_position: 6
title: "Week 6 - Gazebo Simulation"
description: "Master Gazebo physics simulation for robotics development and testing"
keywords: [Gazebo, physics simulation, robotics, virtual testing, ROS 2, digital twin]
last_updated: "2025-12-29"
estimated_reading_time: 20
---

# Week 6: Gazebo Simulation

Welcome to Module 2! In this chapter, we transition from understanding robot descriptions (URDF) to bringing robots to life in a physics-accurate simulation environment. Gazebo is the industry-standard simulator that enables you to test robot behaviors without risking expensive hardware.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the role of simulation in robotics development
- Install and configure Gazebo for ROS 2
- Create and customize simulation worlds
- Spawn URDF robots into Gazebo environments
- Configure physics engines and simulation parameters
- Add sensors to simulated robots and visualize their output

---

## Why Simulation Matters

### The Cost of Physical Testing

Developing robots in the real world is expensive and risky:

| Challenge | Real World | Simulation |
|-----------|------------|------------|
| **Hardware Cost** | $10,000 - $1,000,000+ | $0 (software) |
| **Testing Time** | Hours per iteration | Seconds per iteration |
| **Safety Risk** | Potential damage | Zero risk |
| **Environment Control** | Limited | Complete control |
| **Reproducibility** | Difficult | Perfect repeatability |
| **Parallelization** | One robot | Thousands of instances |

### The Digital Twin Concept

A **digital twin** is a virtual replica of a physical system that mirrors its real-world counterpart in real-time. In robotics, this means:

1. **Design Phase**: Test robot designs before manufacturing
2. **Development Phase**: Develop and debug algorithms safely
3. **Deployment Phase**: Validate behaviors before real-world execution
4. **Operation Phase**: Monitor and predict real robot behavior

```
┌─────────────────────────────────────────────────────────────┐
│                    Digital Twin Workflow                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
│   │  Design  │───▶│ Simulate │───▶│ Validate in Real │     │
│   │  (URDF)  │    │ (Gazebo) │    │     World        │     │
│   └──────────┘    └──────────┘    └──────────────────┘     │
│        │               │                   │                │
│        │               ▼                   │                │
│        │        ┌──────────┐               │                │
│        └───────▶│  Refine  │◀──────────────┘                │
│                 └──────────┘                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Introduction to Gazebo

### What is Gazebo?

**Gazebo** is an open-source 3D robotics simulator that provides:

- **Physics Simulation**: Accurate rigid body dynamics, collision detection
- **Sensor Simulation**: LIDAR, cameras, IMUs, GPS, force/torque sensors
- **Environment Modeling**: Indoor/outdoor worlds with lighting and terrain
- **Robot Models**: Library of pre-built robots and objects
- **ROS 2 Integration**: Seamless communication with ROS 2 nodes

### Gazebo Classic vs Gazebo (Ignition)

The Gazebo ecosystem has evolved significantly:

| Feature | Gazebo Classic | Gazebo (Modern) |
|---------|---------------|-----------------|
| **Version** | Gazebo 11 (EOL 2025) | Gazebo Harmonic (2024+) |
| **Architecture** | Monolithic | Modular (Ignition libraries) |
| **Rendering** | OGRE 1.x | OGRE 2.x (PBR support) |
| **Physics** | ODE default | Multiple engines (DART, Bullet, TPE) |
| **ROS 2 Support** | ros_gazebo | ros_gz (native) |
| **Recommendation** | Legacy projects | New projects |

:::tip Recommendation
For new ROS 2 projects, use **Gazebo Harmonic** or later. This textbook focuses on the modern Gazebo (formerly Ignition Gazebo).
:::

### Gazebo Architecture

Modern Gazebo is built on a collection of **Ignition libraries**:

```
┌─────────────────────────────────────────────────────────────┐
│                     Gazebo Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   gz-sim    │  │  gz-gui     │  │    gz-transport     │ │
│  │ (Simulator) │  │ (Interface) │  │  (Communication)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────────▼──────────┐ │
│  │ gz-physics  │  │ gz-rendering│  │     gz-msgs         │ │
│  │  (Dynamics) │  │  (Graphics) │  │    (Messages)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  gz-math    │  │ gz-sensors  │  │     gz-plugin       │ │
│  │ (Utilities) │  │  (Sensors)  │  │   (Extensions)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Installing Gazebo for ROS 2

### Prerequisites

Ensure you have ROS 2 Humble (or later) installed. Then install Gazebo:

```bash
# Install Gazebo Harmonic (for ROS 2 Humble/Iron/Jazzy)
sudo apt-get update
sudo apt-get install ros-humble-ros-gz

# Verify installation
gz sim --version
```

### ROS-Gazebo Bridge

The `ros_gz` package provides bidirectional communication between ROS 2 and Gazebo:

```bash
# Install the bridge packages
sudo apt-get install ros-humble-ros-gz-bridge
sudo apt-get install ros-humble-ros-gz-sim
sudo apt-get install ros-humble-ros-gz-image
```

### Testing the Installation

Launch a simple world to verify everything works:

```bash
# Terminal 1: Launch Gazebo with an empty world
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:="empty.sdf"

# Terminal 2: List Gazebo topics visible in ROS 2
ros2 topic list
```

---

## Understanding SDF World Files

### What is SDF?

**Simulation Description Format (SDF)** is an XML format for describing:

- Simulation worlds (environments)
- Robot models (alternative to URDF)
- Lights, physics, and plugins

While URDF describes robots, SDF describes entire simulation scenarios.

### Basic World Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="my_first_world">

    <!-- Physics Configuration -->
    <physics name="1ms" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

### Key World Elements

| Element | Purpose | Example |
|---------|---------|---------|
| `<physics>` | Simulation stepping and engine | DART, Bullet, ODE |
| `<light>` | Scene illumination | Sun, point lights, spotlights |
| `<model>` | Objects in the world | Ground, obstacles, robots |
| `<plugin>` | Extend functionality | Sensors, controllers |
| `<scene>` | Rendering settings | Ambient light, shadows, sky |

---

## Creating Your First Simulation World

### Step 1: Create a Package for Worlds

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake my_robot_simulation \
  --dependencies ros_gz_sim

# Create directories
mkdir -p my_robot_simulation/worlds
mkdir -p my_robot_simulation/launch
mkdir -p my_robot_simulation/models
```

### Step 2: Create a Custom World

Create `worlds/warehouse.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="warehouse">

    <!-- Physics: 1kHz simulation rate -->
    <physics name="1kHz" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Plugins for ROS 2 integration -->
    <plugin
      filename="gz-sim-physics-system"
      name="gz::sim::systems::Physics">
    </plugin>
    <plugin
      filename="gz-sim-sensors-system"
      name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin
      filename="gz-sim-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster">
    </plugin>
    <plugin
      filename="gz-sim-user-commands-system"
      name="gz::sim::systems::UserCommands">
    </plugin>

    <!-- Sunlight -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>50 50</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>50 50</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Warehouse Walls -->
    <model name="wall_north">
      <static>true</static>
      <pose>0 10 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>20 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>20 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacle: Box -->
    <model name="obstacle_box">
      <pose>3 2 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>0.167</ixx>
            <iyy>0.167</iyy>
            <izz>0.167</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.4 0.1 1</ambient>
            <diffuse>0.8 0.4 0.1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

### Step 3: Create a Launch File

Create `launch/warehouse.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Paths
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim')
    pkg_my_simulation = FindPackageShare('my_robot_simulation')

    # World file path
    world_file = PathJoinSubstitution([
        pkg_my_simulation, 'worlds', 'warehouse.sdf'
    ])

    # Launch Gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': ['-r ', world_file],
        }.items()
    )

    return LaunchDescription([
        gz_sim,
    ])
```

---

## Spawning Robots into Gazebo

### Converting URDF to Gazebo-Compatible Format

Your URDF from Week 5 needs additional Gazebo-specific elements:

```xml
<robot name="my_humanoid">
  <!-- Standard URDF content -->
  <link name="base_link">
    <!-- ... -->
  </link>

  <!-- Gazebo-specific elements -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <!-- Gazebo plugins for control -->
  <gazebo>
    <plugin filename="gz-sim-joint-state-publisher-system"
            name="gz::sim::systems::JointStatePublisher">
    </plugin>
  </gazebo>
</robot>
```

### Spawning with ros_gz

Create a launch file to spawn your robot:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Get URDF file
    urdf_file = os.path.join(
        FindPackageShare('my_robot_description').find('my_robot_description'),
        'urdf', 'robot.urdf.xacro'
    )

    robot_description = Command(['xacro ', urdf_file])

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5',
        ],
        output='screen',
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen',
    )

    return LaunchDescription([
        robot_state_publisher,
        spawn_robot,
    ])
```

---

## Physics Configuration

### Understanding Physics Engines

Gazebo supports multiple physics engines:

| Engine | Strengths | Use Cases |
|--------|-----------|-----------|
| **DART** | Accurate dynamics, soft contacts | Humanoids, manipulation |
| **Bullet** | Fast, game-oriented | Large environments, many objects |
| **ODE** | Mature, well-tested | General robotics |
| **TPE** | Trivial physics | Sensor testing without dynamics |

### Configuring Physics Parameters

```xml
<physics name="high_accuracy" type="dart">
  <!-- Simulation step size (seconds) -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time factor: 1.0 = real-time, 0.5 = half speed -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Update rate (Hz) -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- DART-specific settings -->
  <dart>
    <collision_detector>fcl</collision_detector>
    <solver>
      <solver_type>dantzig</solver_type>
    </solver>
  </dart>
</physics>
```

### Tuning for Performance vs Accuracy

```
┌─────────────────────────────────────────────────────────────┐
│              Physics Tuning Trade-offs                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Accuracy ◄──────────────────────────────────────► Speed    │
│                                                             │
│  ┌─────────────────┐              ┌─────────────────┐       │
│  │ Small step size │              │ Large step size │       │
│  │ (0.0001s)       │              │ (0.01s)         │       │
│  │ More iterations │              │ Fewer iterations│       │
│  │ Complex solver  │              │ Simple solver   │       │
│  └─────────────────┘              └─────────────────┘       │
│                                                             │
│  Use for:                         Use for:                  │
│  - Contact-rich tasks             - Path planning           │
│  - Manipulation                   - Large-scale simulation  │
│  - Humanoid balance               - Sensor testing          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Adding Sensors in Simulation

### Common Simulated Sensors

Gazebo can simulate all sensors we studied in Week 2:

```xml
<!-- LIDAR Sensor -->
<sensor name="lidar" type="gpu_lidar">
  <pose>0 0 0.5 0 0 0</pose>
  <topic>scan</topic>
  <update_rate>10</update_rate>
  <lidar>
    <scan>
      <horizontal>
        <samples>640</samples>
        <resolution>1</resolution>
        <min_angle>-1.5708</min_angle>
        <max_angle>1.5708</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle>
        <max_angle>0.261799</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100</max>
    </range>
  </lidar>
</sensor>

<!-- RGB Camera -->
<sensor name="camera" type="camera">
  <pose>0 0 0.3 0 0 0</pose>
  <topic>image</topic>
  <update_rate>30</update_rate>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
</sensor>

<!-- IMU Sensor -->
<sensor name="imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <topic>imu</topic>
  <imu>
    <angular_velocity>
      <x><noise type="gaussian"><mean>0</mean><stddev>0.01</stddev></noise></x>
      <y><noise type="gaussian"><mean>0</mean><stddev>0.01</stddev></noise></y>
      <z><noise type="gaussian"><mean>0</mean><stddev>0.01</stddev></noise></z>
    </angular_velocity>
    <linear_acceleration>
      <x><noise type="gaussian"><mean>0</mean><stddev>0.1</stddev></noise></x>
      <y><noise type="gaussian"><mean>0</mean><stddev>0.1</stddev></noise></y>
      <z><noise type="gaussian"><mean>0</mean><stddev>0.1</stddev></noise></z>
    </linear_acceleration>
  </imu>
</sensor>
```

### Bridging Sensor Data to ROS 2

Use the `ros_gz_bridge` to expose Gazebo topics in ROS 2:

```python
# In your launch file
bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
        '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
        '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
    ],
    output='screen'
)
```

---

## Practical Exercise: Complete Simulation Setup

### Goal

Create a simulation with:
1. A warehouse environment
2. A mobile robot with LIDAR
3. ROS 2 integration for sensor data

### Step-by-Step

1. **Build the workspace**:
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_simulation
source install/setup.bash
```

2. **Launch the simulation**:
```bash
ros2 launch my_robot_simulation warehouse.launch.py
```

3. **Verify sensor data in ROS 2**:
```bash
# List all topics
ros2 topic list

# Echo LIDAR data
ros2 topic echo /scan

# Visualize in RViz2
rviz2
```

4. **Control the robot**:
```bash
# Send velocity commands
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5}, angular: {z: 0.1}}"
```

---

## Debugging Simulation Issues

### Common Problems and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Robot falls through ground | Missing collision | Add `<collision>` to all links |
| Robot explodes on spawn | Overlapping links | Check joint origins, spawn higher |
| No sensor data | Missing plugin | Add sensor system plugin to world |
| Slow simulation | High real-time factor | Reduce step size or simplify models |
| Bridge not working | Topic mismatch | Verify topic names and message types |

### Useful Debugging Commands

```bash
# List all Gazebo topics
gz topic -l

# Echo a Gazebo topic
gz topic -e -t /world/warehouse/model/my_robot/link/base_link/sensor/lidar/scan

# Check simulation statistics
gz stats

# Pause/unpause simulation
gz service -s /world/warehouse/control --reqtype gz.msgs.WorldControl \
  --reptype gz.msgs.Boolean --req 'pause: true'
```

---

## Summary

In this chapter, you learned:

- **Why simulation matters**: Cost savings, safety, and accelerated development
- **Gazebo fundamentals**: Architecture, SDF format, physics engines
- **World creation**: Building custom environments with objects and lighting
- **Robot integration**: Spawning URDF robots and configuring Gazebo plugins
- **Sensor simulation**: Adding LIDAR, cameras, and IMUs with realistic noise
- **ROS 2 bridging**: Connecting Gazebo to your ROS 2 nodes

Simulation is the foundation of modern robotics development. Before any algorithm touches real hardware, it should be thoroughly tested in simulation.

---

## Further Reading

- [Gazebo Documentation](https://gazebosim.org/docs) - Official Gazebo tutorials and API reference
- [ros_gz Repository](https://github.com/gazebosim/ros_gz) - ROS 2 integration packages
- [SDF Specification](http://sdformat.org/spec) - Complete SDF format reference
- [DART Physics Engine](https://dartsim.github.io/) - Documentation for DART dynamics
- [Open Robotics Models](https://app.gazebosim.org/fuel) - Free robot and world models

---

## Next Week Preview

In **Week 7**, we'll explore **Unity for Robotics** - a powerful alternative for:
- Photorealistic rendering and synthetic data generation
- Machine learning training environments
- Human-robot interaction scenarios
- Cross-platform deployment
