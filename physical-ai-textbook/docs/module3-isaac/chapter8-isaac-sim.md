---
sidebar_position: 8
title: "Chapter 8 - NVIDIA Isaac Sim"
description: "Master GPU-accelerated robotics simulation with NVIDIA Isaac Sim and Omniverse"
keywords: [NVIDIA Isaac, Isaac Sim, Omniverse, GPU simulation, synthetic data, AI robotics, digital twin]
last_updated: "2025-12-29"
estimated_reading_time: 25
---

# Chapter 8: NVIDIA Isaac Sim

Welcome to Module 3! We now enter the realm of GPU-accelerated simulation with NVIDIA Isaac Sim. Built on the Omniverse platform, Isaac Sim provides unparalleled physics accuracy, photorealistic rendering, and deep integration with NVIDIA's AI ecosystem.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the advantages of GPU-accelerated simulation
- Install and configure NVIDIA Isaac Sim
- Navigate the Isaac Sim interface and workflow
- Create simulation environments using Omniverse USD
- Import robots from URDF and configure physics
- Generate synthetic data with Omniverse Replicator
- Connect Isaac Sim to ROS 2 for robot control

---

## Why Isaac Sim?

### The GPU Simulation Revolution

Traditional simulators run physics on the CPU, limiting parallelism and speed. Isaac Sim leverages NVIDIA GPUs for:

| Capability | CPU Simulation | Isaac Sim (GPU) |
|------------|---------------|-----------------|
| **Physics Parallelism** | Hundreds of contacts | Millions of contacts |
| **Rendering** | Rasterization | Real-time ray tracing |
| **ML Training** | Sequential | Thousands of parallel envs |
| **Sensor Simulation** | Approximations | Physically accurate |
| **Synthetic Data** | Limited scale | Millions of images/hour |

### Isaac Sim in the NVIDIA Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│              NVIDIA Robotics Ecosystem                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Isaac Sim                          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────┐   │   │
│  │  │  PhysX 5  │  │   RTX     │  │  Omniverse    │   │   │
│  │  │ (Physics) │  │(Rendering)│  │  Replicator   │   │   │
│  │  └─────┬─────┘  └─────┬─────┘  └───────┬───────┘   │   │
│  └────────┼──────────────┼────────────────┼───────────┘   │
│           │              │                │                │
│  ┌────────▼──────────────▼────────────────▼───────────┐   │
│  │                 Omniverse Platform                  │   │
│  │         (USD, Nucleus, Connectors)                  │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           │                                 │
│  ┌────────────────────────▼────────────────────────────┐   │
│  │              Isaac ROS / Robot Deployment            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │  VSLAM   │  │ nvblox   │  │  Isaac Manipulator│  │   │
│  │  │          │  │ (3D Map) │  │                   │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Differentiators

1. **PhysX 5**: GPU-accelerated physics with accurate contact dynamics
2. **RTX Rendering**: Real-time ray tracing for photorealistic sensors
3. **Omniverse USD**: Universal Scene Description for collaborative workflows
4. **Replicator**: Domain randomization and synthetic data at scale
5. **Isaac ROS**: Direct integration with ROS 2 perception packages

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 2070 | RTX 3080 / RTX 4080 or higher |
| **VRAM** | 8 GB | 16+ GB |
| **CPU** | Intel i7 / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| **RAM** | 32 GB | 64 GB |
| **Storage** | 50 GB SSD | 100+ GB NVMe SSD |
| **OS** | Ubuntu 20.04/22.04 | Ubuntu 22.04 |

### Software Requirements

- NVIDIA Driver 525.60+ (535+ recommended)
- CUDA 11.8 or 12.x
- Docker (for containerized deployment)
- ROS 2 Humble (for ROS integration)

---

## Installing Isaac Sim

### Option 1: Omniverse Launcher (Recommended)

```bash
# 1. Download Omniverse Launcher
# Visit: https://www.nvidia.com/en-us/omniverse/download/

# 2. Install the launcher
chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage

# 3. Install Isaac Sim from the Exchange tab
# Search for "Isaac Sim" and click Install
```

### Option 2: Docker Container

```bash
# Pull the Isaac Sim container
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1

# Run with GPU access
docker run --name isaac-sim --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  --rm --network=host \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  nvcr.io/nvidia/isaac-sim:2023.1.1
```

### Option 3: Python Package (IsaacLab)

```bash
# Clone Isaac Lab (formerly Isaac Orbit)
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Create conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install Isaac Sim Python packages
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \
  isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app \
  --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab
pip install -e .
```

### Verifying Installation

```bash
# Launch Isaac Sim
~/.local/share/ov/pkg/isaac_sim-2023.1.1/isaac-sim.sh

# Or via Python
python -c "from isaacsim import SimulationApp; app = SimulationApp(); print('Isaac Sim OK')"
```

---

## Isaac Sim Interface Overview

### Main Windows

```
┌─────────────────────────────────────────────────────────────┐
│  File  Edit  View  Window  Tools  Help                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐│
│  │                                                         ││
│  │                    3D Viewport                          ││
│  │              (Scene visualization)                      ││
│  │                                                         ││
│  ├──────────────────────┬──────────────────────────────────┤│
│  │    Stage Panel       │       Property Panel             ││
│  │  (Scene hierarchy)   │    (Selected object props)       ││
│  │                      │                                  ││
│  │  World               │  Transform                       ││
│  │  ├── Environment     │    Position: 0, 0, 0            ││
│  │  ├── Ground          │    Rotation: 0, 0, 0            ││
│  │  └── Robot           │    Scale: 1, 1, 1               ││
│  │      ├── base_link   │                                  ││
│  │      └── joints      │  Physics                         ││
│  │                      │    Mass: 10.0 kg                 ││
│  └──────────────────────┴──────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Content Browser  │  Console  │  Timeline  │  Physics   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Key Panels

| Panel | Purpose |
|-------|---------|
| **Stage** | USD scene hierarchy (prims, transforms) |
| **Property** | Edit selected object attributes |
| **Content Browser** | Navigate assets and files |
| **Console** | Python scripting and logs |
| **Timeline** | Animation and simulation playback |
| **Physics** | Physics settings and debugging |

### Viewport Navigation

| Action | Control |
|--------|---------|
| **Orbit** | Alt + Left Mouse |
| **Pan** | Alt + Middle Mouse |
| **Zoom** | Alt + Right Mouse / Scroll |
| **Focus** | F (on selected object) |
| **Frame All** | A |

---

## Understanding USD (Universal Scene Description)

### What is USD?

**Universal Scene Description (USD)** is Pixar's open-source scene format that enables:

- **Composition**: Layer multiple files non-destructively
- **Collaboration**: Multiple artists work on same scene
- **Scalability**: Handle massive scenes efficiently
- **Interoperability**: Exchange data between tools

### USD in Robotics

```
┌─────────────────────────────────────────────────────────────┐
│                    USD Composition                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                       │
│  │  robot.usd      │  ◄── Robot model (from URDF)          │
│  └────────┬────────┘                                       │
│           │                                                 │
│  ┌────────▼────────┐                                       │
│  │  warehouse.usd  │  ◄── Environment layout               │
│  └────────┬────────┘                                       │
│           │                                                 │
│  ┌────────▼────────┐                                       │
│  │  sensors.usd    │  ◄── Sensor configurations            │
│  └────────┬────────┘                                       │
│           │                                                 │
│  ┌────────▼────────┐                                       │
│  │  physics.usd    │  ◄── Physics materials & settings     │
│  └────────┬────────┘                                       │
│           │                                                 │
│  ┌────────▼────────┐                                       │
│  │  scene.usd      │  ◄── Final composed scene             │
│  └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Basic USD Operations in Python

```python
from pxr import Usd, UsdGeom, UsdPhysics, Gf

# Create a new USD stage
stage = Usd.Stage.CreateNew("my_scene.usd")

# Set up axis and units
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Create a ground plane
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
ground.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
ground.CreateFaceVertexCountsAttr([4])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

# Add physics collision
collision_api = UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim())

# Create a box
box = UsdGeom.Cube.Define(stage, "/World/Box")
box.CreateSizeAttr(1.0)
box.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))

# Add rigid body physics
rigid_body = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
UsdPhysics.CollisionAPI.Apply(box.GetPrim())

# Save
stage.GetRootLayer().Save()
```

---

## Creating Your First Isaac Sim World

### Step 1: Create a New Stage

```python
from isaacsim import SimulationApp

# Initialize Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, GroundPlane
from omni.isaac.core.utils.prims import create_prim
import numpy as np

# Create simulation world
world = World(stage_units_in_meters=1.0)

# Add ground plane
world.scene.add_default_ground_plane()
```

### Step 2: Add Objects

```python
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid

# Add a dynamic box (affected by physics)
dynamic_box = world.scene.add(
    DynamicCuboid(
        prim_path="/World/DynamicBox",
        name="dynamic_box",
        position=np.array([0.0, 0.0, 1.0]),
        scale=np.array([0.5, 0.5, 0.5]),
        color=np.array([1.0, 0.0, 0.0]),  # Red
        mass=1.0
    )
)

# Add a static obstacle
static_obstacle = world.scene.add(
    VisualCuboid(
        prim_path="/World/Obstacle",
        name="obstacle",
        position=np.array([2.0, 0.0, 0.5]),
        scale=np.array([1.0, 1.0, 1.0]),
        color=np.array([0.0, 0.0, 1.0])  # Blue
    )
)
```

### Step 3: Configure Lighting

```python
from omni.isaac.core.utils.prims import create_prim
from pxr import UsdLux

# Create dome light for ambient illumination
create_prim("/World/DomeLight", "DomeLight")
dome_light = UsdLux.DomeLight.Get(world.stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000)

# Create directional light (sun)
create_prim("/World/SunLight", "DistantLight")
sun_light = UsdLux.DistantLight.Get(world.stage, "/World/SunLight")
sun_light.CreateIntensityAttr(3000)
sun_light.CreateAngleAttr(0.53)  # Sun angular diameter
```

### Step 4: Run the Simulation

```python
# Reset the world
world.reset()

# Simulation loop
while simulation_app.is_running():
    world.step(render=True)

    # Get box position
    position, orientation = dynamic_box.get_world_pose()
    print(f"Box position: {position}")

# Cleanup
simulation_app.close()
```

---

## Importing Robots from URDF

### URDF Import Workflow

Isaac Sim provides a robust URDF importer that converts ROS robot descriptions to USD:

```python
from omni.isaac.urdf import _urdf
from omni.isaac.core.utils.extensions import enable_extension

# Enable URDF extension
enable_extension("omni.isaac.urdf")

# Configure import settings
urdf_interface = _urdf.acquire_urdf_interface()

import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = False
import_config.make_default_prim = True
import_config.create_physics_scene = True
import_config.import_inertia_tensor = True
import_config.default_drive_strength = 1000.0
import_config.default_position_drive_damping = 100.0
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

# Import the URDF
urdf_path = "/path/to/robot.urdf"
dest_path = "/World/Robot"

result = urdf_interface.parse_urdf(urdf_path, import_config)
urdf_interface.import_robot(
    urdf_path,
    dest_path,
    import_config,
    ""
)
```

### Articulation Configuration

Isaac Sim uses **Articulations** for robot physics:

```python
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

# Get the robot articulation
robot = world.scene.add(
    Articulation(
        prim_path="/World/Robot",
        name="my_robot"
    )
)

# Reset world to initialize articulation
world.reset()

# Get joint information
num_dof = robot.num_dof
joint_names = robot.dof_names
print(f"Robot has {num_dof} DOF: {joint_names}")

# Get current joint positions
joint_positions = robot.get_joint_positions()
print(f"Current positions: {joint_positions}")

# Set joint targets
target_positions = np.zeros(num_dof)
target_positions[0] = 1.57  # First joint to 90 degrees

action = ArticulationAction(
    joint_positions=target_positions,
    joint_velocities=None,
    joint_efforts=None
)
robot.apply_action(action)
```

### Robot Control Example

```python
from omni.isaac.core.controllers import BaseController
import numpy as np

class SimpleJointController(BaseController):
    def __init__(self, name: str, robot_articulation: Articulation):
        super().__init__(name)
        self._robot = robot_articulation
        self._target_positions = np.zeros(robot_articulation.num_dof)

    def set_target(self, joint_index: int, target: float):
        self._target_positions[joint_index] = target

    def forward(self, observations: dict) -> ArticulationAction:
        return ArticulationAction(
            joint_positions=self._target_positions
        )

# Usage
controller = SimpleJointController("joint_ctrl", robot)
controller.set_target(0, 1.57)  # Move joint 0 to 90 degrees

# In simulation loop
while simulation_app.is_running():
    action = controller.forward({})
    robot.apply_action(action)
    world.step(render=True)
```

---

## Adding Sensors

### Camera Sensor

```python
from omni.isaac.sensor import Camera
import numpy as np

# Create RGB camera
camera = Camera(
    prim_path="/World/Robot/Camera",
    position=np.array([0.0, 0.0, 1.5]),
    frequency=30,
    resolution=(640, 480),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion (w, x, y, z)
)

world.scene.add(camera)
world.reset()

# Initialize camera
camera.initialize()

# Get camera data in simulation loop
while simulation_app.is_running():
    world.step(render=True)

    # Get RGB image
    rgb = camera.get_rgba()[:, :, :3]  # Remove alpha channel

    # Get depth
    depth = camera.get_depth()

    print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
```

### LIDAR Sensor

```python
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.core.utils.rotations import quat_to_euler_angles
import omni

# Create LIDAR
lidar_config = {
    "translation": (0, 0, 0.5),
    "orientation": (1, 0, 0, 0),
    "yawNumRays": 360,
    "pitchNumRays": 1,
    "yawRange": (-180, 180),
    "pitchRange": (0, 0),
    "minRange": 0.1,
    "maxRange": 100.0,
    "rotationRate": 10.0
}

# Create the LIDAR prim
result, lidar = omni.kit.commands.execute(
    "RangeSensorCreateLidar",
    path="/World/Robot/Lidar",
    parent="/World/Robot",
    **lidar_config
)

# Get LIDAR interface
lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

# Get data in simulation loop
while simulation_app.is_running():
    world.step(render=True)

    # Get point cloud
    point_cloud = lidar_interface.get_point_cloud_data("/World/Robot/Lidar")
    print(f"LIDAR points: {len(point_cloud)}")
```

### IMU Sensor

```python
from omni.isaac.sensor import IMUSensor
import numpy as np

# Create IMU
imu = IMUSensor(
    prim_path="/World/Robot/IMU",
    name="imu",
    frequency=100,
    translation=np.array([0, 0, 0.2])
)

world.scene.add(imu)
world.reset()
imu.initialize()

# Get IMU data
while simulation_app.is_running():
    world.step(render=True)

    # Get readings
    lin_acc = imu.get_linear_acceleration()
    ang_vel = imu.get_angular_velocity()
    orientation = imu.get_orientation()

    print(f"Linear Acc: {lin_acc}, Angular Vel: {ang_vel}")
```

---

## Synthetic Data with Omniverse Replicator

### What is Replicator?

**Omniverse Replicator** is a framework for generating synthetic training data with:

- Automatic ground truth annotations
- Domain randomization
- Physically accurate sensor simulation
- Scalable data generation

### Basic Replicator Pipeline

```python
import omni.replicator.core as rep

# Define the scene
with rep.new_layer():
    # Create camera
    camera = rep.create.camera(
        position=(0, -5, 2),
        look_at=(0, 0, 0)
    )

    # Create render product
    render_product = rep.create.render_product(camera, (1024, 1024))

    # Create objects to detect
    cube = rep.create.cube(
        semantics=[("class", "cube")],
        position=(0, 0, 0.5)
    )

    sphere = rep.create.sphere(
        semantics=[("class", "sphere")],
        position=(1, 0, 0.5)
    )

# Define randomizers
with rep.trigger.on_frame(num_frames=100):
    # Randomize cube position
    with cube:
        rep.modify.pose(
            position=rep.distribution.uniform((-2, -2, 0.5), (2, 2, 0.5)),
            rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360))
        )

    # Randomize lighting
    with rep.create.light(light_type="Distant"):
        rep.modify.attribute("intensity", rep.distribution.uniform(500, 3000))
        rep.modify.pose(rotation=rep.distribution.uniform((0, -90, 0), (0, 90, 0)))

# Setup output writers
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="./synthetic_data",
    rgb=True,
    bounding_box_2d_tight=True,
    semantic_segmentation=True,
    instance_segmentation=True,
    distance_to_camera=True
)
writer.attach([render_product])

# Run generation
rep.orchestrator.run()
```

### Domain Randomization for Robotics

```python
import omni.replicator.core as rep

def setup_domain_randomization():
    # Texture randomization
    textures = [
        "omniverse://localhost/NVIDIA/Materials/Base/Wood/Wood_Planks.mdl",
        "omniverse://localhost/NVIDIA/Materials/Base/Metals/Steel_Brushed.mdl",
        "omniverse://localhost/NVIDIA/Materials/Base/Stone/Concrete.mdl",
    ]

    # Floor randomization
    floor = rep.get.prims(path_pattern="/World/Ground")
    with rep.trigger.on_frame():
        with floor:
            rep.randomizer.materials(textures)

    # Camera pose randomization
    camera = rep.get.prims(semantics=[("class", "camera")])
    with rep.trigger.on_frame():
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-3, -3, 1), (3, 3, 3)),
                look_at=(0, 0, 0)
            )

    # Lighting randomization
    lights = rep.get.prims(path_pattern="/World/Lights/*")
    with rep.trigger.on_frame():
        with lights:
            rep.modify.attribute(
                "intensity",
                rep.distribution.uniform(500, 5000)
            )
            rep.modify.attribute(
                "color",
                rep.distribution.uniform((0.8, 0.8, 0.8), (1.0, 1.0, 1.0))
            )

setup_domain_randomization()
```

### Output Annotations

Replicator generates comprehensive annotations:

```json
{
  "frame_id": 42,
  "annotations": {
    "bounding_box_2d_tight": [
      {
        "class": "cube",
        "instance_id": 1,
        "x_min": 120,
        "y_min": 200,
        "x_max": 340,
        "y_max": 450
      }
    ],
    "semantic_segmentation": {
      "colorize": true,
      "mapping": {
        "cube": [255, 0, 0],
        "sphere": [0, 255, 0],
        "background": [0, 0, 0]
      }
    },
    "camera_params": {
      "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "extrinsics": [...]
    }
  }
}
```

---

## ROS 2 Integration

### Isaac Sim ROS 2 Bridge

Isaac Sim provides native ROS 2 integration through the `omni.isaac.ros2_bridge` extension:

```python
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")

from omni.isaac.ros2_bridge import read_camera_info
import rclpy
```

### Publishing Sensor Data to ROS 2

```python
from omni.isaac.sensor import Camera
from omni.isaac.ros2_bridge import CameraROS2Publisher

# Create camera with ROS 2 publisher
camera = Camera(
    prim_path="/World/Robot/Camera",
    frequency=30,
    resolution=(640, 480)
)

# Create ROS 2 publisher for camera
camera_publisher = CameraROS2Publisher(
    camera=camera,
    frame_id="camera_link",
    topic_name="/camera/image_raw",
    queue_size=10
)

# In simulation loop, data automatically published to ROS 2
while simulation_app.is_running():
    world.step(render=True)
```

### Subscribing to ROS 2 Commands

```python
from omni.isaac.ros2_bridge import ROS2Subscriber
from geometry_msgs.msg import Twist

class RobotController:
    def __init__(self, robot: Articulation):
        self.robot = robot
        self.cmd_vel = Twist()

        # Subscribe to cmd_vel
        self.subscriber = ROS2Subscriber(
            topic_name="/cmd_vel",
            msg_type=Twist,
            callback=self.cmd_vel_callback
        )

    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vel = msg

    def update(self):
        # Apply velocity to robot
        linear_vel = self.cmd_vel.linear.x
        angular_vel = self.cmd_vel.angular.z

        # Convert to wheel velocities (differential drive)
        wheel_separation = 0.5
        left_vel = linear_vel - angular_vel * wheel_separation / 2
        right_vel = linear_vel + angular_vel * wheel_separation / 2

        # Apply to robot joints
        self.robot.set_joint_velocities([left_vel, right_vel])
```

### Complete ROS 2 Robot Example

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.sensor import Camera, IMUSensor
from omni.isaac.ros2_bridge import (
    CameraROS2Publisher,
    OdometryROS2Publisher,
    ClockROS2Publisher,
    JointStateROS2Publisher
)
import numpy as np

# Create world
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Import robot (assuming URDF already imported)
robot = world.scene.add(
    Articulation(prim_path="/World/Robot", name="robot")
)

# Add sensors
camera = Camera(
    prim_path="/World/Robot/Camera",
    frequency=30,
    resolution=(640, 480)
)
world.scene.add(camera)

# Create ROS 2 publishers
clock_pub = ClockROS2Publisher()
camera_pub = CameraROS2Publisher(camera, "camera_link", "/camera/image_raw")
joint_pub = JointStateROS2Publisher(robot, "/joint_states")
odom_pub = OdometryROS2Publisher(robot, "odom", "base_link", "/odom")

# Reset
world.reset()
camera.initialize()

# Run simulation
while simulation_app.is_running():
    world.step(render=True)

    # Publishers automatically stream data

simulation_app.close()
```

---

## Practical Exercise: Mobile Robot Simulation

### Goal

Create a complete mobile robot simulation with:
1. Imported robot from URDF
2. LIDAR and camera sensors
3. ROS 2 teleoperation
4. Synthetic data generation

### Step 1: Project Setup

```python
# File: mobile_robot_sim.py
from isaacsim import SimulationApp

CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720
}

simulation_app = SimulationApp(CONFIG)

from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.urdf import _urdf
import numpy as np

# Enable extensions
enable_extension("omni.isaac.ros2_bridge")
enable_extension("omni.isaac.sensor")
```

### Step 2: World Creation

```python
def create_warehouse_world(world: World):
    """Create a warehouse environment."""
    from omni.isaac.core.objects import VisualCuboid, GroundPlane

    # Ground
    world.scene.add_default_ground_plane()

    # Walls
    wall_positions = [
        ([0, 5, 1], [10, 0.2, 2]),   # North wall
        ([0, -5, 1], [10, 0.2, 2]),  # South wall
        ([5, 0, 1], [0.2, 10, 2]),   # East wall
        ([-5, 0, 1], [0.2, 10, 2]),  # West wall
    ]

    for i, (pos, scale) in enumerate(wall_positions):
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/Wall_{i}",
                name=f"wall_{i}",
                position=np.array(pos),
                scale=np.array(scale),
                color=np.array([0.7, 0.7, 0.7])
            )
        )

    # Obstacles
    obstacle_positions = [
        [1.5, 1.5, 0.5],
        [-1.5, 2.0, 0.5],
        [2.0, -1.0, 0.5],
    ]

    for i, pos in enumerate(obstacle_positions):
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/Obstacle_{i}",
                name=f"obstacle_{i}",
                position=np.array(pos),
                scale=np.array([0.5, 0.5, 1.0]),
                color=np.array([0.8, 0.4, 0.1])
            )
        )
```

### Step 3: Robot Integration

```python
def import_robot(urdf_path: str, world: World):
    """Import robot from URDF."""
    from omni.isaac.core.articulations import Articulation

    # URDF import config
    urdf_interface = _urdf.acquire_urdf_interface()
    import_config = _urdf.ImportConfig()
    import_config.fix_base = False
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY

    # Import
    urdf_interface.import_robot(urdf_path, "/World/Robot", import_config, "")

    # Create articulation
    robot = world.scene.add(
        Articulation(prim_path="/World/Robot", name="robot")
    )

    return robot
```

### Step 4: Run Simulation

```python
def main():
    # Create world
    world = World(stage_units_in_meters=1.0)

    # Setup environment
    create_warehouse_world(world)

    # Import robot
    robot = import_robot("/path/to/robot.urdf", world)

    # Add sensors (implementation from earlier sections)
    # ...

    # Setup ROS 2 bridge
    # ...

    # Reset
    world.reset()

    # Simulation loop
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
```

---

## Summary

In this chapter, you learned:

- **Why Isaac Sim**: GPU-accelerated physics, RTX rendering, and NVIDIA ecosystem integration
- **Installation**: Multiple deployment options (Omniverse, Docker, Python)
- **USD Fundamentals**: Scene description and composition for robotics
- **World Creation**: Building environments with physics and lighting
- **Robot Import**: URDF to USD conversion and articulation setup
- **Sensor Simulation**: Cameras, LIDAR, and IMUs with physically accurate models
- **Synthetic Data**: Omniverse Replicator for domain randomization and annotation
- **ROS 2 Integration**: Bidirectional communication with the ROS ecosystem

Isaac Sim represents the cutting edge of robotics simulation, enabling development workflows that were previously impossible.

---

## Further Reading

- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) - Official documentation
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) - RL and robot learning framework
- [Omniverse Replicator](https://docs.omniverse.nvidia.com/replicator/latest/index.html) - Synthetic data generation
- [USD Documentation](https://openusd.org/docs/index.html) - Universal Scene Description
- [PhysX 5 SDK](https://github.com/NVIDIA-Omniverse/PhysX) - GPU physics engine

---

## Next Week Preview

In **Chapter 9**, we dive into **Isaac ROS & VSLAM**:
- Isaac ROS packages for perception
- Visual SLAM with Isaac ROS Visual SLAM
- 3D reconstruction with nvblox
- GPU-accelerated perception pipelines
