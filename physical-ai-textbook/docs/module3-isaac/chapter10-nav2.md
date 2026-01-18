---
sidebar_position: 10
title: "Chapter 10 - Nav2 Path Planning"
description: "Master autonomous robot navigation with ROS 2 Nav2, behavior trees, and advanced path planning"
keywords: [Nav2, path planning, autonomous navigation, ROS 2, behavior trees, costmaps, SLAM, motion planning]
last_updated: "2025-12-29"
estimated_reading_time: 28
---

# Chapter 10: Nav2 Path Planning

With perception systems from Chapter 9 providing localization and mapping, we now tackle the challenge of autonomous navigation. Nav2 (Navigation 2) is the ROS 2 navigation stack that enables robots to move intelligently through their environment, avoiding obstacles and reaching goals efficiently.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the Nav2 architecture and its core components
- Configure costmaps for obstacle avoidance
- Implement and customize path planning algorithms
- Use behavior trees for complex navigation logic
- Integrate Nav2 with Isaac ROS perception
- Test and tune navigation in simulation
- Deploy navigation on real robots

---

## Introduction to Nav2

### What is Nav2?

**Nav2** (Navigation 2) is the second-generation navigation stack for ROS 2, designed for:

- **Autonomous Mobile Robots (AMRs)**: Warehouse robots, delivery robots
- **Service Robots**: Hospital, hotel, restaurant robots
- **Research Platforms**: TurtleBot, custom robots
- **Industrial Applications**: AGVs, inspection robots

### Nav2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Nav2 Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   BT Navigator                       │   │
│  │           (Behavior Tree Orchestration)              │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                                │
│         ┌──────────────────┼──────────────────┐            │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐  │
│  │   Planner   │   │ Controller  │   │    Recovery     │  │
│  │   Server    │   │   Server    │   │    Server       │  │
│  │  (Global)   │   │  (Local)    │   │  (Stuck/Error)  │  │
│  └──────┬──────┘   └──────┬──────┘   └────────┬────────┘  │
│         │                 │                    │           │
│         └─────────────────┼────────────────────┘           │
│                           │                                │
│  ┌────────────────────────▼────────────────────────────┐  │
│  │                   Costmap 2D                         │  │
│  │  ┌──────────────┐        ┌──────────────────────┐  │  │
│  │  │ Global       │        │ Local Costmap        │  │  │
│  │  │ Costmap      │        │ (Rolling Window)     │  │  │
│  │  └──────────────┘        └──────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌─────────────────────────▼───────────────────────────┐   │
│  │                    Inputs                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │   Map    │  │  Odom    │  │  Sensor Data     │  │   │
│  │  │ (SLAM)   │  │ (Pose)   │  │  (LIDAR/Depth)   │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Purpose | Key Plugins |
|-----------|---------|-------------|
| **BT Navigator** | Orchestrates navigation behavior | NavigateToPose, NavigateThroughPoses |
| **Planner Server** | Computes global path | NavFn, Smac, Theta* |
| **Controller Server** | Follows path, avoids obstacles | DWB, TEB, MPPI, RPP |
| **Recovery Server** | Handles stuck situations | Spin, BackUp, Wait |
| **Costmap 2D** | Represents traversability | Static, Inflation, Obstacle layers |

---

## Setting Up Nav2

### Installation

```bash
# Install Nav2 packages
sudo apt-get update
sudo apt-get install -y \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-nav2-simple-commander \
  ros-humble-slam-toolbox

# For simulation testing
sudo apt-get install -y \
  ros-humble-turtlebot3-gazebo \
  ros-humble-turtlebot3-navigation2

# Set TurtleBot3 model
echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
source ~/.bashrc
```

### Basic Nav2 Launch

```bash
# Terminal 1: Launch simulation
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Launch Nav2
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

# Terminal 3: Launch RViz2 with Nav2 config
ros2 launch nav2_bringup rviz_launch.py
```

### Verifying Installation

```bash
# Check Nav2 nodes are running
ros2 node list | grep -E "(planner|controller|bt_navigator|costmap)"

# Check available action servers
ros2 action list

# Expected output:
# /navigate_to_pose
# /navigate_through_poses
# /follow_path
# /compute_path_to_pose
```

---

## Understanding Costmaps

### What are Costmaps?

**Costmaps** represent the traversability of the environment as a 2D grid where each cell has a cost value:

```
┌─────────────────────────────────────────────────────────────┐
│                    Costmap Values                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Cost Value        Meaning              Visualization       │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  0                 Free space           ░░░░░░░░░░░░░       │
│  1-252             Increasing cost      ▒▒▒▒▒▒▒▒▒▒▒▒▒       │
│  253               Inscribed radius     ▓▓▓▓▓▓▓▓▓▓▓▓▓       │
│  254               Lethal obstacle      ████████████████     │
│  255               Unknown              ????????????????     │
│                                                             │
│  Robot Navigation Priority:                                 │
│  • Avoid 254 (lethal) at all costs                         │
│  • Minimize travel through high-cost areas                  │
│  • Prefer paths through 0 (free) space                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Global vs Local Costmap

```
┌─────────────────────────────────────────────────────────────┐
│              Global vs Local Costmap                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Global Costmap                 Local Costmap               │
│  ┌─────────────────────┐       ┌─────────────────────┐     │
│  │░░░░░░░░░░░░░░░░░░░░│       │                     │     │
│  │░░░░████░░░░░░░░░░░░│       │      ░░░░░░░        │     │
│  │░░░░████░░░░░░░░░░░░│       │    ░░░░░░░░░░      │     │
│  │░░░░░░░░░░░░████░░░░│       │   ░░░██░░░░░░░     │     │
│  │░░░░░░░░░░░░████░░░░│       │   ░░░██░░[R]░░     │     │
│  │░░░░░░░░░░░░░░░░░░░░│       │    ░░░░░░░░░░      │     │
│  │░░░████████░░░░░░░░░│       │      ░░░░░░░        │     │
│  │░░░████████░░░░░░░░░│       │                     │     │
│  └─────────────────────┘       └─────────────────────┘     │
│                                                             │
│  • Full environment map         • Rolling window around     │
│  • Used for global planning       robot [R]                 │
│  • Updated less frequently      • Used for local planning   │
│  • Static layer from SLAM       • Real-time sensor data     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Costmap Layers

```yaml
# Global costmap configuration
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      resolution: 0.05
      track_unknown_space: true

      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]

      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

```yaml
# Local costmap configuration
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05

      plugins: ["voxel_layer", "inflation_layer"]

      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan depth
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
        depth:
          topic: /camera/depth/points
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

### Inflation Layer Explained

```
┌─────────────────────────────────────────────────────────────┐
│                   Inflation Layer                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Obstacle                                 │
│                       │                                     │
│                       ▼                                     │
│        ┌─────────────────────────────┐                     │
│        │░░░░░░▒▒▒▒▓▓▓███▓▓▓▒▒▒▒░░░░░░│                     │
│        │░░░▒▒▒▒▓▓▓█████████▓▓▓▒▒▒▒░░░│                     │
│        │░░▒▒▓▓▓███████████████▓▓▓▒▒░░│                     │
│        │░▒▒▓▓████████████████████▓▓▒░│                     │
│        │░▒▓▓█████████████████████▓▓▒░│                     │
│        │░▒▓███████████████████████▓▒░│                     │
│        │░▒▓▓█████████████████████▓▓▒░│                     │
│        │░▒▒▓▓████████████████████▓▓▒░│                     │
│        │░░▒▒▓▓▓███████████████▓▓▓▒▒░░│                     │
│        │░░░▒▒▒▒▓▓▓█████████▓▓▓▒▒▒▒░░░│                     │
│        │░░░░░░▒▒▒▒▓▓▓███▓▓▓▒▒▒▒░░░░░░│                     │
│        └─────────────────────────────┘                     │
│                                                             │
│  ██ Lethal (254)      - Robot center cannot be here        │
│  ▓▓ Inscribed (253)   - Robot footprint touches obstacle   │
│  ▒▒ High cost         - Close to obstacle, penalized       │
│  ░░ Free/Low cost     - Safe for navigation                │
│                                                             │
│  inflation_radius: Distance to inflate around obstacles    │
│  cost_scaling_factor: How quickly cost decreases           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Path Planning Algorithms

### Global Planners

Nav2 provides several global planning algorithms:

```
┌─────────────────────────────────────────────────────────────┐
│                  Global Planners Comparison                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NavFn (Navigation Function)                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Dijkstra's algorithm based                        │   │
│  │ • Guaranteed optimal path                            │   │
│  │ • Good for simple environments                       │   │
│  │ • Slower in large maps                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Smac Planner (State Lattice)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Hybrid A* for non-holonomic robots                │   │
│  │ • 2D A* for holonomic robots                        │   │
│  │ • Considers robot kinematics                         │   │
│  │ • Better for Ackermann/differential drive           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Theta* Planner                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Any-angle path planning                            │   │
│  │ • Smoother paths than grid-based                     │   │
│  │ • Line-of-sight optimization                         │   │
│  │ • Good for open environments                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuring Global Planners

```yaml
# NavFn Planner
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false  # Use Dijkstra
      allow_unknown: true

# Smac Planner 2D (for holonomic robots)
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner/SmacPlanner2D"
      tolerance: 0.25
      downsample_costmap: false
      downsampling_factor: 1
      allow_unknown: true
      max_iterations: 1000000
      max_on_approach_iterations: 1000
      max_planning_time: 2.0
      cost_travel_multiplier: 2.0
      use_final_approach_orientation: false
      smoother:
        max_iterations: 1000
        w_smooth: 0.3
        w_data: 0.2
        tolerance: 1.0e-10

# Smac Hybrid-A* (for non-holonomic robots)
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner/SmacPlannerHybrid"
      tolerance: 0.25
      downsample_costmap: false
      downsampling_factor: 1
      allow_unknown: true
      max_iterations: 1000000
      max_planning_time: 5.0
      motion_model_for_search: "DUBIN"  # DUBIN, REEDS_SHEPP
      angle_quantization_bins: 72
      analytic_expansion_ratio: 3.5
      minimum_turning_radius: 0.40
      reverse_penalty: 2.0
      change_penalty: 0.0
      non_straight_penalty: 1.2
      cost_penalty: 2.0
```

### Local Controllers

```
┌─────────────────────────────────────────────────────────────┐
│                  Local Controllers Comparison                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DWB (Dynamic Window Approach - Base)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Samples velocity space                             │   │
│  │ • Scores trajectories against critics                │   │
│  │ • Flexible and configurable                          │   │
│  │ • Good general-purpose controller                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  TEB (Timed Elastic Band)                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Optimizes trajectory in time                       │   │
│  │ • Handles dynamic obstacles                          │   │
│  │ • Smooth velocity profiles                           │   │
│  │ • Better for tight spaces                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  MPPI (Model Predictive Path Integral)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Sampling-based MPC                                 │   │
│  │ • GPU-accelerated option                             │   │
│  │ • Handles complex dynamics                           │   │
│  │ • State-of-the-art performance                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  RPP (Regulated Pure Pursuit)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Simple and reliable                                │   │
│  │ • Good for path following                            │   │
│  │ • Adaptive lookahead                                 │   │
│  │ • Lower computational cost                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuring Local Controllers

```yaml
# DWB Controller
controller_server:
  ros__parameters:
    controller_frequency: 20.0
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      debug_trajectory_details: True
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.26
      max_vel_y: 0.0
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.26
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      short_circuit_trajectory_evaluation: True
      critics: ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]

# MPPI Controller
controller_server:
  ros__parameters:
    controller_frequency: 30.0
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 56
      model_dt: 0.05
      batch_size: 2000
      vx_std: 0.2
      vy_std: 0.2
      wz_std: 0.4
      vx_max: 0.5
      vx_min: -0.35
      vy_max: 0.5
      wz_max: 1.9
      iteration_count: 1
      prune_distance: 1.7
      transform_tolerance: 0.1
      temperature: 0.3
      gamma: 0.015
      motion_model: "DiffDrive"  # Omni, Ackermann, DiffDrive
      visualize: true
      critics: ["ConstraintCritic", "ObstaclesCritic", "GoalCritic", "GoalAngleCritic", "PathAlignCritic", "PathFollowCritic", "PathAngleCritic", "PreferForwardCritic"]

# Regulated Pure Pursuit
controller_server:
  ros__parameters:
    controller_frequency: 20.0
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.5
      lookahead_dist: 0.6
      min_lookahead_dist: 0.3
      max_lookahead_dist: 0.9
      lookahead_time: 1.5
      rotate_to_heading_angular_vel: 1.8
      transform_tolerance: 0.1
      use_velocity_scaled_lookahead_dist: false
      min_approach_linear_velocity: 0.05
      approach_velocity_scaling_dist: 0.6
      use_collision_detection: true
      max_allowed_time_to_collision_up_to_carrot: 1.0
      use_regulated_linear_velocity_scaling: true
      use_cost_regulated_linear_velocity_scaling: false
      regulated_linear_scaling_min_radius: 0.9
      regulated_linear_scaling_min_speed: 0.25
      use_rotate_to_heading: true
      allow_reversing: false
      rotate_to_heading_min_angle: 0.785
      max_angular_accel: 3.2
      max_robot_pose_search_dist: 10.0
```

---

## Behavior Trees in Nav2

### What are Behavior Trees?

**Behavior Trees (BTs)** are a modular way to organize complex robot behaviors:

```
┌─────────────────────────────────────────────────────────────┐
│                 Behavior Tree Concepts                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Node Types:                                                │
│                                                             │
│  ┌─────────────┐  Control Nodes                            │
│  │  Sequence   │  Execute children left-to-right           │
│  │    [→]      │  SUCCESS if all succeed, FAILURE if any   │
│  └─────────────┘  fails                                    │
│                                                             │
│  ┌─────────────┐  Fallback/Selector                        │
│  │  Fallback   │  Try children until one succeeds          │
│  │    [?]      │  SUCCESS if any succeeds, FAILURE if      │
│  └─────────────┘  all fail                                 │
│                                                             │
│  ┌─────────────┐  Action Nodes                             │
│  │   Action    │  Execute robot actions                    │
│  │    [A]      │  ComputePath, FollowPath, Spin, etc.     │
│  └─────────────┘                                           │
│                                                             │
│  ┌─────────────┐  Condition Nodes                          │
│  │  Condition  │  Check conditions                         │
│  │    [C]      │  IsPathValid, IsBatteryLow, etc.         │
│  └─────────────┘                                           │
│                                                             │
│  ┌─────────────┐  Decorator Nodes                          │
│  │  Decorator  │  Modify child behavior                    │
│  │    [D]      │  RateController, Repeat, Retry, etc.     │
│  └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Default NavigateToPose Behavior Tree

```
┌─────────────────────────────────────────────────────────────┐
│            NavigateToPose Behavior Tree                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      [→] Sequence                           │
│                           │                                 │
│        ┌──────────────────┼──────────────────┐             │
│        │                  │                  │             │
│        ▼                  ▼                  ▼             │
│  ┌───────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ [D] Rate  │    │[?] Fallback │    │    [A]      │      │
│  │ Controller│    │  Recovery   │    │  GoalReached│      │
│  └─────┬─────┘    └──────┬──────┘    └─────────────┘      │
│        │                 │                                  │
│        ▼                 │                                  │
│  ┌───────────┐    ┌──────┴────────────────────┐           │
│  │    [A]    │    │                           │           │
│  │ComputePath│    ┌──────────────┬────────────┐           │
│  │ ToGoal    │    │              │            │           │
│  └───────────┘    ▼              ▼            ▼           │
│                ┌──────┐    ┌──────────┐  ┌────────┐       │
│                │[→]   │    │   [A]    │  │  [A]   │       │
│                │Follow│    │  ClearCM │  │ Spin   │       │
│                │ Path │    │ -Global  │  │        │       │
│                └──────┘    └──────────┘  └────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Custom Behavior Tree XML

```xml
<!-- File: navigate_to_pose_custom.xml -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">

        <!-- Compute path to goal -->
        <RateController hz="1.0">
          <ComputePathToPose goal="{goal}" path="{path}"
                             planner_id="GridBased"/>
        </RateController>

        <!-- Follow the computed path -->
        <ReactiveSequence name="FollowPath">
          <PathExpiringTimer seconds="5.0" path="{path}"/>
          <FollowPath path="{path}" controller_id="FollowPath"/>
        </ReactiveSequence>

      </PipelineSequence>

      <!-- Recovery behaviors when stuck -->
      <ReactiveFallback name="RecoveryFallback">
        <GoalUpdated/>
        <RoundRobin name="RecoveryActions">
          <Sequence name="ClearingActions">
            <ClearEntireCostmap name="ClearLocalCostmap"
                               service_name="local_costmap/clear_entirely_local_costmap"/>
            <ClearEntireCostmap name="ClearGlobalCostmap"
                               service_name="global_costmap/clear_entirely_global_costmap"/>
          </Sequence>
          <Spin spin_dist="1.57"/>
          <Wait wait_duration="5"/>
          <BackUp backup_dist="0.30" backup_speed="0.05"/>
        </RoundRobin>
      </ReactiveFallback>

    </RecoveryNode>
  </BehaviorTree>
</root>
```

### Loading Custom Behavior Trees

```yaml
# In nav2_params.yaml
bt_navigator:
  ros__parameters:
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    # Use custom behavior tree
    default_nav_to_pose_bt_xml: /path/to/navigate_to_pose_custom.xml
    default_nav_through_poses_bt_xml: /path/to/navigate_through_poses.xml
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_rate_controller_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
```

---

## Integration with Isaac ROS

### Using nvblox Costmaps with Nav2

```yaml
# Configure Nav2 to use nvblox occupancy grid
global_costmap:
  global_costmap:
    ros__parameters:
      plugins: ["nvblox_layer", "inflation_layer"]

      nvblox_layer:
        plugin: "nvblox::NvbloxCostmapLayer"
        enabled: True
        nvblox_map_slice_topic: "/nvblox_node/static_map_slice"

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

local_costmap:
  local_costmap:
    ros__parameters:
      plugins: ["nvblox_layer", "inflation_layer"]

      nvblox_layer:
        plugin: "nvblox::NvbloxCostmapLayer"
        enabled: True
        nvblox_map_slice_topic: "/nvblox_node/static_map_slice"
```

### Complete Isaac ROS + Nav2 Launch

```python
# File: isaac_nav2_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    # Parameters file
    nav2_params = os.path.join(
        get_package_share_directory('my_robot_navigation'),
        'config',
        'nav2_isaac_params.yaml'
    )

    return LaunchDescription([
        # cuVSLAM for localization
        Node(
            package='isaac_ros_visual_slam',
            executable='isaac_ros_visual_slam_node',
            name='visual_slam',
            parameters=[{
                'enable_imu_fusion': True,
                'map_frame': 'map',
                'odom_frame': 'odom',
                'base_frame': 'base_link',
            }]
        ),

        # nvblox for 3D mapping
        Node(
            package='nvblox_ros',
            executable='nvblox_node',
            name='nvblox',
            parameters=[{
                'voxel_size': 0.05,
                'global_frame': 'odom',
            }]
        ),

        # Nav2 stack
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                nav2_bringup_dir, '/launch/navigation_launch.py'
            ]),
            launch_arguments={
                'params_file': nav2_params,
                'use_sim_time': 'False',
            }.items()
        ),
    ])
```

---

## Programmatic Navigation

### Using nav2_simple_commander

```python
#!/usr/bin/env python3
"""
Programmatic navigation using Nav2 Simple Commander
"""
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.duration import Duration

def main():
    rclpy.init()
    navigator = BasicNavigator()

    # Set initial pose (if not using AMCL with good initial estimate)
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = 0.0
    initial_pose.pose.position.y = 0.0
    initial_pose.pose.orientation.w = 1.0
    navigator.setInitialPose(initial_pose)

    # Wait for Nav2 to be active
    navigator.waitUntilNav2Active()

    # Define goal pose
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose.pose.position.x = 2.0
    goal_pose.pose.position.y = 1.0
    goal_pose.pose.orientation.w = 1.0

    # Navigate to goal
    navigator.goToPose(goal_pose)

    # Monitor progress
    i = 0
    while not navigator.isTaskComplete():
        i += 1
        feedback = navigator.getFeedback()
        if feedback and i % 5 == 0:
            print(f'Distance remaining: {feedback.distance_remaining:.2f}m')
            print(f'ETA: {Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9:.0f}s')

            # Cancel if taking too long
            if Duration.from_msg(feedback.navigation_time) > Duration(seconds=120.0):
                navigator.cancelTask()

    # Check result
    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print('Goal reached!')
    elif result == TaskResult.CANCELED:
        print('Goal was canceled!')
    elif result == TaskResult.FAILED:
        print('Goal failed!')

    navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Waypoint Following

```python
#!/usr/bin/env python3
"""
Navigate through multiple waypoints
"""
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import rclpy

def create_pose(x, y, yaw=0.0):
    """Create a PoseStamped message"""
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = x
    pose.pose.position.y = y

    # Convert yaw to quaternion (simplified for z-rotation only)
    import math
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)

    return pose

def main():
    rclpy.init()
    navigator = BasicNavigator()
    navigator.waitUntilNav2Active()

    # Define waypoints
    waypoints = [
        create_pose(1.0, 0.0, 0.0),
        create_pose(2.0, 1.0, 1.57),
        create_pose(1.0, 2.0, 3.14),
        create_pose(0.0, 1.0, -1.57),
        create_pose(0.0, 0.0, 0.0),  # Return to start
    ]

    # Navigate through waypoints
    navigator.followWaypoints(waypoints)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        if feedback:
            print(f'Waypoint {feedback.current_waypoint + 1}/{len(waypoints)}')

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print('All waypoints reached!')
    else:
        print(f'Navigation failed with result: {result}')

    navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Coverage Path Planning

```python
#!/usr/bin/env python3
"""
Simple coverage path planning (lawnmower pattern)
"""
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import rclpy
import math

def generate_coverage_path(x_min, x_max, y_min, y_max, spacing=0.5):
    """Generate lawnmower coverage pattern"""
    waypoints = []
    y = y_min
    direction = 1  # 1 = moving right, -1 = moving left

    while y <= y_max:
        if direction == 1:
            x_start, x_end = x_min, x_max
        else:
            x_start, x_end = x_max, x_min

        # Add start of row
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x_start
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        waypoints.append(pose)

        # Add end of row
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x_end
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        waypoints.append(pose)

        y += spacing
        direction *= -1

    return waypoints

def main():
    rclpy.init()
    navigator = BasicNavigator()
    navigator.waitUntilNav2Active()

    # Generate coverage path for a 4x4 meter area
    waypoints = generate_coverage_path(
        x_min=0.0, x_max=4.0,
        y_min=0.0, y_max=4.0,
        spacing=0.5
    )

    print(f'Generated {len(waypoints)} waypoints for coverage')

    # Execute coverage
    navigator.followWaypoints(waypoints)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        if feedback:
            progress = (feedback.current_waypoint / len(waypoints)) * 100
            print(f'Coverage progress: {progress:.1f}%')

    result = navigator.getResult()
    print(f'Coverage complete with result: {result}')

    navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Testing in Simulation

### Gazebo + Nav2 Setup

```bash
# Terminal 1: Launch TurtleBot3 in Gazebo
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Launch SLAM for mapping
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True

# Terminal 3: Launch Nav2
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

# Terminal 4: Launch RViz2
ros2 launch nav2_bringup rviz_launch.py

# Terminal 5: Teleop for initial mapping (optional)
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Isaac Sim + Nav2 Setup

```python
# File: isaac_sim_nav2_test.py
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import DifferentialController
import numpy as np

# Enable extensions
enable_extension("omni.isaac.ros2_bridge")

# Create world
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Add robot (e.g., from URDF)
# ... robot setup code ...

# Add obstacles
from omni.isaac.core.objects import VisualCuboid
obstacles = [
    {"pos": [2.0, 0.0, 0.5], "scale": [0.5, 0.5, 1.0]},
    {"pos": [1.0, 1.5, 0.5], "scale": [1.0, 0.3, 1.0]},
    {"pos": [3.0, -1.0, 0.5], "scale": [0.3, 1.0, 1.0]},
]

for i, obs in enumerate(obstacles):
    world.scene.add(
        VisualCuboid(
            prim_path=f"/World/Obstacle_{i}",
            name=f"obstacle_{i}",
            position=np.array(obs["pos"]),
            scale=np.array(obs["scale"]),
            color=np.array([0.8, 0.2, 0.2])
        )
    )

# Simulation loop
world.reset()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Robot not moving | Controller not receiving path | Check `ros2 topic echo /plan` |
| Path planning fails | Costmap inflation too large | Reduce `inflation_radius` |
| Robot oscillates | Controller gains too high | Tune velocity/acceleration limits |
| Recovery behaviors loop | Robot truly stuck | Check costmap, increase recovery timeout |
| TF errors | Transform not available | Check `ros2 run tf2_tools view_frames` |

### Debugging Commands

```bash
# Check if Nav2 servers are active
ros2 lifecycle list /planner_server
ros2 lifecycle list /controller_server
ros2 lifecycle list /bt_navigator

# Visualize costmaps
ros2 topic echo /global_costmap/costmap
ros2 topic echo /local_costmap/costmap

# Check planned path
ros2 topic echo /plan

# View behavior tree status
ros2 topic echo /behavior_tree_log

# Force clear costmaps
ros2 service call /global_costmap/clear_entirely_global_costmap nav2_msgs/srv/ClearEntireCostmap
ros2 service call /local_costmap/clear_entirely_local_costmap nav2_msgs/srv/ClearEntireCostmap
```

---

## Practical Exercise: Complete Navigation System

### Goal

Build a complete autonomous navigation system that:
1. Uses Isaac ROS perception for localization
2. Navigates through multiple waypoints
3. Handles recovery from stuck situations
4. Reports progress via ROS 2 topics

### Step 1: Create Parameter File

```yaml
# File: config/nav2_params.yaml
# See full configurations in previous sections
amcl:
  ros__parameters:
    # ... AMCL params if not using cuVSLAM

bt_navigator:
  ros__parameters:
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom

planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner/SmacPlanner2D"
      # ... planner params

controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      # ... controller params

global_costmap:
  global_costmap:
    ros__parameters:
      # ... costmap params

local_costmap:
  local_costmap:
    ros__parameters:
      # ... costmap params
```

### Step 2: Create Launch File

```python
# File: launch/navigation.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('my_robot_navigation')
    nav2_dir = get_package_share_directory('nav2_bringup')

    return LaunchDescription([
        # Nav2 with custom params
        IncludeLaunchDescription(
            os.path.join(nav2_dir, 'launch', 'navigation_launch.py'),
            launch_arguments={
                'params_file': os.path.join(pkg_dir, 'config', 'nav2_params.yaml'),
                'use_sim_time': 'True',
            }.items()
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg_dir, 'rviz', 'nav2.rviz')],
        ),
    ])
```

### Step 3: Test Navigation

```bash
# Launch simulation and navigation
ros2 launch my_robot_navigation navigation.launch.py

# Send navigation goal via CLI
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 1.0}, orientation: {w: 1.0}}}}"
```

---

## Summary

In this chapter, you learned:

- **Nav2 Architecture**: BT Navigator, Planner Server, Controller Server, Costmaps
- **Costmap Configuration**: Global vs local, layers (static, obstacle, inflation)
- **Path Planning**: NavFn, Smac, Theta* planners for different scenarios
- **Local Controllers**: DWB, TEB, MPPI, RPP for path following
- **Behavior Trees**: Structure, XML configuration, custom behaviors
- **Isaac ROS Integration**: Using nvblox costmaps with Nav2
- **Programmatic Navigation**: nav2_simple_commander for waypoints and coverage
- **Testing**: Simulation workflows with Gazebo and Isaac Sim

Navigation is the cornerstone of autonomous mobile robotics. With Nav2, you have a flexible, extensible framework that works from simple point-to-point navigation to complex multi-waypoint missions.

---

## Further Reading

- [Nav2 Documentation](https://navigation.ros.org/) - Official Nav2 documentation
- [Nav2 Tutorials](https://navigation.ros.org/tutorials/index.html) - Step-by-step guides
- [Behavior Trees in Robotics](https://arxiv.org/abs/1709.00084) - Academic foundation
- [MPPI Paper](https://arxiv.org/abs/1707.02342) - Model Predictive Path Integral Control
- [Smac Planner](https://navigation.ros.org/configuration/packages/configuring-smac-planner.html) - State lattice planning

---

## Module 3 Complete!

Congratulations on completing Module 3: The AI-Robot Brain! You've learned:

- **Chapter 8**: NVIDIA Isaac Sim for GPU-accelerated simulation
- **Chapter 9**: Isaac ROS for perception (cuVSLAM, nvblox)
- **Chapter 10**: Nav2 for autonomous navigation

---

## Next Module Preview

In **Module 4: The Vision-Language-Action Pipeline**, we explore:

- **Chapter 11**: Voice-to-Action with speech recognition and LLMs
- **Chapter 12**: Cognitive Planning with VLMs for task understanding
- **Chapter 13**: Capstone Project integrating all concepts
