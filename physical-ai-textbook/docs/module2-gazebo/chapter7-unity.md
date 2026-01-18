---
sidebar_position: 7
title: "Chapter 7 - Unity for Robotics"
description: "Leverage Unity's powerful rendering and simulation capabilities for robotics development"
keywords: [Unity, robotics, simulation, synthetic data, machine learning, ROS 2, digital twin]
last_updated: "2025-12-29"
estimated_reading_time: 22
---

# Chapter 7: Unity for Robotics

While Gazebo excels at physics-accurate simulation, Unity brings photorealistic rendering, advanced ML integration, and cross-platform deployment to robotics. In this chapter, we explore Unity Robotics Hub and how it complements traditional simulators.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain when to use Unity vs Gazebo for robotics simulation
- Set up Unity with ROS 2 integration
- Create photorealistic environments for robot testing
- Generate synthetic training data for perception models
- Implement domain randomization for robust ML models
- Build human-robot interaction scenarios

---

## Why Unity for Robotics?

### The Perception Gap

Traditional robotics simulators like Gazebo focus on physics accuracy but often produce visually simplistic environments:

```
┌─────────────────────────────────────────────────────────────┐
│                 The Simulation Reality Gap                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Gazebo Simulation          Real World                      │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ Simple colors   │        │ Complex textures│            │
│  │ Basic shapes    │   →    │ Lighting vary   │            │
│  │ Flat lighting   │  GAP   │ Occlusions      │            │
│  │ No reflections  │        │ Reflections     │            │
│  └─────────────────┘        └─────────────────┘            │
│                                                             │
│  ML models trained on simple visuals fail in real world!   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Unity's Strengths for Robotics

| Capability | Benefit for Robotics |
|------------|---------------------|
| **Photorealistic Rendering** | Train perception models that transfer to real world |
| **Universal Render Pipeline** | Real-time ray tracing, PBR materials |
| **Domain Randomization** | Automatic variation of textures, lighting, objects |
| **Synthetic Data Generation** | Labeled datasets (bounding boxes, segmentation) |
| **Cross-Platform** | Windows, Linux, embedded devices |
| **Asset Store** | Thousands of 3D models, environments |
| **C# Scripting** | Rapid prototyping, custom behaviors |

### When to Use Each Simulator

| Use Case | Recommended Simulator |
|----------|----------------------|
| Physics-critical (contact, dynamics) | Gazebo |
| Perception/ML training | Unity |
| Photorealistic visualization | Unity |
| ROS 2 ecosystem tools | Gazebo |
| Human-robot interaction | Unity |
| Large-scale fleet simulation | Both (hybrid) |
| Real-time control testing | Gazebo |
| Synthetic dataset generation | Unity |

---

## Unity Robotics Hub Overview

### What is Unity Robotics Hub?

**Unity Robotics Hub** is a collection of tools and packages that enable robotics development in Unity:

- **ROS-TCP-Connector**: Bidirectional ROS 2 communication
- **URDF Importer**: Import robot models from URDF
- **Perception Package**: Synthetic data generation with labels
- **ML-Agents**: Reinforcement learning integration
- **Articulation Bodies**: Physics for articulated robots

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Unity Robotics Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Unity Editor                      │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────┐   │   │
│  │  │   Scene   │  │  Physics  │  │   Rendering   │   │   │
│  │  │  Editor   │  │  Engine   │  │   Pipeline    │   │   │
│  │  └─────┬─────┘  └─────┬─────┘  └───────┬───────┘   │   │
│  │        │              │                │            │   │
│  │  ┌─────▼──────────────▼────────────────▼───────┐   │   │
│  │  │              Unity Runtime                   │   │   │
│  │  └─────────────────────┬───────────────────────┘   │   │
│  └────────────────────────┼────────────────────────────┘   │
│                           │                                 │
│                    ┌──────▼──────┐                         │
│                    │ ROS-TCP     │                         │
│                    │ Connector   │                         │
│                    └──────┬──────┘                         │
│                           │                                 │
│  ┌────────────────────────▼────────────────────────────┐   │
│  │                    ROS 2                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │  Nodes   │  │  Topics  │  │  Services        │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Setting Up Unity for Robotics

### Prerequisites

1. **Unity Hub**: Download from [unity.com](https://unity.com/download)
2. **Unity Editor**: Version 2021.3 LTS or later (2022.3 recommended)
3. **ROS 2**: Humble or later installed on your system

### Step 1: Create a New Unity Project

```bash
# Open Unity Hub and create a new project
# Template: 3D (URP) - Universal Render Pipeline
# Project Name: RoboticsSimulation
```

### Step 2: Install Robotics Packages

In Unity, open **Window → Package Manager**, then:

1. Click **+ → Add package from git URL**
2. Add these packages one by one:

```
https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector
https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer
https://github.com/Unity-Technologies/com.unity.perception.git
```

### Step 3: Configure ROS-TCP Connection

Create a connection settings asset:

1. **Assets → Create → Robotics → ROS Connection Prefab**
2. Configure:
   - **ROS IP Address**: `127.0.0.1` (or your ROS machine IP)
   - **ROS Port**: `10000`
   - **Protocol**: ROS2

### Step 4: Start the ROS-TCP Endpoint

On your ROS 2 machine:

```bash
# Install the ROS-TCP-Endpoint package
cd ~/ros2_ws/src
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git -b main-ros2

# Build
cd ~/ros2_ws
colcon build --packages-select ros_tcp_endpoint

# Source and run
source install/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0
```

---

## Importing Robots with URDF

### URDF Importer Workflow

Unity's URDF Importer converts your ROS robot descriptions into Unity GameObjects:

```
┌─────────────────────────────────────────────────────────────┐
│                  URDF Import Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  URDF/Xacro File                                            │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                           │
│  │ Parse Links │──▶ Unity GameObjects with Transforms      │
│  └─────────────┘                                           │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                           │
│  │Parse Joints │──▶ Articulation Bodies (physics joints)   │
│  └─────────────┘                                           │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                           │
│  │Parse Meshes │──▶ MeshFilter + MeshRenderer components   │
│  └─────────────┘                                           │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                           │
│  │ Colliders   │──▶ Unity Colliders (Box, Mesh, etc.)     │
│  └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Importing Your Robot

1. Copy your URDF file and meshes to `Assets/Robots/`
2. In Unity: **Assets → Import Robot from URDF**
3. Select your `.urdf` file
4. Configure import settings:

```csharp
// Import settings (in the import dialog)
public class URDFImportSettings
{
    public bool UseUrdfInertiaData = true;
    public bool UseGravity = true;
    public float GlobalScale = 1.0f;
    public ImportPipelineType Pipeline = ImportPipelineType.ArticulationBody;
}
```

### Articulation Bodies for Robot Physics

Unity's **Articulation Bodies** provide stable physics for articulated robots:

```csharp
using UnityEngine;

public class RobotController : MonoBehaviour
{
    private ArticulationBody[] joints;

    void Start()
    {
        // Get all articulation bodies in the robot
        joints = GetComponentsInChildren<ArticulationBody>();
    }

    public void SetJointTarget(int jointIndex, float targetPosition)
    {
        if (jointIndex < joints.Length)
        {
            var drive = joints[jointIndex].xDrive;
            drive.target = targetPosition * Mathf.Rad2Deg;
            joints[jointIndex].xDrive = drive;
        }
    }

    public float GetJointPosition(int jointIndex)
    {
        if (jointIndex < joints.Length)
        {
            return joints[jointIndex].jointPosition[0];
        }
        return 0f;
    }
}
```

---

## ROS 2 Communication in Unity

### Publishing Topics

Send data from Unity to ROS 2:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class VelocityPublisher : MonoBehaviour
{
    private ROSConnection ros;
    private string topicName = "/cmd_vel";
    public float publishFrequency = 10f;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>(topicName);
        InvokeRepeating("PublishVelocity", 1f, 1f / publishFrequency);
    }

    void PublishVelocity()
    {
        TwistMsg msg = new TwistMsg
        {
            linear = new Vector3Msg { x = 0.5, y = 0, z = 0 },
            angular = new Vector3Msg { x = 0, y = 0, z = 0.1 }
        };
        ros.Publish(topicName, msg);
    }
}
```

### Subscribing to Topics

Receive data from ROS 2 in Unity:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class LaserScanSubscriber : MonoBehaviour
{
    private ROSConnection ros;
    private string topicName = "/scan";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<LaserScanMsg>(topicName, OnLaserScanReceived);
    }

    void OnLaserScanReceived(LaserScanMsg msg)
    {
        // Process laser scan data
        float[] ranges = msg.ranges;
        float angleMin = msg.angle_min;
        float angleIncrement = msg.angle_increment;

        // Visualize or use the data
        Debug.Log($"Received {ranges.Length} laser points");
    }
}
```

### Calling ROS Services

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class ServiceCaller : MonoBehaviour
{
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterRosService<SetBoolRequest, SetBoolResponse>("/enable_motor");
    }

    public void EnableMotor(bool enable)
    {
        SetBoolRequest request = new SetBoolRequest { data = enable };
        ros.SendServiceMessage<SetBoolResponse>(
            "/enable_motor",
            request,
            OnServiceResponse
        );
    }

    void OnServiceResponse(SetBoolResponse response)
    {
        Debug.Log($"Motor enabled: {response.success}, Message: {response.message}");
    }
}
```

---

## Synthetic Data Generation

### Why Synthetic Data?

Training perception models requires massive labeled datasets. Real-world data collection is:

- **Expensive**: Hours of human labeling
- **Limited**: Hard to capture edge cases
- **Biased**: May not cover all scenarios

Synthetic data solves these problems with:

- **Automatic labeling**: Perfect ground truth
- **Unlimited scale**: Generate millions of images
- **Controlled variation**: Test specific scenarios

### Unity Perception Package

The Perception package provides tools for generating labeled training data:

```
┌─────────────────────────────────────────────────────────────┐
│             Synthetic Data Generation Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │   Scene     │   │   Camera    │   │   Labelers      │   │
│  │   Setup     │──▶│   Capture   │──▶│   (Annotation)  │   │
│  └─────────────┘   └─────────────┘   └─────────────────┘   │
│                                              │               │
│                                              ▼               │
│                                    ┌─────────────────┐      │
│                                    │  Output Dataset │      │
│                                    │  - RGB Images   │      │
│                                    │  - Bounding Box │      │
│                                    │  - Segmentation │      │
│                                    │  - Depth Maps   │      │
│                                    └─────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Setting Up Perception

1. **Add Perception Camera**:

```csharp
// Attach to your camera
using UnityEngine.Perception.GroundTruth;

public class PerceptionSetup : MonoBehaviour
{
    void Start()
    {
        var perceptionCamera = gameObject.AddComponent<PerceptionCamera>();

        // Add labelers
        perceptionCamera.AddLabeler(new BoundingBox2DLabeler());
        perceptionCamera.AddLabeler(new SemanticSegmentationLabeler());
        perceptionCamera.AddLabeler(new InstanceSegmentationLabeler());
    }
}
```

2. **Label Objects**:

```csharp
using UnityEngine.Perception.GroundTruth;

// Add to objects you want to detect
public class ObjectLabeler : MonoBehaviour
{
    void Start()
    {
        var labeling = gameObject.AddComponent<Labeling>();
        labeling.labels.Add("robot");
        labeling.labels.Add("humanoid");
    }
}
```

3. **Configure Output**:

```json
// Perception settings (via UI or code)
{
  "outputPath": "PerceptionOutput",
  "captureFormat": "PNG",
  "capturesPerIteration": 1,
  "framesPerCapture": 1
}
```

### Output Format

Unity Perception generates COCO-compatible annotations:

```json
{
  "captures": [
    {
      "id": "frame_001",
      "filename": "rgb/frame_001.png",
      "annotations": [
        {
          "label_id": 1,
          "label_name": "robot",
          "instance_id": 42,
          "bounding_box": {
            "x": 120,
            "y": 80,
            "width": 200,
            "height": 350
          }
        }
      ]
    }
  ]
}
```

---

## Domain Randomization

### What is Domain Randomization?

**Domain Randomization** varies simulation parameters to help ML models generalize to real-world conditions:

```
┌─────────────────────────────────────────────────────────────┐
│               Domain Randomization Strategy                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Randomize during training:                                 │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Lighting   │  │  Textures   │  │  Object Positions   │ │
│  │  - Intensity│  │  - Colors   │  │  - Random spawns    │ │
│  │  - Color    │  │  - Patterns │  │  - Orientations     │ │
│  │  - Direction│  │  - Materials│  │  - Scale variations │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Camera    │  │   Noise     │  │   Distractors       │ │
│  │  - Position │  │  - Gaussian │  │  - Background       │ │
│  │  - FOV      │  │  - Blur     │  │  - Foreground       │ │
│  │  - Exposure │  │  - Occlusion│  │  - Clutter          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  Result: Model learns to handle real-world variations!     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementing Randomizers

```csharp
using UnityEngine;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Parameters;

[AddRandomizerMenu("Custom/Lighting Randomizer")]
public class LightingRandomizer : Randomizer
{
    public FloatParameter lightIntensity = new FloatParameter { value = new UniformSampler(0.5f, 2.0f) };
    public ColorHsvaParameter lightColor = new ColorHsvaParameter();

    private Light sceneLight;

    protected override void OnIterationStart()
    {
        if (sceneLight == null)
            sceneLight = FindObjectOfType<Light>();

        sceneLight.intensity = lightIntensity.Sample();
        sceneLight.color = lightColor.Sample();
    }
}

[AddRandomizerMenu("Custom/Object Position Randomizer")]
public class ObjectPositionRandomizer : Randomizer
{
    public FloatParameter xPosition = new FloatParameter { value = new UniformSampler(-5f, 5f) };
    public FloatParameter zPosition = new FloatParameter { value = new UniformSampler(-5f, 5f) };
    public FloatParameter yRotation = new FloatParameter { value = new UniformSampler(0f, 360f) };

    public GameObject targetObject;

    protected override void OnIterationStart()
    {
        if (targetObject != null)
        {
            targetObject.transform.position = new Vector3(
                xPosition.Sample(),
                targetObject.transform.position.y,
                zPosition.Sample()
            );
            targetObject.transform.rotation = Quaternion.Euler(0, yRotation.Sample(), 0);
        }
    }
}
```

### Texture Randomization

```csharp
using UnityEngine;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Parameters;

[AddRandomizerMenu("Custom/Texture Randomizer")]
public class TextureRandomizer : Randomizer
{
    public Texture2D[] texturePool;
    public CategoricalParameter<Texture2D> textureParameter;

    private Renderer[] targetRenderers;

    protected override void OnScenarioStart()
    {
        textureParameter = new CategoricalParameter<Texture2D>();
        foreach (var tex in texturePool)
            textureParameter.AddOption(tex);

        targetRenderers = FindObjectsOfType<Renderer>();
    }

    protected override void OnIterationStart()
    {
        foreach (var renderer in targetRenderers)
        {
            if (renderer.CompareTag("Randomizable"))
            {
                renderer.material.mainTexture = textureParameter.Sample();
            }
        }
    }
}
```

---

## Creating Photorealistic Environments

### Universal Render Pipeline (URP) Setup

For robotics applications requiring visual realism:

1. **Enable URP features**:
   - Screen Space Ambient Occlusion (SSAO)
   - Screen Space Reflections (SSR)
   - Post-processing (Bloom, Color Grading)

2. **Configure lighting**:

```csharp
using UnityEngine;
using UnityEngine.Rendering.Universal;

public class RealisticLightingSetup : MonoBehaviour
{
    public Light sunLight;
    public ReflectionProbe environmentProbe;

    void Start()
    {
        // Configure sun
        sunLight.type = LightType.Directional;
        sunLight.shadows = LightShadows.Soft;
        sunLight.shadowResolution = UnityEngine.Rendering.LightShadowResolution.VeryHigh;
        sunLight.color = new Color(1f, 0.95f, 0.9f); // Warm sunlight
        sunLight.intensity = 1.5f;

        // Configure environment reflections
        environmentProbe.mode = UnityEngine.Rendering.ReflectionProbeMode.Realtime;
        environmentProbe.refreshMode = UnityEngine.Rendering.ReflectionProbeRefreshMode.EveryFrame;
    }
}
```

### PBR Materials for Robots

Create physically accurate materials:

```csharp
public class MetalMaterialSetup : MonoBehaviour
{
    void Start()
    {
        var renderer = GetComponent<Renderer>();
        var material = new Material(Shader.Find("Universal Render Pipeline/Lit"));

        // Brushed metal appearance
        material.SetFloat("_Metallic", 0.9f);
        material.SetFloat("_Smoothness", 0.7f);
        material.SetColor("_BaseColor", new Color(0.8f, 0.8f, 0.85f));

        renderer.material = material;
    }
}
```

---

## Human-Robot Interaction Scenarios

### Why Unity for HRI?

Unity excels at human-robot interaction research:

- **Character animation**: Realistic human movements
- **Facial expressions**: Emotional responses
- **Voice integration**: Speech synthesis and recognition
- **Social scenarios**: Crowd simulation

### Setting Up Human Characters

```csharp
using UnityEngine;

public class HumanCharacterController : MonoBehaviour
{
    private Animator animator;
    public Transform robotTarget;
    public float interactionDistance = 2f;

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        float distance = Vector3.Distance(transform.position, robotTarget.position);

        if (distance < interactionDistance)
        {
            // Face the robot
            Vector3 direction = robotTarget.position - transform.position;
            direction.y = 0;
            transform.rotation = Quaternion.LookRotation(direction);

            // Trigger interaction animation
            animator.SetBool("IsInteracting", true);
        }
        else
        {
            animator.SetBool("IsInteracting", false);
        }
    }
}
```

### Gesture Recognition Integration

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class GesturePublisher : MonoBehaviour
{
    private ROSConnection ros;
    private Animator humanAnimator;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>("/detected_gesture");
        humanAnimator = GetComponent<Animator>();
    }

    public void OnGestureDetected(string gestureName)
    {
        // Publish gesture to ROS 2
        StringMsg msg = new StringMsg { data = gestureName };
        ros.Publish("/detected_gesture", msg);

        // Trigger corresponding animation
        humanAnimator.SetTrigger(gestureName);
    }
}
```

---

## Practical Exercise: Complete Unity-ROS 2 Pipeline

### Goal

Create a Unity simulation that:
1. Imports a robot from URDF
2. Generates synthetic training data
3. Communicates with ROS 2

### Step-by-Step Implementation

**1. Project Structure**:
```
Assets/
├── Robots/
│   └── my_robot.urdf
├── Scripts/
│   ├── RobotController.cs
│   ├── DataGenerator.cs
│   └── ROSBridge.cs
├── Scenes/
│   └── RoboticsSimulation.unity
└── Randomizers/
    └── CustomRandomizers.cs
```

**2. Main Controller Script**:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class RobotSimulationController : MonoBehaviour
{
    private ROSConnection ros;
    private Camera robotCamera;
    public ArticulationBody robotBase;

    // ROS topics
    private string imageTopic = "/camera/image_raw";
    private string cmdVelTopic = "/cmd_vel";
    private string jointStateTopic = "/joint_states";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Register publishers
        ros.RegisterPublisher<ImageMsg>(imageTopic);
        ros.RegisterPublisher<JointStateMsg>(jointStateTopic);

        // Subscribe to velocity commands
        ros.Subscribe<TwistMsg>(cmdVelTopic, OnCmdVelReceived);

        // Start publishing
        InvokeRepeating("PublishSensorData", 0.1f, 0.033f); // 30Hz
    }

    void OnCmdVelReceived(TwistMsg msg)
    {
        // Apply velocity to robot
        Vector3 linearVel = new Vector3(
            (float)msg.linear.x,
            (float)msg.linear.y,
            (float)msg.linear.z
        );

        Vector3 angularVel = new Vector3(
            (float)msg.angular.x,
            (float)msg.angular.y,
            (float)msg.angular.z
        );

        robotBase.velocity = linearVel;
        robotBase.angularVelocity = angularVel;
    }

    void PublishSensorData()
    {
        // Publish camera image
        // Publish joint states
    }
}
```

**3. Launch ROS 2 Side**:

```bash
# Terminal 1: ROS-TCP Endpoint
ros2 run ros_tcp_endpoint default_server_endpoint

# Terminal 2: Verify connection
ros2 topic list

# Terminal 3: Send commands
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5}, angular: {z: 0.1}}"
```

---

## Hybrid Simulation: Unity + Gazebo

### When to Use Both

For complex projects, combine simulators:

| Component | Simulator | Reason |
|-----------|-----------|--------|
| Physics simulation | Gazebo | More accurate dynamics |
| Visual rendering | Unity | Photorealistic output |
| ML training data | Unity | Domain randomization |
| Control testing | Gazebo | ROS 2 native |
| Demonstration | Unity | Better visuals |

### Architecture for Hybrid Simulation

```
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Simulation Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │     Gazebo      │◀────────▶│      Unity      │          │
│  │  (Physics)      │   Sync   │   (Rendering)   │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │  Joint States   │          │  Camera Images  │          │
│  │  Sensor Data    │          │  Training Data  │          │
│  │  Physics State  │          │  Visualization  │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│           └──────────┬─────────────────┘                   │
│                      ▼                                      │
│              ┌───────────────┐                             │
│              │    ROS 2      │                             │
│              │  Middleware   │                             │
│              └───────────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

In this chapter, you learned:

- **Unity's role in robotics**: Photorealistic simulation and synthetic data
- **ROS 2 integration**: Bidirectional communication via ROS-TCP-Connector
- **URDF import**: Bringing ROS robots into Unity with articulated physics
- **Synthetic data generation**: Using the Perception package for ML training
- **Domain randomization**: Creating robust models that transfer to reality
- **Human-robot interaction**: Building scenarios with human characters
- **Hybrid approaches**: Combining Unity and Gazebo strengths

Unity fills a critical gap in the robotics simulation ecosystem by providing the visual fidelity needed for modern perception systems.

---

## Further Reading

- [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub) - Official Unity robotics resources
- [Unity Perception Package](https://github.com/Unity-Technologies/com.unity.perception) - Synthetic data generation
- [ROS-TCP-Connector](https://github.com/Unity-Technologies/ROS-TCP-Connector) - ROS 2 integration
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) - Reinforcement learning
- [Domain Randomization Paper](https://arxiv.org/abs/1703.06907) - Original research on sim-to-real transfer

---

## Next Week Preview

In **Chapter 8**, we enter **Module 3: The AI-Robot Brain** with **NVIDIA Isaac Sim**:
- GPU-accelerated physics simulation
- Integration with NVIDIA's AI stack
- Advanced perception with Isaac ROS
- Synthetic data at scale with Omniverse Replicator
