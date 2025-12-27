# Data Model: Physical AI Textbook Foundation

**Date**: 2025-12-27
**Feature**: 001-textbook-foundation
**Purpose**: Define content structure, metadata schema, and navigation hierarchy for Docusaurus static site

---

## Overview

For Phase 1 (static site), the "data model" consists of:
1. **Content Structure**: How chapters are organized in the file system
2. **Chapter Metadata**: Frontmatter schema for each markdown file
3. **Navigation Hierarchy**: How sidebar menu is structured
4. **Visual Aid Specifications**: Image/diagram metadata and requirements

No database entities in Phase 1 - all data is file-based markdown.

---

## 1. Content Structure

### Directory Layout

```
docs/
├── intro.md                           # Homepage (Course Overview)
├── module1-ros2/                      # Module 1: The Robotic Nervous System
│   ├── _category_.json                # Module metadata
│   ├── week1-foundations.md           # Week 1: Foundations of Physical AI
│   ├── week2-landscape.md             # Week 2: Physical AI Landscape & Sensors
│   ├── week3-ros2-intro.md            # Week 3: Placeholder (Phase 3)
│   ├── week4-nodes-topics.md          # Week 4: Placeholder (Phase 3)
│   └── week5-urdf.md                  # Week 5: Placeholder (Phase 3)
├── module2-gazebo/                    # Module 2: The Digital Twin
│   ├── _category_.json
│   ├── week6-gazebo-setup.md          # Week 6: Placeholder (Phase 3)
│   └── week7-unity.md                 # Week 7: Placeholder (Phase 3)
├── module3-isaac/                     # Module 3: The AI-Robot Brain
│   ├── _category_.json
│   ├── week8-isaac-sim.md             # Week 8: Placeholder (Phase 3)
│   ├── week9-isaac-ros.md             # Week 9: Placeholder (Phase 3)
│   └── week10-nav2.md                 # Week 10: Placeholder (Phase 3)
├── module4-vla/                       # Module 4: Vision-Language-Action
│   ├── _category_.json
│   ├── week11-voice-action.md         # Week 11: Placeholder (Phase 3)
│   ├── week12-cognitive-planning.md   # Week 12: Placeholder (Phase 3)
│   └── week13-capstone.md             # Week 13: Placeholder (Phase 3)
└── assets/                            # Visual aids organized by week
    ├── week1/
    │   ├── physical-ai-comparison.svg
    │   └── embodied-intelligence-diagram.svg
    └── week2/
        ├── sensor-types-overview.svg
        ├── lidar-point-cloud.webp
        └── imu-axes.svg
```

### File Naming Convention

**Pattern**: `week{N}-{topic-slug}.md`

**Examples**:
- `week1-foundations.md` (Week 1: Foundations of Physical AI)
- `week2-landscape.md` (Week 2: Physical AI Landscape & Sensors)
- `week3-ros2-intro.md` (Week 3: ROS 2 Introduction)

**Rules**:
- Week numbers are always 2 digits (`week01` not `week1`) - **CORRECTION**: Use single digit for Weeks 1-9 to match Docusaurus sorting conventions
- Topic slugs are lowercase with hyphens (kebab-case)
- File extensions are `.md` or `.mdx` (use `.mdx` if React components needed)

---

## 2. Chapter Metadata Schema

Each chapter markdown file includes YAML frontmatter with this structure:

### Frontmatter Template

```yaml
---
sidebar_position: {N}                   # Integer: ordering in sidebar (1, 2, 3...)
title: "Week {N} - {Title}"             # String: display title in sidebar/page
description: "{Brief description}"      # String: SEO meta description, 50-160 chars
keywords: [{keyword1}, {keyword2}]      # Array: SEO keywords
last_updated: "{YYYY-MM-DD}"            # String: ISO date of last content update
estimated_reading_time: {N}             # Integer: minutes to read chapter
---
```

### Week 1 Example

```yaml
---
sidebar_position: 1
title: "Week 1 - Foundations of Physical AI"
description: "Introduction to Physical AI, embodied intelligence, and how AI systems interact with the physical world"
keywords: [Physical AI, embodied intelligence, robotics, AI agents, sensors]
last_updated: "2025-12-27"
estimated_reading_time: 15
---
```

### Week 2 Example

```yaml
---
sidebar_position: 2
title: "Week 2 - Physical AI Landscape & Sensors"
description: "Overview of humanoid robotics landscape and sensor systems including LIDAR, cameras, IMUs, and force/torque sensors"
keywords: [humanoid robots, LIDAR, depth cameras, IMU, sensors, robotics hardware]
last_updated: "2025-12-27"
estimated_reading_time: 18
---
```

### Module Category Metadata

Each module directory includes `_category_.json` for module-level configuration:

```json
{
  "label": "Module 1: The Robotic Nervous System (ROS 2)",
  "position": 1,
  "link": {
    "type": "generated-index",
    "description": "Learn ROS 2 fundamentals, nodes, topics, services, and robot description formats (URDF). Weeks 1-5 of the curriculum."
  },
  "collapsed": false
}
```

---

## 3. Navigation Hierarchy

### Sidebar Structure

```
Physical AI & Humanoid Robotics
├── 📖 Introduction (Homepage)
└── 📚 Curriculum
    ├── 🤖 Module 1: The Robotic Nervous System (ROS 2)
    │   ├── Week 1 - Foundations of Physical AI
    │   ├── Week 2 - Physical AI Landscape & Sensors
    │   ├── Week 3 - ROS 2 Introduction (Placeholder)
    │   ├── Week 4 - Nodes, Topics, and Services (Placeholder)
    │   └── Week 5 - URDF for Humanoids (Placeholder)
    ├── 🎮 Module 2: The Digital Twin (Gazebo & Unity)
    │   ├── Week 6 - Gazebo Simulation (Placeholder)
    │   └── Week 7 - Unity for Robotics (Placeholder)
    ├── 🧠 Module 3: The AI-Robot Brain (NVIDIA Isaac)
    │   ├── Week 8 - Isaac Sim Introduction (Placeholder)
    │   ├── Week 9 - Isaac ROS & VSLAM (Placeholder)
    │   └── Week 10 - Nav2 Path Planning (Placeholder)
    └── 👁️ Module 4: Vision-Language-Action (VLA)
        ├── Week 11 - Voice-to-Action (Placeholder)
        ├── Week 12 - Cognitive Planning with LLMs (Placeholder)
        └── Week 13 - Capstone Project (Placeholder)
```

### Sidebar Configuration (`sidebars.js`)

```javascript
module.exports = {
  curriculumSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: '📖 Course Overview',
    },
    {
      type: 'category',
      label: '🤖 Module 1: ROS 2',
      collapsible: true,
      collapsed: false, // Expanded by default for Week 1-2
      items: [
        'module1-ros2/week1-foundations',
        'module1-ros2/week2-landscape',
        'module1-ros2/week3-ros2-intro',
        'module1-ros2/week4-nodes-topics',
        'module1-ros2/week5-urdf',
      ],
    },
    {
      type: 'category',
      label: '🎮 Module 2: Gazebo & Unity',
      collapsible: true,
      collapsed: true, // Collapsed by default (placeholders)
      items: [
        'module2-gazebo/week6-gazebo-setup',
        'module2-gazebo/week7-unity',
      ],
    },
    {
      type: 'category',
      label: '🧠 Module 3: NVIDIA Isaac',
      collapsible: true,
      collapsed: true,
      items: [
        'module3-isaac/week8-isaac-sim',
        'module3-isaac/week9-isaac-ros',
        'module3-isaac/week10-nav2',
      ],
    },
    {
      type: 'category',
      label: '👁️ Module 4: VLA',
      collapsible: true,
      collapsed: true,
      items: [
        'module4-vla/week11-voice-action',
        'module4-vla/week12-cognitive-planning',
        'module4-vla/week13-capstone',
      ],
    },
  ],
};
```

---

## 4. Visual Aid Specifications

### Image Metadata Schema

Each visual aid must have associated metadata for accessibility and content management:

```json
{
  "filename": "week1-physical-ai-comparison.svg",
  "alt_text": "Side-by-side comparison showing digital AI (cloud servers, data centers) versus Physical AI (robots interacting with real-world objects and environments)",
  "caption": "Figure 1.1: Digital AI vs. Physical AI - Key Differences",
  "source": "Custom created with Excalidraw",
  "license": "CC BY 4.0",
  "file_size": "45KB",
  "dimensions": "1200x800",
  "format": "SVG"
}
```

### Visual Aid Requirements

| Attribute | Requirement | Example |
|-----------|-------------|---------|
| **Alt Text** | Descriptive, 50-150 chars, explains diagram/image content | "Humanoid robot system architecture showing ROS 2 perception, planning, and control nodes" |
| **Caption** | Figure number + brief description | "Figure 2.3: LIDAR Point Cloud Visualization" |
| **File Size** | <200KB (optimized) | 145KB (WebP compressed) |
| **Dimensions** | Max 1200px width, maintain aspect ratio | 1200x800 (landscape), 800x1200 (portrait) |
| **Format** | SVG (diagrams), WebP (photos), PNG (fallback) | `.svg`, `.webp`, `.png` |
| **Naming** | `week{N}-{topic}-{description}.{ext}` | `week2-sensor-types-overview.svg` |
| **License** | CC BY 4.0, Public Domain, or Custom | "CC BY 4.0" or "Custom created" |
| **Lazy Loading** | `loading="lazy"` attribute for all images | `<img src="..." loading="lazy" />` |

### Diagram Types by Week

**Week 1 Visual Aids**:
1. Physical AI vs. Digital AI comparison diagram
2. Embodied intelligence conceptual diagram
3. Robot-environment interaction flowchart

**Week 2 Visual Aids**:
1. Sensor types overview (LIDAR, cameras, IMUs, force/torque)
2. LIDAR point cloud visualization example
3. IMU axis orientation diagram
4. Depth camera working principle

### Markdown Image Syntax

```markdown
![Figure 1.1: Physical AI vs. Digital AI comparison](./assets/week1/physical-ai-comparison.svg "Digital AI operates in virtual environments while Physical AI interacts with the physical world")
*Figure 1.1: Digital AI vs. Physical AI - Key Differences*
```

**Rendered HTML**:
```html
<img
  src="./assets/week1/physical-ai-comparison.svg"
  alt="Figure 1.1: Physical AI vs. Digital AI comparison"
  title="Digital AI operates in virtual environments while Physical AI interacts with the physical world"
  loading="lazy"
/>
<em>Figure 1.1: Digital AI vs. Physical AI - Key Differences</em>
```

---

## 5. Content Template for Chapters

### Standard Chapter Structure

```markdown
---
sidebar_position: {N}
title: "Week {N} - {Title}"
description: "{Brief description}"
keywords: [{keyword1}, {keyword2}, ...]
last_updated: "{YYYY-MM-DD}"
estimated_reading_time: {N}
---

# Week {N}: {Full Title}

**Estimated Reading Time**: {N} minutes

---

## Learning Objectives

By the end of this chapter, you will be able to:

- {Objective 1}
- {Objective 2}
- {Objective 3}

---

## 1. {Section Title}

{Content with paragraphs, lists, code blocks, images}

![{Alt text}](./assets/week{N}/{image-file} "{Hover tooltip}")
*{Caption}*

### 1.1 {Subsection Title}

{More content...}

```{language} title="{filename}"
{Code example}
```

---

## 2. {Next Section Title}

{Content...}

---

## Summary

In this chapter, you learned:

- {Key takeaway 1}
- {Key takeaway 2}
- {Key takeaway 3}

---

## Further Reading

- [{Title}]({URL}) - {Brief description}
- [{Title}]({URL}) - {Brief description}

---

## Exercises

1. **{Exercise Title}**: {Description of task}
2. **{Exercise Title}**: {Description of task}

*(Answers will be provided in Phase 3 - future enhancement)*
```

---

## 6. Placeholder Content for Phase 3

Weeks 3-13 (not implemented in Phase 1) will use this placeholder template:

```markdown
---
sidebar_position: {N}
title: "Week {N} - {Title}"
description: "Content coming soon in Phase 3"
---

# Week {N}: {Title}

**Status**: 🚧 Content Under Development (Phase 3)

---

## Overview

This chapter will cover:

- {Expected topic 1}
- {Expected topic 2}
- {Expected topic 3}

**Estimated Completion**: Phase 3 Content Scale

---

## Curriculum Context

This chapter is part of **Module {X}: {Module Title}** and builds upon concepts from:

- Week {N-1}: {Previous Chapter}

It prepares you for:

- Week {N+1}: {Next Chapter}

---

## Sneak Peek

Key topics that will be covered:

### {Topic 1}

*(Details coming soon)*

### {Topic 2}

*(Details coming soon)*

---

*Check back soon! Content is actively being developed.*
```

---

## Summary

This data model defines:
- ✅ **File organization**: Modules → Weeks, assets by week
- ✅ **Metadata schema**: Frontmatter for titles, descriptions, keywords, dates
- ✅ **Navigation structure**: Sidebar hierarchy with collapsible modules
- ✅ **Visual aid specs**: Alt text, captions, file size, naming, licensing
- ✅ **Content templates**: Standard chapter structure for consistency

**Next Step**: Create `quickstart.md` with developer setup instructions.
