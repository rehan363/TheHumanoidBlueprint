import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 * - create an ordered group of docs
 * - render a sidebar for each doc of that group
 * - provide next/previous navigation
 *
 * The sidebars can be generated from the filesystem, or explicitly defined here.
 *
 * Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
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
      collapsed: false, // Expanded by default for Chapter 1-2
      items: [
        'module1-ros2/chapter1-foundations',
        'module1-ros2/chapter2-landscape',
        'module1-ros2/chapter3-ros2-intro',
        'module1-ros2/chapter4-nodes-topics',
        'module1-ros2/chapter5-urdf',
      ],
    },
    {
      type: 'category',
      label: '🎮 Module 2: Gazebo & Unity',
      collapsible: true,
      collapsed: true, // Collapsed by default (placeholders)
      items: [
        'module2-gazebo/chapter6-gazebo-setup',
        'module2-gazebo/chapter7-unity',
      ],
    },
    {
      type: 'category',
      label: '🧠 Module 3: NVIDIA Isaac',
      collapsible: true,
      collapsed: true,
      items: [
        'module3-isaac/chapter8-isaac-sim',
        'module3-isaac/chapter9-isaac-ros',
        'module3-isaac/chapter10-nav2',
      ],
    },
    {
      type: 'category',
      label: '👁️ Module 4: VLA',
      collapsible: true,
      collapsed: true,
      items: [
        'module4-vla/chapter11-voice-action',
        'module4-vla/chapter12-cognitive-planning',
        'module4-vla/chapter13-capstone',
      ],
    },
  ],
};

export default sidebars;
