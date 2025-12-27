# Research: Physical AI Textbook Foundation

**Date**: 2025-12-27
**Feature**: 001-textbook-foundation
**Purpose**: Resolve technical unknowns and establish best practices for Docusaurus-based educational content

---

## 1. Docusaurus v3.x Setup Best Practices

### Decision: Use Docusaurus 3.0+ with Classic Preset

**Rationale**:
- Mature ecosystem with extensive documentation
- Built-in features align with requirements (sidebar navigation, search, responsive design, MDX support)
- Active community and plugin ecosystem
- Zero-config deployment to GitHub Pages
- Performance-optimized out of the box (meets Lighthouse ≥90 target)

**Alternatives Considered**:
- **VitePress**: Lighter weight, but less feature-complete for educational sites (no built-in search, limited theming)
- **Jekyll**: Older technology, Ruby dependency, less modern developer experience
- **Next.js + MDX**: Over-engineered for static content site, requires custom navigation/search implementation

**Implementation Approach**:
```bash
npx create-docusaurus@latest physical-ai-textbook classic --typescript
```

**Key Configuration Decisions**:
- Use TypeScript for type safety in custom components
- Enable `@docusaurus/preset-classic` for standard features (docs, blog, pages, theme)
- Configure `docs` as the default landing section (educational content primary)
- Set `onBrokenLinks: 'throw'` to catch navigation errors during build

---

## 2. GitHub Pages Deployment Strategy

### Decision: GitHub Actions with gh-pages Branch

**Rationale**:
- Automated deployment on every push to main branch
- Docusaurus official documentation recommends this approach
- No manual deployment steps required
- Integrates with GitHub's free hosting (hackathon requirement)

**Deployment Workflow**:
1. Developer pushes code to `001-textbook-foundation` branch
2. Create PR to `main` branch
3. On merge to `main`, GitHub Actions workflow triggers:
   - Installs Node.js dependencies (`npm ci`)
   - Builds Docusaurus site (`npm run build`)
   - Deploys `/build` directory to `gh-pages` branch
   - GitHub Pages serves site from `gh-pages` branch

**GitHub Actions Configuration** (`.github/workflows/deploy.yml`):
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches:
      - main

permissions:
  contents: write

jobs:
  deploy:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 18
          cache: npm

      - name: Install dependencies
        run: npm ci

      - name: Build website
        run: npm run build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build
          user_name: github-actions[bot]
          user_email: github-actions[bot]@users.noreply.github.com
```

**Custom Domain** (Optional):
- For hackathon demo, use default `<username>.github.io/<repo-name>` URL
- Custom domain can be configured via `CNAME` file in `/static` directory (future enhancement)

**Alternatives Considered**:
- **Netlify/Vercel**: Requires third-party account, adds deployment complexity
- **/docs folder strategy**: Less flexible, requires keeping built assets in source control

---

## 3. Visual Aid Creation Tools

### Decision: Excalidraw + Unsplash/Custom Photography

**Rationale**:
- **Excalidraw** (https://excalidraw.com): Free, browser-based, perfect for technical diagrams (robot architectures, sensor layouts, system diagrams)
- **Unsplash**: High-quality, royalty-free images for robotics/AI concepts
- **Custom screenshots**: For specific ROS 2/NVIDIA Isaac UI demonstrations (Phase 3)

**Image Optimization Strategy**:
- Export Excalidraw diagrams as SVG (scalable, small file size)
- Compress raster images with Squoosh (https://squoosh.app) to <200KB
- Use WebP format for photos (better compression than JPEG)
- Provide descriptive alt text for accessibility (Principle V compliance)

**Alt Text Guidelines**:
- **Bad**: "Image 1" or "Robot diagram"
- **Good**: "Humanoid robot system architecture showing ROS 2 nodes for perception, planning, and control with sensor inputs from LIDAR and cameras"

**Image Specifications**:
- Max width: 1200px (retina displays)
- Max file size: 200KB per image
- Formats: SVG (diagrams), WebP (photos), PNG (fallback)
- Naming convention: `week{N}-{topic}-{description}.{ext}` (e.g., `week1-embodied-intelligence-comparison.svg`)

**Alternatives Considered**:
- **Figma**: Overkill for simple diagrams, requires account
- **Draw.io**: More complex UI, less modern aesthetic
- **Stock photography (paid)**: Unnecessary expense, free alternatives sufficient

---

## 4. MDX Content Authoring Patterns

### Decision: MDX with Prism Syntax Highlighting + KaTeX Math

**Rationale**:
- MDX allows React components in markdown (future interactive elements)
- Prism (Docusaurus default) supports Python, C++, YAML, Bash (required for ROS 2 examples)
- KaTeX renders math equations efficiently (faster than MathJax)

**Code Block Pattern**:
````markdown
```python title="src/robot_controller.py"
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info('Robot controller initialized')
```
````

**Math Equation Pattern**:
```markdown
The robot's forward kinematics are calculated using:

$$
\mathbf{T} = \prod_{i=1}^{n} \mathbf{T}_i(\theta_i)
$$

Where $\mathbf{T}_i$ represents the transformation matrix for joint $i$.
```

**Required Docusaurus Plugins**:
- `remark-math` + `rehype-katex` for LaTeX math rendering
- `@docusaurus/theme-live-codeblock` for interactive code examples (future Phase 3)

**Content Structure Template**:
```markdown
---
sidebar_position: 1
title: Week 1 - Foundations of Physical AI
description: Introduction to embodied intelligence and Physical AI concepts
---

# Week 1: Foundations of Physical AI

## Learning Objectives
- Define Physical AI and embodied intelligence
- Differentiate Physical AI from digital AI systems
- Identify key sensor modalities in robotics

## 1. What is Physical AI?

Physical AI refers to...

[Content sections with headers, paragraphs, code blocks, images]

## Summary

In this chapter, you learned...

## Further Reading
- [Link to authoritative source]
```

**Alternatives Considered**:
- **Plain Markdown**: Lacks component flexibility for Phase 2-4 enhancements
- **MathJax**: Slower rendering than KaTeX, unnecessary features

---

## 5. Performance Optimization Techniques

### Decision: Built-in Optimizations + Lazy Loading

**Rationale**:
- Docusaurus provides optimized production builds by default (code splitting, minification, tree-shaking)
- Native lazy loading for images (`loading="lazy"`) meets performance budget
- Lighthouse CI integration catches regressions

**Optimization Checklist**:
- ✅ **Code Splitting**: Docusaurus automatically splits routes
- ✅ **Image Lazy Loading**: Add `loading="lazy"` attribute to all `<img>` tags
- ✅ **Bundle Size**: Monitor with `npm run build` output, target <500KB gzipped
- ✅ **Font Optimization**: Use system fonts (no web font downloads)
- ✅ **Critical CSS**: Inlined automatically by Docusaurus

**Lighthouse CI Integration**:
```yaml
# .github/workflows/lighthouse-ci.yml
name: Lighthouse CI
on: [pull_request]

jobs:
  lighthouse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 18
      - run: npm ci && npm run build
      - run: npm install -g @lhci/cli
      - run: lhci autorun
```

**Lighthouse Budget** (`.lighthouserc.js`):
```javascript
module.exports = {
  ci: {
    collect: {
      staticDistDir: './build',
    },
    assert: {
      preset: 'lighthouse:recommended',
      assertions: {
        'first-contentful-paint': ['error', {maxNumericValue: 1500}],
        'interactive': ['error', {maxNumericValue: 3500}],
        'categories:performance': ['error', {minScore: 0.9}],
        'categories:accessibility': ['error', {minScore: 0.9}],
      },
    },
  },
};
```

**Alternatives Considered**:
- **Next.js Image Optimization**: Requires backend server, incompatible with GitHub Pages static hosting
- **Manual webpack config**: Unnecessary complexity, Docusaurus defaults sufficient

---

## Summary of Key Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| **Framework** | Docusaurus 3.0 + Classic Preset | Built-in features, educational site optimized, active ecosystem |
| **Deployment** | GitHub Actions → gh-pages branch | Automated, zero-cost, integrates with GitHub Pages |
| **Visual Aids** | Excalidraw (diagrams) + Unsplash (photos) | Free, high-quality, web-optimized |
| **Content Format** | MDX + Prism + KaTeX | Code highlighting, math rendering, component flexibility |
| **Performance** | Docusaurus defaults + lazy loading | Meets Lighthouse ≥90 target, minimal configuration |
| **CI/CD** | GitHub Actions (deploy + Lighthouse) | Automated quality gates, prevents regressions |

**No Unresolved Clarifications** - All technical context items have been researched and decided.

---

## Next Steps

Proceed to **Phase 1: Design** to create:
1. `data-model.md` - Content structure and metadata schema
2. `quickstart.md` - Developer setup guide
3. Update agent context with Docusaurus technology stack
