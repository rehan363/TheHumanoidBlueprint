# Quickstart Guide: Physical AI Textbook Development

**Last Updated**: 2025-12-27
**Feature**: 001-textbook-foundation
**Purpose**: Step-by-step guide for local development, content authoring, and deployment

---

## Prerequisites

### Required Software

| Software | Version | Installation Link |
|----------|---------|-------------------|
| **Node.js** | 18.x LTS or higher | [nodejs.org](https://nodejs.org/) |
| **npm** | 9.x or higher (comes with Node.js) | Included with Node.js |
| **Git** | 2.x or higher | [git-scm.com](https://git-scm.com/) |
| **Code Editor** | Any (VS Code recommended) | [code.visualstudio.com](https://code.visualstudio.com/) |

### Verify Installation

```bash
node --version  # Should output v18.x.x or higher
npm --version   # Should output 9.x.x or higher
git --version   # Should output 2.x.x or higher
```

---

## Initial Setup

### 1. Clone Repository

```bash
git clone <repository-url> physical-ai-textbook
cd physical-ai-textbook
```

### 2. Install Dependencies

```bash
npm install
```

**What this does**:
- Installs Docusaurus v3.x and all required packages
- Sets up React, MDX, syntax highlighting, math rendering
- Configures development tools

**Expected output**:
```
added 1234 packages in 45s
```

### 3. Start Development Server

```bash
npm start
```

**What this does**:
- Starts local Docusaurus development server
- Opens browser at `http://localhost:3000`
- Enables hot-reload (changes auto-refresh)

**Expected output**:
```
[SUCCESS] Docusaurus website is running at: http://localhost:3000/
```

---

## Content Authoring Workflow

### Creating a New Chapter

**Step 1**: Create markdown file in appropriate module directory

```bash
# Example: Creating Week 1 content
touch docs/module1-ros2/week1-foundations.md
```

**Step 2**: Add frontmatter metadata

```yaml
---
sidebar_position: 1
title: "Week 1 - Foundations of Physical AI"
description: "Introduction to Physical AI and embodied intelligence"
keywords: [Physical AI, embodied intelligence, robotics]
last_updated: "2025-12-27"
estimated_reading_time: 15
---
```

**Step 3**: Write content following the [data-model.md](./data-model.md) template

```markdown
# Week 1: Foundations of Physical AI

**Estimated Reading Time**: 15 minutes

---

## Learning Objectives

- Define Physical AI and embodied intelligence
- [Additional objectives...]

---

## 1. What is Physical AI?

Physical AI refers to artificial intelligence systems that interact directly with the physical world...

[Continue with sections, images, code examples...]
```

**Step 4**: Save file and verify in browser (hot-reload should update automatically)

### Adding Visual Aids

**Step 1**: Create/export diagram (use Excalidraw, export as SVG)

**Step 2**: Optimize image size

```bash
# For PNG/JPG/WebP:
# Use https://squoosh.app (browser-based tool)
# Target: <200KB per image
```

**Step 3**: Save to appropriate assets directory

```bash
mkdir -p docs/assets/week1
cp physical-ai-comparison.svg docs/assets/week1/
```

**Step 4**: Reference in markdown

```markdown
![Physical AI vs Digital AI](./assets/week1/physical-ai-comparison.svg "Comparison of AI paradigms")
*Figure 1.1: Digital AI vs. Physical AI - Key Differences*
```

**Step 5**: Verify alt text and caption for accessibility

### Adding Code Examples

Use fenced code blocks with language identifier and optional title:

````markdown
```python title="src/robot_controller.py"
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info('Robot initialized')

if __name__ == '__main__':
    rclpy.init()
    controller = RobotController()
    rclpy.spin(controller)
```
````

**Supported languages**: python, cpp, yaml, bash, javascript, typescript, xml, json

### Adding Math Equations

**Inline math**: Use single dollar signs

```markdown
The transformation matrix $\mathbf{T}$ represents...
```

**Block math**: Use double dollar signs

```markdown
$$
\mathbf{T} = \prod_{i=1}^{n} \mathbf{T}_i(\theta_i)
$$
```

**Requirements**: KaTeX plugin (automatically configured)

---

## Project Structure Reference

```
physical-ai-textbook/
├── docs/                       # All content here
│   ├── intro.md               # Homepage
│   ├── module1-ros2/          # Module 1 chapters
│   ├── module2-gazebo/        # Module 2 chapters
│   ├── module3-isaac/         # Module 3 chapters
│   ├── module4-vla/           # Module 4 chapters
│   └── assets/                # Images, diagrams, videos
├── src/                       # Custom React components (if needed)
│   ├── components/
│   └── css/
├── static/                    # Static assets (logo, favicon)
├── docusaurus.config.js       # Site configuration
├── sidebars.js                # Navigation structure
├── package.json               # Dependencies
└── .github/
    └── workflows/
        └── deploy.yml         # Deployment automation
```

---

## Building for Production

### Build Command

```bash
npm run build
```

**What this does**:
- Compiles all markdown to HTML
- Bundles JavaScript/CSS (minified, optimized)
- Generates static files in `build/` directory
- Validates all links (fails on broken links)

**Expected output**:
```
[SUCCESS] Generated static files in "build".
[SUCCESS] Use `npm run serve` to test your build locally.
```

### Test Production Build Locally

```bash
npm run serve
```

Opens production build at `http://localhost:3000` (no hot-reload, exact production behavior)

---

## Deployment to GitHub Pages

### Automatic Deployment (Recommended)

**Trigger**: Push to `main` branch automatically deploys via GitHub Actions

```bash
git add .
git commit -m "feat: add Week 1 content"
git push origin main
```

**GitHub Actions workflow** (`.github/workflows/deploy.yml`) handles:
1. Install dependencies
2. Build site
3. Deploy to `gh-pages` branch
4. GitHub Pages serves from `gh-pages`

**View deployment status**: GitHub repo → Actions tab

**Live URL**: `https://<username>.github.io/<repo-name>/`

### Manual Deployment (If Needed)

```bash
# Build site
npm run build

# Deploy to GitHub Pages
GIT_USER=<your-github-username> npm run deploy
```

**Note**: Automatic deployment via GitHub Actions is preferred (no manual steps required)

---

## Configuration Files

### `docusaurus.config.js`

Key settings to customize:

```javascript
module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A Comprehensive Guide to Physical AI Education',
  url: 'https://<username>.github.io',
  baseUrl: '/<repo-name>/',
  organizationName: '<github-username>',
  projectName: '<repo-name>',

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/<username>/<repo>/edit/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
```

### `sidebars.js`

Defines navigation structure (see [data-model.md](./data-model.md#sidebar-configuration-sidebarsjs) for full example)

---

## Troubleshooting

### Issue: `npm install` fails

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

### Issue: Hot-reload not working

**Solution**:
```bash
# Stop dev server (Ctrl+C)
# Restart
npm start
```

### Issue: Broken link error during build

**Error message**:
```
Error: Docusaurus found broken links!
```

**Solution**:
- Check file paths in markdown (case-sensitive)
- Verify all referenced images exist
- Fix relative links (use `./` for same directory, `../` for parent)

### Issue: Math equations not rendering

**Solution**:
```bash
# Install math plugins
npm install remark-math@3 rehype-katex@5
```

Then add to `docusaurus.config.js`:
```javascript
presets: [
  [
    'classic',
    {
      docs: {
        remarkPlugins: [require('remark-math')],
        rehypePlugins: [require('rehype-katex')],
      },
    },
  ],
],
stylesheets: [
  {
    href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
    type: 'text/css',
    integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
    crossorigin: 'anonymous',
  },
],
```

### Issue: GitHub Pages not updating

**Solution**:
1. Check GitHub Actions tab for deployment status
2. Verify `gh-pages` branch exists and has recent commits
3. Check GitHub repo settings → Pages → Source is set to `gh-pages` branch
4. Clear browser cache and hard refresh (Ctrl+Shift+R)

### Issue: Images not loading

**Solution**:
- Verify image path is relative to markdown file
- Check file extension matches actual file (`.svg` not `.SVG`)
- Ensure image is committed to Git and pushed

---

## Content Quality Checklist

Before committing chapters, verify:

- [ ] Frontmatter includes all required fields (title, description, keywords, date, reading time)
- [ ] All code examples are syntax-highlighted and tested
- [ ] Images include descriptive alt text (50-150 chars)
- [ ] Images are optimized (<200KB each)
- [ ] Math equations render correctly
- [ ] No broken links (run `npm run build` to verify)
- [ ] Chapter follows template structure from [data-model.md](./data-model.md)
- [ ] Content verified against authoritative sources (Constitution Principle III)

---

## Performance Monitoring

### Lighthouse Audit (Local)

```bash
# Build production site
npm run build

# Serve locally
npm run serve

# In Chrome DevTools:
# 1. Open DevTools (F12)
# 2. Navigate to "Lighthouse" tab
# 3. Select "Performance", "Accessibility", "SEO"
# 4. Click "Analyze page load"

# Target scores:
# - Performance: ≥90 (desktop), ≥80 (mobile)
# - Accessibility: ≥90
# - SEO: ≥90
```

### Lighthouse CI (Automated)

GitHub Actions workflow runs Lighthouse on every PR:
- Performance budget enforced
- Fails PR if scores drop below thresholds
- Reports available in Actions tab

---

## Useful Commands

| Command | Purpose |
|---------|---------|
| `npm start` | Start development server with hot-reload |
| `npm run build` | Build production site (validates links) |
| `npm run serve` | Serve production build locally |
| `npm run deploy` | Manual deployment to GitHub Pages |
| `npm run clear` | Clear Docusaurus cache (if build issues) |
| `npm run swizzle` | Customize Docusaurus components (advanced) |

---

## Additional Resources

- **Docusaurus Documentation**: https://docusaurus.io/docs
- **MDX Syntax**: https://mdxjs.com/
- **KaTeX Math**: https://katex.org/docs/supported.html
- **Excalidraw (Diagrams)**: https://excalidraw.com/
- **Squoosh (Image Optimization)**: https://squoosh.app/
- **GitHub Pages Setup**: https://docs.github.com/en/pages

---

## Next Steps

1. **Install dependencies**: `npm install`
2. **Start dev server**: `npm start`
3. **Create Week 1 content**: Follow content authoring workflow
4. **Add visual aids**: Create diagrams with Excalidraw
5. **Build and test**: `npm run build && npm run serve`
6. **Deploy**: Push to `main` branch (GitHub Actions deploys automatically)

**Questions?** Refer to [data-model.md](./data-model.md) for content structure or [research.md](./research.md) for technical decisions.
