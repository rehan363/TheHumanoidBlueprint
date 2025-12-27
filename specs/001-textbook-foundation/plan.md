# Implementation Plan: Physical AI Textbook Foundation

**Branch**: `001-textbook-foundation` | **Date**: 2025-12-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-textbook-foundation/spec.md`

## Summary

Phase 1 Foundation delivers a statically-generated educational website using Docusaurus v3.x, featuring the first two weeks of Physical AI curriculum content (Introduction to Physical AI). The site will be deployed to GitHub Pages with responsive design, sidebar navigation, search functionality, and visual aids. This establishes the content architecture and quality template for future chapters (Weeks 3-13 in Phase 3).

**Primary Requirement**: Prospective students and educators can browse a professional, mobile-responsive textbook site with comprehensive Week 1-2 content covering Physical AI foundations, embod

ied intelligence, robotics landscape, and sensor systems.

**Technical Approach**: Leverage Docusaurus's built-in features (sidebar navigation, search, MDX support, responsive theming) to minimize custom development. Focus effort on content authoring, visual aid creation, and deployment automation via GitHub Actions.

## Technical Context

**Language/Version**: JavaScript/TypeScript (Node.js 18.x LTS), MDX for content authoring
**Primary Dependencies**:
- Docusaurus v3.x (static site generator)
- React 18.x (Docusaurus runtime)
- @docusaurus/preset-classic (navigation, search, theming)
- Remark/Rehype plugins for markdown enhancement (code highlighting, math rendering)

**Storage**: Static markdown/MDX files in repository; no database for Phase 1
**Testing**:
- Docusaurus build validation (ensures no broken links/references)
- Lighthouse CI for performance/accessibility auditing
- Manual content review against constitution checklist

**Target Platform**: Static site hosted on GitHub Pages (Linux environment for build); accessed via modern browsers (Chrome, Firefox, Safari)
**Project Type**: Web application (frontend-only static site for Phase 1)
**Performance Goals**:
- First Contentful Paint <1.5s
- Time to Interactive <3.5s on 3G
- Lighthouse Performance ≥90 (desktop), ≥80 (mobile)
- Bundle size <500KB gzipped

**Constraints**:
- No backend/database in Phase 1 (RAG is Phase 2)
- Content must work without JavaScript (progressive enhancement)
- GitHub Pages deployment limitations (static files only, no server-side logic)
- Visual aids must be <200KB each for performance

**Scale/Scope**:
- Phase 1: 2 chapters (Weeks 1-2), ~3000+ words total content
- Future: 11 additional chapters (Phase 3)
- Expected users: <1000 concurrent (hackathon demo scale)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: AI-Driven Content Creation ✅
- Following Spec-Kit Plus workflow (Constitution → Spec → Plan → Tasks)
- PHRs being created for all significant decisions
- No architectural decisions requiring ADR in Phase 1 (using standard Docusaurus patterns)

### Principle II: Interactive Learning Experience ⏭️
- RAG chatbot deferred to Phase 2 (per Constitution's recommended implementation order)
- Phase 1 focuses on content foundation as prerequisite

### Principle III: Content Quality & Technical Accuracy ✅
- Week 1-2 content will be verified against authoritative Physical AI sources
- Code examples (if included) will follow ROS 2 Humble/Iron official documentation
- Visual aids will be custom-created or sourced from Creative Commons
- Content review checklist will be applied before merge

### Principle IV: Modular Architecture ✅
- Frontend (Docusaurus) is independently deployable
- No backend in Phase 1, so backend/DB modularity deferred to Phase 2
- Content structure uses Docusaurus's plugin system for future extensibility

### Principle V: Accessibility & Personalization ✅
- Docusaurus provides keyboard navigation by default
- Static site works without JavaScript (content readable, navigation functional)
- Personalization/translation deferred to Phase 4 (bonus features)

### Principle VI: Security & Privacy ✅
- No user data collected in Phase 1 (anonymous read-only access)
- No authentication, so no credential storage concerns
- Environment variables for deployment secrets (GitHub Actions)

### Principle VII: Progressive Enhancement ✅
- Implementing Phase 1 (Foundation) before Phase 2 (RAG) as specified
- Feature flags not needed in Phase 1 (no conditional features)

### Principle VIII: Documentation-First & Traceability ✅
- Spec and plan artifacts present
- Deployment process will be documented in quickstart.md
- Setup instructions will be reproducible

### Principle IX: Graceful Degradation ✅
- Static site inherently reliable (no dynamic backend to fail)
- Broken image links will show alt text
- 404 pages will include navigation back to homepage

**Gate Status**: PASS - All applicable principles satisfied for Phase 1 scope

## Project Structure

### Documentation (this feature)

```text
specs/001-textbook-foundation/
├── spec.md              # Feature specification
├── plan.md              # This file (/sp.plan output)
├── research.md          # Phase 0 research findings (to be created)
├── data-model.md        # Phase 1 data structures (to be created)
├── quickstart.md        # Phase 1 setup guide (to be created)
├── contracts/           # API contracts (N/A for Phase 1 - no APIs)
├── checklists/
│   └── requirements.md  # Spec quality checklist (existing)
└── tasks.md             # Phase 2 task list (/sp.tasks output - future)
```

### Source Code (repository root)

```text
docs/                       # Docusaurus source directory
├── intro.md               # Homepage content
├── module1-ros2/          # Module 1: ROS 2
│   ├── week1-foundations.md
│   └── week2-landscape.md
├── module2-gazebo/        # Module 2: Gazebo & Unity (placeholders)
├── module3-isaac/         # Module 3: NVIDIA Isaac (placeholders)
├── module4-vla/           # Module 4: VLA (placeholders)
└── assets/                # Images, diagrams, videos
    ├── week1/
    └── week2/

src/                       # Docusaurus custom components (if needed)
├── components/
│   └── HomepageFeatures.js  # Custom homepage sections
└── css/
    └── custom.css         # Theme customization

static/                    # Static assets served directly
└── img/
    └── logo.svg           # Site logo/branding

docusaurus.config.js       # Docusaurus configuration
sidebar.js                 # Sidebar navigation structure
package.json               # Node.js dependencies
.github/
└── workflows/
    └── deploy.yml         # GitHub Actions deployment workflow

tests/                     # Future: Lighthouse CI configuration
└── lighthouse-ci.js
```

**Structure Decision**: Using Docusaurus's standard "docs" directory structure for content organization. Modules map to folders, weeks map to markdown files. This structure scales naturally to Weeks 3-13 (Phase 3) and supports Docusaurus's automatic sidebar generation.

## Complexity Tracking

*No constitution violations - this section intentionally left empty.*

---

## Phase 0: Research & Technology Decisions

This phase resolves technical unknowns and establishes best practices for Docusaurus content authoring.

### Research Tasks

1. **Docusaurus v3.x Setup Best Practices**
   - Official getting-started guide
   - Recommended project structure for educational content
   - MDX features for interactive content

2. **GitHub Pages Deployment Strategy**
   - GitHub Actions workflow for automated deployment
   - Branch strategy (gh-pages vs. /docs folder)
   - Custom domain configuration (if needed)

3. **Visual Aid Creation Tools**
   - Diagramming tools compatible with Physical AI topics (robots, sensors, architectures)
   - Image optimization for web (<200KB per image)
   - Accessible alt text best practices

4. **MDX Content Authoring Patterns**
   - Code block syntax highlighting for Python/C++/YAML
   - Math equation rendering (KaTeX or MathJax)
   - Embedding diagrams and videos

5. **Performance Optimization Techniques**
   - Image lazy loading
   - Bundle size optimization
   - Lighthouse CI integration for automated audits

### Research Output

See [research.md](./research.md) for detailed findings and decisions.

---

## Phase 1: Design & Contracts

### Data Model

For Phase 1 (static site), the "data model" is the content structure and metadata schema:

See [data-model.md](./data-model.md) for:
- Chapter metadata structure (title, module, week number, topics, learning objectives)
- Navigation hierarchy (modules > weeks > sections)
- Visual aid specifications (size, format, alt text requirements)
- Content template for consistent chapter authoring

### API Contracts

**N/A for Phase 1** - No backend APIs. RAG chatbot API contracts will be defined in Phase 2.

Future `contracts/` directory will contain:
- OpenAPI spec for RAG query endpoints (Phase 2)
- Auth API contracts (Phase 4 bonus feature)

### Quickstart Guide

See [quickstart.md](./quickstart.md) for:
- Local development setup (Node.js installation, npm dependencies)
- Running Docusaurus dev server
- Content authoring workflow
- Deployment to GitHub Pages
- Troubleshooting common issues

---

## Next Steps

After this plan is complete:

1. **Run `/sp.tasks`** to generate actionable task list from this plan
2. **Run `/sp.implement`** to execute tasks systematically
3. **Content authoring**: Write Week 1 and Week 2 markdown files following data model template
4. **Visual aid creation**: Create diagrams for Physical AI concepts, sensors, robotics landscape
5. **Deployment**: Configure GitHub Actions and deploy to GitHub Pages
6. **Validation**: Run Lighthouse audit, verify constitution compliance checklist

**Estimated Effort**: 8-12 hours (2-3 hours setup, 4-6 hours content writing, 2-3 hours visual aids and deployment)
