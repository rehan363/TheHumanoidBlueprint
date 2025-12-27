---
id: 002
title: Create Implementation Plan
stage: plan
date: 2025-12-27
surface: agent
model: claude-sonnet-4-5
feature: 001-textbook-foundation
branch: 001-textbook-foundation
user: developer
command: /sp.plan
labels: ["plan", "architecture", "docusaurus", "phase-1", "research"]
links:
  spec: specs/001-textbook-foundation/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-textbook-foundation/plan.md
  - specs/001-textbook-foundation/research.md
  - specs/001-textbook-foundation/data-model.md
  - specs/001-textbook-foundation/quickstart.md
tests:
  - N/A
---

## Prompt

/sp.plan

User requested implementation plan for Physical AI Textbook Foundation (Phase 1). Following Spec-Kit Plus workflow: Constitution → Spec → Plan → Tasks → Implementation.

## Response snapshot

Created comprehensive implementation plan with Phase 0 (Research) and Phase 1 (Design) artifacts:

**plan.md - Implementation Plan**
- **Summary**: Docusaurus v3.x static site with Week 1-2 content deployed to GitHub Pages
- **Technical Context**: Node.js 18.x, Docusaurus 3.x, MDX, React 18.x, GitHub Pages deployment
- **Performance Goals**: FCP <1.5s, TTI <3.5s, Lighthouse ≥90 (desktop), ≥80 (mobile), bundle <500KB
- **Constitution Check**: All 9 principles evaluated - PASS (all applicable requirements satisfied for Phase 1 scope)
- **Project Structure**: Docusaurus standard `docs/` directory with module/week hierarchy
- **Complexity Tracking**: No violations (standard Docusaurus patterns, no over-engineering)

**research.md - Phase 0 Research Findings**
Resolved 5 technical unknowns:

1. **Framework Selection**: Docusaurus 3.0 + Classic Preset
   - Rationale: Built-in navigation, search, responsive design, MDX support, educational site optimized
   - Alternatives rejected: VitePress (less feature-complete), Jekyll (older tech), Next.js (over-engineered)

2. **Deployment Strategy**: GitHub Actions → gh-pages branch
   - Rationale: Automated, zero-cost, integrates with GitHub Pages, official Docusaurus recommendation
   - GitHub Actions workflow defined for CI/CD automation

3. **Visual Aid Tools**: Excalidraw (diagrams) + Unsplash (photos)
   - Rationale: Free, high-quality, web-optimized, SVG export for diagrams
   - Image specs: <200KB, max 1200px width, WebP/SVG formats, descriptive alt text

4. **Content Format**: MDX + Prism syntax highlighting + KaTeX math rendering
   - Rationale: Code highlighting for Python/C++/YAML, math equations, React component flexibility
   - Supports future Phase 2-4 interactive enhancements

5. **Performance Optimization**: Docusaurus defaults + lazy loading
   - Rationale: Built-in code splitting, minification, tree-shaking meets Lighthouse ≥90 target
   - Lighthouse CI integration for automated quality gates

**data-model.md - Content Structure**
Defined file-based "data model" for static site:

- **Directory Layout**: `docs/module{N}-{name}/week{N}-{topic}.md` structure
- **Chapter Metadata Schema**: YAML frontmatter with title, description, keywords, dates, reading time
- **Navigation Hierarchy**: 4 modules → 13 weeks, collapsible sidebar, module categories
- **Visual Aid Specifications**: Alt text requirements, file size limits, naming conventions, lazy loading
- **Content Template**: Standard chapter structure with learning objectives, sections, summary, exercises
- **Placeholder Strategy**: Weeks 3-13 use "coming soon" template for Phase 3

**quickstart.md - Developer Guide**
Comprehensive setup and authoring guide:

- **Prerequisites**: Node.js 18.x, npm 9.x, Git 2.x verification commands
- **Initial Setup**: Clone, install dependencies, start dev server workflow
- **Content Authoring**: Step-by-step for creating chapters, adding visual aids, code examples, math equations
- **Build & Deploy**: Production build commands, GitHub Actions automatic deployment
- **Troubleshooting**: Common issues (broken links, hot-reload, math rendering, GitHub Pages updates)
- **Quality Checklist**: Pre-commit verification (frontmatter, alt text, image optimization, link validation)
- **Performance Monitoring**: Lighthouse audit instructions (local + CI)

**Constitution Re-Evaluation**: PASS ✅
- Principle I (AI-Driven): Following Spec-Kit Plus workflow, PHRs created
- Principle II (RAG): Deferred to Phase 2 (per Constitution recommended order)
- Principle III (Quality): Content review checklist defined, authoritative source verification planned
- Principle IV (Modular): Frontend independently deployable, backend deferred to Phase 2
- Principle V (Accessibility): Keyboard navigation, works without JS, personalization deferred to Phase 4
- Principle VI (Security): No user data in Phase 1 (anonymous read-only), env vars for deployment secrets
- Principle VII (Progressive Enhancement): Implementing Phase 1 before Phase 2 as specified
- Principle VIII (Documentation): Spec/plan/quickstart artifacts present, reproducible setup
- Principle IX (Graceful Degradation): Static site inherently reliable, 404 pages with navigation

**Technical Decisions Summary**:
- Framework: Docusaurus 3.0 (TypeScript, classic preset)
- Deployment: GitHub Actions + gh-pages branch
- Content: MDX with Prism + KaTeX
- Visual Aids: Excalidraw SVG diagrams, Unsplash photos, <200KB optimization
- Performance: Lazy loading, Lighthouse CI, bundle size monitoring

**No Unresolved Clarifications** - All technical unknowns researched and decided

## Outcome

- ✅ Impact: Phase 1 architecture fully defined; ready for task breakdown
- 🧪 Tests: Lighthouse CI integration planned for automated quality gates
- 📁 Files: Created plan.md, research.md, data-model.md, quickstart.md
- 🔁 Next prompts: Run `/sp.tasks` to generate actionable task list from plan
- 🧠 Reflection: Plan successfully balances feature completeness with Phase 1 scope; Docusaurus minimizes custom development, focuses effort on content quality

## Evaluation notes (flywheel)

- Failure modes observed: None - all research tasks resolved with clear decisions and rationale
- Graders run and results (PASS/FAIL): Constitution Check - PASS (all 9 principles compliant for Phase 1)
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Monitor if Docusaurus 3.0 performance meets Lighthouse ≥90 target without additional optimization; adjust if needed during implementation
