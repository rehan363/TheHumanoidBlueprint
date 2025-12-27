# Tasks: Physical AI Textbook Foundation

**Input**: Design documents from `/specs/001-textbook-foundation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, quickstart.md

**Tests**: No test tasks included (not requested in spec; Phase 1 focuses on content delivery)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Web app (static site)**: `docs/` for content, `src/` for custom components, `static/` for assets
- Paths follow Docusaurus conventions per plan.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize Docusaurus project and configure foundational structure

- [ ] T001 Verify Node.js 18.x and npm 9.x are installed (run `node --version` and `npm --version`)
- [ ] T002 Initialize Docusaurus project with TypeScript template using `npx create-docusaurus@latest physical-ai-textbook classic --typescript`
- [ ] T003 [P] Install additional dependencies: remark-math, rehype-katex for math rendering (run `npm install remark-math@3 rehype-katex@5`)
- [ ] T004 [P] Configure docusaurus.config.js with site metadata (title: "Physical AI & Humanoid Robotics", tagline, URL, baseUrl per research.md)
- [ ] T005 [P] Configure math rendering plugins in docusaurus.config.js (add remark-math and rehype-katex to preset configuration)
- [ ] T006 [P] Add KaTeX CSS stylesheet to docusaurus.config.js stylesheets array
- [ ] T007 [P] Create module directory structure: docs/module1-ros2/, docs/module2-gazebo/, docs/module3-isaac/, docs/module4-vla/
- [ ] T008 [P] Create assets directory structure: docs/assets/week1/, docs/assets/week2/
- [ ] T009 [P] Create _category_.json files for each module directory (Module 1-4 with labels and descriptions per data-model.md)
- [ ] T010 Configure sidebars.js with 4-module curriculum structure (collapsible categories, Week 1-2 under Module 1 expanded, others collapsed)
- [ ] T011 [P] Update src/css/custom.css with project branding colors (optional: use default Docusaurus theme)
- [ ] T012 [P] Replace static/img/logo.svg with project logo (use default or create simple logo)
- [ ] T013 Test local development server with `npm start` and verify site loads at http://localhost:3000

**Checkpoint**: Docusaurus project initialized, configured, and running locally

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core content structure and deployment infrastructure that MUST be complete before user story content can be authored

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T014 Create GitHub Actions workflow file at .github/workflows/deploy.yml with Node.js 18, npm ci, build, and gh-pages deployment (per research.md deployment strategy)
- [ ] T015 [P] Create placeholder files for Weeks 3-13 using placeholder template from data-model.md (module2/week6-7, module3/week8-10, module4/week11-13)
- [ ] T016 [P] Create .lighthouserc.js configuration file with performance budgets (FCP <1.5s, TTI <3.5s, performance ≥0.9)
- [ ] T017 [P] Create GitHub Actions workflow file at .github/workflows/lighthouse-ci.yml for automated performance audits on PRs
- [ ] T018 Test production build with `npm run build` and verify no broken links or build errors
- [ ] T019 Test production serve with `npm run serve` and verify site functionality at http://localhost:3000

**Checkpoint**: Foundation ready - user story content authoring can now begin in parallel

---

## Phase 3: User Story 1 - Browse Physical AI Curriculum Overview (Priority: P1) 🎯 MVP

**Goal**: Prospective students and educators can view homepage with course overview, 4-module structure, 13-week breakdown, and learning outcomes

**Independent Test**: Deploy to GitHub Pages and verify homepage displays course title, 4 modules, 13 weeks, learning outcomes, and hardware requirements overview

### Implementation for User Story 1

- [ ] T020 [P] [US1] Create docs/intro.md homepage file with course title "Physical AI & Humanoid Robotics"
- [ ] T021 [P] [US1] Add course tagline and description to docs/intro.md (educational value proposition)
- [ ] T022 [US1] Add 4-module overview section to docs/intro.md with module titles (Module 1: ROS 2, Module 2: Gazebo & Unity, Module 3: NVIDIA Isaac, Module 4: VLA)
- [ ] T023 [US1] Add 13-week breakdown section to docs/intro.md listing all week titles organized by module
- [ ] T024 [US1] Add learning outcomes section to docs/intro.md (6 learning outcomes from hackathon document)
- [ ] T025 [US1] Add hardware requirements overview section to docs/intro.md (brief summary referencing Week 1-2 for details)
- [ ] T026 [US1] Add "Why Physical AI Matters" section to docs/intro.md explaining humanoid robotics significance
- [ ] T027 [US1] Configure docs/intro.md as homepage in docusaurus.config.js (set docs plugin routeBasePath to '/')
- [ ] T028 [US1] Verify homepage renders correctly with `npm start` and test navigation to homepage from any page

**Checkpoint**: At this point, User Story 1 should be fully functional - homepage displays complete curriculum overview

---

## Phase 4: User Story 2 - Read Introduction to Physical AI Chapters (Priority: P1)

**Goal**: Students can read comprehensive Week 1-2 content covering Physical AI foundations, embodied intelligence, robotics landscape, and sensor systems

**Independent Test**: Navigate to Week 1 and Week 2 chapter pages and verify educational content is present (≥1500 words each), well-formatted, includes visual aids, and covers specified topics

### Implementation for User Story 2

#### Week 1 Content Creation

- [ ] T029 [P] [US2] Create docs/module1-ros2/week1-foundations.md with frontmatter (sidebar_position: 1, title, description, keywords, last_updated, estimated_reading_time: 15)
- [ ] T030 [US2] Write Week 1 section 1: "What is Physical AI?" in week1-foundations.md (definition, characteristics, ≥300 words)
- [ ] T031 [US2] Write Week 1 section 2: "Embodied Intelligence" in week1-foundations.md (concept explanation, examples, ≥300 words)
- [ ] T032 [US2] Write Week 1 section 3: "Digital AI vs Physical AI" in week1-foundations.md (key differences, comparison table, ≥250 words)
- [ ] T033 [US2] Write Week 1 section 4: "Real-World Applications" in week1-foundations.md (use cases, industry examples, ≥300 words)
- [ ] T034 [US2] Write Week 1 section 5: "The Challenge of Physical Interaction" in week1-foundations.md (technical challenges, safety considerations, ≥250 words)
- [ ] T035 [US2] Add Learning Objectives section to week1-foundations.md (3-5 bullet points)
- [ ] T036 [US2] Add Summary section to week1-foundations.md (key takeaways)
- [ ] T037 [US2] Add Further Reading section to week1-foundations.md (3-5 authoritative sources with URLs)

#### Week 1 Visual Aids

- [ ] T038 [P] [US2] Create "Physical AI vs Digital AI" comparison diagram using Excalidraw, export as SVG, save to docs/assets/week1/physical-ai-comparison.svg (<200KB)
- [ ] T039 [P] [US2] Create "Embodied Intelligence" conceptual diagram using Excalidraw, export as SVG, save to docs/assets/week1/embodied-intelligence-diagram.svg (<200KB)
- [ ] T040 [US2] Embed physical-ai-comparison.svg in week1-foundations.md section 3 with descriptive alt text and caption "Figure 1.1"
- [ ] T041 [US2] Embed embodied-intelligence-diagram.svg in week1-foundations.md section 2 with descriptive alt text and caption "Figure 1.2"

#### Week 2 Content Creation

- [ ] T042 [P] [US2] Create docs/module1-ros2/week2-landscape.md with frontmatter (sidebar_position: 2, title, description, keywords, last_updated, estimated_reading_time: 18)
- [ ] T043 [US2] Write Week 2 section 1: "Humanoid Robotics Landscape" in week2-landscape.md (overview, key players, current state, ≥350 words)
- [ ] T044 [US2] Write Week 2 section 2: "Sensor Systems Overview" in week2-landscape.md (importance, categories, ≥250 words)
- [ ] T045 [US2] Write Week 2 section 3: "LIDAR Sensors" in week2-landscape.md (working principle, applications, advantages/limitations, ≥300 words)
- [ ] T046 [US2] Write Week 2 section 4: "Camera Systems" in week2-landscape.md (RGB cameras, depth cameras, stereo vision, ≥300 words)
- [ ] T047 [US2] Write Week 2 section 5: "Inertial Measurement Units (IMUs)" in week2-landscape.md (accelerometers, gyroscopes, applications, ≥250 words)
- [ ] T048 [US2] Write Week 2 section 6: "Force/Torque Sensors" in week2-landscape.md (tactile feedback, manipulation, ≥200 words)
- [ ] T049 [US2] Add Learning Objectives section to week2-landscape.md (3-5 bullet points)
- [ ] T050 [US2] Add Summary section to week2-landscape.md (key takeaways)
- [ ] T051 [US2] Add Further Reading section to week2-landscape.md (3-5 authoritative sources with URLs)

#### Week 2 Visual Aids

- [ ] T052 [P] [US2] Create "Sensor Types Overview" diagram using Excalidraw showing LIDAR, cameras, IMUs, force sensors, export as SVG, save to docs/assets/week2/sensor-types-overview.svg (<200KB)
- [ ] T053 [P] [US2] Find or create LIDAR point cloud visualization image, optimize to WebP format, save to docs/assets/week2/lidar-point-cloud.webp (<200KB)
- [ ] T054 [P] [US2] Create "IMU Axes" orientation diagram using Excalidraw (X/Y/Z axes, roll/pitch/yaw), export as SVG, save to docs/assets/week2/imu-axes.svg (<200KB)
- [ ] T055 [US2] Embed sensor-types-overview.svg in week2-landscape.md section 2 with descriptive alt text and caption "Figure 2.1"
- [ ] T056 [US2] Embed lidar-point-cloud.webp in week2-landscape.md section 3 with descriptive alt text and caption "Figure 2.2"
- [ ] T057 [US2] Embed imu-axes.svg in week2-landscape.md section 5 with descriptive alt text and caption "Figure 2.3"

#### Content Quality Verification

- [ ] T058 [US2] Verify Week 1 content meets ≥1500 words requirement (run word count)
- [ ] T059 [US2] Verify Week 2 content meets ≥1500 words requirement (run word count)
- [ ] T060 [US2] Verify all images have descriptive alt text (50-150 characters) and loading="lazy" attribute
- [ ] T061 [US2] Verify all images are <200KB and display correctly in browser
- [ ] T062 [US2] Verify all external links in Further Reading sections return 200 status
- [ ] T063 [US2] Run `npm run build` and verify no broken links or markdown errors for Week 1-2 content

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - homepage and Week 1-2 chapters fully readable with visual aids

---

## Phase 5: User Story 3 - Navigate Between Chapters and Modules (Priority: P2)

**Goal**: Students can easily navigate between chapters, modules, and sections using sidebar navigation with current page highlighting

**Independent Test**: Click through all navigation links in sidebar, verify routing accuracy, confirm current page is highlighted, test module expand/collapse

### Implementation for User Story 3

- [ ] T064 [P] [US3] Verify sidebars.js configuration includes all Week 1-2 chapters with correct paths (module1-ros2/week1-foundations, module1-ros2/week2-landscape)
- [ ] T065 [P] [US3] Verify Module 1 (ROS 2) is set to `collapsed: false` in sidebars.js (expanded by default to show Week 1-2)
- [ ] T066 [P] [US3] Verify Modules 2-4 are set to `collapsed: true` in sidebars.js (collapsed by default since placeholders)
- [ ] T067 [US3] Test navigation from homepage to Week 1, verify URL changes and content loads without errors
- [ ] T068 [US3] Test navigation from Week 1 to Week 2, verify sidebar highlights Week 2 as current page
- [ ] T069 [US3] Test navigation from Week 2 back to homepage, verify breadcrumbs work correctly
- [ ] T070 [US3] Test collapsing Module 1 in sidebar, verify Week 1-2 chapters hide
- [ ] T071 [US3] Test expanding Module 1 again, verify Week 1-2 chapters reappear
- [ ] T072 [US3] Test navigation to placeholder pages (Week 3-13), verify "Content coming soon" message displays
- [ ] T073 [US3] Verify sidebar visual highlighting (bold/color) for current page works across all Week 1-2 pages
- [ ] T074 [US3] Verify browser back/forward buttons work correctly with sidebar navigation
- [ ] T075 [US3] Test search functionality: search for "embodied intelligence", verify Week 1 appears in results

**Checkpoint**: All navigation links functional, sidebar highlighting works, search indexes Week 1-2 content

---

## Phase 6: User Story 4 - Access Site on Mobile Devices (Priority: P3)

**Goal**: Students can read textbook on mobile phones and tablets with responsive design, hamburger menu, and readable text without horizontal scrolling

**Independent Test**: Open site on mobile browser or dev tools mobile emulation (iPhone, Android, tablet), verify responsive layout, hamburger menu, and text readability

### Implementation for User Story 4

- [ ] T076 [P] [US4] Test site on mobile viewport 375px width (iPhone SE) using browser dev tools, verify content reflows correctly
- [ ] T077 [P] [US4] Test site on mobile viewport 320px width (minimum spec requirement), verify no horizontal scrolling
- [ ] T078 [P] [US4] Verify hamburger menu icon appears on mobile viewports (<997px per Docusaurus default breakpoint)
- [ ] T079 [US4] Tap hamburger menu on mobile viewport, verify sidebar navigation opens as overlay
- [ ] T080 [US4] Navigate to Week 1 from hamburger menu, verify page loads and menu closes automatically
- [ ] T081 [US4] Verify text is readable without zooming on mobile (font size ≥16px for body text)
- [ ] T082 [US4] Verify images scale proportionally on mobile and don't overflow viewport width
- [ ] T083 [US4] Test site on tablet viewport 768px width (iPad), verify sidebar remains visible alongside content
- [ ] T084 [US4] Test site on tablet landscape mode 1024px width, verify desktop layout renders
- [ ] T085 [US4] Verify touch targets (links, buttons) are ≥44px for accessibility on mobile
- [ ] T086 [US4] Test scrolling performance on mobile device (smooth scrolling, no jank)

**Checkpoint**: All user stories (1-4) should now be independently functional - site works on desktop, tablet, and mobile

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final quality improvements, deployment, and validation

- [ ] T087 [P] Run Lighthouse audit on homepage, verify Performance ≥90, Accessibility ≥90, SEO ≥90
- [ ] T088 [P] Run Lighthouse audit on Week 1 page, verify Performance ≥90, Accessibility ≥90
- [ ] T089 [P] Run Lighthouse audit on Week 2 page, verify Performance ≥90, Accessibility ≥90
- [ ] T090 [P] Run Lighthouse audit on mobile (375px), verify Performance ≥80, Accessibility ≥90
- [ ] T091 [P] Verify all Constitution Principle III content quality checklist items (code examples executable if present, hardware prices sourced, external refs accessible, terminology consistent)
- [ ] T092 Run final `npm run build` and verify build completes with zero errors and zero warnings
- [ ] T093 Commit all changes to feature branch 001-textbook-foundation with conventional commit message "feat: add Phase 1 foundation with Week 1-2 content"
- [ ] T094 Push feature branch to GitHub remote repository
- [ ] T095 Create Pull Request from 001-textbook-foundation to main with description referencing spec.md and plan.md
- [ ] T096 Verify GitHub Actions deployment workflow triggers automatically after merge to main
- [ ] T097 Wait for GitHub Actions deployment to complete (check Actions tab for green checkmark)
- [ ] T098 Verify live site is accessible at GitHub Pages URL (https://<username>.github.io/<repo-name>/)
- [ ] T099 Test live site on actual mobile device (iOS or Android), verify functionality matches local testing
- [ ] T100 Run quickstart.md validation: verify another developer can clone repo, run `npm install`, `npm start`, and view site locally

**Checkpoint**: Phase 1 complete - textbook foundation deployed to GitHub Pages with Week 1-2 content, ready for Phase 2 (RAG integration)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion (T001-T013) - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion (T014-T019)
  - User Story 1 (Phase 3): Can start after Foundational
  - User Story 2 (Phase 4): Can start after Foundational (independent of US1, but logically follows for content flow)
  - User Story 3 (Phase 5): Depends on US2 completion (needs content to navigate)
  - User Story 4 (Phase 6): Depends on US1-3 completion (needs full site to test responsive)
- **Polish (Phase 7)**: Depends on all user stories (US1-4) being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Logically follows US1 but technically independent
- **User Story 3 (P2)**: Depends on US2 completion (T063) - Requires content to exist for navigation testing
- **User Story 4 (P3)**: Depends on US1-3 completion (T075) - Requires full site for comprehensive responsive testing

### Within Each User Story

**User Story 1 (Homepage)**:
- All tasks T020-T027 can be worked in parallel (different sections of same file)
- T028 (verification) must come last

**User Story 2 (Week 1-2 Content)**:
- Week 1 content (T029-T037) independent of Week 2 content (T042-T051)
- Visual aids (T038-T041, T052-T057) can be created in parallel with writing
- Content quality verification (T058-T063) must come after all content/images complete

**User Story 3 (Navigation)**:
- Configuration tasks (T064-T066) can run in parallel
- Testing tasks (T067-T075) must run sequentially to verify each navigation path

**User Story 4 (Mobile)**:
- All mobile testing tasks (T076-T086) can run in parallel if multiple devices/viewports available

### Parallel Opportunities

- **Phase 1 (Setup)**: T003, T004, T005, T006 can run in parallel (different config files)
- **Phase 1 (Setup)**: T007, T008, T009, T011, T012 can run in parallel (different directories/files)
- **Phase 2 (Foundational)**: T014, T015, T016, T017 can run in parallel (independent files)
- **US1**: T020, T021 can run together; T022, T023, T024, T025, T026 can run together (different sections)
- **US2**: Week 1 content (T029-T037) and Week 2 content (T042-T051) can run in parallel
- **US2**: Visual aids T038, T039, T052, T053, T054 can all be created in parallel (different images)
- **US2**: T058, T059, T060, T061, T062 verification tasks can run in parallel
- **US3**: T064, T065, T066 configuration tasks can run in parallel
- **US4**: T076, T077, T078, T082, T083, T084, T085, T086 can run in parallel (different viewports/devices)
- **Phase 7**: T087, T088, T089, T090 Lighthouse audits can run in parallel; T091 can run independently

---

## Parallel Example: User Story 2 (Week 1-2 Content)

```bash
# Launch Week 1 and Week 2 content creation in parallel:
Task: "Write Week 1 section 1-5 in week1-foundations.md"
Task: "Write Week 2 section 1-6 in week2-landscape.md"

# Launch all visual aids creation in parallel:
Task: "Create physical-ai-comparison.svg in docs/assets/week1/"
Task: "Create embodied-intelligence-diagram.svg in docs/assets/week1/"
Task: "Create sensor-types-overview.svg in docs/assets/week2/"
Task: "Create or find lidar-point-cloud.webp in docs/assets/week2/"
Task: "Create imu-axes.svg in docs/assets/week2/"
```

---

## Implementation Strategy

### MVP First (User Stories 1 & 2 Only)

1. Complete Phase 1: Setup (T001-T013)
2. Complete Phase 2: Foundational (T014-T019) - CRITICAL
3. Complete Phase 3: User Story 1 (T020-T028) - Homepage
4. Complete Phase 4: User Story 2 (T029-T063) - Week 1-2 Content
5. **STOP and VALIDATE**: Test independently, deploy, demo
6. This gives you 100 hackathon base points (Docusaurus site + RAG-ready content)

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy (Homepage live!)
3. Add User Story 2 → Test independently → Deploy (Content readable!)
4. Add User Story 3 → Test independently → Deploy (Navigation enhanced!)
5. Add User Story 4 → Test independently → Deploy (Mobile-friendly!)
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T019)
2. Once Foundational is done:
   - Developer A: User Story 1 (T020-T028)
   - Developer B: User Story 2 Week 1 (T029-T041)
   - Developer C: User Story 2 Week 2 (T042-T057)
3. Stories complete and integrate independently

---

## Estimated Effort

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1: Setup | T001-T013 (13 tasks) | 2-3 hours |
| Phase 2: Foundational | T014-T019 (6 tasks) | 1-2 hours |
| Phase 3: US1 Homepage | T020-T028 (9 tasks) | 1-2 hours |
| Phase 4: US2 Content | T029-T063 (35 tasks) | 6-8 hours |
| Phase 5: US3 Navigation | T064-T075 (12 tasks) | 1-2 hours |
| Phase 6: US4 Mobile | T076-T086 (11 tasks) | 1-2 hours |
| Phase 7: Polish | T087-T100 (14 tasks) | 2-3 hours |
| **TOTAL** | **100 tasks** | **14-22 hours** |

**MVP (US1-2 only)**: 8-12 hours
**Full Phase 1**: 14-22 hours

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- No automated tests in Phase 1 (focus on content delivery, manual validation)
- Commit frequently (after each logical task group or at each checkpoint)
- Stop at any checkpoint to validate story independently before continuing
- Week 1-2 content establishes template for future Weeks 3-13 (Phase 3 - Content Scale)
- Phase 2 (RAG integration) will build upon this foundation
