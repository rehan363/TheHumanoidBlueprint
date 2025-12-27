# Physical AI & Humanoid Robotics Textbook Constitution

<!--
Sync Impact Report:
Version change: 1.0.0 → 1.1.1
Modified sections:
  - Added Principle IX: Graceful Degradation
  - Enhanced Principle III with Content Quality Assurance process
  - Enhanced Principle VII with Feature Priority Matrix + Recommended Implementation Order
  - Added Performance Budgets section
  - Added Testing Requirements section
  - Enhanced Deployment Gates with Compliance Checklist
Rationale: MINOR version bump (1.0→1.1) - added new principle, quantifiable metrics.
          PATCH bump (1.1.0→1.1.1) - clarified implementation order (RAG after content).
Templates requiring updates: ⚠ pending validation
Follow-up TODOs: Extract technology stack decisions to ADRs (post-hackathon)
-->

## Core Principles

### I. AI-Driven Content Creation (Spec-Driven Development)

**MUST Requirements:**
- All content generation MUST use Claude Code with Spec-Kit Plus workflow
- Every feature MUST follow: Constitution → Spec → Plan → Tasks → Implementation
- All significant decisions MUST be documented as Prompt History Records (PHRs)
- Architectural decisions MUST be captured as ADRs when meeting significance criteria

**Rationale:** Ensures systematic, traceable, and high-quality content development that
leverages AI capabilities while maintaining human oversight and decision control.

### II. Interactive Learning Experience (RAG Integration)

**MUST Requirements:**
- Textbook MUST include embedded RAG chatbot for content questions
- Chatbot MUST support both full-content and text-selection-based queries
- RAG system MUST use: OpenAI Agents/ChatKit SDK, FastAPI, Neon Postgres, Qdrant vector DB
- Responses MUST be contextually relevant and cite source sections

**Rationale:** Enhances learning by providing on-demand clarification and exploration,
making complex Physical AI concepts more accessible to diverse learners.

### III. Content Quality & Technical Accuracy (NON-NEGOTIABLE)

**MUST Requirements:**
- All technical content MUST be verified against authoritative sources
- Code examples MUST be tested and functional
- Hardware specifications MUST reflect current market availability (≤90 days old)
- ROS 2, Gazebo, Unity, and NVIDIA Isaac content MUST follow official documentation
- Mathematical formulas and physics simulations MUST be validated

**SHOULD Requirements:**
- Content SHOULD include practical examples from real-world robotics applications
- Visual aids (diagrams, animations) SHOULD accompany complex concepts

**Content Quality Assurance Process:**
- **ROS 2 Code:** Tested against ROS 2 Humble/Iron official tutorials
- **Hardware Specs:** Verified against manufacturer datasheets (dated within 90 days)
- **Physics Equations:** Peer-reviewed or cited from authoritative textbooks
- **NVIDIA Isaac:** Validated against Isaac Sim 2023.1+ documentation
- **External Links:** Automated check for 200 status (broken link detection)
- **Mathematical Notation:** LaTeX validated for consistency

**Review Checklist per Chapter:**
- ☐ Code examples executable in described environment
- ☐ Hardware prices updated and sourced
- ☐ All external references accessible
- ☐ Technical terminology consistent with industry standards

**Rationale:** Physical AI education requires precision; incorrect information could lead
to dangerous hardware implementations or failed learning outcomes.

### IV. Modular Architecture (Separation of Concerns)

**MUST Requirements:**
- Frontend (Docusaurus) MUST be independently deployable
- Backend API (FastAPI) MUST expose well-defined REST/GraphQL endpoints
- Database layer (Neon Postgres + Qdrant) MUST be abstracted behind service interfaces
- Authentication (Better-Auth) MUST be modular and replaceable
- Each module MUST have clear interface contracts documented in spec files

**Rationale:** Enables parallel development, easier testing, maintainability, and
component replacement without cascading changes.

### V. Accessibility & Personalization

**MUST Requirements:**
- Content MUST be readable and navigable via keyboard only
- Core textbook MUST work without JavaScript (progressive enhancement)
- Logged-in users MUST be able to personalize content based on background profile
- Translation feature MUST accurately render technical terms in Urdu

**SHOULD Requirements:**
- Content SHOULD adapt complexity based on user's declared software/hardware background
- Font sizes and contrast SHOULD be adjustable

**Rationale:** Democratizes access to Physical AI education across different learning
styles, languages, and accessibility needs.

### VI. Security & Privacy

**MUST Requirements:**
- User credentials MUST be hashed using bcrypt or Argon2
- Authentication tokens MUST use JWT with secure signing and expiration
- Database connections MUST use SSL/TLS encryption
- User background data MUST be stored with consent and used only for personalization
- API endpoints MUST implement rate limiting to prevent abuse
- Environment variables MUST never be committed to version control

**MUST NOT Requirements:**
- API keys and secrets MUST NOT be hardcoded
- User personal data MUST NOT be shared with third parties without explicit consent

**Rationale:** Protects learner data and system integrity, especially important when
handling educational profiles and authentication.

### VII. Progressive Enhancement (Core First, Bonus Layered)

**MUST Requirements:**
- Base deliverables (Docusaurus book + RAG chatbot) MUST be completed first
- Bonus features (auth, personalization, translation, subagents) MUST NOT block core
- Each bonus feature MUST be independently testable and toggleable via feature flags

**Feature Priority Classification:**

**Core Requirements (100 points total - MUST complete):**
- **Textbook Content:** Complete Physical AI curriculum (13 weeks, 4 modules)
- **Docusaurus Site:** Professional static site deployed to GitHub Pages
- **RAG Chatbot:** Embedded chatbot for content Q&A (full-text + selection-based)

**Bonus Features (50 points each - Independent):**
- **P1:** Better-Auth signup/signin with user background profiling
- **P2:** Content personalization based on declared background
- **P2:** Urdu translation with technical term accuracy
- **P3:** Claude Code subagents/skills for reusability

**Recommended Implementation Order:**
1. **Phase 1 - Foundation:** Docusaurus setup + 2 sample chapters → Deploy MVP
2. **Phase 2 - Intelligence:** RAG chatbot integration (now has content to index)
3. **Phase 3 - Content Scale:** Complete all 13 weeks of curriculum content
4. **Phase 4 - Enhancements:** Add bonus features in priority order (P1→P2→P3)

**Rationale for Order:**
- Early deployment validates infrastructure
- RAG needs content to be useful (can't index empty chapters)
- Sample chapters allow iterative testing of RAG quality
- Bonus features are truly independent and don't block core

**Feature Flag Naming Convention:**
- `FEATURE_AUTH_ENABLED=true|false`
- `FEATURE_PERSONALIZATION_ENABLED=true|false`
- `FEATURE_TRANSLATION_URDU_ENABLED=true|false`
- `FEATURE_SUBAGENTS_ENABLED=true|false`

**Rationale:** Guarantees hackathon base requirements (100 points) are met before
pursuing bonus points; prevents scope creep from jeopardizing core functionality.

### VIII. Documentation-First & Traceability

**MUST Requirements:**
- Every user interaction MUST generate a PHR in appropriate directory
- Component interfaces MUST be documented in spec.md files

- API endpoints MUST have OpenAPI/Swagger documentation
- Setup instructions MUST be reproducible on clean environments
- Deployment process MUST be documented with rollback procedures

**Rationale:** Enables knowledge transfer, debugging, and long-term maintenance;
critical for educational projects that may be forked or extended.

### IX. Graceful Degradation & Reliability

**MUST Requirements:**
- Core textbook MUST remain readable if RAG backend is unavailable
- Personalization failure MUST NOT block content access (fallback to default view)
- Translation service failure MUST show original English content with error notice
- Database unavailable MUST allow read-only static site access
- All user-facing errors MUST be actionable (not stack traces or error codes alone)

**Error Handling Standards:**
- Backend errors MUST be logged with correlation IDs for debugging
- Rate limit errors MUST suggest retry timing (e.g., "Try again in 60 seconds")
- Failed API calls MUST timeout within 10s and show fallback UI
- Chatbot unavailable MUST display: "AI assistant temporarily offline. Browse content normally."

**Monitoring Requirements:**
- Uptime target: 99% for static content, 95% for dynamic features
- Error budget: <5% of RAG queries can fail per 24h period
- Alert thresholds: >10 errors/min, >5s p95 latency, >80% resource usage

**Rationale:** Educational platform must prioritize content access over feature richness;
users should never be blocked from learning due to optional feature failures.

## Technology Stack Requirements

### Frontend Stack
- **Framework:** Docusaurus (v3.x recommended)
- **Deployment:** GitHub Pages
- **Styling:** CSS Modules or Tailwind CSS (choose one, document in ADR)
- **Build Tool:** Webpack (Docusaurus default) or Vite

### Backend Stack
- **API Framework:** FastAPI (Python 3.10+)
- **Authentication:** Better-Auth (if implementing bonus feature)
- **AI Integration:** OpenAI SDK (Agents/ChatKit)
- **Deployment:** Vercel, Railway, or Render (document choice in spec)

### Database Stack
- **Relational DB:** Neon Serverless Postgres (for user profiles, auth)
- **Vector DB:** Qdrant Cloud Free Tier (for RAG embeddings)
- **ORM:** SQLAlchemy or Prisma (if using Node.js layer)

### Development Tools
- **Spec Management:** Spec-Kit Plus
- **AI Assistant:** Claude Code
- **Version Control:** Git with conventional commits
- **CI/CD:** GitHub Actions

### Quality Gates
- **Linting:** Ruff (Python), ESLint (JavaScript/TypeScript)
- **Formatting:** Black (Python), Prettier (JS/TS)
- **Type Checking:** Mypy (Python), TypeScript strict mode
- **Testing:** Pytest (backend), Jest (frontend)

### Testing Requirements

**Coverage Targets:**
- Backend: ≥80% line coverage (critical paths: 100%)
- Frontend components: ≥70% coverage
- Integration tests: All API endpoints, RAG pipeline end-to-end

**Test Types Required:**
- **Unit Tests:** Pure functions, data transformations, utility modules
- **Integration Tests:** API contracts, database queries, vector search
- **E2E Tests:** Critical user journeys (P1 user stories from spec)
- **Performance Tests:** Load testing RAG endpoints with k6 or Locust

**Test Naming Convention:**
- `test_<function>_<scenario>_<expected_result>`
- Example: `test_rag_query_with_text_selection_returns_relevant_context`

### Performance Budgets

**Frontend Performance:**
- First Contentful Paint (FCP): <1.5s
- Time to Interactive (TTI): <3.5s on 3G connection
- Lighthouse Performance Score: ≥90
- Bundle size: <500KB (gzipped)

**Backend Performance:**
- RAG query response: p95 <2s, p99 <5s
- Chapter content load: <500ms
- Auth token validation: <50ms
- Concurrent users supported: ≥50 (demo requirement)

**Database Performance:**
- Vector similarity search: <100ms for top 10 results
- Postgres query response: <200ms for user profile fetch
- Embedding generation: <1s per 1000 tokens

## Development Workflow

### 1. Spec-Driven Development Cycle

**For Every Feature:**
1. Create/Update Constitution (this document) if principles change
2. Run `/sp.specify` to create detailed spec.md
3. Run `/sp.clarify` to resolve ambiguities
4. Run `/sp.plan` to generate architectural plan.md
5. Run `/sp.tasks` to break down into testable tasks.md
6. Run `/sp.implement` to execute tasks
7. Run `/sp.analyze` for consistency checks
8. Run `/sp.git.commit_pr` when ready for review

**After Significant Decisions:**
- Run `/sp.adr <decision-title>` when suggested by agent

**After Every User Interaction:**
- PHR MUST be created automatically (agent responsibility)

### 2. Branching Strategy

- **main:** Production-ready code deployed to GitHub Pages
- **develop:** Integration branch for completed features
- **feature/<feature-name>:** Individual feature branches (match spec/<feature>/)

### 3. Code Review Requirements

**MUST Pass Before Merge:**
- All tests passing (unit + integration)
- Linting and formatting checks passing
- No hardcoded secrets or API keys
- PHR created and filed appropriately
- Spec/plan/tasks artifacts updated if behavior changed

### 4. Deployment Gates

**Before Deploying to Production:**

**Constitution Compliance Checklist:**
- ☐ **Principle I:** Spec/Plan/Tasks artifacts present in `specs/<feature>/`
- ☐ **Principle II:** RAG integration tested with ≥10 diverse sample queries
- ☐ **Principle III:** All code examples tested and executable
- ☐ **Principle IV:** Frontend deploys independently (verified in staging)
- ☐ **Principle V:** Keyboard navigation tested (Tab through entire UI)
- ☐ **Principle VI:** Security scan passing (no HIGH/CRITICAL vulnerabilities)
- ☐ **Principle VII:** Feature flags configured and documented in `.env.example`
- ☐ **Principle VIII:** OpenAPI docs generated and accessible at `/api/docs`
- ☐ **Principle IX:** Graceful degradation tested (simulate backend failures)

**Quality Assurance Gates:**
- ☐ All tests passing (unit + integration + E2E)
- ☐ Accessibility audit: Lighthouse score ≥90
- ☐ Performance: FCP <1.5s, TTI <3.5s measured
- ☐ Security: OWASP ZAP scan completed (report attached)
- ☐ Load testing: ≥50 concurrent users handled (if backend deployed)
- ☐ No hardcoded secrets or API keys in codebase
- ☐ PHRs created and filed appropriately
- ☐ Spec/plan/tasks updated to reflect actual implementation

## Governance

### Amendment Process

**MINOR Version Bump (New Principle Added):**
1. Propose change via issue or discussion
2. Update constitution with new principle
3. Run consistency propagation checks (templates, docs)
4. Create ADR documenting rationale for governance change
5. Require 1 maintainer approval

**MAJOR Version Bump (Breaking Change to Governance):**
1. RFC (Request for Comments) period of 3 days minimum
2. Document migration path for existing work
3. Update all dependent templates and guidance files
4. Require consensus from all active maintainers

**PATCH Version Bump (Clarification/Typo Fix):**
1. Direct PR with clear description
2. No approval required if non-semantic

### Compliance Verification

- Constitution supersedes all other practices and preferences
- All PRs MUST include checklist verifying principle compliance
- Complexity introduced MUST be justified against constitution principles
- Violations MUST be documented as technical debt with remediation plan

### Living Document Philosophy

This constitution is a living document that evolves with project understanding.
When real-world constraints conflict with ideals, document the trade-off as an
ADR and amend constitution if pattern emerges.

**Version**: 1.1.1 | **Ratified**: 2025-12-26 | **Last Amended**: 2025-12-27
