# Tasks: RAG Integration for Physical AI Textbook

**Input**: Design documents from `/specs/002-rag-integration/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/chat-api.md

**Tests**: Tests are NOT explicitly requested in the spec, so test tasks are EXCLUDED from this task list. Focus is on implementation only.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

This is a web app with:
- **Backend**: `backend/src/` (FastAPI)
- **Frontend**: `physical-ai-textbook/src/` (Docusaurus + React)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create backend directory structure per plan.md (backend/src/, backend/scripts/, backend/tests/, backend/data/)
- [x] T002 Initialize Python project with requirements.txt (FastAPI 0.115+, Qdrant Client 1.7+, LiteLLM 1.50+, Google GenerativeAI SDK 0.8+, slowapi, asyncpg, pytest)
- [x] T003 [P] Create .env.example file in backend/ with all required environment variables (GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, NEON_DATABASE_URL)
- [x] T004 [P] Create backend/README.md with setup instructions from quickstart.md
- [x] T005 [P] Configure CORS in backend/src/main.py with allowed origins from contracts/chat-api.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create backend/src/config.py for environment variable management and settings
- [x] T007 [P] Create Pydantic models in backend/src/models/chat.py (ChatQueryRequest, ChatQueryResponse, Citation, SelectedTextContext, ErrorResponse)
- [x] T008 [P] Create Pydantic models in backend/src/models/chunk.py (TextChunk, ChunkMetadata)
- [x] T009 [P] Create Pydantic models in backend/src/models/health.py (HealthCheckResponse)
- [x] T010 Create backend/src/utils/error_handlers.py for custom exception handling (RateLimitExceeded, ServiceUnavailable, InvalidRequest)
- [x] T011 [P] Create backend/src/utils/rate_limiter.py with slowapi configuration (3 req/min per IP)
- [x] T012 Implement backend/src/services/embedding_service.py using Google gemini-embedding-001 for vector generation
- [x] T013 Implement backend/src/services/vector_store.py as Qdrant client wrapper with collection setup (cosine similarity, 768 dimensions)
- [x] T014 Create backend/scripts/setup_qdrant.py to initialize Qdrant collection with indexed payload fields (week, module)
- [x] T015 Implement backend/src/utils/chunking.py with MarkdownHeaderTextSplitter (512-768 tokens, 10-20% overlap)
- [x] T016 Create backend/scripts/index_docs.py CLI tool to index documentation from physical-ai-textbook/docs/
- [x] T017 Implement backend/src/services/llm_service.py with LiteLLM + Gemini integration, LRU caching (100 entries), exponential backoff
- [x] T018 Implement backend/src/services/rag_pipeline.py orchestrating retrieval (Qdrant) + generation (Gemini) with citation extraction
- [x] T019 Create FastAPI app entry point in backend/src/main.py with CORS, rate limiting, error handlers
- [x] T020 Create backend/src/routers/health.py with GET /api/health endpoint per contracts/chat-api.md
- [x] T021 [P] Create backend/data/index_metadata.json file structure for indexing state tracking

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Ask Questions About Textbook Content (Priority: P1) 🎯 MVP

**Goal**: Students can ask full-text questions about textbook content and receive AI-generated answers with source citations from the RAG system.

**Independent Test**: Enter a question like "What is DDS middleware?" in the chat interface and verify that the response includes relevant content from the textbook with proper citations (e.g., "[Source: Week 3, ROS 2 Architecture]").

### Implementation for User Story 1

- [x] T022 [US1] Implement POST /api/chat/query endpoint in backend/src/routers/chat.py for full_text query_type
- [x] T023 [US1] Add query validation in backend/src/routers/chat.py (1-500 chars, sanitization)
- [x] T024 [US1] Integrate RAG pipeline in backend/src/routers/chat.py (call embedding_service → vector_store → llm_service)
- [x] T025 [US1] Implement citation formatting in backend/src/services/rag_pipeline.py with chunk_index positioning ("[Week X, Section Y, Part N of M]")
- [x] T026 [US1] Add fallback message handling in backend/src/routers/chat.py for queries with no relevant content ("I couldn't find information about that in the textbook")
- [x] T027 [US1] Add rate limiting to /api/chat/query endpoint (3 req/min per IP)
- [x] T028 [US1] Add error handling for 429, 503, 500 responses per contracts/chat-api.md
- [x] T029 [P] [US1] Create React component physical-ai-textbook/src/components/RAGChatbot/index.tsx (main chat widget with floating action button)
- [x] T030 [P] [US1] Create React component physical-ai-textbook/src/components/RAGChatbot/ChatModal.tsx (modal dialog)
- [x] T031 [P] [US1] Create React component physical-ai-textbook/src/components/RAGChatbot/MessageList.tsx (chat history display)
- [x] T032 [P] [US1] Create React component physical-ai-textbook/src/components/RAGChatbot/QueryInput.tsx (user input field with debouncing)
- [x] T033 [P] [US1] Create React component physical-ai-textbook/src/components/RAGChatbot/Citation.tsx (source citation display)
- [x] T034 [P] [US1] Create TypeScript interfaces in physical-ai-textbook/src/components/RAGChatbot/types.ts (ChatMessage, Citation, QueryRequest, QueryResponse)
- [x] T035 [P] [US1] Create CSS module physical-ai-textbook/src/components/RAGChatbot/styles.module.css for component styling
- [x] T036 [US1] Create custom hook physical-ai-textbook/src/hooks/useChatAPI.ts for API client (fetch /api/chat/query)
- [x] T037 [US1] Create custom hook physical-ai-textbook/src/hooks/useSessionStorage.ts for chat history persistence
- [x] T038 [US1] Swizzle Root component in physical-ai-textbook/src/theme/Root.tsx to add global chat widget
- [x] T039 [US1] Configure API proxy in physical-ai-textbook/docusaurus.config.ts (/api → http://localhost:8000/api)
- [x] T040 [US1] Add graceful degradation in useChatAPI.ts (handle 503, display "AI assistant temporarily offline")
- [x] T041 [US1] Add client-side rate limiting in useChatAPI.ts (3 req/min, display countdown timer on 429)
- [x] T042 [US1] Add loading states and error messages in ChatModal.tsx
- [ ] T043 [US1] Test full query flow: frontend → backend → Qdrant → Gemini → response with citations

**Checkpoint**: At this point, User Story 1 (core RAG chatbot) should be fully functional and testable independently. Students can ask questions and receive cited answers.

---

## Phase 4: User Story 2 - Get Context-Specific Help from Selected Text (Priority: P2)

**Goal**: Students can highlight text in the documentation, click "Ask AI about this," and receive contextually relevant explanations or simplifications.

**Independent Test**: Select a paragraph in a chapter, click "Ask AI about this selection," and verify that the response is contextually relevant to the selected passage with proper citations including the selected text source.

### Implementation for User Story 2

- [x] T044 [US2] Update POST /api/chat/query endpoint in backend/src/routers/chat.py to handle text_selection query_type with context parameter
- [x] T045 [US2] Implement context validation in backend/src/routers/chat.py (text, file_path, selection_range required when query_type is text_selection)
- [x] T046 [US2] Enhance RAG pipeline in backend/src/services/rag_pipeline.py to prepend selected text to prompt for context-aware generation
- [x] T047 [US2] Add special citation handling for selected text in backend/src/services/rag_pipeline.py ("[Selected Text, Week X]" with relevance_score 1.0)
- [x] T048 [P] [US2] Create custom hook physical-ai-textbook/src/hooks/useTextSelection.ts to track user text selection on documentation pages
- [x] T049 [US2] Add "Ask AI about this" button in physical-ai-textbook/src/components/RAGChatbot/index.tsx that appears on text selection
- [x] T050 [US2] Update useChatAPI.ts in physical-ai-textbook/src/hooks/ to support text_selection query_type with context parameter
- [x] T051 [US2] Update QueryInput.tsx to pre-fill with selected text context when "Ask AI about this" is clicked
- [ ] T052 [US2] Test text selection flow: select text → click button → query sent with context → response includes selected text citation

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Students can ask general questions AND get help with specific selected text.

---

## Phase 5: User Story 3 - Browse Conversation History (Priority: P3)

**Goal**: Students can scroll through their previous questions and answers in the chat interface, with history persisted across page refreshes using browser storage.

**Independent Test**: Ask 5 questions in the chat, scroll up to verify all exchanges are visible, then refresh the page and verify that conversation history is restored.

### Implementation for User Story 3

- [x] T053 [US3] Enhance useSessionStorage.ts in physical-ai-textbook/src/hooks/ to persist full conversation history to sessionStorage
- [x] T054 [US3] Update MessageList.tsx in physical-ai-textbook/src/components/RAGChatbot/ to load conversation history on mount from sessionStorage
- [x] T055 [US3] Add scroll restoration in MessageList.tsx (scroll to bottom on new messages, allow scroll up for history)
- [x] T056 [US3] Add "Clear history" button in ChatModal.tsx to reset conversation (clear sessionStorage)
- [x] T057 [US3] Add message timestamps in MessageList.tsx for each exchange
- [ ] T058 [US3] Test history persistence: ask questions → refresh page → verify history restored

**Checkpoint**: All user stories should now be independently functional. Students have full RAG chat experience with history.



## Phase 7: Database Integration for Session Storage (Priority: P2)

**Goal**: Integrate Neon Postgres for anonymous session storage to persist conversation history in the database (as mandated by Constitution Principle II).

**Independent Test**: Start a conversation, refresh the page, and verify that conversation history is restored from the database (not just sessionStorage).

### Implementation for Database Integration

- [x] T066 [P] Create backend/src/services/database_service.py with asyncpg client for Neon Postgres
- [x] T067 Create database migration script backend/scripts/setup_database.py with query_sessions and session_messages table schemas from data-model.md
- [x] T068 [P] Create backend/src/models/session.py with Pydantic models (QuerySession, SessionMessage)
- [x] T069 Implement session creation in backend/src/services/database_service.py (generate anonymous session_token)
- [x] T070 Implement message persistence in backend/src/services/database_service.py (save user and assistant messages)
- [x] T071 Implement message retrieval in backend/src/services/database_service.py (get session history by token)
- [x] T072 Update POST /api/chat/query in backend/src/routers/chat.py to persist messages to database after response
- [x] T073 Add session_id parameter to ChatQueryRequest in backend/src/models/chat.py (optional, auto-generated if missing)
- [x] T074 Update useChatAPI.ts in physical-ai-textbook/src/hooks/ to include session_id in requests
- [x] T075 Generate and store anonymous session_token in sessionStorage on first query
- [ ] T076 Add GET /api/chat/history endpoint in backend/src/routers/chat.py to retrieve conversation history by session_token
- [ ] T077 Update useSessionStorage.ts to fetch history from /api/chat/history on mount (fallback to sessionStorage if offline)
- [ ] T078 Test database persistence: ask questions → refresh page → verify history loaded from database

**Checkpoint**: Conversation history is now persisted in Neon Postgres, satisfying Constitution Principle II requirement.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T079 [P] Add comprehensive docstrings to all backend services (embedding_service.py, vector_store.py, llm_service.py, rag_pipeline.py, database_service.py)
- [ ] T080 [P] Add TypeScript JSDoc comments to all React components and hooks
- [ ] T081 Add logging middleware in backend/src/main.py for request/response tracking
- [ ] T082 [P] Add error logging in all backend services with structured logging
- [ ] T083 Update backend/README.md with deployment instructions for Vercel/Railway
- [ ] T084 [P] Create deployment configuration backend/Dockerfile for containerized deployment
- [ ] T085 [P] Create GitHub Actions workflow .github/workflows/backend-deploy.yml for backend deployment
- [ ] T086 [P] Update physical-ai-textbook README with RAG feature documentation
- [ ] T087 Add accessibility features: keyboard navigation (Tab to chat button, Esc to close modal) in ChatModal.tsx
- [ ] T088 Add ARIA labels to all chat components for screen reader support
- [ ] T089 Test responsive design on mobile devices (chat widget, modal sizing)
- [ ] T090 Run quickstart.md validation (setup backend → index docs → test queries)
- [ ] T091 Security scan: verify no API keys in codebase, check .env.example completeness
- [ ] T092 Performance optimization: verify p95 latency <3s for queries
- [ ] T093 Load test: verify 10-15 concurrent users without degradation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Admin Endpoint (Phase 6)**: Depends on Foundational - can proceed in parallel with user stories
- **Database Integration (Phase 7)**: Depends on Foundational + US1 completion - enhances US3
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1) - Ask Questions**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2) - Text Selection**: Depends on US1 completion (extends the query endpoint) - integrates with US1 components
- **User Story 3 (P3) - Conversation History**: Can start after Foundational (Phase 2) - independently implements browser storage feature
- **Admin Endpoint (Phase 6)**: Independent of user stories - can proceed in parallel
- **Database Integration (Phase 7)**: Depends on US1 and US3 - enhances both with persistent storage

### Within Each User Story

- Backend endpoint before frontend components (to enable API contract testing)
- Frontend hooks before components (hooks provide data/logic to UI)
- Individual React components can be built in parallel ([P] marked)
- Integration testing after all components complete

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel (T003, T004, T005)
- All Foundational Pydantic models marked [P] can run in parallel (T007, T008, T009, T011, T021)
- Within US1: All React components marked [P] can run in parallel (T029-T035)
- Within US2: Text selection hook can be built in parallel with backend changes
- All Polish tasks marked [P] can run in parallel (T079-T086)

---

## Parallel Example: User Story 1

```bash
# Launch all React components for User Story 1 together:
Task T029: "Create RAGChatbot/index.tsx (main chat widget)"
Task T030: "Create RAGChatbot/ChatModal.tsx"
Task T031: "Create RAGChatbot/MessageList.tsx"
Task T032: "Create RAGChatbot/QueryInput.tsx"
Task T033: "Create RAGChatbot/Citation.tsx"
Task T034: "Create types.ts"
Task T035: "Create styles.module.css"

# Then integrate with hooks sequentially (hooks depend on types):
Task T036: "Create useChatAPI.ts hook"
Task T037: "Create useSessionStorage.ts hook"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. **Run indexing**: Execute `backend/scripts/index_docs.py` to populate Qdrant
4. Complete Phase 3: User Story 1
5. **STOP and VALIDATE**: Test User Story 1 independently
   - Ask question: "What is DDS middleware?"
   - Verify response with citations
   - Test rate limiting (>3 req/min)
   - Test offline graceful degradation
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Run indexing → Documentation searchable
3. Add User Story 1 → Test independently → Deploy/Demo (MVP! Core RAG functionality)
4. Add User Story 2 → Test independently → Deploy/Demo (Enhanced with text selection)
5. Add User Story 3 → Test independently → Deploy/Demo (Full chat experience with history)
6. Add Database Integration → Test independently → Deploy/Demo (Persistent storage)
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. One developer runs indexing while others start on tasks
3. Once Foundational is done:
   - Developer A: User Story 1 (backend endpoint + RAG pipeline)
   - Developer B: User Story 1 (frontend components)
   - Developer C: User Story 2 (after US1 backend complete)
4. Stories complete and integrate independently

---

## Indexing Requirement

**⚠️ CRITICAL**: Before testing any user story, you MUST run the indexing script:

```bash
cd backend
python scripts/setup_qdrant.py  # Create Qdrant collection
python scripts/index_docs.py --docs-path ../physical-ai-textbook/docs/
```

This populates the vector database with textbook content. Without this, queries will return "I couldn't find information about that in the textbook."

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Tests are NOT included (not requested in spec) - focus is on implementation
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Remember: Gemini free tier has 5 RPM limit - LRU caching and rate limiting are critical

---

## Task Summary

- **Total Tasks**: 93
- **Setup Tasks**: 5
- **Foundational Tasks**: 16
- **User Story 1 Tasks**: 22 (MVP - core RAG chatbot)
- **User Story 2 Tasks**: 9 (text selection queries)
- **User Story 3 Tasks**: 6 (conversation history)
- **Admin Endpoint Tasks**: 7
- **Database Integration Tasks**: 13
- **Polish Tasks**: 15
- **Parallel Opportunities**: 23 tasks marked [P]
- **Suggested MVP Scope**: Setup + Foundational + User Story 1 (43 tasks)
