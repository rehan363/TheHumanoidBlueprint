# Backend Debugging Plan & Restoration Strategy

**Objective**: Systematically diagnose why the backend is "not working," populate missing data, and handle API limitations.

## Phase 1: Diagnostics (Environment & Connectivity)
The goal is to isolate whether the failure is network, authentication, or quota-related.

- [ ] **1.1. Validate Environment Variables**
  - Check `.env` for existence of:
    - `GEMINI_API_KEY`
    - `QDRANT_URL` & `QDRANT_API_KEY`
    - `NEON_DATABASE_URL` (for chat history)
  - *Action*: User to confirm keys are present in valid format.

- [ ] **1.2. Test Qdrant Cloud Connection**
  - **Task**: Run a connection test script.
  - **Success Criteria**: Successful handshake with Qdrant Cloud.
  - **Failure Mode**: Network timeout or 401 Unauthorized.

- [ ] **1.3. Test Gemini API (Quota & Validity)**
  - **Task**: Run a minimal "Hello World" generation script.
  - **Success Criteria**: Receive a text response.
  - **Failure Mode**: 
    - `401 Unauthorized`: Invalid Key.
    - `429 Too Many Requests`: Quota Exceeded.
  - **Decision Point**: If `429` is hit, immediately trigger **Phase 4 (OpenRouter Migration)**.

## Phase 2: Data Restoration (Vector Database)
The backend cannot work if the "brain" (Qdrant) is empty.

- [ ] **2.1. Inspect Collection Status**
  - **Task**: Check if collection `physical-ai-textbook` exists and verify vector count.
  - **Expected**: Collection exists, Count > 0.
  - *Symptom*: RAG returns "No relevant context found" for everything if count is 0.

- [ ] **2.2. Re-Initialize & Index (If needed)**
  - **Task**:
    1. Run `python scripts/setup_qdrant.py` (Recreate structure).
    2. Run `python scripts/index_docs.py` (Populate data).
  - *Verification*: Check vector count again.

## Phase 3: Service Integration Check
Verify the application logic handles the connections correctly.

- [ ] **3.1. Test Database Service (Neon Postgres)**
  - **Task**: Attempt to create a dummy session in the database.
  - **Fix**: Run `scripts/setup_database.py` if tables are missing.

- [ ] **3.2. Test Full RAG Pipeline**
  - **Task**: Run an internal query via python shell (bypassing HTTP layer) to verify `process_query` returns citations.

## Phase 4: Contingency - OpenRouter Migration (If Gemini Failed)
Trigger this ONLY if Step 1.3 fails due to quota or preference.

- [ ] **4.1. Configuration Update**
  - Add `OPENROUTER_API_KEY` to `.env`.
  - Update `LLM_MODEL` to an OpenRouter supported model (e.g., `google/gemini-2.0-flash-exp:free` or `deepseek/deepseek-chat`).

- [ ] **4.2. Refactor `llm_service.py`**
  - Modify `LLMService` to use `openai` client (OpenAI-compatible endpoint) instead of `google.generativeai` SDK.
  - Set `base_url="https://openrouter.ai/api/v1"`.

- [ ] **4.3. Validation**
  - Rerun generation test with new provider.

## Execution Order
1. Run **Phase 1** checks immediately.
2. If Qdrant is empty, run **Phase 2**.
3. If Gemini is broken, execute **Phase 4**.
4. Finally, verify with **Phase 3**.
