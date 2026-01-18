# Backend Debugging Summary - 2026-01-18

## 🔍 Diagnostic Results

### Phase 1: Environment & Connectivity - COMPLETED ✓

| Component | Status | Details |
|-----------|--------|---------|
| **Environment Variables** | ✅ FIXED | `.env` had `GEMINI_API_KEY_1` instead of `GEMINI_API_KEY` - now corrected |
| **Qdrant Connection** | ❌ FAILED | 404 Not Found - Cluster appears to be deleted or URL is invalid |
| **Gemini API** | ❌ FAILED | 400 API_KEY_INVALID - All 3 API keys appear to be invalid/expired |
| **Neon Database** | ⏸️ NOT TESTED | Will test after fixing Qdrant and Gemini |

---

## 🚨 Critical Issues Found

### Issue #1: Qdrant Cluster Not Found (404)
**Error**: `Unexpected Response: 404 (Not Found)`

**Root Cause**: The Qdrant cluster at the URL in `.env` doesn't exist or was deleted.

**Impact**: 
- Vector search completely broken
- RAG system cannot retrieve context
- All queries will fail with "No relevant content found"

**Resolution Options**:
1. **Check Qdrant Cloud Dashboard** - Verify if cluster exists
2. **Create New Cluster** - If deleted, create new free tier cluster
3. **Update `.env`** - Update `QDRANT_URL` and `QDRANT_API_KEY` with new cluster credentials

---

### Issue #2: Invalid Gemini API Keys
**Error**: `400 API key not valid. Please pass a valid API key.`

**Root Cause**: All 3 Gemini API keys in `.env` are invalid/expired/revoked.

**Impact**:
- LLM generation completely broken
- Cannot generate responses to user queries
- Backend will return 503 Service Unavailable

**Resolution Options**:
1. **Generate New Gemini Key** - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Switch to OpenRouter** - Use free models via OpenRouter (recommended for reliability)

---

## 📋 Next Steps (Priority Order)

### IMMEDIATE (Blocking)

- [ ] **1. Fix Gemini API Key**
  - **Option A**: Get new Gemini API key from Google AI Studio
  - **Option B**: Switch to OpenRouter (RECOMMENDED)
    - Add `OPENROUTER_API_KEY` to `.env`
    - Modify `llm_service.py` to use OpenRouter endpoint
    - Use free model: `google/gemini-2.0-flash-exp:free` or `deepseek/deepseek-chat`

- [ ] **2. Fix Qdrant Cluster**
  - Login to [Qdrant Cloud](https://cloud.qdrant.io/)
  - Check if cluster exists
  - If not, create new free tier cluster
  - Update `.env` with new credentials
  - Run `python scripts/setup_qdrant.py` to initialize collection

### AFTER FIXES

- [ ] **3. Populate Vector Database**
  - Run `python scripts/index_docs.py --docs-path ../physical-ai-textbook/docs`
  - Verify vectors are indexed

- [ ] **4. Test Neon Database**
  - Test connection to Postgres
  - Run `python scripts/setup_database.py` if needed

- [ ] **5. Integration Test**
  - Start backend: `uvicorn rag_backend.main:app --reload`
  - Test `/api/health` endpoint
  - Test `/api/chat/query` with sample question

---

## 🔧 Scripts Created for Debugging

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/debug_connectivity.py` | Full connectivity check | ✅ Complete |
| `scripts/test_qdrant.py` | Detailed Qdrant diagnostics | ✅ Complete |
| `scripts/test_gemini.py` | Gemini API validation | ✅ Complete |

---

## 💡 Recommendation: Switch to OpenRouter

**Why OpenRouter?**
- ✅ More reliable than free Gemini tier
- ✅ No quota issues (better rate limits)
- ✅ Multiple free models available
- ✅ OpenAI-compatible API (easy migration)

**Migration Effort**: ~15 minutes (modify `llm_service.py` only)

---

## 📝 User Action Required

**Please choose one of the following paths:**

### Path A: Fix Gemini + Qdrant (Traditional)
1. Get new Gemini API key from Google AI Studio
2. Login to Qdrant Cloud and verify/create cluster
3. Update `.env` with new credentials

### Path B: Switch to OpenRouter + Fix Qdrant (RECOMMENDED)
1. Get OpenRouter API key from [openrouter.ai](https://openrouter.ai/)
2. Login to Qdrant Cloud and verify/create cluster
3. Update `.env` and modify `llm_service.py`

**Which path would you like to take?**
