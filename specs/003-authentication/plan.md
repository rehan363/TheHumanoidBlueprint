# Plan: 003-authentication (STRICT COMPLIANCE)

## Architecture: Dual-Service Unified Database
To strictly follow Requirement #5, we will deploy a **Hybrid Backend**.

### 1. Better-Auth Service (`backend/auth`)
- **Runtime**: Node.js managed by `npm`.
- **Primary Tool**: Better-Auth SDK.
- **Database**: Drizzle Adapter connecting to Neon Postgres.
- **Responsibility**: Signup (including custom background fields), Login, Session management, and CORS protection.

### 2. RAG Backend (`backend/rag_backend`)
- **Runtime**: Python managed by `uv`.
- **Database**: Shared access to the same Neon Postgres instance.
- **Responsibility**: Retrieving user background from the `user` table using the session token provided by the frontend.

## Implementation Workflow

### Phase 1: Better-Auth Foundation (Node.js)
- [ ] Create `backend/auth` workspace.
- [ ] Configure `auth.ts` with custom fields for: `software_background`, `hardware_background`.
- [ ] Initialize the Neon Postgres schema via Better-Auth CLI.

### Phase 2: Python Integration (uv)
- [ ] Implement a `SessionValidator` utility in Python.
- [ ] This utility will directly query the `session` and `user` tables in Neon to verify the frontend's token.
- [ ] Update `rag_backend` config to include `BETTER_AUTH_SECRET` if needed for shared logic.

### Phase 3: Frontend Swizzling (Docusaurus)
- [ ] Add `@better-auth/react` to the Docusaurus project.
- [ ] Create the Signup/Login UI components with the custom background questionnaire.
- [ ] Swizzle the `Root` component to wrap the textbook in an `AuthProvider`.

## Security & Compliance
- **ADR 009: Strict Tooling**: Use Better-Auth exclusively for IDP (Identity Provider) functionality to ensure hackathon eligibility.
- **ADR 010: Database Share**: Both services connect to Neon. Use connection pooling (PgBouncer/Neon Pooler) to manage the dual-connection load.
