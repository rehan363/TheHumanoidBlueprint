# Tasks: 003-authentication (STRICT COMPLIANCE)

## Milestone 1: Better-Auth Node.js Initialization
- [x] **T1.1**: Create `backend/auth` directory.
- [x] **T1.2**: Initialize Node project and install `@better-auth/cli`, `better-auth`, `drizzle-orm`, `@neondatabase/serverless`.
- [x] **T1.3**: Configure `auth.ts` with custom user fields: `python_experience` (integer), `hardware_experience` (integer), `onboarding_complete` (boolean).
- [x] **T1.4**: Run `npx @better-auth/cli generate` to create the compliant schema.

## Milestone 2: Python Session Bridge (uv)
- [x] **T2.1**: Update `backend/pyproject.toml` with `asyncpg` or `psycopg2` if not present (using `uv add`).
- [x] **T2.2**: Create `rag_backend/services/auth_verifier.py`.
- [x] **T2.3**: Implement `verify_session(token: str)` that queries the `session` table in Neon.
- [x] **T2.4**: Implement `get_user_profile(user_id: str)` to fetch background data for personalization.

## Milestone 3: Frontend Component Swizzling
- [x] **T3.1**: Swizzle Docusaurus `Root` for the global `AuthProvider`.
- [x] **T3.2**: Create custom `RegisterForm` that includes sliders for Python and Hardware background.
- [x] **T3.3**: Update `Navbar` with an `AuthButton` for Login/Logout state.

## Milestone 4: Personalization Logic
- [x] **T4.1**: Modify the RAG System Prompt to include user profile data: *"The user has [X] level of Python and [Y] level of Hardware knowledge. Adjust your explanation accordingly."*
- [x] **T4.2**: Verify that Urdu translation (bonus) only triggers for authenticated users if desired.

## Verification Checklist
- [ ] Registration in Node server reflects correctly in the Neon database.
- [ ] Python backend successfully retrieves the same user record.
- [ ] Chatbot responses change when background levels are modified.
