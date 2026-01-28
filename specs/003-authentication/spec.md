# Specification: 003-authentication (STRICT COMPLIANCE)

## Overview
Implement the authentication system strictly following **Hackathon Requirement #5**, which mandates the use of **Better-Auth** for signup and signin. This system will be integrated with the existing Python RAG backend while maintaining a dedicated Better-Auth service for full requirement compliance.

## Goals
- **Strict Compliance**: Implement signup/signin using [Better-Auth](https://www.better-auth.com/).
- **Data Collection**: Gather user background data (hardware/software) during signup for personalization.
- **Unified Identity**: Enable the Python RAG backend to recognize the Better-Auth session for localized explanations.
- **Traceability**: All auth tables must exist in the shared Neon Postgres database as specified in the Better-Auth documentation.

## Success Criteria (Hackathon Scoring)
- [ ] **Bonus Points (+50)**: Better-Auth is functional and managing sessions.
- [ ] **Bonus Points (+50)**: Personalization based on background (Python/Hardware) is implemented.
- [ ] **Core Requirement**: Session persistence in Neon Postgres is verified.

## Tech Stack
- **Authentication Engine**: Better-Auth (Node.js/TypeScript).
- **RAG Engine**: FastAPI (Python/uv).
- **Shared Database**: Neon Serverless Postgres.
- **Frontend**: Docusaurus 3 (React).

## User Stories
- **US1**: As a user, I sign up using the Better-Auth form. I answer questions about my Python and Hardware experience.
- **US2**: As an authenticated user, my session is stored in the `session` table in Neon.
- **US3**: As a user, when I ask the RAG chatbot a question, the Python backend reads my background from the database to simplify or enrich the explanation.

## Constraints
- **Library Lock-in**: Better-Auth MUST be used for the auth engine (Requirement #5).
- **Data Sovereignty**: User profile and background data must be stored in the primary `user` table.
- **Environment**: Must support cross-origin authentication from GitHub Pages to the deployed backend.
