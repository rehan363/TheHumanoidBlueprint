---
name: authentication
description: |
  Production-grade implementation of Better-Auth with Neon Postgres and Drizzle ORM.
  This skill should be used when implementing secure signup, login, session management,
  and role-based access control for the Physical AI & Robotics platform.
---

# Authentication

A comprehensive guide and toolset for implementing end-to-end authentication using Better-Auth, Drizzle ORM, and Neon Postgres.

## Before Implementation

Gather context to ensure successful implementation:

| Source | Gather |
|--------|--------|
| **Codebase** | Project structure (Mono-repo vs separate), existing Drizzle config, Docusaurus setup |
| **Conversation** | Auth providers needed (Email, GitHub, etc.), role requirements, specific onboarding fields |
| **Skill References** | Drizzle schemas, Auth client configuration, Docusaurus integration patterns |
| **User Guidelines** | .env naming conventions, security standards |

## Workflows

### 1. Project Initialization
- Ensure `Better-Auth`, `Drizzle-ORM`, and `@neondatabase/serverless` are installed.
- Configure `.env` with `DATABASE_URL`, `BETTER_AUTH_SECRET`, and `BETTER_AUTH_URL`.

### 2. Database & Schema Setup
- Define the auth schema in `src/db/schema.ts` (see `references/database_schema.md`).
- Initialize Drizzle with Neon HTTP/Websocket based on environment.
- Run `npx @better-auth/cli generate` and apply migrations.

### 3. Auth Server Configuration
- Set up `auth.ts` library to initialize Better-Auth with the Drizzle adapter.
- Configure plugins (OIDC if needed, Email/Password, etc.).

### 4. Client-Side Integration
- Initialize the Better-Auth client in the frontend.
- Swizzle Docusaurus `Root` or `Navbar` to provide auth context (see `references/frontend_integration.md`).
- Implement Login/Signup UI components with proper error handling.

## Component Patterns

### Auth Client Setup
```typescript
import { createAuthClient } from "better-auth/react";
export const authClient = createAuthClient({
    baseURL: process.env.BETTER_AUTH_URL
});
```

### Protected Route Hook
```typescript
export function useRequireAuth() {
  const { data: session, isPending } = authClient.useSession();
  useEffect(() => {
    if (!isPending && !session) {
      window.location.href = "/login";
    }
  }, [session, isPending]);
  return { session, isPending };
}
```

## References

| File | Content |
|------|---------|
| `references/database_schema.md` | Drizzle schema definitions for People, Sessions, and Accounts |
| `references/frontend_integration.md` | Docusaurus swizzling and React context patterns |
| `references/better_auth_api.md` | Core API reference for server and client |
| `references/security_best_practices.md` | PKCE, CORS, and Secret management |

## Scripts

- `scripts/validate_env.py` - Validates that all required environment variables are set.

## Common Pitfalls

- **Neon Cold Starts**: Always use the serverless-aware Drizzle drivers for edge functions.
- **CORS Mismatch**: `BETTER_AUTH_URL` must match the actual frontend URL exactly.
- **Docusaurus SSR**: Ensure auth client only runs on the client-side to avoid hydration mismatches.

## Implementation Checklist

- [ ] `.env` variables verified
- [ ] Drizzle migrations applied and tables exist in Neon
- [ ] `auth.ts` exported and correctly configured
- [ ] Frontend `authClient` initialized
- [ ] `Root` component swizzled for global auth state
- [ ] Login/Logout flows tested end-to-end
- [ ] Role-based access verified (if applicable)
