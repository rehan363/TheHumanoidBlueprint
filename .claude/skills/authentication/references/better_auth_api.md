# Better-Auth API Reference

## Server API (`auth.ts`)

The primary entry point for server-side auth logic.

### 1. Initialization
```typescript
import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { db } from "./db";

export const auth = betterAuth({
    database: drizzleAdapter(db, {
        provider: "pg", // "mysql", "sqlite"
    }),
    emailAndPassword: {
        enabled: true
    },
    socialProviders: {
        github: {
            clientId: process.env.GITHUB_CLIENT_ID,
            clientSecret: process.env.GITHUB_CLIENT_SECRET,
        }
    }
});
```

### 2. Session Management
- `auth.api.getSession(req)`: Get current session from request.
- `auth.api.listSessions(userId)`: List all active sessions for a user.
- `auth.api.revokeSession(sessionId)`: Revoke a specific session.

---

## Client API (`auth-client.ts`)

Hooks and methods for the frontend.

### 1. Hooks
- `useSession()`: Returns `{ data: Session, isPending: boolean, error: Error }`.
- `useListSessions()`: Returns active sessions for the user.

### 2. Auth Actions
- `signIn.email({ email, password, callbackURL })`
- `signIn.social({ provider, callbackURL })`
- `signUp.email({ email, password, name, image, role, ...customFields })`
- `signOut()`

### 3. Account Actions
- `updateUser({ name, image, ... })`
- `changePassword({ newPassword, currentPassword })`
- `linkAccount({ provider })`

---

## Drizzle Adapter Specifics

Better-Auth expects specific table structures. Use the CLI to generate them or see `database_schema.md`.

```typescript
// Customizing table names
database: drizzleAdapter(db, {
    provider: "pg",
    usePlural: true, // Uses 'users' instead of 'user'
    mapping: {
        user: "custom_user_table"
    }
})
```
