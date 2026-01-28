# Security Best Practices for Better-Auth

## 1. Environment Variables

Never hardcode secrets. Ensure these are in your `.env`:

| Variable | Requirement |
|----------|-------------|
| `BETTER_AUTH_SECRET` | 32+ characters, random string |
| `BETTER_AUTH_URL` | Canonical URL of your auth server |
| `DATABASE_URL` | Neon connection string with SSL |

## 2. CORS and Trusted Origins

Better-Auth protects against CSRF and unauthorized redirects.

```typescript
// auth.ts config
export const auth = betterAuth({
    // ...
    trustedOrigins: [
        "http://localhost:3000",
        "https://your-production-domain.com"
    ],
    // For Docusaurus/SPAs, ensure advanced security is on
    advanced: {
        disableDefaultCookieCache: true, // Recommended for SPAs
    }
});
```

## 3. PKCE (Proof Key for Code Exchange)

When using OAuth providers (Google, GitHub) in a frontend-only app (like Docusaurus):
- Always use the Authorization Code Flow with PKCE.
- Better-Auth handles this automatically if you use their client SDK.
- If manually constructing URLs, store `code_verifier` in `sessionStorage`.

## 4. Session Security

- **Session Expiry**: Keep session durations reasonable (e.g., 30 days for 'Remember Me', 24 hours otherwise).
- **Secure Cookies**: Better-Auth uses `HttpOnly`, `Secure`, and `SameSite=Lax` by default.
- **CSRF**: Ensure `BETTER_AUTH_URL` is configured correctly to enable proper CSRF protection markers.

## 5. Rate Limiting

Implement rate limiting on your auth endpoints (especially `/signin` and `/signup`) at the load balancer or application layer to prevent brute-force attacks.

## 6. Role-Based Access Control (RBAC)

- Never trust role information from the client.
- Always verify roles on the server-side before performing sensitive operations or returning protected data.
- Use `session.user.role` from the verified session object only.
