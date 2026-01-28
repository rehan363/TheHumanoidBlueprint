# Docusaurus Frontend Integration

Integrating Better-Auth with Docusaurus requires careful handling of React lifecycle and Docusaurus's static/dynamic nature.

## 1. Auth Client Initialization

Create a shared client instance.

```typescript
// src/lib/auth-client.ts
import { createAuthClient } from "better-auth/react";

export const authClient = createAuthClient({
  baseURL: "http://localhost:3000", // Should be env var
});

export const { useSession, signIn, signOut, signUp } = authClient;
```

## 2. Global Provider (Swizzling Root)

To make auth state available everywhere, swizzle the `Root` component.

```bash
npm run swizzle @docusaurus/theme-classic Root -- --danger
```

```tsx
// src/theme/Root.tsx
import React from 'react';
import { AuthProvider } from '../context/AuthContext';

export default function Root({children}) {
  return (
    <AuthProvider>
      {children}
    </AuthProvider>
  );
}
```

## 3. Navbar Integration (Swizzling NavbarItem)

```tsx
// src/theme/NavbarItem/ComponentTypes.js
// Add custom AuthButton here or swizzle specific items
```

### AuthButton Component
```tsx
import React from 'react';
import { useSession, signOut } from '../../lib/auth-client';
import Link from '@docusaurus/Link';

export default function AuthButton() {
  const { data: session, isPending } = useSession();

  if (isPending) return <div>Loading...</div>;

  if (session) {
    return (
      <div className="navbar__item dropdown dropdown--hoverable">
        <a className="navbar__link">👤 {session.user.name}</a>
        <ul className="dropdown__menu">
          <li><a className="dropdown__link" onClick={() => signOut()}>Logout</a></li>
        </ul>
      </div>
    );
  }

  return <Link to="/login" className="button button--primary">Sign In</Link>;
}
```

## 4. Protecting Pages

Use a wrapper component or a hook.

```tsx
// src/components/ProtectedRoute.tsx
import React from 'react';
import { useSession } from '../lib/auth-client';
import Layout from '@theme/Layout';

export default function ProtectedRoute({ children }) {
  const { data: session, isPending } = useSession();

  if (isPending) return <Layout><div>Loading...</div></Layout>;
  
  if (!session) {
    if (typeof window !== 'undefined') {
      window.location.href = '/login';
    }
    return null;
  }

  return <>{children}</>;
}
```

## 5. Handling SSR

Docusaurus builds are static. LocalStorage/Cookies are only available in the browser.
Always check `typeof window !== 'undefined'` if doing manual redirects outside of `useEffect`.
Using Better-Auth hooks is safe as they handle this internally.
