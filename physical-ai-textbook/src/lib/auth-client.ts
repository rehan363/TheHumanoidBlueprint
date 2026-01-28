import { createAuthClient } from "better-auth/react";

export const authClient = createAuthClient({
    // In production this will be the deployed URL, in dev it points to our node sidecar
    baseURL: "http://localhost:4000"
});

export const { signIn, signUp, signOut, useSession } = authClient;
