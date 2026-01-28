import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as dotenv from "dotenv";
import * as schema from "./auth-schema";

dotenv.config();

const sql = neon(process.env.NEON_DATABASE_URL!);
const db = drizzle(sql, { schema });

export const auth = betterAuth({
    database: drizzleAdapter(db, {
        provider: "pg",
        schema: {
            ...schema
        }
    }),
    // The base URL of the auth server itself
    baseURL: process.env.BETTER_AUTH_URL || "http://localhost:4000",
    // Origins allowed to access this auth server
    trustedOrigins: [
        "http://localhost:3000",
        "https://rehan363.github.io"
    ],
    emailAndPassword: {
        enabled: true,
    },
    user: {
        additionalFields: {
            python_experience: {
                type: "number",
                defaultValue: 0,
            },
            hardware_experience: {
                type: "number",
                defaultValue: 0,
            },
            onboarding_complete: {
                type: "boolean",
                defaultValue: false,
            }
        }
    },
    // We can add social providers (GitHub, Google) here if needed later
});
