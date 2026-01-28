import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { auth } from "./auth";

const app = new Hono();

// Enable CORS for frontend access (Local and GitHub Pages)
app.use("*", cors({
    origin: ["http://localhost:3000", "https://rehan363.github.io"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
    exposeHeaders: ["set-cookie"],
    credentials: true,
}));

// Root route to show the server is alive
app.get("/", (c) => {
    return c.text("Better-Auth Sidecar is running! Use /api/auth for authentication.");
});

app.on(["POST", "GET"], "/api/auth/*", (c) => {
    return auth.handler(c.req.raw);
});

console.log("Better-Auth server running on http://localhost:4000");

serve({
    fetch: app.fetch,
    port: 4000,
});
