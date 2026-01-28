import { neon } from "@neondatabase/serverless";

async function testConnection() {
    // Modified URL: removed -pooler and channel_binding
    const url = "postgresql://neondb_owner:npg_w2ns1Sqcgktu@ep-frosty-block-ad9kj8rc.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require";
    console.log("Testing connection to:", url);
    const sql = neon(url);
    try {
        const result = await sql`SELECT 1 as test`;
        console.log("Connection successful:", result);
    } catch (error) {
        console.error("Connection failed:", error);
    }
}

testConnection();
