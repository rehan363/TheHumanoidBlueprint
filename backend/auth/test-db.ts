import { neon } from "@neondatabase/serverless";
import * as dotenv from "dotenv";
dotenv.config();

async function testConnection() {
    console.log("Testing connection to:", process.env.NEON_DATABASE_URL);
    const sql = neon(process.env.NEON_DATABASE_URL!);
    try {
        const result = await sql`SELECT 1 as test`;
        console.log("Connection successful:", result);
    } catch (error) {
        console.error("Connection failed:", error);
    }
}

testConnection();
