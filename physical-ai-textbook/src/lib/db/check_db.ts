import { neon } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-http';
import * as schema from './schema';
import * as dotenv from 'dotenv';

dotenv.config();

async function main() {
    const sql = neon(process.env.DATABASE_URL!);
    const db = drizzle(sql, { schema });

    try {
        const users = await db.select().from(schema.user).limit(1);
        print("Successfully connected to DB and queried 'user' table.");
        print("Existing users count: " + users.length);
    } catch (e) {
        print("Failed to query DB: " + e.message);
    }
}

function print(msg: string) {
    console.log(msg);
}

main();
