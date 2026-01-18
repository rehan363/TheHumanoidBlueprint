import asyncio
import asyncpg
import os
import sys

# Add the parent directory of backend/src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from services.database_service import db_service

async def setup_database():
    """
    Connects to the database and creates the necessary tables.
    Drops existing tables if they exist for easy development/reset.
    """
    print("Starting database setup...")

    try:
        await db_service.connect()
        print("Database service connected.")

        # Drop tables if they exist (for development/reset)
        await db_service.execute("DROP TABLE IF EXISTS session_messages CASCADE;")
        await db_service.execute("DROP TABLE IF EXISTS query_sessions CASCADE;")
        print("Existing tables (if any) dropped.")

        # Create query_sessions table
        await db_service.execute("""
            CREATE TABLE query_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_token VARCHAR(255) UNIQUE NOT NULL,
                user_id UUID NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                last_activity TIMESTAMP DEFAULT NOW(),
                message_count INT DEFAULT 0
            );
        """)
        print("Table 'query_sessions' created.")

        # Create session_messages table
        await db_service.execute("""
            CREATE TABLE session_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES query_sessions(id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                citations JSONB,
                query_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        print("Table 'session_messages' created.")

        # Create indexes
        await db_service.execute("CREATE INDEX idx_session_messages_session ON session_messages(session_id);")
        await db_service.execute("CREATE INDEX idx_query_sessions_token ON query_sessions(session_token);")
        await db_service.execute("CREATE INDEX idx_query_sessions_activity ON query_sessions(last_activity DESC);")
        print("Indexes created.")


        print("Database setup complete.")

    except Exception as e:
        print(f"An error occurred during database setup: {e}")
        sys.exit(1) # Exit with an error code

    finally:
        await db_service.disconnect()
        print("Database service disconnected.")

if __name__ == "__main__":
    # Load environment variables (e.g., NEON_DATABASE_URL)
    # from .env file if it exists, to allow local execution
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
        print("Loaded .env file.")
    except ImportError:
        print("python-dotenv not installed. Skipping .env loading.")
    except Exception as e:
        print(f"Error loading .env file: {e}")

    asyncio.run(setup_database())
