import asyncpg
import os
import json
import logging # Added logging import
from typing import Optional, List, Literal
from uuid import UUID, uuid4
from datetime import datetime

from rag_backend.models.session import QuerySession, SessionMessage, Citation # Import QuerySession

logger = logging.getLogger(__name__) # Initialize logger

from rag_backend.config import settings

class DatabaseService:
    """
    Service for managing database connections and operations using asyncpg.
    Connects to a Neon Postgres database using the NEON_DATABASE_URL environment variable.
    """
    _instance: Optional['DatabaseService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseService, cls).__new__(cls)
            cls._instance.connection_pool: Optional[asyncpg.Pool] = None
        return cls._instance

    async def connect(self):
        """
        Establishes a connection pool to the Neon Postgres database.
        """
        if self.connection_pool:
            return

        db_url = settings.neon_database_url
        if not db_url:
            logger.error("NEON_DATABASE_URL environment variable not set.", extra={"error_type": "ConfigurationError"})
            raise ValueError("NEON_DATABASE_URL environment variable not set.")

        try:
            self.connection_pool = await asyncpg.create_pool(db_url)
            logger.info("Successfully connected to Neon Postgres database.", extra={"db_url_prefix": db_url.split('@')[-1] if '@' in db_url else db_url})
        except Exception as e:
            logger.error(
                f"Failed to connect to Neon Postgres database: {e}",
                exc_info=True,
                extra={"error_type": "DatabaseConnectionError", "db_url_prefix": db_url.split('@')[-1] if '@' in db_url else db_url, "exception_message": str(e)}
            )
            raise

    async def disconnect(self):
        """
        Closes the database connection pool.
        """
        if self.connection_pool:
            logger.info("Disconnecting from Neon Postgres database.", extra={"status": "disconnecting"})
            await self.connection_pool.close()
            self.connection_pool = None

    async def execute(self, query: str, *args):
        """
        Executes a database query without returning results (e.g., INSERT, UPDATE, DELETE).
        """
        if not self.connection_pool:
            await self.connect()

        try:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(query, *args)
        except Exception as e:
            logger.error(
                f"Failed to execute query: {query[:100]}... Error: {e}",
                exc_info=True,
                extra={"error_type": "DatabaseExecuteError", "query_preview": query[:100], "exception_message": str(e)}
            )
            raise

    async def fetch(self, query: str, *args):
        """
        Executes a database query and returns all results (e.g., SELECT).
        """
        if not self.connection_pool:
            await self.connect()

        try:
            async with self.connection_pool.acquire() as connection:
                return await connection.fetch(query, *args)
        except Exception as e:
            logger.error(
                f"Failed to fetch results for query: {query[:100]}... Error: {e}",
                exc_info=True,
                extra={"error_type": "DatabaseFetchError", "query_preview": query[:100], "exception_message": str(e)}
            )
            raise

    async def fetchrow(self, query: str, *args):
        """
        Executes a database query and returns a single row.
        """
        if not self.connection_pool:
            await self.connect()

        try:
            async with self.connection_pool.acquire() as connection:
                return await connection.fetchrow(query, *args)
        except Exception as e:
            logger.error(
                f"Failed to fetch row for query: {query[:100]}... Error: {e}",
                exc_info=True,
                extra={"error_type": "DatabaseFetchRowError", "query_preview": query[:100], "exception_message": str(e)}
            )
            raise

    async def create_session(self) -> QuerySession:
        """
        Creates a new anonymous chat session in the database.
        Returns the created QuerySession object.
        """
        session_token = str(uuid4()) # Generate a unique token
        query = """
            INSERT INTO query_sessions (session_token, created_at, last_activity, message_count)
            VALUES ($1, $2, $3, $4)
            RETURNING id, session_token, user_id, created_at, last_activity, message_count;
        """
        # Use datetime.utcnow() for timezone-naive timestamps in UTC
        now = datetime.utcnow()
        record = await self.fetchrow(query, session_token, now, now, 0)

        if record:
            return QuerySession(
                id=record['id'],
                session_token=record['session_token'],
                user_id=record['user_id'],
                created_at=record['created_at'],
                last_activity=record['last_activity'],
                message_count=record['message_count']
            )
        else:
            error_msg = "Failed to create new session."
            logger.error(
                error_msg,
                extra={"error_type": "CreateSessionError", "session_token": session_token}
            )
            raise Exception(error_msg)

    async def save_message(
        self,
        session_id: UUID,
        role: Literal['user', 'assistant'],
        content: str,
        citations: Optional[List[Citation]] = None,
        query_type: Optional[Literal['full_text', 'text_selection']] = None
    ) -> SessionMessage:
        """
        Saves a chat message to the database and updates the session's activity.
        """
        try:
            if not self.connection_pool:
                await self.connect()

            async with self.connection_pool.acquire() as connection:
                async with connection.transaction():
                    # Convert citations to JSON-serializable format
                    citations_json = None
                    if citations:
                        citations_json = json.dumps([c.dict() if hasattr(c, 'dict') else c.model_dump() for c in citations])
                    
                    insert_query = """
                        INSERT INTO session_messages (session_id, role, content, citations, query_type, created_at)
                        VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                        RETURNING id, session_id, role, content, citations, query_type, created_at;
                    """
                    now = datetime.utcnow()
                    message_record = await connection.fetchrow(
                        insert_query, session_id, role, content, citations_json, query_type, now
                    )

                    update_session_query = """
                        UPDATE query_sessions
                        SET message_count = message_count + 1, last_activity = $1
                        WHERE id = $2;
                    """
                    await connection.execute(update_session_query, now, session_id)

                    if message_record:
                        citations_data = message_record['citations']
                        if isinstance(citations_data, str):
                            citations_data = json.loads(citations_data)
                            
                        return SessionMessage(
                            id=message_record['id'],
                            session_id=message_record['session_id'],
                            role=message_record['role'],
                            content=message_record['content'],
                            citations=[Citation(**c) for c in citations_data] if citations_data else None,
                            query_type=message_record['query_type'],
                            created_at=message_record['created_at']
                        )
                    else:
                        error_msg = "Failed to save message."
                        logger.error(
                            error_msg,
                            extra={"error_type": "SaveMessageError", "session_id": str(session_id), "role": role, "query_type": query_type, "content_preview": content[:100]}
                        )
                        raise Exception(error_msg)
        except Exception as e:
            logger.error(
                f"Failed to save message for session {session_id}: {e}",
                exc_info=True,
                extra={"error_type": "SaveMessageException", "session_id": str(session_id), "role": role, "query_type": query_type, "exception_message": str(e)}
            )
            raise

    async def get_session_messages(self, session_id: UUID) -> List[SessionMessage]:
        """
        Retrieves all messages for a given session, ordered by creation time.
        """
        try:
            if not self.connection_pool:
                await self.connect()

            query = """
                SELECT id, session_id, role, content, citations, query_type, created_at
                FROM session_messages
                WHERE session_id = $1
                ORDER BY created_at;
            """
            records = await self.fetch(query, session_id)

            messages = []
            for record in records:
                citations_data = record['citations']
                if isinstance(citations_data, str):
                    citations_data = json.loads(citations_data)

                messages.append(SessionMessage(
                    id=record['id'],
                    session_id=record['session_id'],
                    role=record['role'],
                    content=record['content'],
                    citations=[Citation(**c) for c in citations_data] if citations_data else None,
                    query_type=record['query_type'],
                    created_at=record['created_at']
                ))
            return messages

        except Exception as e:
            logger.error(
                f"Failed to retrieve messages for session {session_id}: {e}",
                exc_info=True,
                extra={"error_type": "GetSessionMessagesError", "session_id": str(session_id), "exception_message": str(e)}
            )
            raise


# Global instance to be used throughout the application
db_service = DatabaseService()
