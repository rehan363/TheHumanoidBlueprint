from typing import Optional, Dict, Any
from asyncpg import Pool
import logging
from rag_backend.services.database_service import db_service

logger = logging.getLogger(__name__)

class AuthVerifier:
    """
    Service for verifying Better-Auth sessions in Python/FastAPI.
    Reads directly from the shared Neon Postgres database.
    """
    
    async def verify_session(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verifies a session token against the Better-Auth 'session' table.
        Returns user data and background if valid, else None.
        """
        # Note: Better-Auth/Drizzle columns are often camelCase and require quotes in Postgres
        query = """
            SELECT 
                u.id, u.name, u.email, 
                u.python_experience, u.hardware_experience, 
                s."expiresAt"
            FROM "session" s
            JOIN "user" u ON s."userId" = u.id
            WHERE s.token = $1 AND s."expiresAt" > NOW();
        """
        try:
            record = await db_service.fetchrow(query, token)
            if record:
                return dict(record)
            return None
        except Exception as e:
            logger.error(f"Error verifying authentication session: {e}", exc_info=True)
            return None

auth_verifier = AuthVerifier()
